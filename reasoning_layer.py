"""
reasoning_layer.py - the brain of Deep OQ: reliability, human tone, and an AI agent panel.

Files (keep them in the same folder as deep_oq.py):
    providers.py        Anthropic / OpenAI / Gemini / Groq over plain HTTPS, model auto-pick, failover
    solvers.py          deterministic solvers (calculus, algebra, linear algebra, number theory,
                        statistics, finance, units, puzzles ...), each re-checked a second way
    reasoning_layer.py  this file: loop removal, completion, depth, agent panel, verification

Quick start (set your API keys as environment variables first, see providers.py):

    from reasoning_layer import build_engine
    engine = build_engine()                       # 7 agents by default; build_engine(panel_agents=30) for more
    reply = engine.respond(history, user_message) # history = [{"role": "user"|"assistant", "content": str}, ...]
    reply.text         # show this
    reply.confidence   # "high" | "medium" | "low" | "n/a"
    reply.notes        # for your logs only (which solver/agents ran); do not show users

What happens to each message:
  1. "1) ... 2) ..." questions are split and each part is handled on its own.
  2. If a deterministic solver recognises the problem and its independent check passes, a real
     computed result is handed to the lead model, which explains it in a human voice.
     (If the check fails, the result is thrown away.)
  3. Other maths / logic / science questions go to the agent panel. There are 65 agent roles
     (12 generalists + 53 domain specialists: calculus analyst, organic chemist, debugger,
     economist, Hindi communicator ...). A router picks the specialists that fit the question,
     agents on different providers solve independently, answers are weight-voted, a split vote
     triggers a debate round, a program is run as a cross-check, and the lead model judges.
     Disagreement lowers the confidence and the reply says so honestly.
  4. Normal conversation gets full-depth, human replies: loops removed, cut-off answers
     continued, thin one-liners expanded.

Limits: reply length, continuations (40), panel size, workers (64), debate size and the multi-part
cap (40) are all parameters of ReasoningEngine / build_engine, and timeouts and the token ceiling are
environment variables (see providers.py). Two limits are deliberately kept: the sandbox around
model-written code (blocked imports, CPU/memory/file caps) and the input checks in solvers.py.

Run `python reasoning_layer.py --selftest` for the offline tests.
"""
from __future__ import annotations

import collections
import concurrent.futures as cf
import itertools
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

try:  # optional modules living next to this file
    import solvers
except Exception:  # noqa: BLE001
    solvers = None

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

# Edit this block to say whatever you want Deep OQ to say about itself.
IDENTITY = (
    "You are Deep OQ, an AI assistant. When someone asks who you are or what you can do, "
    "give a real, friendly introduction in several natural paragraphs, not one line. Cover: "
    "who you are; what you can do - Chat (talk anything through), Code (write, debug and "
    "explain code), Photo and Video (create and work with images and video), Lab (hands-on "
    "experiments and tools), News, Music, VR, Knowledge (your own saved documents), Memory "
    "(remembering useful things about the person across chats) and Research (deeper "
    "multi-source answers); how you solve hard maths, science and logic problems (several "
    "attempts, checking by running code, and saying honestly when you are unsure); and an "
    "invitation to try something specific. Edit this text to match what is really live."
)

HUMAN_SYSTEM_PROMPT = IDENTITY + """

How you talk:
- Sound like a thoughtful person, not a manual. Plain words, contractions are fine.
- Answer the actual question first, then add detail only if it helps.
- Reply in the language the user writes in (English, Hindi, Hinglish, etc.).
- Do not use labels like "Answer" or "Step-by-Step Execution" for simple questions.
- Never repeat a sentence or paragraph. Always finish what you start.

Depth and conversation:
- Never answer a real question with one line. For explanations, advice, ideas, "tell me
  about...", comparisons and questions about yourself, give a substantial, well-organised
  answer: the direct answer, the reasoning behind it, a concrete example, and any catch
  or alternative worth knowing. Several paragraphs is normal when the topic deserves it.
- Only greetings, thanks and truly tiny factual questions get a short reply.
- Keep the conversation going like a person would: react to what they said, connect it to
  earlier messages, and end with one natural follow-up question or offer when it fits.
  Never more than one question.

How you think:
- For maths, logic, physics and code: work step by step, then check the result
  (substitute back, test a special case, or use a second method) before answering.
- If a question has a false premise or a trick, point it out.
- Never claim you ran code, a simulation or a search unless a tool result is
  actually given to you in this conversation.
- If you are not sure, say so plainly and say what you would check.
- When a problem has a single final result, end with one last line exactly like:
  FINAL: <the result>
"""

CONTINUE_PROMPT = (
    "Continue exactly where you stopped. Do not repeat anything you already wrote "
    "and do not restart. Finish the answer."
)

ANTI_LOOP_NOTE = (
    "\n\n(Your previous draft got stuck repeating itself. Write one clean, complete "
    "answer with no repeated passages.)"
)

CODE_CHECK_PROMPT = """Write a short, self-contained Python program that independently
computes the answer to the problem below. Use only the standard library and sympy.
Do not read files, use the network or use input(). Compute it numerically or
symbolically rather than copying a guess. If the problem is multiple choice, print
the letter of the correct option. Print exactly one line: RESULT: <answer>
Reply with only the code in a ```python block.

Problem:
{question}
"""

RECONCILE_PROMPT = """Problem:
{question}

Draft solution (backed by {votes} of {total} weighted votes from an independent agent panel):
{draft}

Other answers the panel proposed: {alts}

Result of an independent program run: {code_result}

Task: if the program result and the draft's final answer agree, rewrite the draft as a
clear, natural, complete explanation. If they disagree, re-derive the problem carefully,
decide which one is wrong, and write the corrected solution. Be honest if you still
cannot tell. Do not mention these instructions. End with one line: FINAL: <result>"""

CRITIQUE_PROMPT = """Problem:
{question}

Proposed solution:
{draft}

Act as a strict reviewer. Look for wrong assumptions, algebra or arithmetic slips,
missed cases and unsupported claims. If you find a real error, write the corrected
full solution. If it is correct, restate it cleanly. Natural tone, no repetition.
End with one line: FINAL: <result>"""

# --------------------------------------------------------------------------
# Text repair: repetition, incompleteness, overlap
# --------------------------------------------------------------------------

_TOKEN = re.compile(r"\S+")


def find_loop_start(text: str, max_period: int = 60) -> Optional[int]:
    """Return the character index where a degenerate repetition loop begins
    (the end of the first copy of the repeated chunk), or None.

    A loop = some chunk of n words repeated at least 3 times in a row.
    """
    toks = list(_TOKEN.finditer(text))
    n_tok = len(toks)
    words = [t.group(0) for t in toks]
    best: Optional[int] = None
    for n in range(1, min(max_period, n_tok // 3) + 1):
        run = 0
        for j in range(n_tok - n):
            if words[j] == words[j + n]:
                run += 1
                need = 2 * n if n > 1 else 4
                if run >= need:
                    first_copy_end = j - run + 1 + n  # token index after first copy
                    if first_copy_end < n_tok:
                        pos = toks[first_copy_end].start()
                        if best is None or pos < best:
                            best = pos
                    break
            else:
                run = 0
    return best


def strip_repetition(text: str) -> Tuple[str, bool]:
    """Cut a repetition loop off, keeping one copy of the repeated chunk."""
    start = find_loop_start(text)
    if start is None:
        return text, False
    return text[:start].rstrip(), True


def _norm_sentence(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def dedupe_sentences(text: str, min_len: int = 25) -> str:
    """Drop sentences that are exact repeats of an earlier one (code fences are left alone)."""
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    seen = set()
    out = []
    for part in parts:
        if part.startswith("```"):
            out.append(part)
            continue
        pieces = re.split(r"(?<=[.!?])\s+", part)
        kept = []
        for p in pieces:
            key = _norm_sentence(p)
            if len(key) >= min_len and key in seen:
                continue
            seen.add(key)
            kept.append(p)
        out.append(" ".join(kept))
    return "".join(out)


_LABEL_LINE = re.compile(
    r"^\s*(?:#+\s*)?\**\s*(answer|step-by-step execution|solution)\s*:?\s*\**\s*$",
    re.I | re.M,
)


def clean_formatting(text: str) -> str:
    text = _LABEL_LINE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def looks_incomplete(text: str) -> bool:
    t = text.rstrip()
    if not t:
        return True
    if t.count("```") % 2 == 1:
        return True
    if t.count("$$") % 2 == 1:
        return True
    if t.endswith((":", ",", " and", " or", " the", " of", " to", " then", " but", " so", " because", "(", "-")):
        return True
    return False


def trim_overlap(existing: str, new: str, max_overlap: int = 400) -> str:
    """If `new` starts by re-typing the end of `existing`, drop that part."""
    tail = existing[-max_overlap:]
    for k in range(min(len(tail), len(new)), 15, -1):
        if new.startswith(tail[-k:]):
            return new[k:]
    return new


_FINAL = re.compile(r"^\s*\**\s*FINAL\s*\**\s*:\s*\**\s*(.+?)\s*\**\s*$", re.I | re.M)


def extract_final(text: str) -> Optional[str]:
    m = _FINAL.findall(text)
    return m[-1].strip() if m else None


def strip_final_line(text: str) -> str:
    return _FINAL.sub("", text).rstrip()


def _norm_answer(a: str) -> str:
    a = a.lower()
    a = re.sub(r"\\(left|right|text|mathrm|displaystyle)", "", a)
    a = re.sub(r"[\s$\\{}()\[\]`*_.,;:]", "", a)
    a = a.replace("ln", "log")
    return a


def _to_float(a: str) -> Optional[float]:
    m = re.fullmatch(r"\s*[-+]?\d+(\.\d+)?([eE][-+]?\d+)?\s*", a.replace(",", ""))
    return float(a.replace(",", "")) if m else None


def answers_match(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    fa, fb = _to_float(a), _to_float(b)
    if fa is not None and fb is not None:
        return math.isclose(fa, fb, rel_tol=1e-6, abs_tol=1e-9)
    na, nb = _norm_answer(a), _norm_answer(b)
    if na == nb:
        return True
    if min(len(na), len(nb)) >= 3 and (na in nb or nb in na):
        return True
    # multiple choice: "C" vs "C) g'(0) = cos(log 2)"
    la = re.match(r"^\(?([a-e])\)?(?:[\s).:]|$)", a.strip(), re.I)
    lb = re.match(r"^\(?([a-e])\)?(?:[\s).:]|$)", b.strip(), re.I)
    return bool(la and lb and la.group(1).lower() == lb.group(1).lower())


# --------------------------------------------------------------------------
# Running generated code (for cross-checking maths / logic)
# --------------------------------------------------------------------------

_BLOCKED = re.compile(
    r"(\bimport\s+(os|sys|subprocess|socket|shutil|requests|urllib|http|ctypes|"
    r"pathlib|multiprocessing|threading|signal|builtins)\b"
    r"|\bfrom\s+(os|sys|subprocess|socket|shutil|requests|urllib|http|ctypes|"
    r"pathlib|multiprocessing|threading|signal|builtins)\b"
    r"|__import__|\beval\s*\(|\bexec\s*\(|\bopen\s*\(|\bcompile\s*\(|\binput\s*\()"
)


def extract_code(text: str) -> Optional[str]:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S | re.I)
    return m.group(1) if m else None


def run_python(code: str, timeout: int = 20) -> Tuple[Optional[str], str]:
    """Run code in a throw-away subprocess with CPU/memory/file limits.

    Returns (stdout or None, status). NOTE: this is defence in depth, not a
    hardened sandbox. For a public site, run it in a separate worker/container
    without your DATABASE_URL or other secrets in its environment.
    """
    if _BLOCKED.search(code):
        return None, "blocked: uses a disallowed import or call"

    def _limits():  # pragma: no cover (POSIX only)
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "check.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=d,
                env={"PATH": os.environ.get("PATH", "")},
                preexec_fn=_limits if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return None, "timeout"
        except Exception as e:  # noqa: BLE001
            return None, f"error: {e}"
    if proc.returncode != 0:
        return None, "failed: " + proc.stderr.strip()[-300:]
    m = re.findall(r"RESULT:\s*(.+)", proc.stdout)
    return (m[-1].strip() if m else None), ("ok" if m else "no RESULT line")


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

_HARD = re.compile(
    r"(\bsolve\b|\bprove\b|\bcalculate\b|\bcompute\b|\bevaluate\b|\bsimplify\b|"
    r"\bderivative\b|\bintegral\b|\blimit\b|\bprobability\b|\bexpected\b|"
    r"\bhow many\b|\bhow can you\b|\bpuzzle\b|\briddle\b|\bwhich statement\b|"
    r"\bfind (the|all|x|n)\b|\bjee\b|\bolympiad\b|\bequation\b|\btheorem\b|"
    r"[A-D]\)\s|\d\s*[-+*/^=]\s*\d|\bsin\b|\bcos\b|\blog\b|∫|∑|√)",
    re.I,
)


def needs_verification(question: str) -> bool:
    return bool(_HARD.search(question)) and len(question) > 25


# --------------------------------------------------------------------------
# Depth control: no thin one-liners for real questions
# --------------------------------------------------------------------------

DEPTH_NOTE = (
    "\n\n(Your last draft was too short. Give a fuller, genuinely useful answer: the direct "
    "answer, the reasoning, a concrete example, and what to watch out for. Talk to me like "
    "a knowledgeable person having a real conversation, and finish with one natural "
    "follow-up question or offer.)"
)

_SMALL_TALK = re.compile(
    r"^\s*(hi+|hello+|hey+|yo|hola|namaste|thanks?( you)?|thx|ok(ay)?|cool|nice|great|"
    r"bye|good (morning|night|evening)|how are you\??)\W*$",
    re.I,
)

_OPEN_QUESTION = re.compile(
    r"(tell me|explain|describe|about (you|yourself)|who are you|what can you|what do you|"
    r"features?|how (do|does|can|to|should)|why|difference|compare|versus|\bvs\b|"
    r"help me|suggest|ideas?|advice|should i|plan|introduc|story|review|opinion|"
    r"what is|what are|what's)",
    re.I,
)


def is_small_talk(message: str) -> bool:
    return bool(_SMALL_TALK.match(message))


def wants_depth(message: str) -> bool:
    """True when a short reply would be a disappointing answer."""
    if is_small_talk(message):
        return False
    return bool(_OPEN_QUESTION.search(message)) or len(message.split()) >= 8


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


# --------------------------------------------------------------------------
# Panel roles, solver explanation, multi-part questions
# --------------------------------------------------------------------------

AGENT_ROLES = [
    "Careful step-by-step solver: derive the result methodically and double-check each step.",
    "Alternative-method solver: solve it by a different route than the obvious one.",
    "Sceptic: assume the obvious answer is a trap, hunt for the catch, then answer.",
    "Edge-case tester: test small and extreme cases, signs, domains and special values before answering.",
    "Verifier: solve it, then substitute the answer back or sanity-check units and magnitudes.",
    "First-principles thinker: go back to definitions and basic laws, avoid memorised shortcuts.",
    "Estimator: get a rough order-of-magnitude answer first, then refine it exactly.",
    "Counter-example hunter: try to break the likely answer with a concrete counter-example.",
    "Formaliser: restate the problem precisely with symbols and constraints, then solve.",
    "Simplifier: reduce the problem to a smaller equivalent one, then solve that.",
    "Exam-setter: think about how the question was designed and what mistake it targets.",
    "Programmer: reason as if you were writing the algorithm, and trace it by hand.",
]

@dataclass(frozen=True)
class Specialist:
    name: str
    tags: Tuple[str, ...]
    prompt: str


def _sp(name, tags, prompt):
    return Specialist(name, tuple(tags.split()), prompt)


# 53 domain specialists. Each is a prompt persona; the router below picks the ones that fit the question.
SPECIALISTS: List[Specialist] = [
    # --- mathematics (16)
    _sp("Algebraist", "math algebra", "Manipulate expressions and equations exactly; check every sign and every domain restriction."),
    _sp("Calculus analyst", "math calculus", "Differentiate, integrate and take limits carefully; name each rule and verify by differentiating back."),
    _sp("Linear algebraist", "math linear_algebra", "Think in vector spaces, ranks and eigenstructure; verify with a small numeric example."),
    _sp("Number theorist", "math number_theory", "Use divisibility, modular arithmetic and prime structure; test small cases to confirm."),
    _sp("Combinatorialist", "math combinatorics probability", "Count carefully: define the sample space, avoid double counting, verify with a smaller case."),
    _sp("Probabilist", "math probability statistics", "Define events precisely, use conditional probability, and sanity-check that probabilities sum to 1."),
    _sp("Statistician", "statistics probability math", "Identify the right estimator or test, state assumptions, and interpret the result in plain words."),
    _sp("Geometer", "math geometry trig", "Draw the figure mentally, label every length and angle, and use coordinates to double-check."),
    _sp("Trigonometry specialist", "math trig geometry", "Use identities and exact values; check the quadrant and the range of each function."),
    _sp("Discrete-maths and graph theorist", "math discrete algorithms puzzle", "Model the problem as a graph, recurrence or invariant; look for a pigeonhole or parity argument."),
    _sp("Proof writer", "math proof logic", "Prove each claim step by step, state what is assumed, and look for the gap a sceptic would find."),
    _sp("Inequality and olympiad expert", "math exam algebra", "Try AM-GM, Cauchy-Schwarz, substitution and symmetry; check equality cases."),
    _sp("Differential-equations specialist", "math ode calculus physics", "Classify the equation, pick the right method, and substitute the solution back to verify."),
    _sp("Mathematical modeller", "math physics finance", "Translate the words into equations, state the assumptions, solve, and interpret the units."),
    _sp("JEE/NEET exam coach", "exam math physics chemistry", "Solve the way a top exam candidate would: identify the concept being tested and the common trap, then eliminate options."),
    _sp("Competitive-programming mathematician", "math code algorithms", "Find the clean closed form or the efficient algorithm; reason about complexity and edge cases."),
    # --- physics (7)
    _sp("Mechanics specialist", "physics mechanics", "Draw a free-body diagram, choose axes, and use conservation laws where they simplify the problem."),
    _sp("Electromagnetism specialist", "physics em", "Track fields, charges and circuits carefully; check signs, directions and units."),
    _sp("Thermodynamics specialist", "physics thermo chemistry", "Identify the process and the system, apply the first and second laws, and keep temperature in kelvin."),
    _sp("Waves and optics specialist", "physics waves", "Use wave relations and ray or interference geometry; check limiting cases."),
    _sp("Modern-physics specialist", "physics modern", "Apply quantum, relativity and nuclear relations with correct constants and units."),
    _sp("Fluid-dynamics specialist", "physics fluids", "Use continuity, Bernoulli and pressure relations; check that the units balance."),
    _sp("Dimensional-analysis and units checker", "physics units", "Check the dimensions of every formula and convert units explicitly; flag any mismatch."),
    # --- chemistry (4)
    _sp("Physical chemist", "chemistry physchem", "Use equilibrium, kinetics and thermodynamic relations; keep track of units and significant figures."),
    _sp("Organic chemist", "chemistry organic", "Reason about functional groups, mechanisms and stereochemistry before naming products."),
    _sp("Inorganic chemist", "chemistry inorganic", "Use periodic trends, oxidation states and coordination chemistry; check charge balance."),
    _sp("Stoichiometry and analytical chemist", "chemistry stoich", "Balance first, convert to moles, find the limiting reagent, then compute; check mass balance."),
    # --- life science and medicine (2)
    _sp("Biology tutor", "biology", "Explain the mechanism from first principles and connect it to the bigger system."),
    _sp("Medical-information explainer", "medicine biology", "Give general educational information, be clear about uncertainty, and say when a professional should be consulted."),
    # --- computing (8)
    _sp("Algorithm designer", "code algorithms", "Choose the data structure, state the complexity, and test the idea on a tiny input by hand."),
    _sp("Debugger", "code debugging", "Reproduce the failure mentally, form hypotheses, find the root cause, and propose the smallest safe fix."),
    _sp("Software architect", "code systems", "Weigh trade-offs, keep the design simple, and name the failure modes."),
    _sp("Security reviewer (defensive)", "code security", "Look for the weaknesses a careful defender must close and explain how to fix them; do not write attack tools."),
    _sp("SQL and database expert", "code database", "Think in sets and indexes; check joins, nulls and edge cases."),
    _sp("Web and API engineer", "code web", "Consider HTTP semantics, validation, errors, and deployment realities."),
    _sp("Data scientist", "code ml statistics", "Check data quality, leakage and evaluation before trusting a model."),
    _sp("Code verifier", "code debugging algorithms", "Write test cases, including edge cases, and trace the code against them."),
    # --- money (3)
    _sp("Quantitative analyst", "finance math", "Use the right time-value or risk formula and check the result by an independent calculation."),
    _sp("Accountant", "finance accounting", "Keep the books balanced; state each assumption about treatment and period."),
    _sp("Economist", "finance economics", "Separate positive from normative claims; use marginal reasoning and note the assumptions."),
    # --- language (5)
    _sp("Editor", "writing", "Tighten the prose, fix structure, and keep the author's voice."),
    _sp("Translator", "translation writing", "Preserve meaning, tone and idiom, and note where a literal translation would mislead."),
    _sp("Storyteller", "story writing", "Build vivid scenes with a clear arc, concrete detail and a distinct voice."),
    _sp("Summariser", "summary writing", "Capture the core argument faithfully and briefly, without adding claims."),
    _sp("Hindi and Hinglish communicator", "hindi writing", "Answer naturally in the user's mix of Hindi and English, in the script they used."),
    # --- general thinking (8)
    _sp("Fact-checker", "research factcheck", "Separate what is well established from what is uncertain; flag claims that need a source."),
    _sp("Historian", "history research", "Give context and causes, and distinguish evidence from legend."),
    _sp("Geographer", "geography research", "Use places, scale and physical and human geography to ground the answer."),
    _sp("Philosopher and ethicist", "philosophy logic", "Lay out the strongest positions fairly and test each against objections."),
    _sp("Planner", "planning", "Break the goal into steps with order, dependencies, risks and a first action."),
    _sp("Devil's advocate", "critique logic", "Argue against the leading answer as hard as honesty allows, then say whether it survives."),
    _sp("Plain-language teacher", "teaching", "Explain as to a bright beginner: simple words, one good example, then the precise statement."),
    _sp("Synthesiser", "research planning", "Combine the strongest points from different views into one coherent answer."),
]

_DOMAIN_PATTERNS = {
    "math": r"\b(solve|prove|calculate|compute|equation|integral|derivative|limit|matrix|probability|theorem|polynomial|sequence|series|modulo|prime|geometry|algebra|trigonometr)\w*",
    "algebra": r"\b(equation|polynomial|inequality|quadratic|factor|simplify|expand|roots?)\b",
    "calculus": r"\b(derivative|differentiate|integral|integrate|limit|taylor|maxima|minima|d/dx)\b|∫",
    "linear_algebra": r"\b(matrix|matrices|eigen\w*|determinant|vector|rank|linear (system|map)|basis)\b",
    "number_theory": r"\b(prime|gcd|lcm|modulo|mod|divisib\w*|congruen\w*|remainder|diophantine)\b",
    "combinatorics": r"\b(permutations?|combinations?|arrangements?|ways to|choose|binomial|counting)\b",
    "probability": r"\b(probabilit\w*|expected|dice|coin|cards?|random|odds|bayes)\b",
    "statistics": r"\b(mean|median|variance|standard deviation|regression|correlation|hypothesis|p-value|distribution|sample)\b",
    "geometry": r"\b(triangle|circle|angle|polygon|area|volume|perimeter|radius|sphere|cone|cylinder)\b",
    "trig": r"\b(sin|cos|tan|sec|csc|cot|trigonometr\w*|radians?)\b",
    "discrete": r"\b(graph|tree|node|edge|path|pigeonhole|recurrence|induction|invariant)\b",
    "proof": r"\b(prove|proof|show that|demonstrate|hence)\b",
    "ode": r"\b(ode|differential equation|y'|dy/dx)\b",
    "exam": r"\b(jee|neet|olympiad|imo|gate|cat|upsc|exam|which statement|mcq)\b|\b[A-D]\)\s",
    "physics": r"\b(force|velocity|acceleration|energy|momentum|torque|circuit|voltage|current|wave|photon|gravity|pressure|physics)\b",
    "mechanics": r"\b(force|velocity|acceleration|momentum|torque|friction|projectile|collision|pendulum)\b",
    "em": r"\b(electric|magnetic|charge|capacitor|inductor|circuit|voltage|current|resistance|ohm)\b",
    "thermo": r"\b(heat|temperature|entropy|enthalpy|thermodynamic\w*|gas law|carnot)\b",
    "waves": r"\b(wave|frequency|wavelength|refraction|diffraction|interference|lens|mirror|optics)\b",
    "modern": r"\b(quantum|relativity|photon|nucleus|nuclear|radioactive|half-life|electron)\b",
    "fluids": r"\b(fluid|bernoulli|viscosity|buoyan\w*|flow rate|streamline)\b",
    "units": r"\b(units?|convert|dimension\w*|si )\b",
    "chemistry": r"\b(chemical|molecule|molar|mole|acid|base|reaction|compound|element|ph\b|bond|oxidation)\b",
    "physchem": r"\b(equilibrium|rate law|kinetics|gibbs|enthalpy|electrochem\w*)\b",
    "organic": r"\b(organic|alkane|alkene|alcohol|ketone|aldehyde|benzene|polymer|isomer)\b",
    "inorganic": r"\b(periodic table|coordination|ionic|lattice|oxidation state|transition metal)\b",
    "stoich": r"\b(balance|stoichiometr\w*|limiting reagent|molarity|yield|mole)\b",
    "biology": r"\b(cell|dna|rna|protein|gene|evolution|species|organism|photosynthesis|enzyme)\b",
    "medicine": r"\b(symptom|disease|diagnos\w*|medicine|drug|dose|treatment|infection)\b",
    "code": r"\b(code|function|python|javascript|java|c\+\+|sql|bug|error|compile|api|algorithm|class|loop|script)\b",
    "algorithms": r"\b(algorithm|complexity|sort|search|dynamic programming|recursion|big o|graph)\b",
    "debugging": r"\b(bug|error|exception|traceback|crash|fix|not working|fails?)\b",
    "systems": r"\b(architecture|scalab\w*|microservice|deploy\w*|server|latency|database design|system design)\b",
    "security": r"\b(security|vulnerab\w*|encrypt\w*|password|auth\w*|injection|xss|csrf)\b",
    "database": r"\b(sql|query|database|postgres\w*|index|join|table|schema)\b",
    "web": r"\b(html|css|react|flask|django|http|rest|endpoint|frontend|backend|website)\b",
    "ml": r"\b(machine learning|neural|model|training|dataset|classifier|llm|embedding)\b",
    "finance": r"\b(interest|loan|emi|invest\w*|stock|bond|npv|irr|return|portfolio|profit|loss|tax)\b",
    "accounting": r"\b(balance sheet|ledger|depreciation|revenue|expense|accounting|gst|invoice)\b",
    "economics": r"\b(inflation|gdp|demand|supply|market|economy|monetary|fiscal|elasticity)\b",
    "writing": r"\b(write|essay|email|letter|story|poem|blog|rewrite|edit|proofread|paragraph)\b",
    "translation": r"\b(translate|translation|in (hindi|french|spanish|german|tamil|urdu)|meaning of)\b",
    "story": r"\b(story|novel|character|plot|poem|screenplay)\b",
    "summary": r"\b(summar\w*|tl;dr|key points|brief)\b",
    "hindi": r"[\u0900-\u097F]|\b(kya|kaise|kyun|hai|nahi|batao|samjhao|mujhe)\b",
    "research": r"\b(research|compare|difference between|pros and cons|history of|explain)\b",
    "factcheck": r"\b(is it true|true or false|fact.?check|really|myth)\b",
    "history": r"\b(history|ancient|medieval|empire|war|dynasty|independence|century)\b",
    "geography": r"\b(country|capital|river|mountain|continent|climate|map|population)\b",
    "philosophy": r"\b(ethic\w*|moral|philosoph\w*|meaning of life|free will|consciousness)\b",
    "planning": r"\b(plan|schedule|roadmap|steps to|strategy|how do i start|study plan)\b",
    "critique": r"\b(review|critique|feedback|what's wrong|weakness|flaw)\b",
    "teaching": r"\b(explain|teach|eli5|simple terms|beginner|understand)\b",
    "logic": r"\b(logic|puzzle|riddle|paradox|if and only if|syllogism|truth table|deduce)\b",
    "puzzle": r"\b(puzzle|riddle|brain ?teaser|trick question|switches|jugs?|river crossing)\b",
}
_DOMAIN_RE = {k: re.compile(v, re.I) for k, v in _DOMAIN_PATTERNS.items()}


def detect_domains(question: str) -> set:
    return {k for k, rx in _DOMAIN_RE.items() if rx.search(question)}


def pick_roles(question: str, n: int) -> List[str]:
    """n role prompts for n agents: a careful generalist first, then specialists that match the
    question's domains interleaved with generalists, then everyone else as filler."""
    tags = detect_domains(question)
    scored = sorted(((len(tags & set(sp_.tags)), i, sp_) for i, sp_ in enumerate(SPECIALISTS)),
                    key=lambda t: (-t[0], t[1]))
    matched = [f"{sp_.name}: {sp_.prompt}" for sc, _, sp_ in scored if sc > 0]
    rest = [f"{sp_.name}: {sp_.prompt}" for sc, _, sp_ in scored if sc == 0]
    general = list(AGENT_ROLES)
    order = [general[0]]
    for a, b in itertools.zip_longest(matched, general[1:]):
        order += [x for x in (a, b) if x]
    order += rest
    return [order[i % len(order)] for i in range(max(0, n))]


DEBATE_PROMPT = """Question:
{question}

Positions from other solvers (each followed by an excerpt of its reasoning):
{positions}

Examine every position critically. Find the actual error in the wrong ones, decide which answer is
correct (or find a better one), and solve the problem again properly. End with: FINAL: <result>"""


SOLVER_NOTE = """A trusted program just computed and independently verified the following. Explain it to
the user in your own natural voice: say what the result is, walk through the key idea in plain
language, and keep every number, expression and grid exactly as given (put grids or boards in a
code block). You may say it was computed and cross-checked by running code, because that is true.
Do not add facts that contradict it. If the computed result clearly does not answer what the
user actually asked, ignore it and answer normally.

{block}"""

VISION_NOTE = ("The user has attached an image. Read it carefully first and transcribe any problem text "
               "exactly in your head, then answer what they ask. If part of the image is unclear or cut off, "
               "say exactly what you could not read instead of guessing.")

VISION_REVIEW_PROMPT = """Re-read the problem in the attached image carefully, then check the draft answer below line by
line: your reading of the problem, every step and every claim. Fix any error or gap. If the draft is correct,
restate it cleanly. Do not mention that you were given a draft. Natural, human tone.

Draft:
{draft}"""

REFUSAL_TEXT = "I can't help with that one, but I'm happy to help with something close to it."


def split_parts(q: str):
    """'Intro 1) a 2) b 3) c' -> ('Intro', ['a', 'b', 'c']) when numbered 1,2,3...; else None."""
    ms = list(re.finditer(r"(?:^|(?<=\s))(\d{1,2})\)\s+(?=\S)", q))
    nums = [int(m.group(1)) for m in ms]
    if len(ms) >= 2 and nums == list(range(1, len(nums) + 1)):
        intro = q[:ms[0].start()].strip()
        parts = [q[m.end(): (ms[i + 1].start() if i + 1 < len(ms) else len(q))].strip()
                 for i, m in enumerate(ms)]
        return intro, parts
    return None


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


@dataclass
class Reply:
    text: str
    confidence: str = "n/a"  # high | medium | low | n/a
    mode: str = "chat"  # chat | solve
    notes: List[str] = field(default_factory=list)


Agent = Tuple[Callable, int, str]  # (call_fn, weight, label)

_CONF_RANK = {"low": 0, "medium": 1, "n/a": 2, "high": 3}


class ReasoningEngine:
    def __init__(
        self,
        call_model: Callable,
        samples: int = 3,
        max_continuations: int = 40,
        run_code: bool = True,
        code_timeout: int = 20,
        system_prompt: str = HUMAN_SYSTEM_PROMPT,
        workers: int = 8,
        min_words: int = 90,
        max_tokens: int = 16000,
        agents: Optional[List[Agent]] = None,
        use_solvers: bool = True,
        chat_effort: str = "medium",
        solve_effort: str = "high",
        debate_rounds: int = 1,
        vision_call: Optional[Callable] = None,
        debate_agents: int = 4,
    ):
        self.call_model = call_model
        self.samples = max(1, samples)
        self.max_continuations = max_continuations
        self.run_code = run_code
        self.code_timeout = code_timeout
        self.system_prompt = system_prompt
        self.workers = max(1, workers)
        self.min_words = min_words
        self.max_tokens = max_tokens
        self.agents = agents or []
        self.use_solvers = use_solvers and solvers is not None
        self.chat_effort = chat_effort
        self.solve_effort = solve_effort
        self.debate_rounds = debate_rounds
        self.vision_call = vision_call
        self.debate_agents = max(1, debate_agents)

    # ---- model plumbing ---------------------------------------------------
    def _call(self, messages, **opts) -> Tuple[str, str]:
        fn = opts.pop("_fn", None) or self.call_model
        out = fn(messages, **opts)
        if isinstance(out, tuple):
            return str(out[0]), (out[1] or "stop")
        if isinstance(out, dict):
            return str(out.get("text", "")), (out.get("finish_reason") or "stop")
        return str(out), "stop"

    @staticmethod
    def _with_note(messages, note):
        msgs = [dict(m) for m in messages]
        for m in reversed(msgs):
            if m["role"] == "user":
                m["content"] += note
                break
        return msgs

    def _build(self, history, user_message, extra: str = ""):
        msgs = [{"role": "system", "content": self.system_prompt + extra}]
        msgs += [{"role": m["role"], "content": m["content"]} for m in (history or [])
                 if m.get("role") in ("user", "assistant")]
        msgs.append({"role": "user", "content": user_message})
        return msgs

    # ---- 1. complete, loop-free generation -------------------------------
    def generate_complete(self, messages, **opts) -> str:
        text, finish = self._call(messages, **opts)
        if finish == "refusal":
            return text.strip() or REFUSAL_TEXT
        text, looped = strip_repetition(text)
        if looped:
            retry_opts = dict(opts)
            retry_opts["repetition_penalty"] = 1.15
            retry_opts["temperature"] = max(opts.get("temperature", 0.7), 0.8)
            t2, finish2 = self._call(self._with_note(messages, ANTI_LOOP_NOTE), **retry_opts)
            t2, looped2 = strip_repetition(t2)
            if len(t2) >= len(text) or not looped2:
                text, finish = t2, finish2

        heuristic_used = 0
        for _ in range(self.max_continuations):
            cut = finish == "length"
            if not cut:
                if heuristic_used >= 1 or not looks_incomplete(text):
                    break
                heuristic_used += 1
            cont = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": CONTINUE_PROMPT},
            ]
            more, finish = self._call(cont, **opts)
            more, _ = strip_repetition(more)
            more = trim_overlap(text, more)
            if not more.strip():
                break
            joiner = "" if text.endswith((" ", "\n")) or more.startswith((" ", "\n")) else " "
            text = text + joiner + more
        return text

    # ---- 2. the agent panel ------------------------------------------------
    def _attempts(self, history, question):
        if self.agents:
            roles = pick_roles(question, len(self.agents))
            jobs = [(fn, w, label, roles[i]) for i, (fn, w, label) in enumerate(self.agents)]
        else:
            roles = pick_roles(question, self.samples)
            jobs = [(None, 1, "lead", roles[i]) for i in range(self.samples)]

        def run(job):
            fn, _w, _label, role = job
            msgs = self._build(history, question,
                               extra=f"\n\nYour role on this panel: {role}\n"
                                     "Solve it independently and finish with: FINAL: <result>")
            opts = dict(temperature=0.7, max_tokens=self.max_tokens, effort=self.solve_effort)
            if fn:
                opts["_fn"] = fn
            return self.generate_complete(msgs, **opts)

        texts: List[str] = []
        weights: List[int] = []
        labels: List[str] = []
        with cf.ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = [(ex.submit(run, j), j) for j in jobs]
            for f, j in futs:
                try:
                    t = f.result()
                except Exception:  # noqa: BLE001
                    continue
                if t.strip():
                    texts.append(t)
                    weights.append(j[1])
                    labels.append(j[2])
        return texts, weights, labels

    @staticmethod
    def _vote(candidates: List[str], weights: Optional[List[int]] = None):
        """Weighted vote over final answers. Returns (draft, winning_weight, total_weight, alternatives)."""
        weights = weights or [1] * len(candidates)
        finals = [extract_final(c) for c in candidates]
        groups: List[List[int]] = []
        for i, f in enumerate(finals):
            if f is None:
                continue
            for g in groups:
                if answers_match(f, finals[g[0]]):
                    g.append(i)
                    break
            else:
                groups.append([i])
        total = sum(weights)
        if not groups:
            return max(candidates, key=len), 0, total, "none"
        wt = lambda g: sum(weights[i] for i in g)  # noqa: E731
        ranked = sorted(groups, key=wt, reverse=True)
        best = ranked[0]
        pick = max(best, key=lambda i: len(candidates[i]))  # most thorough write-up among the winners
        alts = "; ".join(f"{finals[g[0]]} (weight {wt(g)})" for g in ranked[1:4]) or "none"
        return candidates[pick], wt(best), total, alts

    def _debate(self, question: str, candidates: List[str]):
        """Show the strongest competing answers to a few agents and let them re-solve."""
        finals = [extract_final(c) for c in candidates]
        groups: List[List[int]] = []
        for i, f in enumerate(finals):
            if f is None:
                continue
            for g in groups:
                if answers_match(f, finals[g[0]]):
                    g.append(i)
                    break
            else:
                groups.append([i])
        if len(groups) < 2:
            return [], []
        groups.sort(key=len, reverse=True)
        positions = "\n\n".join(
            f"Position {chr(65 + n)} (answer: {finals[g[0]]}; {len(g)} solver(s)):\n"
            f"...{candidates[max(g, key=lambda i: len(candidates[i]))][-700:]}"
            for n, g in enumerate(groups[:4]))
        prompt = DEBATE_PROMPT.format(question=question, positions=positions)
        pool = (self.agents[:self.debate_agents] if self.agents else [(None, 1, "lead")] * 2)
        outs: List[str] = []
        ws: List[int] = []

        def run(a):
            fn = a[0]
            msgs = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
            opts = dict(temperature=0.5, max_tokens=self.max_tokens, effort=self.solve_effort)
            if fn:
                opts["_fn"] = fn
            return self.generate_complete(msgs, **opts)

        with cf.ThreadPoolExecutor(max_workers=max(1, len(pool))) as ex:
            futs = [(ex.submit(run, a), a) for a in pool]
            for f, a in futs:
                try:
                    t = f.result()
                except Exception:  # noqa: BLE001
                    continue
                if extract_final(t):
                    outs.append(t)
                    ws.append(a[1])
        return outs, ws

    def _code_cross_check(self, question: str, notes: List[str]) -> Optional[str]:
        if not self.run_code:
            return None
        try:
            raw = self.generate_complete(
                [{"role": "user", "content": CODE_CHECK_PROMPT.format(question=question)}],
                temperature=0.2, effort=self.solve_effort, max_tokens=self.max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            notes.append(f"code check unavailable: {type(e).__name__}")
            return None
        code = extract_code(raw)
        if not code:
            notes.append("code check: model returned no program")
            return None
        result, status = run_python(code, self.code_timeout)
        notes.append(f"code check: {status}")
        return result

    def _explain_solver(self, history, question, res, notes) -> Reply:
        notes.append(f"solver: {res.solver} (verified: {res.check})")
        block = solvers.format_for_llm(res)
        msgs = self._build(history, question, extra="\n\n" + SOLVER_NOTE.format(block=block))
        try:
            text = self.generate_complete(msgs, temperature=0.4, max_tokens=self.max_tokens,
                                          effort=self.chat_effort)
        except Exception:  # noqa: BLE001
            text = f"{res.answer}\n\n(Checked independently: {res.check}.)"
        text = dedupe_sentences(clean_formatting(strip_final_line(text)))
        return Reply(text, "high", "solve", notes)

    def solve(self, history, question: str, pre=None) -> Reply:
        notes: List[str] = []
        res = pre if pre is not None else (solvers.try_solve(question) if self.use_solvers else None)
        if res is not None:
            if res.verified:
                return self._explain_solver(history, question, res, notes)
            notes.append(f"solver '{res.solver}' failed its own check and was discarded")

        candidates, weights, labels = self._attempts(history, question)
        if not candidates:
            return Reply("Sorry, I couldn't get a solution out just now. Please try again.",
                         "low", "solve", notes + ["no candidates"])
        draft, votes, total, alts = self._vote(candidates, weights)
        notes.append(f"panel: {len(candidates)} agents answered; weighted vote {votes}/{total}")
        roles_used = [r.split(":", 1)[0] for r in pick_roles(question, min(6, len(candidates)))]
        notes.append("roles: " + ", ".join(roles_used))
        for _ in range(self.debate_rounds):
            if total and votes / total < 0.66 and len(candidates) >= 3:
                extra, extra_w = self._debate(question, candidates)
                if not extra:
                    break
                candidates, weights = candidates + extra, weights + extra_w
                draft, votes, total, alts = self._vote(candidates, weights)
                notes.append(f"debate round: {len(extra)} agents re-solved; weighted vote now {votes}/{total}")

        code_result = self._code_cross_check(question, notes)

        final_text, _ = self._call(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": RECONCILE_PROMPT.format(
                    question=question, draft=draft, votes=votes, total=total, alts=alts,
                    code_result=code_result if code_result else "not available")},
            ],
            temperature=0.2, effort=self.solve_effort, max_tokens=self.max_tokens,
        )
        final_text, _ = strip_repetition(final_text)
        final_answer = extract_final(final_text)

        code_agrees = bool(code_result) and answers_match(final_answer, code_result)
        strong_vote = total > 0 and votes / total >= 0.66

        if not code_agrees and not strong_vote:
            reviewed, _ = self._call(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": CRITIQUE_PROMPT.format(question=question, draft=final_text)},
                ],
                temperature=0.2, effort=self.solve_effort, max_tokens=self.max_tokens,
            )
            reviewed, _ = strip_repetition(reviewed)
            if reviewed.strip():
                final_text = reviewed
                notes.append("review pass applied")

        if code_agrees and strong_vote:
            confidence = "high"
        elif code_agrees or strong_vote:
            confidence = "medium"
        else:
            confidence = "low"

        body = dedupe_sentences(clean_formatting(strip_final_line(final_text)))
        if confidence == "low":
            body += ("\n\nHeads up: my own checks didn't fully agree on this one, so treat "
                     "this as my best attempt rather than a sure thing.")
        return Reply(body, confidence, "solve", notes)

    # ---- 3. entry points ---------------------------------------------------
    def _respond_single(self, history, user_message: str) -> Reply:
        pre = solvers.try_solve(user_message) if self.use_solvers else None
        if pre is not None or needs_verification(user_message):
            return self.solve(history, user_message, pre=pre)
        msgs = self._build(history, user_message)
        opts = dict(temperature=0.7, max_tokens=self.max_tokens, effort=self.chat_effort)
        text = self.generate_complete(msgs, **opts)
        notes: List[str] = []
        if wants_depth(user_message):
            for _ in range(2):
                if word_count(text) >= self.min_words:
                    break
                fuller = self.generate_complete(self._with_note(msgs, DEPTH_NOTE), **opts)
                notes.append(f"expanded thin reply ({word_count(text)} -> {word_count(fuller)} words)")
                if word_count(fuller) > word_count(text):
                    text = fuller
        text = dedupe_sentences(clean_formatting(strip_final_line(text)))
        return Reply(text, "n/a", "chat", notes)

    def _respond_parts(self, history, intro: str, parts: List[str]) -> Reply:
        replies = []
        for i, part in enumerate(parts, 1):
            q = (intro + "\n\n" if intro else "") + part
            replies.append(self._respond_single(history, q))
        text = "\n\n".join(f"**{i})** {r.text}" for i, r in enumerate(replies, 1))
        conf = min((r.confidence for r in replies), key=lambda c: _CONF_RANK.get(c, 2))
        notes = [f"part {i}: {n}" for i, r in enumerate(replies, 1) for n in r.notes]
        return Reply(text, conf, "multi", notes)

    def _respond_vision(self, history, user_message: str, images) -> Reply:
        from providers import normalize_images
        try:
            imgs = normalize_images(images)
        except ValueError as e:
            return Reply(f"I couldn't use that image: {e}.", "n/a", "error", [str(e)])
        if not self.vision_call:
            return Reply("I can't look at pictures right now because no image-capable model is connected "
                         "(an Anthropic, OpenAI or Gemini key is needed).", "n/a", "error", ["no vision provider"])
        question = user_message.strip() or "Describe this image and answer anything it asks."
        msgs = self._build(history, question, extra="\n\n" + VISION_NOTE)
        msgs[-1]["images"] = imgs
        opts = dict(temperature=0.4, max_tokens=self.max_tokens, effort=self.solve_effort, _fn=self.vision_call)
        text = self.generate_complete(msgs, **opts)
        notes = [f"vision: read {len(imgs)} image(s)"]
        if re.search(r"\b(prove|show that|solve|find|calculate|evaluate|simplify|verify|derive|answer)\b", question, re.I) \
                or needs_verification(question):
            review = [{"role": "system", "content": self.system_prompt},
                      {"role": "user", "content": VISION_REVIEW_PROMPT.format(draft=strip_final_line(text)),
                       "images": imgs}]
            checked = self.generate_complete(review, **opts)
            if checked.strip():
                text = checked
                notes.append("review pass: re-read the image and re-checked every step")
        text = dedupe_sentences(clean_formatting(strip_final_line(text)))
        return Reply(text, "n/a", "vision", notes)

    def respond(self, history, user_message: str, images=None) -> Reply:
        """images: optional list of data-URL strings or {"media_type", "data"(base64)} dicts."""
        try:
            if images:
                return self._respond_vision(history, user_message, images)
            sp = split_parts(user_message)
            if sp and len(sp[1]) <= 40:
                return self._respond_parts(history, *sp)
            return self._respond_single(history, user_message)
        except Exception as e:  # noqa: BLE001  (never show a stack trace to a user)
            return Reply("I couldn't reach my AI models just now. Please try again in a moment.",
                         "n/a", "error", [f"{type(e).__name__}: {str(e)[:200]}"])


def build_engine(panel_agents: int = 7, strong_agents: int = 4, **kwargs) -> ReasoningEngine:
    """Engine wired to whatever API keys are set (Anthropic, OpenAI, Gemini, Groq).

    The lead model (chat, judge, explanations) is the strongest available, with automatic
    failover. The solving panel spreads `panel_agents` agents over your providers: the first
    `strong_agents` use each provider's strongest model, the rest its cheaper fast tier.
    Every agent is a separate paid API call, so cost grows with panel_agents; beyond roughly
    10 to 15 agents the answers stop getting better and you are mostly paying for repeats.
    """
    from providers import ProviderPool
    pool = ProviderPool()
    agents = pool.agents(panel_agents, strong_agents) if panel_agents else None
    kwargs.setdefault("workers", min(64, max(4, panel_agents)))
    return ReasoningEngine(pool.lead_call(), agents=agents, vision_call=pool.vision_call(), **kwargs)


# --------------------------------------------------------------------------
# Self-test (no model needed)
# --------------------------------------------------------------------------


def _selftest() -> None:
    # 1. the exact loop from the screenshot
    looped = "Problem: Let $f: " * 8
    cleaned, found = strip_repetition("Here is one. " + looped)
    assert found and cleaned.count("Problem:") == 1, cleaned

    # 2. normal text is untouched
    ok = "The quick brown fox jumps over the lazy dog. It was a sunny day, and nothing repeated."
    assert strip_repetition(ok) == (ok, False)

    # 3. repeated sentence + stray labels (screenshot 4 style)
    messy = ("Answer\nYou can determine which switch controls each bulb by using the heat "
             "generated by the bulbs. You can determine which switch controls each bulb by "
             "using the heat generated by the bulbs.")
    fixed = dedupe_sentences(clean_formatting(messy))
    assert "Answer" not in fixed and fixed.count("heat generated") == 1, fixed

    # 4. incomplete detection and overlap trimming
    assert looks_incomplete("Step 1: do this, then") and not looks_incomplete("All done.")
    assert trim_overlap("abc " * 20 + "the end of the line is here", "the end of the line is here and more") \
        == " and more"

    # 5. answer matching
    assert answers_match("C", "C) g'(0) = cos(log 2)")
    assert answers_match("0.7071067812", "0.70710678118")
    assert answers_match("cos(ln 2)", "cos(log 2)")
    assert not answers_match("B", "C")

    # 6. FINAL parsing
    assert extract_final("blah\nFINAL: **C**") == "C"

    # 7. code runner: good code, blocked code, timeout
    r, s = run_python("print('RESULT: 42')")
    assert r == "42" and s == "ok", (r, s)
    r, s = run_python("import os\nprint('RESULT: 1')")
    assert r is None and s.startswith("blocked")
    r, s = run_python("while True:\n    pass", timeout=2)
    assert r is None

    # 8. end-to-end with a scripted fake model
    calls = {"n": 0}

    def fake(messages, **opts):
        calls["n"] += 1
        last = messages[-1]["content"]
        if "Write a short, self-contained Python program" in last:
            return "```python\nprint('RESULT: C')\n```"
        if "Task: if the program result" in last:
            return "So g'(0) = cos(log 2), option C.\nFINAL: C"
        if last == CONTINUE_PROMPT:
            return " and that finishes it."
        return "Working it out, the derivative gives cos(log 2).\nFINAL: C"

    eng = ReasoningEngine(fake)
    rep = eng.respond([], "For x in R let f(x)=|log 2 - sin x|, g(x)=f(f(x)). Which statement is "
                          "correct at x = 0? A) g'(0)=-sin(log 2) B) not differentiable "
                          "C) g'(0)=cos(log 2) D) g'(0)=-cos(log 2)")
    assert rep.mode == "solve" and rep.confidence == "high", (rep, calls)
    assert "FINAL" not in rep.text
    chat = ReasoningEngine(lambda m, **o: ("Hi there! What's on your mind?", "stop")).respond([], "hi")
    assert chat.mode == "chat" and "Hi there" in chat.text
    # 9. depth control: a one-liner to "features" gets expanded, small talk does not
    assert wants_depth("features") and wants_depth("tell me about yourself and feature")
    assert not wants_depth("hi") and not wants_depth("thanks!")
    seq = iter(["I am Deep OQ, an advanced AI", " ".join(f"idea{i} explains feature{i} clearly." for i in range(40)) + " Want to try one?"])
    thin = ReasoningEngine(lambda m, **o: (next(seq), "stop")).respond([], "features")
    assert word_count(thin.text) >= 90 and thin.notes, thin
    # 10. agent panel: weighted vote, code cross-check, judge
    def agent(ans):
        return lambda m, **o: {"text": f"Working...\nFINAL: {ans}", "finish_reason": "stop"}

    panel = [(agent("C"), 2, "a"), (agent("C"), 2, "b"), (agent("C"), 2, "c"),
             (agent("B"), 1, "d"), (agent("B"), 1, "e")]

    def lead(messages, **opts):
        last = messages[-1]["content"]
        if "Write a short, self-contained Python program" in last:
            return "```python\nprint('RESULT: C')\n```"
        if "Task: if the program result" in last:
            assert "B (weight 2)" in last, last          # dissent is shown to the judge
            return "It comes out as C.\nFINAL: C"
        return "fine"

    rep = ReasoningEngine(lead, agents=panel).respond([], "Which statement is correct? A) x B) y C) z D) w for this equation")
    assert rep.mode == "solve" and rep.confidence == "high", rep
    assert any("weighted vote 6/8" in n for n in rep.notes), rep.notes

    split_panel = [(agent("C"), 1, "a"), (agent("B"), 1, "b")]          # no majority, no code => low
    low = ReasoningEngine(lambda m, **o: "No clear result.\nFINAL: C" if "Task:" in m[-1]["content"] else "x",
                          agents=split_panel, run_code=False).respond([], "Which statement is correct? A) x B) y C) z D) w for this equation")
    assert low.confidence == "low" and "Heads up" in low.text, low

    # 11. a verified solver result is explained by the lead model and the check is reported
    seen = {}

    def explainer(messages, **opts):
        seen["sys"] = messages[0]["content"]
        return "The derivative is x*(x*cos(x) + 2*sin(x)); I computed it and cross-checked it with finite differences."

    ex = ReasoningEngine(explainer).respond([], "find the derivative of x^2*sin(x)")
    assert ex.mode == "solve" and ex.confidence == "high" and "Product rule" in seen["sys"], (ex, seen)
    assert any(n.startswith("solver: derivative") for n in ex.notes)

    # 12. numbered multi-part questions are answered part by part
    multi = ReasoningEngine(lambda m, **o: "Handled this part.").respond(
        [], "Please answer: 1) derivative of sin(x^2) 2) gcd of 12 and 18 3) is 97 prime")
    assert multi.mode == "multi" and all(f"**{i})**" in multi.text for i in (1, 2, 3)), multi.text
    assert split_parts("A) one B) two") is None and split_parts("price is 1) odd") is None

    # 13. refusals and provider outages degrade politely
    assert ReasoningEngine(lambda m, **o: ("", "refusal")).respond([], "tell me a story about dragons").text == REFUSAL_TEXT

    def boom(m, **o):
        raise RuntimeError("provider down")

    down = ReasoningEngine(boom).respond([], "tell me about quantum computing in detail")
    assert down.mode == "error" and "try again" in down.text

    # 14. 50+ specialists, routed by domain; a split vote triggers a debate round
    assert len(SPECIALISTS) >= 50 and len({sp_.name for sp_ in SPECIALISTS}) == len(SPECIALISTS)
    assert len(SPECIALISTS) + len(AGENT_ROLES) >= 60
    assert "Calculus analyst" in " ".join(pick_roles("find the limit and the derivative of this function", 8))
    assert "Debugger" in " ".join(pick_roles("my python code throws an error and crashes", 8))
    assert "Stoichiometry" in " ".join(pick_roles("balance this chemical reaction and find the limiting reagent", 8))
    assert len(pick_roles("anything", 200)) == 200 and pick_roles("anything", 3)[0].startswith("Careful")
    assert "hindi" in detect_domains("mujhe samjhao kaise hota hai")

    def debater(own):
        def fn(m, **o):
            if "Positions from other solvers" in m[-1]["content"]:
                return {"text": "Re-checking each position... the second is right.\nFINAL: C", "finish_reason": "stop"}
            return {"text": f"Working...\nFINAL: {own}", "finish_reason": "stop"}
        return fn

    split = [(debater("C"), 1, "a"), (debater("C"), 1, "b"), (debater("B"), 1, "c"), (debater("B"), 1, "d")]

    def lead2(messages, **opts):
        last = messages[-1]["content"]
        if "Write a short, self-contained Python program" in last:
            return "```python\nprint('RESULT: C')\n```"
        return "Settled.\nFINAL: C"

    deb = ReasoningEngine(lead2, agents=split).respond([], "Which statement is correct? A) x B) y C) z D) w for this equation")
    assert any("debate round" in n for n in deb.notes) and deb.confidence == "high", deb.notes
    nodeb = ReasoningEngine(lead2, agents=split, debate_rounds=0).respond([], "Which statement is correct? A) x B) y C) z D) w for this equation")
    assert not any("debate" in n for n in nodeb.notes)

    # 15. pictures: routed to the vision model, reviewed, and handled politely when impossible
    seen_msgs = []

    def vision_model(messages, **opts):
        seen_msgs.append(messages)
        has_img = any(m.get("images") for m in messages)
        assert has_img, "every vision call must carry the image"
        return "Proof: both bounds follow from the AM-GM inequality." if len(seen_msgs) == 1 else "Checked proof: it holds."

    png = {"media_type": "image/png", "data": "QUJD"}
    ve = ReasoningEngine(lambda m, **o: "x", vision_call=vision_model)
    vr = ve.respond([], "prove this", images=[png])
    assert vr.mode == "vision" and vr.text == "Checked proof: it holds." and len(seen_msgs) == 2, (vr, len(seen_msgs))
    seen_msgs.clear()
    vr2 = ve.respond([], "what is in this picture?", images=["data:image/png;base64,QUJD"])
    assert vr2.mode == "vision" and len(seen_msgs) == 1                      # no review for plain description
    no_vis = ReasoningEngine(lambda m, **o: "x").respond([], "prove this", images=[png])
    assert no_vis.mode == "error" and "image-capable" in no_vis.text
    bad = ve.respond([], "look", images=[{"media_type": "image/bmp", "data": "QUJD"}])
    assert bad.mode == "error" and "PNG" in bad.text
    print("selftest passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
