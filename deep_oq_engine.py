"""
Deep OQ Engine v2.1 (conversational + smarter)
==============================================
Drop-in upgrade for your FastAPI chatbot. Fixes the problems seen in your screenshots:

  * "Answer ... Answer ..." / "Problem: Let $f: Problem: Let $f:"  -> RepetitionGuard
  * Replies that stop mid-sentence                                  -> auto-continue on finish_reason=length
  * "I am Deep OQ, an advanced AI" (robotic, one line)              -> human persona prompt + style cleaner
  * Claims of checking without checking                             -> real tools (SymPy, Python sandbox)
  * Wrong/unchecked maths                                           -> draft -> verify -> revise pipeline
  * Broken JEE problem generation                                   -> verified problem generator (SymPy answers)
  * Works even when the LLM is down                                 -> local SymPy fallback solver

Run:
    pip install fastapi uvicorn httpx sympy mpmath pydantic
    export LLM_BASE_URL=https://api.openai.com/v1      # any OpenAI-compatible endpoint
    export LLM_API_KEY=...
    export LLM_MODEL=your-model-name
    uvicorn deep_oq_engine:app --host 0.0.0.0 --port 8000

Self test (no API key needed):
    python deep_oq_engine.py
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import httpx
import mpmath as mp
import sympy as sp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# ---------------------------------------------------------------------------
# 1. CONFIG  (nothing is capped unless you set it)
# ---------------------------------------------------------------------------


@dataclass
class Config:
    base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    api_key: str = os.getenv("LLM_API_KEY", "")
    model: str = os.getenv("LLM_MODEL", "gpt-4o")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.8"))
    # None => let the provider use its own maximum output length.
    max_tokens: Optional[int] = int(os.getenv("LLM_MAX_TOKENS", "0")) or None
    frequency_penalty: float = 0.35      # reduces word loops
    presence_penalty: float = 0.15
    max_continuations: int = 50          # auto-continue rounds when output is cut off
    max_tool_steps: int = 25             # tool calls per question
    verify_math: bool = True
    think_first: bool = True            # private planning pass before hard answers
    enforce_depth: bool = True          # expand thin answers automatically
    sandbox_timeout_s: int = 20
    sandbox_memory_mb: int = 1024
    sympy_timeout_s: int = 30
    history_compress_chars: int = 120_000  # compress only far beyond normal chats
    request_timeout_s: float = 300.0


CFG = Config()

# ---------------------------------------------------------------------------
# 2. PERSONA  (talks like a person, not a template)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Deep OQ. People talk to you the way they would talk to a brilliant,
warm friend who happens to know maths, code, science, logic and a lot about life.

HOW YOU TALK
- Real conversation, not a help-desk. Use natural sentences, contractions, and some personality.
  React to what the person actually said before you answer ("Oh, that's a good one", "Ugh, that
  sounds stressful", "Nice, JEE prep, let's go").
- Never give a one-line answer unless the person clearly wants one. Give substance: the answer,
  the reasoning behind it, a concrete example, and one useful extra thought they didn't ask for.
- Write in flowing paragraphs. Use a short list only when steps or options truly need it.
- Match their language and mood. If they write Hinglish, answer in Hinglish. If they are playful,
  be playful. If they are upset, slow down and be gentle.
- End most replies with ONE natural follow-up (a question, an offer, or a next step) so the
  conversation keeps going. Never end with the generic "How can I help you today?".
- Never open with labels like "Answer", "Step-by-Step Execution" or "Problem:". Never repeat yourself.
- Never say "As an AI language model". When asked about yourself, introduce yourself properly:
  your personality, what you are great at with concrete examples, and honestly what you are not.

HOW YOU THINK
- Work out what the person really needs, not just the literal words. Notice hidden assumptions.
- For maths, physics, logic and code: solve it, then check it with your tools before you commit.
- Show your thinking in a way a good teacher would: the key idea first, then the steps, why each
  step works, the final result stated clearly, and a common mistake to watch for.
- If a question is ambiguous, make the most sensible assumption, say what you assumed, and answer.
  Do not stall by asking questions you could answer yourself.
- If you are unsure, say so plainly and say how you would find out. Honesty beats sounding smart.
- Always finish what you start. Never trail off.

TOOLS: to call one, output exactly one block per call:
<tool>{"name": "math", "op": "diff", "expr": "Abs(log(2)-sin(x))", "var": "x"}</tool>
<tool>{"name": "python", "code": "print(sum(range(10)))"}</tool>
<tool>{"name": "puzzle", "query": "3 switches 3 bulbs"}</tool>
<tool>{"name": "generate_problem", "topic": "derivative", "difficulty": 3}</tool>
math ops: simplify, expand, factor, diff, integrate, limit, solve, series, numeric, check_derivative, matrix_det, matrix_inv.
After tool results arrive, continue in your normal conversational voice and give the final answer
with no tool blocks.
"""

# ---------------------------------------------------------------------------
# 3. REPETITION GUARD
# ---------------------------------------------------------------------------

_REPEAT_RE = re.compile(r"(.{6,400}?)(?:\s*\1){2,}", re.S)
_ROBOT_PHRASES = [
    r"as an ai language model,?\s*",
    r"i am just an ai,?\s*",
    r"i don't have personal (opinions|feelings),?\s*but\s*",
]


class RepetitionGuard:
    """Detects and removes the looping seen in the screenshots."""

    @staticmethod
    def collapse(text: str) -> str:
        prev = None
        out = text
        for _ in range(6):                       # run until stable
            prev = out
            out = _REPEAT_RE.sub(r"\1", out)
            if out == prev:
                break
        # drop duplicate consecutive sentences
        parts = re.split(r"(?<=[.!?])\s+", out)
        cleaned: list[str] = []
        for p in parts:
            if not cleaned or p.strip().lower() != cleaned[-1].strip().lower():
                cleaned.append(p)
        out = " ".join(cleaned)
        # drop a leading lone label like "Answer"
        out = re.sub(r"^\s*(\*\*)?answer(\*\*)?[:\s]+", "", out, flags=re.I)
        return out.strip()

    @staticmethod
    def is_looping(text: str) -> bool:
        words = text.split()
        if len(words) < 24:
            return False
        for n in (3, 4, 6, 8):
            grams: dict[tuple, int] = {}
            for i in range(len(words) - n):
                g = tuple(words[i : i + n])
                grams[g] = grams.get(g, 0) + 1
                if grams[g] >= 4:
                    return True
        return False

    @staticmethod
    def humanize(text: str) -> str:
        for pat in _ROBOT_PHRASES:
            text = re.sub(pat, "", text, flags=re.I)
        return text.strip()


# ---------------------------------------------------------------------------
# 4. LLM CLIENT  (auto-continue, loop retry)
# ---------------------------------------------------------------------------


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @property
    def available(self) -> bool:
        return bool(self.cfg.api_key)

    async def _once(self, messages: list[dict], **over: Any) -> tuple[str, str]:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": over.get("temperature", self.cfg.temperature),
            "frequency_penalty": over.get("frequency_penalty", self.cfg.frequency_penalty),
            "presence_penalty": over.get("presence_penalty", self.cfg.presence_penalty),
        }
        if self.cfg.max_tokens:
            payload["max_tokens"] = self.cfg.max_tokens
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        async with httpx.AsyncClient(timeout=self.cfg.request_timeout_s) as client:
            r = await client.post(f"{self.cfg.base_url}/chat/completions", json=payload, headers=headers)
        if r.status_code != 200:
            raise LLMError(f"LLM HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        choice = data["choices"][0]
        return choice["message"]["content"] or "", choice.get("finish_reason", "stop")

    async def complete(self, messages: list[dict], **over: Any) -> str:
        """Full answer: retries on loops, continues if cut off."""
        text, finish = await self._once(messages, **over)
        attempts = 0
        while RepetitionGuard.is_looping(text) and attempts < 3:
            attempts += 1
            bump = {"temperature": min(1.0, self.cfg.temperature + 0.15 * attempts),
                    "frequency_penalty": min(1.2, self.cfg.frequency_penalty + 0.3 * attempts)}
            text, finish = await self._once(messages, **bump)
        text = RepetitionGuard.collapse(text)

        rounds = 0
        while finish == "length" and rounds < self.cfg.max_continuations:
            rounds += 1
            cont = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Continue exactly where you stopped. Do not repeat anything."},
            ]
            more, finish = await self._once(cont, **over)
            if RepetitionGuard.is_looping(more):
                break
            text = (text + more).strip()
        return RepetitionGuard.collapse(text)


# ---------------------------------------------------------------------------
# 5. MATH TOOLKIT (SymPy + mpmath)
# ---------------------------------------------------------------------------

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_X, _Y, _Z, _T, _N = sp.symbols("x y z t n", real=True)
_LOCALS = {
    "x": _X, "y": _Y, "z": _Z, "t": _T, "n": _N,
    "log": sp.log, "ln": sp.log, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "sqrt": sp.sqrt, "exp": sp.exp,
    "Abs": sp.Abs, "abs": sp.Abs, "pi": sp.pi, "e": sp.E, "E": sp.E, "oo": sp.oo,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh, "factorial": sp.factorial,
    "binomial": sp.binomial, "Matrix": sp.Matrix,
}
_FORBIDDEN = re.compile(r"__|import|exec|eval|open\(|os\.|sys\.|subprocess|lambda", re.I)


def safe_parse(expr: str) -> sp.Expr:
    if _FORBIDDEN.search(expr):
        raise ValueError("Expression contains blocked tokens.")
    return parse_expr(expr, local_dict=_LOCALS, transformations=_TRANSFORMS)


def _fmt(e: Any) -> str:
    try:
        num = sp.N(e, 15)
        s = f"{sp.simplify(e)}"
        if getattr(e, "is_number", False):
            return f"{s}  (≈ {num})"
        return s
    except Exception:
        return str(e)


class MathTool:
    OPS = {"simplify", "expand", "factor", "diff", "integrate", "limit", "solve",
           "series", "numeric", "check_derivative", "matrix_det", "matrix_inv"}

    @classmethod
    def run(cls, call: dict) -> str:
        op = call.get("op", "simplify")
        if op not in cls.OPS:
            return f"Unknown math op '{op}'. Valid: {sorted(cls.OPS)}"
        vname = call.get("var", "x")
        var = _LOCALS.get(vname) or sp.Symbol(vname, real=True)
        expr = safe_parse(call["expr"]) if "expr" in call else None
        if op == "simplify":
            return _fmt(sp.simplify(expr))
        if op == "expand":
            return str(sp.expand(expr))
        if op == "factor":
            return str(sp.factor(expr))
        if op == "diff":
            order = int(call.get("order", 1))
            d = sp.diff(expr, var, order)
            at = call.get("at")
            res = f"d/d{var}: {d}"
            if at is not None:
                val = d.subs(var, safe_parse(str(at)))
                res += f"\nat {var}={at}: {_fmt(val)}"
            return res
        if op == "integrate":
            if "lower" in call and "upper" in call:
                lo, hi = safe_parse(str(call["lower"])), safe_parse(str(call["upper"]))
                return _fmt(sp.integrate(expr, (var, lo, hi)))
            return str(sp.integrate(expr, var)) + " + C"
        if op == "limit":
            to = safe_parse(str(call.get("to", "0")))
            return _fmt(sp.limit(expr, var, to, call.get("dir", "+-")))
        if op == "solve":
            sol = sp.solve(expr, var)
            return f"solutions for {var}: {sol}"
        if op == "series":
            return str(sp.series(expr, var, safe_parse(str(call.get("at", "0"))), int(call.get("n", 6))))
        if op == "numeric":
            mp.mp.dps = 30
            sub = {sp.Symbol(k): safe_parse(str(v)) for k, v in call.get("subs", {}).items()}
            return str(sp.N(expr.subs(sub), 20))
        if op == "check_derivative":
            return cls.check_derivative(expr, var, call.get("at", "0"))
        if op == "matrix_det":
            return _fmt(sp.Matrix(expr).det())
        if op == "matrix_inv":
            return str(sp.Matrix(expr).inv())
        return "unsupported"

    @staticmethod
    def check_derivative(expr: sp.Expr, var: sp.Symbol, at: Any) -> str:
        """Compare symbolic derivative to a high-precision numeric derivative."""
        mp.mp.dps = 40
        point = sp.N(safe_parse(str(at)), 40)
        symbolic = sp.N(sp.diff(expr, var).subs(var, point), 25)
        f = sp.lambdify(var, expr, "mpmath")
        numeric = mp.diff(f, mp.mpf(str(point)))
        ok = abs(mp.mpf(str(symbolic)) - numeric) < mp.mpf("1e-15")
        return f"symbolic={symbolic}  numeric={mp.nstr(numeric, 20)}  match={ok}"


# ---------------------------------------------------------------------------
# 6. PYTHON SANDBOX
# ---------------------------------------------------------------------------


class Sandbox:
    """Runs Python in a separate process with CPU/memory/time limits and a throwaway dir."""

    @staticmethod
    def run(code: str, timeout: int | None = None) -> str:
        timeout = timeout or CFG.sandbox_timeout_s

        def limits() -> None:  # POSIX only
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_CPU, (timeout, timeout))
                mem = CFG.sandbox_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            except Exception:
                pass

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "main.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(code)
            try:
                p = subprocess.run(
                    [sys.executable, "-I", path], cwd=d, capture_output=True, text=True,
                    timeout=timeout, preexec_fn=limits if os.name == "posix" else None,
                )
            except subprocess.TimeoutExpired:
                return f"Timed out after {timeout}s."
        out = p.stdout[-20000:]
        err = p.stderr[-5000:]
        return out + (f"\n[stderr]\n{err}" if err else "") or "(no output)"


# ---------------------------------------------------------------------------
# 7. PUZZLE LIBRARY
# ---------------------------------------------------------------------------

PUZZLES: dict[str, dict[str, Any]] = {
    "three switches": {
        "keys": ["switch", "bulb", "closed room", "enter the room only once"],
        "solution": (
            "Turn switch 1 on and leave it for about 10 minutes. Turn it off, turn switch 2 on, "
            "and walk in straight away. The bulb that is lit belongs to switch 2. Of the two dark "
            "bulbs, the warm one belongs to switch 1 (it was on long enough to heat up). "
            "The cold dark one belongs to switch 3. Three states (on, warm-off, cold-off) separate "
            "three bulbs with a single visit."
        ),
    },
    "two doors": {
        "keys": ["two guards", "one always lies", "two doors"],
        "solution": (
            "Ask either guard: 'What would the other guard say is the safe door?' Then pick the "
            "opposite. A liar reporting the truthful guard's answer, or a truth-teller reporting the "
            "liar's, both point to the wrong door."
        ),
    },
    "river crossing": {
        "keys": ["wolf", "goat", "cabbage"],
        "solution": (
            "Take the goat across. Go back alone. Take the wolf across and bring the goat back. "
            "Take the cabbage across, go back alone, then take the goat across again."
        ),
    },
    "nine coins": {
        "keys": ["balance", "heavier coin", "9 coins", "nine coins"],
        "solution": (
            "Split into three groups of 3. Weigh two groups. If one is heavier, it holds the odd "
            "coin; if they balance, it is in the third group. Then weigh two coins from that group "
            "the same way. Two weighings total."
        ),
    },
}


def find_puzzle(query: str) -> Optional[str]:
    q = query.lower()
    for name, p in PUZZLES.items():
        if sum(k in q for k in p["keys"]) >= 2:
            return p["solution"]
    return None


# ---------------------------------------------------------------------------
# 8. VERIFIED PROBLEM GENERATOR (fixes "Problem: Let $f: Problem: ...")
# ---------------------------------------------------------------------------


@dataclass
class Problem:
    topic: str
    statement: str
    options: list[str]
    correct: int
    solution: str


class ProblemGenerator:
    """Every answer is computed by SymPy, so the key is always right."""

    TOPICS = ["derivative", "integral", "limit", "quadratic", "composition"]

    @classmethod
    def make(cls, topic: str = "random", difficulty: int = 2, seed: int | None = None) -> Problem:
        rnd = random.Random(seed)
        if topic == "random" or topic not in cls.TOPICS:
            topic = rnd.choice(cls.TOPICS)
        return getattr(cls, f"_{topic}")(rnd, max(1, min(5, difficulty)))

    @staticmethod
    def _options(correct: Any, rnd: random.Random, wrong: list[Any]) -> tuple[list[str], int]:
        pool = [correct] + [w for w in wrong if sp.simplify(w - correct) != 0][:3]
        while len(pool) < 4:
            pool.append(correct + rnd.choice([1, -1, 2, -2]) * rnd.randint(1, 5))
        rnd.shuffle(pool)
        idx = next(i for i, v in enumerate(pool) if sp.simplify(v - correct) == 0)
        return [str(sp.nsimplify(v)) for v in pool], idx

    @classmethod
    def _derivative(cls, rnd, d):
        a, b = rnd.randint(2, 4 + d), rnd.randint(1, 5)
        f = _X**a + b * sp.sin(_X) * (sp.exp(_X) if d >= 3 else 1)
        x0 = rnd.choice([0, 1])
        ans = sp.diff(f, _X).subs(_X, x0)
        opts, idx = cls._options(ans, rnd, [ans + 1, ans - b, ans * 2])
        return Problem("derivative", f"Let f(x) = {f}. Find f'({x0}).", opts, idx,
                       f"f'(x) = {sp.diff(f, _X)}, so f'({x0}) = {ans}.")

    @classmethod
    def _integral(cls, rnd, d):
        a, b = rnd.randint(1, 3), rnd.randint(2, 3 + d)
        f = a * _X ** (b - 1) + sp.cos(_X) * (1 if d >= 2 else 0)
        ans = sp.integrate(f, (_X, 0, 1))
        opts, idx = cls._options(ans, rnd, [ans + 1, ans - sp.Rational(1, 2), 2 * ans])
        return Problem("integral", f"Evaluate the integral of {f} from x = 0 to x = 1.", opts, idx,
                       f"Antiderivative {sp.integrate(f, _X)}; evaluating 0 to 1 gives {ans}.")

    @classmethod
    def _limit(cls, rnd, d):
        k = rnd.randint(2, 3 + d)
        f = (sp.exp(k * _X) - 1) / _X
        ans = sp.limit(f, _X, 0)
        opts, idx = cls._options(ans, rnd, [ans + 1, ans - 1, 1])
        return Problem("limit", f"Find the limit as x → 0 of {f}.", opts, idx,
                       f"This is the derivative of e^({k}x) at 0, which is {ans}.")

    @classmethod
    def _quadratic(cls, rnd, d):
        r1, r2 = rnd.randint(-6, 6), rnd.randint(-6, 6)
        poly = sp.expand((_X - r1) * (_X - r2))
        ans = r1 + r2
        opts, idx = cls._options(sp.Integer(ans), rnd, [sp.Integer(r1 * r2), sp.Integer(-ans), sp.Integer(ans + 1)])
        return Problem("quadratic", f"Find the sum of the roots of {poly} = 0.", opts, idx,
                       f"Roots are {r1} and {r2}; their sum is {ans}.")

    @classmethod
    def _composition(cls, rnd, d):
        c = rnd.choice([2, 3, 5])
        f = sp.Abs(sp.log(c) - sp.sin(_X))
        g = f.subs(_X, f)
        ans = sp.N(sp.diff(g, _X).subs(_X, 0), 15)
        true = sp.cos(sp.log(c))
        opts, idx = cls._options(true, rnd, [-true, sp.sin(sp.log(c)), -sp.sin(sp.log(c))])
        return Problem("composition",
                       f"For x in R, let f(x) = |log {c} - sin x| and g(x) = f(f(x)). Find g'(0).",
                       opts, idx, f"g'(0) = f'(f(0))·f'(0) = {true} ≈ {ans}.")

    @staticmethod
    def render(p: Problem) -> str:
        letters = "ABCD"
        lines = [p.statement, ""] + [f"{letters[i]}) {o}" for i, o in enumerate(p.options)]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 9. TOOL DISPATCH
# ---------------------------------------------------------------------------

TOOL_RE = re.compile(r"<tool>(.*?)</tool>", re.S)


async def run_tool(call: dict) -> str:
    name = call.get("name")
    try:
        if name == "math":
            return await asyncio.wait_for(asyncio.to_thread(MathTool.run, call), CFG.sympy_timeout_s)
        if name == "python":
            return await asyncio.to_thread(Sandbox.run, call.get("code", ""))
        if name == "puzzle":
            return find_puzzle(call.get("query", "")) or "No stored puzzle matched. Reason it out directly."
        if name == "generate_problem":
            p = ProblemGenerator.make(call.get("topic", "random"), int(call.get("difficulty", 2)))
            return (ProblemGenerator.render(p) + f"\n\n[KEY: {'ABCD'[p.correct]}]\n[SOLUTION: {p.solution}]")
        return f"Unknown tool '{name}'."
    except asyncio.TimeoutError:
        return "Tool timed out."
    except Exception as exc:  # report, never crash
        return f"Tool error: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# 10. LOCAL FALLBACK SOLVER  (works with no LLM)
# ---------------------------------------------------------------------------

_LOCAL_PATTERNS = [
    (re.compile(r"(?:derivative|differentiate|diff)\s+(?:of\s+)?(.+)", re.I), "diff"),
    (re.compile(r"(?:integrate|integral of)\s+(.+)", re.I), "integrate"),
    (re.compile(r"(?:solve)\s+(.+?)(?:\s*=\s*0)?$", re.I), "solve"),
    (re.compile(r"(?:simplify)\s+(.+)", re.I), "simplify"),
    (re.compile(r"(?:factor(?:ise|ize)?)\s+(.+)", re.I), "factor"),
]


def local_solve(text: str) -> Optional[str]:
    for pat, op in _LOCAL_PATTERNS:
        m = pat.search(text.strip())
        if m:
            expr = m.group(1).strip().rstrip("?.")
            if "=" in expr and op == "solve":
                lhs, rhs = expr.split("=", 1)
                expr = f"({lhs})-({rhs})"
            try:
                return MathTool.run({"op": op, "expr": expr})
            except Exception as exc:
                return f"Couldn't parse that expression ({exc})."
    puzzle = find_puzzle(text)
    return puzzle


# ---------------------------------------------------------------------------
# 11. SESSIONS / MEMORY
# ---------------------------------------------------------------------------


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    messages: list[dict] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    last_intent: str = ""

    def remember(self, text: str) -> None:
        m = re.search(r"my name is ([A-Za-z]+)", text, re.I)
        if m:
            self.facts["name"] = m.group(1)
        m = re.search(r"i(?:'m| am) (?:in class|in) (\d{1,2}(?:th)?)", text, re.I)
        if m:
            self.facts["class"] = m.group(1)
        m = re.search(r"i (?:like|love|enjoy) ([A-Za-z\- ]{3,30})", text, re.I)
        if m:
            self.facts["likes"] = m.group(1).strip()
        m = re.search(r"i(?:'m| am) preparing for ([A-Za-z\- ]+)", text, re.I)
        if m:
            self.facts["preparing_for"] = m.group(1).strip()

    def memory_note(self) -> str:
        if not self.facts:
            return ""
        return "Things you know about the user: " + "; ".join(f"{k}={v}" for k, v in self.facts.items())


SESSIONS: dict[str, Session] = {}

# ---------------------------------------------------------------------------
# 12. THE ENGINE
# ---------------------------------------------------------------------------

_MATHY = re.compile(r"(derivative|integral|integrate|limit|solve|prove|probability|matrix|equation|"
                    r"g'\(|f\(x\)|jee|calculate|simplify|log|sin|cos)", re.I)



# ---------------------------------------------------------------------------
# 11b. INTENT, MOOD AND DEPTH CONTROL  (makes it talk, not just answer)
# ---------------------------------------------------------------------------

_GREET_RE = re.compile(r"^\s*(hi+|hello+|hey+|yo|namaste|hola|sup|good (morning|evening|afternoon|night))\b[\s!.?]*$", re.I)
_ABOUT_RE = re.compile(r"(tell me about (you|yourself)|who are you|what are you|full introduction|"
                       r"introduce yourself|what can you do|about you)", re.I)
_EMO_RE = re.compile(r"(i feel|i'm (so )?(sad|tired|stressed|anxious|scared|lonely|worried|nervous|low)|"
                     r"i am (so )?(sad|tired|stressed|anxious|scared|lonely|worried|nervous|low)|depress|"
                     r"can't sleep|exam stress|i failed|nobody (likes|cares)|feeling (low|down|lost))", re.I)
_CODE_RE = re.compile(r"(\bcode\b|python|javascript|typescript|\bbug\b|traceback|error|function|\bapi\b|"
                      r"fastapi|html|css|\bsql\b|debug|github|render|deploy)", re.I)
_PUZZLE_RE = re.compile(r"(puzzle|riddle|switches|bulbs|brain ?teaser|how can you determine)", re.I)


def classify(text: str) -> str:
    t = text.strip()
    if _GREET_RE.match(t):
        return "greeting"
    if _ABOUT_RE.search(t):
        return "about_me"
    if _EMO_RE.search(t):
        return "emotional"
    if _PUZZLE_RE.search(t) or find_puzzle(t):
        return "puzzle"
    if _MATHY.search(t):
        return "math"
    if _CODE_RE.search(t):
        return "code"
    if len(t.split()) <= 6 and not t.endswith("?"):
        return "casual"
    return "question"


GUIDE: dict[str, str] = {
    "greeting": (
        "They are just saying hello. Reply warmly in 2-4 sentences like a friend would. Add a touch "
        "of personality and ask one natural question about what they are up to or working on."),
    "about_me": (
        "Introduce yourself properly in 2-3 short paragraphs: who you are, your personality, what you "
        "are genuinely good at (give 3-4 concrete things they could ask, like solving a JEE problem "
        "step by step, debugging code, cracking a logic puzzle, or helping them plan a study week), "
        "and honestly what you cannot do (you can be wrong, you verify important things, you do not "
        "see the internet unless connected). Sound like a person at a first meeting. End by inviting "
        "them to try something."),
    "emotional": (
        "The person may be struggling. Lead with empathy and reflect what they said in your own words. "
        "Do not rush to fix or lecture. Offer one or two gentle, practical ideas, then ask how they are "
        "doing. If they hint at self-harm or being unsafe, take it seriously, encourage them to reach "
        "out right now to someone they trust or a local crisis helpline, and stay with them."),
    "math": (
        "Be a great teacher. Restate the problem in a line, name the key idea, solve step by step "
        "explaining WHY each step works, verify with tools, state the final answer clearly, then "
        "mention one common mistake or a shortcut. Offer a similar practice problem at the end."),
    "code": (
        "Explain your approach in plain words first, then give complete, working code, then explain "
        "the tricky parts, edge cases, and how to test it. Mention likely pitfalls on their setup."),
    "puzzle": (
        "Think out loud like someone enjoying the puzzle: what the constraint really is, the key "
        "insight, the full solution step by step, why it works, and why obvious alternatives fail."),
    "casual": (
        "Chat naturally. Respond to what they said with real interest, add something of your own "
        "(an opinion, a related thought, a fun fact), and keep the conversation moving."),
    "question": (
        "Give a genuinely useful answer: direct answer first, then the reasoning, a concrete example, "
        "and one 'next thing worth considering'. Aim for roughly 150-400 words unless the topic is "
        "truly tiny."),
}

TEMPS: dict[str, float] = {"greeting": 0.9, "about_me": 0.85, "emotional": 0.8, "math": 0.3,
                           "code": 0.4, "puzzle": 0.5, "casual": 0.9, "question": 0.7}
MIN_WORDS: dict[str, int] = {"greeting": 25, "about_me": 120, "emotional": 70, "math": 110,
                             "code": 90, "puzzle": 90, "casual": 45, "question": 130}
THINK_INTENTS = {"math", "code", "puzzle", "question"}


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


OFFLINE_REPLIES: dict[str, str] = {
    "greeting": ("Hey, good to see you! I'm Deep OQ. I'm running in a lighter mode at the moment, "
                 "but I'm still here. What are you working on today? If it's a maths problem, throw it at me."),
    "about_me": ("I'm Deep OQ, an AI built to be someone you can actually talk to, not just a search box. "
                 "I like maths (derivatives, integrals, JEE-style problems), logic puzzles, and debugging code. "
                 "Right now my language model isn't connected, so I can only do direct calculations and a few "
                 "classic puzzles. Once the API key is set I can explain things properly, in full conversation. "
                 "Try me with something like 'derivative of x^3 sin x'."),
    "emotional": ("That sounds heavy, and I'm glad you said it out loud. I'm in a limited mode right now, "
                  "so I can't give you the kind of full conversation you deserve, but please don't carry it alone. "
                  "Talking to a friend, someone in your family, or a counsellor can really help. "
                  "How long has it been feeling like this?"),
}


class DeepOQ:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.llm = LLMClient(cfg)

    # -- history compression (only when truly huge) -------------------------
    async def _compress(self, s: Session) -> None:
        total = sum(len(m["content"]) for m in s.messages)
        if total < self.cfg.history_compress_chars or not self.llm.available:
            return
        old, recent = s.messages[:-10], s.messages[-10:]
        summary = await self.llm.complete([
            {"role": "system", "content": "Summarise this chat, keeping every fact, number and decision."},
            {"role": "user", "content": json.dumps(old)[:200_000]},
        ])
        s.messages = [{"role": "system", "content": f"Earlier conversation summary: {summary}"}] + recent

    # -- agent loop -----------------------------------------------------------
    async def _agent(self, msgs: list[dict], **over: Any) -> str:
        for _ in range(self.cfg.max_tool_steps):
            text = await self.llm.complete(msgs, **over)
            calls = TOOL_RE.findall(text)
            if not calls:
                return text
            results = []
            for raw in calls:
                try:
                    call = json.loads(raw)
                except json.JSONDecodeError:
                    results.append("Bad tool JSON; resend as valid JSON.")
                    continue
                results.append(f"[{call.get('name')}] {await run_tool(call)}")
            msgs = msgs + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "TOOL RESULTS:\n" + "\n".join(results)
                 + "\nUse these facts. Finish the answer; no more tool blocks unless needed."},
            ]
        return await self.llm.complete(msgs + [{"role": "user", "content": "Give your final answer now."}], **over)

    # -- verification ---------------------------------------------------------
    async def _verify(self, question: str, draft: str) -> str:
        review = await self.llm.complete([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Question:\n{question}\n\nDraft answer:\n{draft}\n\n"
                "Check the draft like a strict examiner. Re-verify every calculation with tools where "
                "possible. If everything is right reply exactly: VERDICT: OK. Otherwise reply "
                "'VERDICT: FIX' followed by the corrected full answer in a natural voice.")},
        ])
        if "VERDICT: FIX" in review:
            return review.split("VERDICT: FIX", 1)[1].strip(" :\n")
        return draft

    # -- private planning pass --------------------------------------------------
    async def _plan(self, history: list[dict], user_text: str) -> str:
        try:
            return await self.llm.complete([
                {"role": "system", "content": (
                    "You are a private reasoning scratchpad for an assistant. The user will never see this. "
                    "In under 160 words list: what is REALLY being asked, what the person probably cares "
                    "about, key facts or steps, likely traps, and what an excellent answer includes.")},
                *history[-6:],
                {"role": "user", "content": user_text},
            ], temperature=0.3)
        except LLMError:
            return ""

    # -- expand thin answers ----------------------------------------------------
    async def _ensure_depth(self, user_text: str, draft: str, intent: str) -> str:
        need = MIN_WORDS.get(intent, 100)
        for _ in range(2):
            if word_count(draft) >= need:
                return draft
            try:
                richer = await self.llm.complete([
                    {"role": "system", "content": SYSTEM_PROMPT + "\n" + GUIDE[intent]},
                    {"role": "user", "content": (
                        f"The user said:\n{user_text}\n\nYour draft was too thin:\n{draft}\n\n"
                        f"Rewrite it as a fuller, warmer, conversational reply of at least {need} words. "
                        "Keep every correct fact from the draft, add reasoning, an example, and a natural "
                        "follow-up. Do not mention the draft.")},
                ], temperature=TEMPS.get(intent, 0.8))
            except LLMError:
                return draft
            if word_count(richer) > word_count(draft) and not RepetitionGuard.is_looping(richer):
                draft = richer
            else:
                break
        return draft

    # -- public entry ---------------------------------------------------------
    async def reply(self, session: Session, user_text: str) -> str:
        session.remember(user_text)
        intent = classify(user_text)
        session.messages.append({"role": "user", "content": user_text})

        if not self.llm.available:
            out = (local_solve(user_text) if intent in ("math", "puzzle") else None) \
                or OFFLINE_REPLIES.get(intent) \
                or ("My language model isn't connected right now, so I can only do direct maths "
                    "(derivatives, integrals, solving equations) and a few classic puzzles. "
                    "Once LLM_API_KEY is set on the server, I'll be able to talk about anything. "
                    "Want to try a maths question in the meantime?")
            session.messages.append({"role": "assistant", "content": out})
            return out

        await self._compress(session)
        system = SYSTEM_PROMPT + "\n\nSITUATION GUIDE FOR THIS MESSAGE:\n" + GUIDE[intent]
        if session.facts:
            system += "\n" + session.memory_note() + " Use it naturally, like a friend who remembers."
        history = session.messages[:-1]
        try:
            if self.cfg.think_first and intent in THINK_INTENTS and word_count(user_text) > 6:
                notes = await self._plan(history, user_text)
                if notes:
                    system += "\n\nYour private notes (never mention them):\n" + notes
            msgs = [{"role": "system", "content": system}] + session.messages
            draft = await self._agent(msgs, temperature=TEMPS[intent])
            final = draft
            if self.cfg.verify_math and intent in ("math", "puzzle", "code") and word_count(draft) > 25:
                final = await self._verify(user_text, draft)
            if self.cfg.enforce_depth:
                final = await self._ensure_depth(user_text, final, intent)
        except LLMError as exc:
            final = local_solve(user_text) or f"I hit a snag reaching my model ({exc}). Give me a moment and try again?"
        final = RepetitionGuard.humanize(RepetitionGuard.collapse(final))
        session.messages.append({"role": "assistant", "content": final})
        session.last_intent = intent
        return final


ENGINE = DeepOQ()

# ---------------------------------------------------------------------------
# 13. FASTAPI APP
# ---------------------------------------------------------------------------

app = FastAPI(title="Deep OQ Engine", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None


class MathIn(BaseModel):
    op: str
    expr: Optional[str] = None
    var: str = "x"
    extra: dict = {}


class CodeIn(BaseModel):
    code: str


class GenIn(BaseModel):
    topic: str = "random"
    difficulty: int = 2


def _session(sid: Optional[str]) -> Session:
    if sid and sid in SESSIONS:
        return SESSIONS[sid]
    s = Session()
    SESSIONS[s.id] = s
    return s


@app.post("/chat")
async def chat(body: ChatIn):
    s = _session(body.session_id)
    reply = await ENGINE.reply(s, body.message)
    return {"session_id": s.id, "reply": reply, "intent": s.last_intent}


@app.post("/chat/stream")
async def chat_stream(body: ChatIn):
    s = _session(body.session_id)
    text = await ENGINE.reply(s, body.message)

    async def gen() -> AsyncIterator[str]:
        for tok in re.findall(r"\S+\s*", text):
            yield f"data: {json.dumps({'t': tok})}\n\n"
            await asyncio.sleep(0.008)
        yield f"data: {json.dumps({'done': True, 'session_id': s.id})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/solve/math")
async def solve_math(body: MathIn):
    call = {"op": body.op, "var": body.var, **body.extra}
    if body.expr:
        call["expr"] = body.expr
    return {"result": await run_tool({"name": "math", **call})}


@app.post("/run/python")
async def run_python(body: CodeIn):
    return {"output": await run_tool({"name": "python", "code": body.code})}


@app.post("/problems/generate")
def generate(body: GenIn):
    p = ProblemGenerator.make(body.topic, body.difficulty)
    return {"topic": p.topic, "question": p.statement, "options": p.options,
            "answer": "ABCD"[p.correct], "solution": p.solution}


@app.get("/health")
def health():
    return {"ok": True, "llm": ENGINE.llm.available, "model": CFG.model}


# ---------------------------------------------------------------------------
# 14. SELF TEST
# ---------------------------------------------------------------------------


def _selftest() -> None:
    print("1) Repetition guard")
    bad = "Answer You can use heat. " * 5 + "Problem: Let $f: " * 6
    print("   ", RepetitionGuard.collapse(bad)[:80])

    print("2) Your JEE question: f(x)=|log2 - sin x|, g=f(f(x)), g'(0)")
    f = sp.Abs(sp.log(2) - sp.sin(_X))
    g = f.subs(_X, f)
    print("   ", MathTool.check_derivative(g, _X, 0))
    print("    cos(log 2) =", sp.N(sp.cos(sp.log(2)), 15), "-> option C")

    print("3) Puzzle:", (find_puzzle("3 switches outside a closed room, 3 bulbs, enter once") or "")[:70], "...")
    print("4) Generated problem:")
    print(textwrap.indent(ProblemGenerator.render(ProblemGenerator.make("derivative", 3, seed=1)), "    "))
    print("5) Local solver:", local_solve("derivative of x^3*sin(x)"))
    print("6) Sandbox:", Sandbox.run("print(sum(range(101)))"))
    print("7) Intent routing:")
    for q in ["hi", "tell me about you", "full introduction", "give me jee problem", "i feel so stressed about exams",
              "my fastapi code has an error", "how does a transformer work?", "nice"]:
        print(f"    {q!r:42} -> {classify(q)}")


if __name__ == "__main__":
    _selftest()
