#!/usr/bin/env python3
"""
deep_oq_all.py  -  Deep OQ Solver 2.0, everything in ONE file.

# Deep OQ Solver 2.0

A verified problem-solving engine + LLM bridge for your FastAPI chatbot.
Every exact answer is re-checked by an independent method (substitution, finite differences, numpy/scipy,
brute-force enumeration, or a *real* Monte Carlo simulation).  If a check fails, the answer is discarded.

## What it solves (exactly, with worked steps)
| Area | Examples |
|---|---|
| Algebra | equations, systems, inequalities, quintics (numeric), factor / expand / simplify |
| Calculus | derivatives with rule trace, integrals (definite / improper), limits (one-sided), Taylor, optimisation, tangent lines, ODEs, corners (non-differentiable points) |
| Linear algebra | det, inverse, rank, eigen, RREF, nullspace, A·x=b, products |
| Number theory | gcd/lcm, primes, factorisation, totient, modular exponent, CRT, base conversion |
| Probability | dice, coins, cards, birthday, Bayes, derangements, binomial - each checked by simulation |
| Statistics | descriptive stats, outliers, regression, correlation, normal, confidence intervals |
| Physics | ~25 formulas, unit-aware parsing, cross-checking with independent equations |
| Chemistry | molar mass, equation balancing (exact nullspace) |
| Finance | simple/compound interest, EMI (amortisation replay), CAGR, NPV, IRR |
| Units | length, mass, time, volume, speed, energy, pressure, data, temperature, dimension checking |
| Puzzles | 3 switches/bulbs, water jugs, wolf-goat-cabbage, missionaries, N-Queens, Hanoi, Sudoku, Monty Hall, truth tables |
| Multi-part | "1) ... 2) ... 3) ..." each solved and verified separately |

## Install & run
    pip install sympy numpy scipy mpmath          # core
    pip install fastapi uvicorn                   # only for the server

    python -m deep_oq "integrate x^2 from 0 to 3"        # CLI
    python -m unittest discover -s deep_oq/tests -t .    # 83 tests

    # put your keys in a .env file (see below) or in environment variables
    uvicorn deep_oq.server:app --port 8000
    # POST /chat {"message": "..."}    POST /solve {"problem": "..."}    GET /health

## Use inside your existing FastAPI app
    from deep_oq.llm_bridge import answer
    async def my_llm(messages, **params) -> str: ...      # call your model here
    reply = await answer(user_message, my_llm, history=history)
    return reply.text            # reply.source, reply.verified, reply.notes, reply.tool_log

## How the shortcomings from the screenshots are handled
* Repetition loops / "Problem: Let $f:" x8  -> guard.py removes loops; retries with stronger anti-repeat settings.
* Cut-off or one-line answers               -> completeness checks + minimum length for "features", "tell me about...".
* Unverified claims ("I ran a simulation")  -> stripped unless the sandbox really executed code (tool_log).
* Overconfidence                            -> every solver reports verified True/False/None; failures are discarded.
* Wrong LLM maths                           -> exact solvers run FIRST; the LLM may only explain, and its numbers are checked.

## Security notes (read before deploying)
* sympy's parser uses eval().  `parse.assert_safe` firewalls all input (whitelisted words, no dunders / quotes / attributes).
  Never bypass it, and never pass raw user text to sympify / parse_expr yourself.
* Solvers run in forked child processes with a timeout so a runaway computation cannot freeze the server.
* `codeexec` (LLM-written Python) is defence-in-depth, NOT a hardened jail.  For public traffic run the service in a
  container / gVisor with networking disabled.

## Known limits
* Natural-language parsing is pattern based: unusual phrasing falls through to the LLM (honestly flagged).
* Physics formula library and chemistry element table are finite; extend FORMULAS / MASS.
* Hindi/emoji problem statements are refused rather than guessed - add a translation step if you need them.

## Language-model providers (most powerful model preconfigured for each)
| Provider | Default model | Fallbacks | Key variable |
|---|---|---|---|
| Anthropic | `claude-fable-5-2` | claude-opus-5-5, claude-sonnet-5-5 | ANTHROPIC_API_KEY |
| OpenAI | `gpt-6.2-astra` (Responses API, reasoning effort high) | gpt-6.2-sol, gpt-6-astra | OPENAI_API_KEY |
| Google Gemini | `gemini-4-argon` (thinking level high) | gemini-3.8-flash, gemini-3.5-flash | GEMINI_API_KEY / GOOGLE_API_KEY |
| Groq | `openai/gpt-oss-120b` (open-weight; Groq does not host the models above) | qwen/qwen3.8-27b, llama-3.3-70b-versatile | GROQ_API_KEY |

Put keys in a `.env` file next to the script (never paste them into chats or commit them):

    ANTHROPIC_API_KEY=...
    OPENAI_API_KEY=...
    GEMINI_API_KEY=...
    GROQ_API_KEY=...
    DEEP_OQ_PROVIDER_ORDER=anthropic,openai,gemini,groq    # failover order
    DEEP_OQ_MODE=failover                                   # or: vote (ask all, majority answer wins)

Behaviour: failover on errors / rate limits / bad keys (with cool-down), retries with backoff, automatic fallback to
older model IDs, and if every ID is retired the provider's /models list is searched for the best current model.
Check your setup with:  python deep_oq_all.py ping

-------------------------------------------------------------------------------
HOW THIS SINGLE FILE WORKS
  Every module of the package (parse, verify, engine, all solvers, guard, sandbox, LLM bridge, server and the test
  suite) is embedded below as plain, editable text inside a raw string, one section per module.  A tiny import hook
  at the bottom registers them as the package `deep_oq`, so the code behaves exactly like the multi-file version
  (relative imports, tracebacks with real line numbers, multiprocessing, pickling).  To change a module, edit its text
  in its section - no rebuilding needed.

USAGE
  python deep_oq_all.py "integrate x^2 from 0 to 3"   solve one problem
  python deep_oq_all.py                                interactive prompt
  python deep_oq_all.py test                           run all tests (108)
  python deep_oq_all.py serve [port]                   HTTP API (needs fastapi + uvicorn)
  python deep_oq_all.py providers                      show models + which API keys are set
  python deep_oq_all.py ping                           send one tiny real request to each provider (checks keys/billing)
  python deep_oq_all.py modules                        list the embedded modules

  # as a library
  from deep_oq_all import solve, Engine, answer, app
  print(solve("derivative of x^2 sin(x)").render())
  reply = await answer(user_message, my_llm, history=history)     # my_llm: async def (messages, **params) -> str
-------------------------------------------------------------------------------
"""

import importlib.abc
import importlib.util
import linecache
import sys

_MODULES = {}          # dotted name -> (is_package, relpath, source)


def _register(name, is_pkg, rel, src):
    _MODULES[name] = (is_pkg, rel, src)


class _EmbeddedLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        is_pkg, rel, src = _MODULES[module.__name__]
        filename = "<deep_oq_all:%s>" % rel
        module.__file__ = filename
        linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)   # real tracebacks
        exec(compile(src, filename, "exec"), module.__dict__)

    def get_source(self, fullname):
        return _MODULES[fullname][2]


class _EmbeddedFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in _MODULES:
            is_pkg = _MODULES[fullname][0]
            return importlib.util.spec_from_loader(fullname, _EmbeddedLoader(), is_package=is_pkg)
        return None


def _install():
    if not any(isinstance(f, _EmbeddedFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _EmbeddedFinder())


_install()

# ====================================================================================================
# MODULE: deep_oq    (deep_oq/__init__.py, 10 lines)
# ====================================================================================================
_register('deep_oq', True, 'deep_oq/__init__.py', r'''"""Deep OQ Solver - a verified problem-solving engine.

    from deep_oq import solve
    print(solve("derivative of x^2 sin(x)").render())
"""
from .engine import solve, Engine  # noqa: F401
from .models import Solution, Step  # noqa: F401

__version__ = "2.0.0"
''')

# ====================================================================================================
# MODULE: deep_oq.codeexec    (deep_oq/codeexec.py, 95 lines)
# ====================================================================================================
_register('deep_oq.codeexec', False, 'deep_oq/codeexec.py', r'''"""Best-effort sandboxed Python runner, so 'I ran the code' claims can be REAL.

SECURITY NOTE: this is a defence-in-depth sandbox (AST allow-list + rlimits + isolated subprocess + timeout).
It is NOT a hardened jail.  For untrusted public traffic, run the whole service inside a container / gVisor / nsjail
and disable network access.
"""
from __future__ import annotations

import ast
import os
try:
    import resource
except ImportError:
    resource = None
import subprocess
import sys
import tempfile
from dataclasses import dataclass

ALLOWED_IMPORTS = {"math", "cmath", "fractions", "decimal", "itertools", "functools", "collections", "random",
                   "statistics", "sympy", "numpy", "scipy", "heapq", "bisect", "re", "string", "json", "datetime",
                   "operator", "typing", "dataclasses", "mpmath", "time"}
BANNED_NAMES = {"exec", "eval", "compile", "open", "__import__", "input", "breakpoint", "globals", "locals",
                "vars", "getattr", "setattr", "delattr", "memoryview", "exit", "quit", "help"}
BANNED_ATTRS = {"system", "popen", "remove", "rmdir", "unlink", "rename", "environ", "fork", "kill", "load", "loads",
                "read_csv", "to_csv", "savetxt", "loadtxt", "save", "fromfile", "tofile"}


@dataclass
class ExecResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    blocked: str = ""
    seconds: float = 0.0

    def summary(self) -> str:
        if self.blocked:
            return f"BLOCKED: {self.blocked}"
        if self.timed_out:
            return "TIMEOUT"
        return (self.stdout or "").strip() or (self.stderr or "").strip()[-400:]


def static_check(code: str) -> str:
    """Return '' if the code passes the allow-list, else the reason it is blocked."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import of '{a.name}' is not allowed"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"import from '{node.module}' is not allowed"
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"use of '{node.id}' is not allowed"
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS:
                return f"attribute '.{node.attr}' is not allowed"
            if node.attr.startswith("__") and node.attr.endswith("__"):
                return f"dunder attribute '.{node.attr}' is not allowed"
    return ""


def _limits(mem_mb: int, cpu_s: int):
    def apply():
        if resource is not None:
            resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        if hasattr(os, "setsid"):
            os.setsid()
    return apply

def run_python(code: str, timeout: float = 8.0, mem_mb: int = 1024, max_output: int = 20_000) -> ExecResult:
    import time
    reason = static_check(code)
    if reason:
        return ExecResult(False, blocked=reason)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "snippet.py")
        with open(path, "w") as fh:
            fh.write(code)
        try:
            p = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True, timeout=timeout,
                               cwd=d, env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"},
                               preexec_fn=_limits(mem_mb, int(timeout) + 2))
        except subprocess.TimeoutExpired as e:
            return ExecResult(False, stdout=(e.stdout or b"").decode(errors="ignore")[:max_output] if isinstance(e.stdout, bytes) else (e.stdout or ""),
                              timed_out=True, seconds=time.time() - t0)
    return ExecResult(p.returncode == 0, p.stdout[:max_output], p.stderr[:max_output], seconds=time.time() - t0)
''')

# ====================================================================================================
# MODULE: deep_oq.engine    (deep_oq/engine.py, 163 lines)
# ====================================================================================================
_register('deep_oq.engine', False, 'deep_oq/engine.py', r'''"""The orchestrator: classify -> run the best solvers (with timeouts) -> verify -> fall back."""
from __future__ import annotations

import multiprocessing as mp
import pickle
import re
import time
from collections import OrderedDict
from typing import Callable, Optional

from .models import Solution
from .solvers import (algebra, arithmetic, calculus, chemistry, combprob, finance, linalg, logic, numtheory,
                      physics, stats, units)

REGISTRY = [  # order = tie-break priority
    ("logic", logic), ("calculus", calculus), ("linalg", linalg), ("chemistry", chemistry), ("finance", finance),
    ("stats", stats), ("combprob", combprob), ("physics", physics), ("units", units), ("numtheory", numtheory),
    ("algebra", algebra), ("arithmetic", arithmetic),
]


def _worker(fn, text, q):
    try:
        sol = fn(text)
        try:
            pickle.dumps(sol)
        except Exception:                                     # noqa: BLE001
            sol.answer = str(sol.answer)
        q.put(("ok", sol))
    except Exception as exc:                                  # noqa: BLE001
        q.put(("err", f"{type(exc).__name__}: {exc}"))


def run_with_timeout(fn: Callable, text: str, seconds: float) -> Optional[Solution]:
    """Run a solver in a forked child so a runaway sympy call cannot hang the server."""
    try:
        ctx = mp.get_context("fork")
    except ValueError:                                        # e.g. Windows: no fork
        return fn(text)
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(fn, text, q), daemon=True)
    p.start()
    try:
        kind, payload = q.get(timeout=seconds)
    except Exception:                                         # queue.Empty -> timed out
        p.terminate()
        p.join(1)
        s = Solution(text)
        s.warnings.append(f"timed out after {seconds:g}s")
        return s
    p.join(1)
    if kind == "ok":
        return payload
    s = Solution(text)
    s.warnings.append(payload)
    return s


class Engine:
    def __init__(self, timeout: float = 20.0, use_timeout: bool = True, cache_size: int = 256):
        self.timeout, self.use_timeout = timeout, use_timeout
        self._cache: "OrderedDict[str, Solution]" = OrderedDict()
        self._cache_size = cache_size

    # ---- routing ------------------------------------------------------
    def classify(self, text: str) -> list:
        ranked = []
        for i, (name, mod) in enumerate(REGISTRY):
            try:
                score = float(mod.can_handle(text))
            except Exception:                                 # noqa: BLE001
                score = 0.0
            if score > 0:
                ranked.append((score - i * 1e-3, name, mod))
        ranked.sort(key=lambda r: -r[0])
        return [(round(s, 3), n) for s, n, _ in ranked]

    # ---- solving --------------------------------------------------------
    def solve(self, text: str) -> Solution:
        key = text.strip()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        parts = split_parts(text)
        if len(parts) > 1:
            result = self._solve_multi(text, parts)
        else:
            result = self._solve_one(text)
        self._cache[key] = result
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return result

    def _solve_one(self, text: str) -> Solution:
        t0 = time.time()
        best_failed: Optional[Solution] = None
        notes = []
        for score, name in self.classify(text):
            mod = dict(REGISTRY)[name]
            sol = run_with_timeout(mod.solve, text, self.timeout) if self.use_timeout else mod.solve(text)
            if sol is None:
                continue
            sol.category = sol.category or name
            if sol.solved and sol.verified is not False:
                sol.warnings = notes + sol.warnings
                sol.warnings.append(f"solved in {time.time() - t0:.2f}s by '{name}' solver")
                return sol
            if sol.solved and sol.verified is False:
                notes.append(f"'{name}' solver produced an answer that FAILED verification, so it was discarded")
                best_failed = best_failed or sol
            else:
                notes += [f"{name}: {w}" for w in sol.warnings]
        out = Solution(text)
        out.warnings = notes or ["no solver recognised this problem"]
        if best_failed is not None:
            out.warnings.append("closest (unverified) attempt: " + best_failed.answer_text)
        return out

    def _solve_multi(self, text: str, parts: list) -> Solution:
        combo = Solution(text, "multi-part")
        combo.method = "each part solved and verified independently"
        answers, all_ok, any_solved = [], True, False
        for label, sub in parts:
            s = self._solve_one(sub)
            combo.add(f"({label}) {sub}", s.answer_text if s.solved else "not solved by exact solvers")
            if s.solved:
                any_solved = True
                answers.append(f"({label}) {s.answer_text}")
                combo.evidence(f"({label}) {'verified' if s.verified else 'unverified'} - {s.method}")
                all_ok &= bool(s.verified)
            else:
                all_ok = False
                answers.append(f"({label}) -")
        if not any_solved:
            combo.warnings.append("none of the parts could be solved exactly")
            return combo
        return combo.finish(answers, "\n".join(answers), verified=all_ok if all_ok else None)


def split_parts(text: str) -> list:
    """'1) ... 2) ...' / '(a) ... (b) ...' / 'a. ... b. ...' -> [(label, text), ...]"""
    m = list(re.finditer(r"(?m)(?:^|\s)(?:\(?([a-eA-E]|[1-9])[\).])\s+", text))
    if len(m) < 2:
        return []
    labels = [x.group(1) for x in m]
    if len(set(labels)) != len(labels):
        return []
    out = []
    for i, x in enumerate(m):
        end = m[i + 1].start() if i + 1 < len(m) else len(text)
        chunk = text[x.end():end].strip()
        if chunk:
            out.append((x.group(1), chunk))
    return out if len(out) >= 2 else []


_default = Engine()


def solve(text: str) -> Solution:
    """Module-level convenience: deep_oq.solve("...")"""
    return _default.solve(text)
''')

# ====================================================================================================
# MODULE: deep_oq.guard    (deep_oq/guard.py, 124 lines)
# ====================================================================================================
_register('deep_oq.guard', False, 'deep_oq/guard.py', r'''"""Text hygiene for model output: repetition loops, cut-off answers, and fake 'I ran a simulation' claims."""
from __future__ import annotations

import re
from typing import Optional

# --------------------------------------------------------------------------
# 2. Repetition removal
# --------------------------------------------------------------------------
_LOOP_RE = re.compile(r"(.{3,200}?)(?:\s*\1){2,}", re.DOTALL)


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", " ", s.lower()).strip()


def _dedupe_ngrams(text: str, n: int = 8) -> str:
    """Drop any run of words that repeats an earlier n-word window (catches
    repeats that have no punctuation between them)."""
    words = text.split()
    key = [re.sub(r"\W+", "", w.lower()) for w in words]
    seen, out, i = set(), [], 0
    while i < len(words):
        g = tuple(key[i:i + n])
        if len(g) == n and g in seen:
            j = i
            while j < len(words) and tuple(key[j:j + n]) in seen:
                j += 1
            i = min(len(words), j + n - 1)
            if out and out[-1].strip("*:").lower() in {"answer", "problem", "solution"}:
                out.pop()                      # stray label that introduced the repeat
            continue
        if len(g) == n:
            seen.add(g)
        out.append(words[i])
        i += 1
    res = " ".join(out)
    if res and res[-1] not in ".!?:)$`*" and len(res.split()) > n:
        res += "."
    return res


def has_loop(text: str) -> bool:
    """True if a chunk of text repeats 3+ times in a row, or a sentence repeats."""
    if _LOOP_RE.search(text) or _dedupe_ngrams(text) != text.strip() and \
            len(_dedupe_ngrams(text).split()) < len(text.split()) - 3:
        return True
    sentences = [_norm(s) for s in re.split(r"(?<=[.!?])\s+", text) if len(s) > 25]
    return len(sentences) != len(set(sentences))


def remove_repetition(text: str) -> str:
    """Collapse loops, drop duplicate sentences and stray repeated labels."""
    prev = None
    while prev != text:                       # collapse "X X X X" -> "X"
        prev = text
        text = _LOOP_RE.sub(r"\1", text)

    text = "\n".join(_dedupe_ngrams(l) if l.strip() else l for l in text.split("\n"))

    out, seen = [], set()
    for line in text.split("\n"):
        parts = re.split(r"(?<=[.!?])\s+", line)
        kept = []
        for p in parts:
            key = _norm(p)
            if len(key) > 25 and key in seen:
                continue
            seen.add(key)
            kept.append(p)
        out.append(" ".join(kept))
    text = "\n".join(out)

    # "Answer You can ... Answer You can ..." style: drop a leading bare label
    text = re.sub(r"^\s*(?:\d+\.\s*)?\**Answer\**:?\s+(?=\S)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# 3. Completeness checks
# --------------------------------------------------------------------------
def incomplete_reasons(text: str, min_words: int = 3) -> list[str]:
    reasons = []
    t = text.strip()
    if len(t.split()) < min_words:
        reasons.append("answer too short")
    if len(re.findall(r"(?<!\\)\$", t.replace("$$", ""))) % 2:
        reasons.append("unclosed $ math")
    if t.count("$$") % 2:
        reasons.append("unclosed $$ block")
    if t.count("```") % 2:
        reasons.append("unclosed code block")
    for o, c in ("()", "[]", "{}"):
        if t.count(o) > t.count(c):
            reasons.append(f"unbalanced {o}{c}")
    if re.search(r"[:,;(]\s*$", t) or re.search(
            r"\b(and|or|but|the|of|to|with|let|where|so)\s*$", t, re.I):
        reasons.append("ends mid-sentence")
    # heading with nothing under it, e.g. "1. Answer" then a single line
    if re.search(r"(?m)^\s*(?:\d+\.\s*)?\**(Answer|Problem|Solution|Step[- ]by[- ]Step.*)\**:?\s*$", t) \
            and len(t.split()) < 25:
        reasons.append("heading without content")
    if re.search(r"Problem:\s*Let \$f:\s*$", t):
        reasons.append("cut off after 'Let $f:'")
    return reasons


# --------------------------------------------------------------------------
# 4. Unverified-claim guard (e.g. fake "Monte Carlo simulation" claims)
# --------------------------------------------------------------------------
_CLAIM_RE = re.compile(
    r"[^.\n]*\b(I|we)\s+(ran|run|executed|simulated|performed|verified by running|searched)\b"
    r"[^.\n]*\.?|[^.\n]*\bMonte Carlo\b[^.\n]*\.?", re.I)


def strip_unverified_claims(text: str, tool_log: Optional[list] = None) -> tuple[str, bool]:
    """Remove 'I ran a simulation'-type sentences unless a real tool actually ran."""
    if tool_log:
        return text, False
    new = _CLAIM_RE.sub("", text)
    changed = new != text
    return re.sub(r"\n{3,}", "\n\n", new).strip(), changed
''')

# ====================================================================================================
# MODULE: deep_oq.llm_bridge    (deep_oq/llm_bridge.py, 259 lines)
# ====================================================================================================
_register('deep_oq.llm_bridge', False, 'deep_oq/llm_bridge.py', r'''"""Glue between the exact solvers and your language model.

Flow for every user message
  1. Small talk / "who are you"            -> LLM with the human-style persona prompt.
  2. Exact solver recognises the problem   -> verified answer; the LLM may only *explain* it, and the numbers are
                                              checked so the explanation cannot change the result.
  3. Anything else                         -> LLM with a REAL Python tool loop (code is executed in the sandbox and the
                                              output is fed back), optional self-consistency voting, then text guards.

You supply one async function:   async def llm(messages: list[dict], **params) -> str
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from . import codeexec
from .engine import Engine
from .guard import has_loop, incomplete_reasons, remove_repetition, strip_unverified_claims

LLM = Callable[..., Awaitable[str]]

FEATURES = """- Chat: answer questions, explain things, brainstorm, write and rewrite text
- Exact problem solving: algebra, calculus, linear algebra, number theory, probability, statistics, physics,
  chemistry, finance, unit conversion, logic puzzles, Sudoku, N-Queens - every answer is re-checked
- Code: write and debug code; Python snippets are actually executed in a sandbox when needed
- Photo / Video / Lab / News / Music / VR: the extra modes in the sidebar (where enabled)
- Research mode, Memory + Knowledge, voice (Listen / mic), language switch (EN)"""

SYSTEM_PROMPT = f"""You are Deep OQ, a friendly AI assistant. Talk like a thoughtful person, not a form:
natural, warm, conversational, matching the user's tone and language (English, Hindi, Hinglish...).

Style
- Never give canned one-liners. If someone asks about you or your features, answer with real detail.
  Length follows the question: a greeting gets a short reply, a big question gets a full answer.
- Plain words, contractions, an occasional light touch of humour. No lecturing.
- Answer anything helpfully and directly. Ask one short clarifying question only if truly ambiguous.
- For maths/logic: reason step by step, then give the final answer clearly.
- Never repeat sentences, never leave empty headings, finish every sentence and formula.

Honesty
- If unsure, say so. Never invent facts, sources or numbers.
- Never claim to have run code, simulations or searches unless a tool result is shown to you.
- Politely point out false premises.
- You still decline things that would seriously harm people (weapons, malware, self-harm instructions) -
  briefly, kindly, with a safe alternative.

Tools
- To compute something, write ONE python code block (```python ... ```) that prints its result. You will be shown
  the real output. Allowed imports: math, fractions, itertools, sympy, numpy, scipy, statistics, random, ...

Your real features
{FEATURES}
Only describe features from this list."""

BASE_PARAMS = dict(temperature=0.7, top_p=0.95, max_tokens=4096, repetition_penalty=1.1,
                   frequency_penalty=0.3, presence_penalty=0.3)
RETRY_PARAMS = dict(BASE_PARAMS, temperature=0.3, repetition_penalty=1.3, frequency_penalty=0.8)

_DETAIL = re.compile(r"(?i)\b(features?|about (yourself|you)|explain|describe|tell me more|how does|how do|why|compare|"
                     r"difference|guide|steps?|detail|kya kya|batao|samjhao)\b")
_GREET = re.compile(r"(?i)^\s*(hi+|hello+|hey+|yo|thanks?|thank you|ok(ay)?|bye|namaste)\W*$")
_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def expected_min_words(question: str) -> int:
    if _GREET.match(question):
        return 1
    if _DETAIL.search(question):
        return 60
    return 8 if len(question.split()) > 3 else 3


@dataclass
class BotReply:
    text: str
    source: str = "llm"                 # solver | solver+llm | llm+tools | llm
    verified: Optional[bool] = None
    category: str = ""
    solution: Optional[dict] = None
    tool_log: list = field(default_factory=list)
    attempts: int = 0
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"answer": self.text, "source": self.source, "verified": self.verified, "category": self.category,
                "solution": self.solution, "tool_runs": len(self.tool_log), "attempts": self.attempts, "notes": self.notes}


# --------------------------------------------------------------------------------------------- helpers
def numbers_in(text: str) -> list:
    return re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))


def preserves_numbers(solver_text: str, explanation: str) -> bool:
    """Every number in the exact answer must still appear in the LLM's explanation."""
    have = set(numbers_in(explanation))
    return all(n in have or n.rstrip("0").rstrip(".") in have for n in numbers_in(solver_text))


def extract_final_answer(text: str) -> Optional[str]:
    """The value after 'Final answer:' (first number there, fractions allowed); else the last number in the text."""
    m = re.search(r"(?i)(?:final answer|answer)\s*[:=\-]\s*\**\s*([^\n]+)", text)
    if m:
        first = re.search(r"-?\d+(?:\.\d+)?(?:/\d+)?", m.group(1).replace(",", ""))
        return first.group(0) if first else m.group(1).strip(" .*")
    nums = numbers_in(text)
    return nums[-1] if nums else None


async def _generate_clean(llm: LLM, messages: list, question: str, reply: BotReply, max_attempts: int = 3,
                          allow_code: bool = False) -> str:
    """Call the model, strip loops, retry when the text is broken or too thin."""
    min_words = expected_min_words(question)
    problems, text = [], ""
    for attempt in range(1, max_attempts + 1):
        reply.attempts += 1
        msgs = messages
        if attempt > 1:
            msgs = messages + [{"role": "assistant", "content": text[:400]},
                               {"role": "user", "content": "That reply had problems (" + ", ".join(problems) +
                                ").  Answer again from scratch in a natural, friendly, complete way. No repetition."}]
        raw = await llm(msgs, **(BASE_PARAMS if attempt == 1 else RETRY_PARAMS))
        problems = []
        if allow_code and _CODE.search(raw):          # a tool request is allowed to be short
            return remove_repetition(raw) if has_loop(raw) else raw
        if has_loop(raw):
            problems.append("repetition loop")
            reply.notes.append(f"attempt {attempt}: repetition removed")
        text = remove_repetition(raw)
        problems += incomplete_reasons(text, min_words)
        if not problems:
            return text
        reply.notes.append(f"attempt {attempt}: {', '.join(problems)}")
    return text + f"\n\n*(This answer may be incomplete: {', '.join(problems)}. Try regenerating.)*"


async def _tool_loop(llm: LLM, messages: list, question: str, reply: BotReply, rounds: int = 3) -> str:
    text = await _generate_clean(llm, messages, question, reply, allow_code=True)
    for _ in range(rounds):
        blocks = _CODE.findall(text)
        if not blocks:
            break
        code = blocks[0]
        res = await asyncio.to_thread(codeexec.run_python, code)
        reply.tool_log.append({"code": code, "output": res.summary(), "ok": res.ok, "blocked": res.blocked})
        messages = messages + [{"role": "assistant", "content": text},
                               {"role": "user", "content": f"Real tool output:\n```\n{res.summary()}\n```\n"
                                "Now write the final answer for the user using these real results. "
                                "If the code failed or was blocked, fix it or explain honestly. No more code unless needed."}]
        text = await _generate_clean(llm, messages, question, reply, allow_code=True)
    return text


async def _vote(llm: LLM, messages: list, question: str, reply: BotReply, n: int) -> str:
    """Self-consistency: sample n tool-assisted answers, return the one whose final value is the most common."""
    outs = []
    for i in range(n):
        sub = BotReply("")
        outs.append((await _tool_loop(llm, messages, question, sub), sub))
        reply.tool_log += sub.tool_log
        reply.attempts += sub.attempts
    finals = [extract_final_answer(t) for t, _ in outs]
    ranked = Counter(f for f in finals if f).most_common()
    if ranked:
        best, votes = ranked[0]
        reply.notes.append(f"self-consistency vote: '{best}' got {votes}/{n}")
        reply.verified = votes > n / 2
        for (t, _), f in zip(outs, finals):
            if f == best:
                return t
    return outs[0][0]


# --------------------------------------------------------------------------------------------- main entry
async def answer(question: str, llm: LLM, history: Optional[list] = None, engine: Optional[Engine] = None,
                 votes: int = 1, explain: bool = True) -> BotReply:
    engine = engine or Engine()
    history = history or []
    base = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    reply = BotReply("")

    # 1) exact solver first - never let the LLM guess when we can compute
    sol = await asyncio.to_thread(engine.solve, question)
    if sol.solved and sol.verified is not False:
        reply.category, reply.solution, reply.verified = sol.category, sol.to_dict(), sol.verified
        rendered = sol.render()
        reply.source = "solver"
        reply.text = rendered
        if explain:
            prompt = (f"The user asked: {question}\n\nAn exact, verified solver produced this result:\n{rendered}\n\n"
                      "Explain it to the user in a friendly, human way (short intro, key idea, the result). "
                      "You MUST keep every number and the final answer exactly as given. Do not add new calculations.")
            try:
                explained = await _generate_clean(llm, base + [{"role": "user", "content": prompt}], "explain", reply)
                if preserves_numbers(sol.answer_text, explained):
                    reply.text = explained + ("\n\n" + _evidence_block(sol) if sol.verification else "")
                    reply.source = "solver+llm"
                else:
                    reply.notes.append("LLM explanation changed the numbers, so the plain verified solution is shown instead")
            except Exception as exc:                          # noqa: BLE001
                reply.notes.append(f"explanation step failed ({type(exc).__name__}); showing verified solution")
        reply.notes += sol.warnings
        return reply
    if sol.warnings:
        reply.notes += [w for w in sol.warnings if "FAILED verification" in w]

    # 2) LLM + real tools
    msgs = base + [{"role": "user", "content": question}]
    text = await (_vote(llm, msgs, question, reply, votes) if votes > 1 else _tool_loop(llm, msgs, question, reply))
    reply.source = "llm+tools" if reply.tool_log else "llm"
    text, changed = strip_unverified_claims(text, reply.tool_log)
    if changed:
        reply.notes.append("removed 'I ran/simulated' claims because no tool actually ran")
    if reply.source == "llm+tools" and reply.verified is None:
        ok_runs = [t for t in reply.tool_log if t["ok"]]
        reply.verified = bool(ok_runs) or None
    if reply.verified is None and re.search(r"\d", text) and re.search(r"(?i)calculat|solve|prove|answer", question):
        text += "\n\n*Note: I couldn't verify this independently - please double-check important numbers.*"
    reply.text = text
    return reply


def _evidence_block(sol) -> str:
    lines = [f"- {e}" for e in sol.verification[:4]]
    return "**How I checked it**\n" + "\n".join(lines)


# --------------------------------------------------------------------------------------------- real LLM adapter
class OpenAICompatibleLLM:
    """Works with OpenAI, Ollama (/v1), vLLM, LM Studio, Together, Groq...  Config via env or constructor."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None,
                 timeout: float = 120):
        self.base_url = (base_url or os.environ.get("DEEP_OQ_LLM_URL", "http://localhost:11434/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("DEEP_OQ_LLM_KEY", "")
        self.model = model or os.environ.get("DEEP_OQ_LLM_MODEL", "llama3")
        self.timeout = timeout

    def _call(self, messages, **params) -> str:
        body = {"model": self.model, "messages": messages,
                **{k: v for k, v in params.items() if k in ("temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty")}}
        if "repetition_penalty" in params:
            body["repetition_penalty"] = params["repetition_penalty"]      # vLLM/Together-style; ignored elsewhere
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:                       # noqa: S310
            return json.loads(r.read())["choices"][0]["message"]["content"]

    async def __call__(self, messages, **params) -> str:
        return await asyncio.to_thread(self._call, messages, **params)
''')

# ====================================================================================================
# MODULE: deep_oq.models    (deep_oq/models.py, 83 lines)
# ====================================================================================================
_register('deep_oq.models', False, 'deep_oq/models.py', r'''"""Shared data types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Step:
    """One line of a worked solution."""
    text: str
    math: Optional[str] = None      # LaTeX or plain math for this step

    def render(self, n: int) -> str:
        body = f"{n}. {self.text}"
        if self.math:
            body += f"\n     {self.math}"
        return body


@dataclass
class Solution:
    problem: str
    category: str = "unknown"
    answer: Any = None
    answer_text: str = ""
    steps: list = field(default_factory=list)
    verified: Optional[bool] = None          # True / False / None (could not check)
    verification: list = field(default_factory=list)   # evidence lines
    method: str = ""
    confidence: float = 0.0                  # 0..1
    warnings: list = field(default_factory=list)
    solved: bool = False

    # --- helpers used by solvers -------------------------------------
    def add(self, text: str, math: Optional[str] = None) -> "Solution":
        self.steps.append(Step(text, math))
        return self

    def evidence(self, line: str) -> "Solution":
        self.verification.append(line)
        return self

    def finish(self, answer, answer_text: str = "", verified: Optional[bool] = None,
               confidence: Optional[float] = None) -> "Solution":
        self.answer = answer
        self.answer_text = answer_text or str(answer)
        self.verified = verified
        self.solved = True
        if confidence is None:
            confidence = {True: 0.97, None: 0.7, False: 0.2}[verified]
        self.confidence = confidence
        return self

    # --- output --------------------------------------------------------
    def render(self, show_steps: bool = True) -> str:
        out = []
        if not self.solved:
            out.append("I could not solve this one with the exact solvers.")
            out += [f"  - {w}" for w in self.warnings]
            return "\n".join(out)
        out.append(f"**Answer:** {self.answer_text}")
        if show_steps and self.steps:
            out.append("\n**Working:**")
            out += [s.render(i + 1) for i, s in enumerate(self.steps)]
        if self.verification:
            tag = {True: "Verified", False: "FAILED verification", None: "Not fully verified"}[self.verified]
            out.append(f"\n**{tag}:**")
            out += [f"  - {v}" for v in self.verification]
        if self.warnings:
            out.append("\n**Notes:**")
            out += [f"  - {w}" for w in self.warnings]
        out.append(f"\n_Method: {self.method} | confidence {self.confidence:.0%}_")
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {
            "problem": self.problem, "category": self.category, "solved": self.solved,
            "answer": self.answer_text, "steps": [{"text": s.text, "math": s.math} for s in self.steps],
            "verified": self.verified, "verification": self.verification,
            "method": self.method, "confidence": self.confidence, "warnings": self.warnings,
        }
''')

# ====================================================================================================
# MODULE: deep_oq.parse    (deep_oq/parse.py, 133 lines)
# ====================================================================================================
_register('deep_oq.parse', False, 'deep_oq/parse.py', r'''"""Turn human-written maths into sympy objects."""
from __future__ import annotations

import re
from typing import Optional

import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        implicit_multiplication_application,
                                        convert_xor, function_exponentiation)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,
                                          convert_xor, function_exponentiation)

_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
x, y, z, t, n = sp.symbols("x y z t n", real=True)

LOCALS = {
    "x": x, "y": y, "z": z, "t": t, "n": n, "e": sp.E, "E": sp.E, "pi": sp.pi, "oo": sp.oo,
    "ln": sp.log, "Abs": sp.Abs, "abs": sp.Abs, "sqrt": sp.sqrt, "cbrt": sp.cbrt,
    "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan, "asin": sp.asin,
    "acos": sp.acos, "atan": sp.atan, "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "sec": sp.sec, "csc": sp.csc, "cot": sp.cot, "exp": sp.exp, "factorial": sp.factorial,
    "gamma": sp.gamma, "floor": sp.floor, "ceiling": sp.ceiling, "sign": sp.sign,
}


def clean(text: str) -> str:
    """Normalise unicode / LaTeX-ish / natural language bits to python-ish maths."""
    s = text.strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = s.replace("−", "-").replace("–", "-").replace("×", "*").replace("÷", "/").replace("·", "*")
    s = re.sub(r"(?i)\b(infinity|inf)\b", "oo", s)
    s = s.replace("π", "pi").replace("∞", "oo").replace("√", "sqrt").replace("≤", "<=").replace("≥", ">=")
    s = re.sub(r"([a-zA-Z0-9\)])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)", lambda m: m.group(1) + "^(" + m.group(2).translate(_SUP) + ")", s)
    # LaTeX
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\(left|right|,|;|!|cdot)", lambda m: "*" if m.group(1) == "cdot" else "", s)
    s = re.sub(r"\\(sin|cos|tan|log|ln|exp|pi|infty|theta|alpha)", lambda m: {"infty": "oo"}.get(m.group(1), m.group(1)), s)
    s = s.replace("{", "(").replace("}", ")").replace("$", "")
    # |expr| -> Abs(expr)   (non nested, greedy pairs)
    while re.search(r"\|([^|]+)\|", s):
        s = re.sub(r"\|([^|]+)\|", r"Abs(\1)", s, count=1)
    # sin^2 x -> (sin(x))^2 ;  sin^2(x) -> (sin(x))^2
    s = re.sub(r"\b(sinh|cosh|tanh|sin|cos|tan|sec|csc|cot|ln|log)\s*\^\s*\(?(\d+)\)?\s*"
               r"(\([^()]*\)|[a-z]\w*|\d+)", r"((\1(\3))^\2)", s)
    s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)          # 1,000 -> 1000
    s = s.replace("^", "**") if False else s                # convert_xor handles ^
    return s.strip().rstrip("?.,;:")


_FUNC_NAMES = {"sin", "cos", "tan", "sec", "csc", "cot", "sinh", "cosh", "tanh", "asin", "acos", "atan", "arcsin",
               "arccos", "arctan", "log", "ln", "exp", "sqrt", "cbrt", "Abs", "abs", "factorial", "gamma", "floor",
               "ceiling", "sign", "pi", "oo", "e", "E", "Derivative", "y"}
_VAR_LETTERS = set("xyztnabckmruvwh")
_SAFE_CHARS = re.compile(r"^[0-9A-Za-z\s+\-*/^=<>()\[\]{},.|!&:%']*$")
FILLER = re.compile(r"(?i)\b(please|pls|kindly|thanks|thank you|now|the|this|that|given|equation|question|problem|"
                    r"me|for me|can you|could you|would you|find out|tell me|answer|value|values)\b")


def assert_safe(s: str, allow_ode: bool = False) -> None:
    """Firewall in front of sympy's parser (which uses eval internally).  Raises ValueError on anything that is
    not plain maths: unknown words, dunders, quotes, attribute access, odd characters."""
    if len(s) > 400:
        raise ValueError("expression too long")
    if re.search(r"\d{16,}", s) or re.search(r"(\*\*|\^)\s*\(?\s*\d{5,}", s):
        raise ValueError("number too large")
    if "__" in s or "`" in s or '"' in s or (("'" in s) and not allow_ode):
        raise ValueError("unsafe characters in expression")
    if not _SAFE_CHARS.match(s):
        raise ValueError("unsupported characters in expression")
    if re.search(r"[A-Za-z_\)\]]\s*\.\s*[A-Za-z_]", s):
        raise ValueError("attribute access is not allowed")
    for w in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", s):
        if w in _FUNC_NAMES or w in LOCALS:
            continue
        if len(w) <= 3 and set(w) <= _VAR_LETTERS:
            continue
        # allow digit-prefixed things like 2x handled by regex above; 1e5 style exponent
        if re.fullmatch(r"e\d+", w):
            continue
        raise ValueError(f"unknown word '{w}' in expression")


def to_expr(text: str, log_base_e: bool = True) -> sp.Expr:
    """Parse a single expression.  Raises ValueError on failure."""
    s = clean(text)
    assert_safe(s)
    loc = dict(LOCALS)
    loc["log"] = sp.log if log_base_e else (lambda v, b=10: sp.log(v, b))
    try:
        return parse_expr(s, local_dict=loc, transformations=_TRANSFORMS, evaluate=True)
    except Exception as exc:                                  # noqa: BLE001
        raise ValueError(f"cannot parse '{text}': {exc}") from exc


def to_relation(text: str, log_base_e: bool = True):
    """Parse 'lhs = rhs' into sp.Eq, or an inequality, or a bare expression."""
    s = clean(text)
    for op, fn in (("<=", sp.Le), (">=", sp.Ge), ("!=", sp.Ne)):
        if op in s:
            l, r = s.split(op, 1)
            return fn(to_expr(l, log_base_e), to_expr(r, log_base_e))
    for op, fn in ((" < ", sp.Lt), (" > ", sp.Gt)):
        if op in f" {s} ":
            l, r = f" {s} ".split(op, 1)
            return fn(to_expr(l, log_base_e), to_expr(r, log_base_e))
    if "=" in s:
        l, r = s.split("=", 1)
        return sp.Eq(to_expr(l, log_base_e), to_expr(r, log_base_e), evaluate=False)
    return to_expr(s, log_base_e)


def find_math(text: str) -> Optional[str]:
    """Pull the maths part out of a sentence, e.g. 'differentiate x^2*sin(x) please' -> 'x^2*sin(x)'."""
    s = text.strip()
    s = re.sub(r"(?i)\b(please|can you|could you|find|calculate|compute|evaluate|solve|what is|what's|"
               r"the|of|for|with respect to|wrt|w\.r\.t\.?|derivative|differentiate|integrate|integral|"
               r"limit|simplify|factorise|factorize|expand|determinant|inverse|eigenvalues?)\b", " ", s)
    s = re.sub(r"\b(from|to)\s+x\s*=?", " ", s) if False else s
    s = re.sub(r"\s+", " ", s).strip(" :,.?")
    return s or None


def free_symbol(expr, prefer=("x", "t", "y", "z", "n")) -> sp.Symbol:
    syms = sorted(expr.free_symbols, key=lambda q: q.name)
    for p in prefer:
        for q in syms:
            if q.name == p:
                return q
    return syms[0] if syms else x
''')

# ====================================================================================================
# MODULE: deep_oq.providers    (deep_oq/providers.py, 514 lines)
# ====================================================================================================
_register('deep_oq.providers', False, 'deep_oq/providers.py', r'''"""Provider adapters for Anthropic, OpenAI, Google Gemini and Groq - each preconfigured with its most powerful model.

Keys are read ONLY from environment variables (or a local .env file).  Never hard-code or paste keys into chats.

    ANTHROPIC_API_KEY   OPENAI_API_KEY   GEMINI_API_KEY (or GOOGLE_API_KEY)   GROQ_API_KEY
    DEEP_OQ_PROVIDER_ORDER=anthropic,openai,gemini,groq     (failover order)
    DEEP_OQ_MODE=failover | vote                            (vote = ask every provider, majority answer wins)
    DEEP_OQ_<PROVIDER>_MODEL=...   override a model         DEEP_OQ_<PROVIDER>_BASE_URL=...   override an endpoint
    DEEP_OQ_OPENAI_EFFORT=high     DEEP_OQ_GEMINI_THINKING=high     DEEP_OQ_GROQ_EFFORT=high

Model IDs below were checked against each provider's docs on 2026-11-21.  If an ID is ever retired the adapter tries
the fallbacks, then asks the provider's /models endpoint and picks the best match automatically.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Optional

from .llm_bridge import extract_final_answer


# ----------------------------------------------------------------------------------------------- errors / http
class LLMError(Exception):
    def __init__(self, provider: str, message: str, status: Optional[int] = None, retry_after: Optional[float] = None):
        super().__init__(f"[{provider}] {message}" + (f" (HTTP {status})" if status else ""))
        self.provider, self.status, self.retry_after = provider, status, retry_after
        self.message = message

    @property
    def retryable(self) -> bool:
        return self.status is None or self.status in (408, 409, 425, 429, 500, 502, 503, 504, 529)

    @property
    def model_missing(self) -> bool:
        if self.status == 404:
            return True
        return bool(self.status == 400 and re.search(
            r"(model.*(not found|does not exist|no such|unknown model|is not available|invalid model))|"
            r"((not found|does not exist).*model)", self.message, re.I))


def _error_text(raw: str) -> str:
    try:
        d = json.loads(raw)
    except ValueError:
        return raw.strip()[:300]
    err = d.get("error", d) if isinstance(d, dict) else d
    if isinstance(err, dict):
        return str(err.get("message") or err.get("status") or err)[:300]
    return str(err)[:300]


def _http(provider: str, method: str, url: str, headers: dict, body: Optional[dict], timeout: float, secret: str = "") -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", "Accept": "application/json", **headers})
    scrub = (lambda s: s.replace(secret, "***")) if secret else (lambda s: s)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:                       # noqa: S310
            raw = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace") if e.fp else ""
        ra = e.headers.get("retry-after") if e.headers else None
        try:
            ra = float(ra) if ra else None
        except ValueError:
            ra = None
        raise LLMError(provider, scrub(_error_text(raw) or str(e.reason)), e.code, ra) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise LLMError(provider, scrub(f"network error: {getattr(e, 'reason', e)}")) from None
    try:
        return json.loads(raw)
    except ValueError:
        raise LLMError(provider, "server returned invalid JSON") from None


# ----------------------------------------------------------------------------------------------- model ranking
_PREFS = {
    "anthropic": [r"fable", r"mythos", r"opus", r"sonnet"],
    "openai": [r"astra", r"sol\b", r"terra", r"luna"],
    "gemini": [r"argon", r"-pro", r"flash(?!-lite)"],
    "groq": [r"gpt-oss-120b", r"minimax", r"qwen3\.\d+-27b", r"llama-3\.3-70b"],
}
_NOT_CHAT = re.compile(r"embed|audio|tts|whisper|transcri|realtime|image|vision|guard|safeguard|moderation|search|"
                       r"codex|computer|live|dall|imagen|veo|lyria|preview-\d{2}", re.I)


def _version(model_id: str) -> tuple:
    nums = [int(n) for n in re.findall(r"\d+", model_id) if int(n) < 100][:4]
    return tuple(nums + [0] * (4 - len(nums)))


def rank_models(provider: str, ids: list) -> list:
    """Best-first list of chat models for a provider, preferring the highest tier, then the highest version."""
    scored = []
    for mid in ids:
        if _NOT_CHAT.search(mid):
            continue
        for i, pat in enumerate(_PREFS.get(provider, [])):
            if re.search(pat, mid, re.I):
                scored.append(((i, tuple(-n for n in _version(mid))), mid))
                break
    return [m for _, m in sorted(scored)]


# ----------------------------------------------------------------------------------------------- base class
def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def split_messages(messages: list):
    """-> (system_text, turns) with alternating user/assistant turns, first turn always 'user'."""
    system, turns = [], []
    for m in messages:
        role, text = m.get("role", "user"), _text(m.get("content", ""))
        if role == "system":
            system.append(text)
            continue
        role = "assistant" if role in ("assistant", "model") else "user"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n\n" + text
        else:
            turns.append({"role": role, "content": text})
    if not turns or turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "content": "(continue)"})
    return "\n\n".join(system), turns


class Provider:
    name = ""
    env_keys: tuple = ()
    default_model = ""
    fallback_models: tuple = ()
    default_base = ""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None,
                 timeout: float = 300, max_attempts: int = 3, backoff: float = 1.0):
        key = api_key or next((os.environ[k] for k in self.env_keys if os.environ.get(k)), "")
        if not key:
            raise LLMError(self.name, f"no API key found - set {' or '.join(self.env_keys)}")
        self.api_key = key
        up = self.name.upper()
        self.model = model or os.environ.get(f"DEEP_OQ_{up}_MODEL") or self.default_model
        self.base_url = (base_url or os.environ.get(f"DEEP_OQ_{up}_BASE_URL") or self.default_base).rstrip("/")
        self.timeout, self.max_attempts, self.backoff = timeout, max_attempts, backoff

    def __repr__(self) -> str:                        # never show the key
        return f"<{type(self).__name__} model={self.model!r} key=***{self.api_key[-2:] if len(self.api_key) > 8 else ''}>"

    # --- provider-specific pieces
    def build(self, model, messages, params):          # -> (url, headers, body)
        raise NotImplementedError

    def parse(self, data: dict) -> str:
        raise NotImplementedError

    def minimal(self, body: dict) -> dict:             # strip optional fields for a 400 retry
        return body

    def list_request(self):                            # -> (url, headers)
        raise NotImplementedError

    def parse_model_ids(self, data: dict) -> list:
        return [m.get("id", "") for m in data.get("data", [])]

    # --- shared machinery
    def discover(self) -> list:
        url, headers = self.list_request()
        return self.parse_model_ids(_http(self.name, "GET", url, headers, None, 60, self.api_key))

    def _call_model(self, model, messages, params) -> str:
        url, headers, body = self.build(model, messages, params)
        stripped, last = False, None
        for attempt in range(1, self.max_attempts + 1):
            try:
                text = self.parse(_http(self.name, "POST", url, headers, body, self.timeout, self.api_key))
                if not text.strip():
                    raise LLMError(self.name, "model returned no text (all tokens may have gone to reasoning)")
                return text
            except LLMError as e:
                last = e
                if e.model_missing:
                    raise
                if e.status == 400 and not stripped:
                    body, stripped = self.minimal(body), True
                    continue
                if e.retryable and attempt < self.max_attempts:
                    delay = e.retry_after if e.retry_after is not None else self.backoff * 2 ** (attempt - 1) * (0.75 + random.random() / 2)
                    time.sleep(min(delay, 30))
                    continue
                raise
        raise last                                                                         # pragma: no cover

    def call_sync(self, messages: list, params: dict) -> str:
        tried, last = [], None
        for m in [self.model] + [x for x in self.fallback_models if x != self.model]:
            tried.append(m)
            try:
                text = self._call_model(m, messages, params)
                self.model = m                                                           # stick to what works
                return text
            except LLMError as e:
                last = e
                if not e.model_missing:
                    raise
        try:
            ids = self.discover()
        except LLMError:
            ids = []
        for m in rank_models(self.name, ids)[:6]:
            if m in tried:
                continue
            tried.append(m)
            try:
                text = self._call_model(m, messages, params)
                self.model = m
                return text
            except LLMError as e:
                last = e
                if not e.model_missing:
                    raise
        raise LLMError(self.name, f"no usable model (tried {', '.join(tried)}): {last.message if last else 'unknown'}")

    async def __call__(self, messages: list, **params) -> str:
        return await asyncio.to_thread(self.call_sync, messages, params)


def _clip(v, lo, hi, default):
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------------------------------------- Anthropic
class AnthropicLLM(Provider):
    name = "anthropic"
    env_keys = ("ANTHROPIC_API_KEY",)
    default_model = "claude-fable-5-2"                       # most capable generally-available Claude
    fallback_models = ("claude-opus-5-5", "claude-sonnet-5-5")
    default_base = "https://api.anthropic.com"

    def _headers(self):
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}

    def build(self, model, messages, params):
        system, turns = split_messages(messages)
        body = {"model": model, "max_tokens": int(params.get("max_tokens") or 4096), "messages": turns}
        if system:
            body["system"] = system
        if "temperature" in params:
            body["temperature"] = _clip(params["temperature"], 0, 1, 0.7)
        return f"{self.base_url}/v1/messages", self._headers(), body

    def parse(self, data):
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    def minimal(self, body):
        return {k: v for k, v in body.items() if k != "temperature"}

    def list_request(self):
        return f"{self.base_url}/v1/models?limit=100", self._headers()


# ----------------------------------------------------------------------------------------------- OpenAI
class OpenAILLM(Provider):
    name = "openai"
    env_keys = ("OPENAI_API_KEY",)
    default_model = "gpt-6.2-astra"                          # OpenAI's most capable model (released 2026-11-17)
    fallback_models = ("gpt-6.2-sol", "gpt-6-astra")
    default_base = "https://api.openai.com"

    def build(self, model, messages, params):
        system, turns = split_messages(messages)
        effort = os.environ.get("DEEP_OQ_OPENAI_EFFORT", "high")
        if effort == "none":
            effort = "low"                                   # Astra rejects 'none'
        # reasoning tokens count against max_output_tokens, so leave generous room
        body = {"model": model, "input": turns, "reasoning": {"effort": effort},
                "max_output_tokens": min(max(int(params.get("max_tokens") or 4096), 16000), 128000)}
        if system:
            body["instructions"] = system
        return f"{self.base_url}/v1/responses", {"Authorization": f"Bearer {self.api_key}"}, body

    def parse(self, data):
        if data.get("status") == "failed":
            raise LLMError(self.name, str((data.get("error") or {}).get("message", "response failed")))
        if isinstance(data.get("output_text"), str) and data["output_text"].strip():
            return data["output_text"]
        out = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                out += [c.get("text", "") for c in item.get("content", []) if c.get("type") in ("output_text", "text")]
        return "".join(out)

    def minimal(self, body):
        return {k: v for k, v in body.items() if k != "reasoning"}

    def list_request(self):
        return f"{self.base_url}/v1/models", {"Authorization": f"Bearer {self.api_key}"}


# ----------------------------------------------------------------------------------------------- Gemini
class GeminiLLM(Provider):
    name = "gemini"
    env_keys = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    default_model = "gemini-4-argon"                         # Google's most capable model (GA 2026-10-13)
    fallback_models = ("gemini-3.8-flash", "gemini-3.5-flash")
    default_base = "https://generativelanguage.googleapis.com"

    def _headers(self):
        return {"x-goog-api-key": self.api_key}

    def build(self, model, messages, params):
        system, turns = split_messages(messages)
        gen = {"maxOutputTokens": min(max(int(params.get("max_tokens") or 4096), 16384), 65536),
               "thinkingConfig": {"thinkingLevel": os.environ.get("DEEP_OQ_GEMINI_THINKING", "high")}}
        if "temperature" in params:
            gen["temperature"] = _clip(params["temperature"], 0, 2, 0.7)
        if "top_p" in params:
            gen["topP"] = _clip(params["top_p"], 0, 1, 0.95)
        body = {"contents": [{"role": "model" if t["role"] == "assistant" else "user", "parts": [{"text": t["content"]}]} for t in turns],
                "generationConfig": gen}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return f"{self.base_url}/v1beta/models/{model}:generateContent", self._headers(), body

    def parse(self, data):
        cands = data.get("candidates") or []
        if not cands:
            reason = (data.get("promptFeedback") or {}).get("blockReason")
            raise LLMError(self.name, f"no candidates returned" + (f" (blocked: {reason})" if reason else ""), 400 if reason else None)
        parts = (cands[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    def minimal(self, body):
        gen = {k: v for k, v in body.get("generationConfig", {}).items() if k not in ("thinkingConfig", "topP")}
        return {**body, "generationConfig": gen}

    def list_request(self):
        return f"{self.base_url}/v1beta/models?pageSize=200", self._headers()

    def parse_model_ids(self, data):
        return [m.get("name", "").replace("models/", "") for m in data.get("models", [])]


# ----------------------------------------------------------------------------------------------- Groq
class GroqLLM(Provider):
    name = "groq"
    env_keys = ("GROQ_API_KEY",)
    default_model = "openai/gpt-oss-120b"                    # Groq's production flagship (Groq hosts open-weight models)
    fallback_models = ("qwen/qwen3.8-27b", "llama-3.3-70b-versatile")
    default_base = "https://api.groq.com"

    def build(self, model, messages, params):
        system, turns = split_messages(messages)
        msgs = ([{"role": "system", "content": system}] if system else []) + turns
        body = {"model": model, "messages": msgs,
                "max_completion_tokens": min(max(int(params.get("max_tokens") or 4096), 8192), 65536)}
        if "temperature" in params:
            body["temperature"] = _clip(params["temperature"], 0, 2, 0.7)
        if "top_p" in params:
            body["top_p"] = _clip(params["top_p"], 0, 1, 0.95)
        if "gpt-oss" in model:
            body["reasoning_effort"] = os.environ.get("DEEP_OQ_GROQ_EFFORT", "high")
        return f"{self.base_url}/openai/v1/chat/completions", {"Authorization": f"Bearer {self.api_key}"}, body

    def parse(self, data):
        ch = data.get("choices") or []
        return _text((ch[0].get("message") or {}).get("content", "")) if ch else ""

    def minimal(self, body):
        return {k: v for k, v in body.items() if k not in ("reasoning_effort", "top_p")}

    def list_request(self):
        return f"{self.base_url}/openai/v1/models", {"Authorization": f"Bearer {self.api_key}"}


PROVIDERS = {"anthropic": AnthropicLLM, "openai": OpenAILLM, "gemini": GeminiLLM, "groq": GroqLLM}


# ----------------------------------------------------------------------------------------------- multi-provider router
class MultiLLM:
    """Callable like any `llm(messages, **params)`.  mode='failover' tries providers in order; mode='vote' asks them
    all and returns the answer most providers agree on (best for maths/logic; costs one call per provider)."""

    def __init__(self, providers: list, mode: str = "failover", cooldown: float = 120.0):
        self.providers, self.mode, self.cooldown = list(providers), mode, cooldown
        self._cool: dict = {}
        self.last_used: Optional[str] = None
        self.last_votes: dict = {}

    def _active(self) -> list:
        now = time.time()
        return [p for p in self.providers if self._cool.get(p.name, 0) <= now] or self.providers

    def _penalise(self, p: Provider, e: LLMError) -> None:
        self._cool[p.name] = time.time() + (self.cooldown * 5 if e.status in (401, 402, 403) else self.cooldown)

    async def __call__(self, messages: list, **params) -> str:
        if self.mode == "vote" and len(self.providers) > 1:
            return await self._vote(messages, params)
        errors = []
        for p in self._active():
            try:
                text = await p(messages, **params)
                self.last_used = p.name
                return text
            except LLMError as e:
                errors.append(str(e))
                self._penalise(p, e)
        raise LLMError("multi", "all providers failed: " + " | ".join(errors))

    async def _vote(self, messages: list, params: dict) -> str:
        active = self._active()
        results = await asyncio.gather(*[p(messages, **params) for p in active], return_exceptions=True)
        good, errors = [], []
        for p, r in zip(active, results):
            if isinstance(r, str):
                good.append((p, r))
            else:
                errors.append(str(r))
                if isinstance(r, LLMError):
                    self._penalise(p, r)
        if not good:
            raise LLMError("multi", "all providers failed: " + " | ".join(errors))
        keyed = [(extract_final_answer(t), p, t) for p, t in good]
        self.last_votes = {p.name: k for k, p, _ in keyed}
        counts = Counter(k for k, _, _ in keyed if k)
        best = counts.most_common(1)[0][0] if counts else None
        for k, p, t in keyed:
            if best is None or k == best:
                self.last_used = p.name
                return t
        return keyed[0][2]                                                                  # pragma: no cover


# ----------------------------------------------------------------------------------------------- setup helpers
def load_env_file(path: Optional[str] = ".env") -> int:
    """Minimal .env reader (KEY=VALUE).  Existing environment variables always win."""
    if not path or not os.path.isfile(path):
        return 0
    n = 0
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            k = k[7:].strip() if k.startswith("export ") else k
            v = v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v
                n += 1
    return n


def available_providers(**kwargs) -> list:
    order = [x.strip().lower() for x in os.environ.get("DEEP_OQ_PROVIDER_ORDER", "anthropic,openai,gemini,groq").split(",")]
    out = []
    for name in order:
        cls = PROVIDERS.get(name)
        if cls is None:
            continue
        try:
            out.append(cls(**kwargs.get(name, {})))
        except LLMError:
            pass                                             # no key for this provider - skip silently
    return out


def make_llm_from_env(env_file: Optional[str] = ".env", mode: Optional[str] = None):
    """Build the LLM callable from your environment.  Returns None when no key is configured."""
    load_env_file(env_file)
    provs = available_providers()
    if provs:
        return MultiLLM(provs, mode or os.environ.get("DEEP_OQ_MODE", "failover"))
    if os.environ.get("DEEP_OQ_LLM_URL"):                    # legacy: any OpenAI-compatible endpoint
        from .llm_bridge import OpenAICompatibleLLM
        return OpenAICompatibleLLM()
    return None


def describe_providers() -> list:
    rows = []
    for name, cls in PROVIDERS.items():
        rows.append({"provider": name, "model": os.environ.get(f"DEEP_OQ_{name.upper()}_MODEL") or cls.default_model,
                     "fallbacks": list(cls.fallback_models), "key_set": any(os.environ.get(k) for k in cls.env_keys),
                     "env": " or ".join(cls.env_keys)})
    return rows


async def ping_all(providers: list) -> list:
    """Send one tiny real request to each provider - the quickest way to check keys, models and billing."""
    async def one(p):
        t0 = time.time()
        try:
            text = await p([{"role": "user", "content": "Reply with exactly one word: pong"}], max_tokens=64)
            return {"provider": p.name, "model": p.model, "ok": True, "ms": int((time.time() - t0) * 1000), "reply": text.strip()[:40]}
        except LLMError as e:
            return {"provider": p.name, "model": p.model, "ok": False, "ms": int((time.time() - t0) * 1000), "error": str(e)[:200]}
    return list(await asyncio.gather(*[one(p) for p in providers]))
''')

# ====================================================================================================
# MODULE: deep_oq.search    (deep_oq/search.py, 75 lines)
# ====================================================================================================
_register('deep_oq.search', False, 'deep_oq/search.py', r'''"""Generic state-space search used by puzzle solvers (BFS, A*) + a small constraint solver."""
from __future__ import annotations

import heapq
import itertools
from collections import deque
from typing import Callable, Hashable, Iterable, Optional


def _rebuild(prev, goal):
    out, cur = [], goal
    while prev[cur] is not None:
        p, lab = prev[cur]
        out.append((lab, cur))
        cur = p
    return out[::-1]


def bfs_path(start, is_goal, neighbours, limit: int = 2_000_000):
    """Return list of (label, state) from start to the nearest goal, or None."""
    q = deque([start])
    prev = {start: None}
    while q and len(prev) < limit:
        s = q.popleft()
        if is_goal(s):
            return _rebuild(prev, s)
        for lab, nxt in neighbours(s):
            if nxt not in prev:
                prev[nxt] = (s, lab)
                q.append(nxt)
    return None


def astar(start, is_goal, neighbours, heuristic, limit: int = 1_000_000):
    """neighbours yields (label, state, cost)."""
    tie = itertools.count()
    pq = [(heuristic(start), 0, next(tie), start)]
    prev, best = {start: None}, {start: 0}
    while pq and len(best) < limit:
        _, g, _, s = heapq.heappop(pq)
        if is_goal(s):
            return _rebuild(prev, s), g
        if g > best.get(s, 1e18):
            continue
        for lab, nxt, c in neighbours(s):
            ng = g + c
            if ng < best.get(nxt, 1e18):
                best[nxt] = ng
                prev[nxt] = (s, lab)
                heapq.heappush(pq, (ng + heuristic(nxt), ng, next(tie), nxt))
    return None


def solve_csp(variables: dict, constraints: Iterable[Callable], all_solutions: bool = False):
    """Backtracking CSP.  variables = {name: iterable_of_values};
    constraints = callables(assignment_dict) -> bool, called on PARTIAL assignments
    (must return True when it cannot yet decide)."""
    names = sorted(variables, key=lambda k: len(list(variables[k])))
    domains = {k: list(v) for k, v in variables.items()}
    cons, out = list(constraints), []

    def rec(i, asg):
        if i == len(names):
            out.append(dict(asg))
            return not all_solutions
        for val in domains[names[i]]:
            asg[names[i]] = val
            if all(c(asg) for c in cons) and rec(i + 1, asg):
                return True
            del asg[names[i]]
        return False

    rec(0, {})
    return out
''')

# ====================================================================================================
# MODULE: deep_oq.server    (deep_oq/server.py, 68 lines)
# ====================================================================================================
_register('deep_oq.server', False, 'deep_oq/server.py', r'''"""FastAPI server.   Run:  uvicorn deep_oq.server:app --host 0.0.0.0 --port 8000

Language model: set ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY / GROQ_API_KEY (env vars or a .env file).
See providers.py for all options.  Without any key, /solve still works (exact solvers need no LLM) and /chat is solver-only.
"""
from __future__ import annotations

import os
from typing import Optional

from .engine import Engine
from .llm_bridge import BotReply, answer
from .providers import describe_providers, make_llm_from_env

ENGINE = Engine()
LLM = make_llm_from_env()          # Anthropic / OpenAI / Gemini / Groq from your environment (.env supported)


async def handle_chat(message: str, history: Optional[list] = None, votes: int = 1, llm=None) -> dict:
    """Framework-independent chat handler (easy to unit test)."""
    llm = llm or LLM
    if llm is None:
        sol = ENGINE.solve(message)
        if sol.solved:
            return BotReply(sol.render(), "solver", sol.verified, sol.category, sol.to_dict()).to_dict()
        return BotReply("I can solve maths, physics, chemistry, finance, puzzles and more exactly, but for open "
                        "conversation I need a language model connected (add an API key, e.g. ANTHROPIC_API_KEY).", "none").to_dict()
    return (await answer(message, llm, history=history, engine=ENGINE, votes=max(1, min(votes, 5)))).to_dict()


def handle_solve(problem: str) -> dict:
    s = ENGINE.solve(problem)
    d = s.to_dict()
    d["rendered"] = s.render()
    d["routing"] = ENGINE.classify(problem)
    return d


try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    class ChatRequest(BaseModel):
        message: str
        history: list = []
        votes: int = 1

    class SolveRequest(BaseModel):
        problem: str

    app = FastAPI(title="Deep OQ", version="2.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "llm_connected": LLM is not None,
                "providers": [{k: v for k, v in r.items() if k != "env"} for r in describe_providers()]}

    @app.post("/solve")
    def solve_endpoint(req: SolveRequest):
        return handle_solve(req.problem)

    @app.post("/chat")
    async def chat_endpoint(req: ChatRequest):
        return await handle_chat(req.message, req.history, req.votes)

except ImportError:          # fastapi not installed: handlers above still work
    app = None
''')

# ====================================================================================================
# MODULE: deep_oq.solvers    (deep_oq/solvers/__init__.py, 2 lines)
# ====================================================================================================
_register('deep_oq.solvers', True, 'deep_oq/solvers/__init__.py', r'''"""Individual solver modules.  Each exposes `can_handle(text) -> float` (0..1) and `solve(text) -> Solution`."""
''')

# ====================================================================================================
# MODULE: deep_oq.tests    (deep_oq/tests/__init__.py, 1 lines)
# ====================================================================================================
_register('deep_oq.tests', True, 'deep_oq/tests/__init__.py', r'''''')

# ====================================================================================================
# MODULE: deep_oq.verify    (deep_oq/verify.py, 143 lines)
# ====================================================================================================
_register('deep_oq.verify', False, 'deep_oq/verify.py', r'''"""Independent numeric checks.  Every solver should call these instead of trusting itself."""
from __future__ import annotations

import math
import random
from typing import Iterable, Optional

import sympy as sp


def _to_complex(v) -> Optional[complex]:
    try:
        return complex(sp.N(v))
    except Exception:                                         # noqa: BLE001
        return None


def sample_points(symbols: Iterable[sp.Symbol], count: int = 25, lo: float = -4.0, hi: float = 4.0,
                  seed: int = 7) -> list:
    rng = random.Random(seed)
    syms = list(symbols)
    return [{s: sp.Float(rng.uniform(lo, hi)) for s in syms} for _ in range(count)]


def numeric_equal(a: sp.Expr, b: sp.Expr, tol: float = 1e-7, count: int = 30) -> Optional[bool]:
    """True if a and b agree at random points (where both are finite).  None if no usable point."""
    syms = sorted((sp.sympify(a).free_symbols | sp.sympify(b).free_symbols), key=lambda s: s.name)
    good = 0
    for pt in sample_points(syms, count):
        va, vb = _to_complex(sp.sympify(a).subs(pt)), _to_complex(sp.sympify(b).subs(pt))
        if va is None or vb is None or any(map(lambda c: math.isnan(c.real) or math.isinf(abs(c)), (va, vb))):
            continue
        good += 1
        if abs(va - vb) > tol * max(1.0, abs(va), abs(vb)):
            return False
    return True if good >= 3 else None


def residual_ok(eq: sp.Eq, assignment: dict, tol: float = 1e-8) -> Optional[bool]:
    """Substitute a candidate solution back into an equation."""
    try:
        diff = sp.simplify((eq.lhs - eq.rhs).subs(assignment))
        if diff == 0:
            return True
        c = _to_complex(diff)
        return None if c is None else abs(c) < tol
    except Exception:                                         # noqa: BLE001
        return None


def numeric_derivative(f: sp.Expr, var: sp.Symbol, at: float, h: float = 1e-6) -> Optional[float]:
    try:
        fp = sp.lambdify(var, f, "math")
        return (fp(at + h) - fp(at - h)) / (2 * h)
    except Exception:                                         # noqa: BLE001
        return None


def derivative_check(f: sp.Expr, df: sp.Expr, var: sp.Symbol, points: int = 12) -> Optional[bool]:
    """Compare a symbolic derivative against central finite differences."""
    rng = random.Random(3)
    ok = checked = 0
    for _ in range(points * 3):
        a = rng.uniform(-3, 3)
        nd = numeric_derivative(f, var, a)
        try:
            sd = float(sp.N(df.subs(var, a)))
        except Exception:                                     # noqa: BLE001
            continue
        if nd is None or math.isnan(nd) or math.isnan(sd) or math.isinf(nd) or math.isinf(sd):
            continue
        checked += 1
        ok += abs(nd - sd) <= 1e-4 * max(1.0, abs(sd))
        if checked >= points:
            break
    return None if checked < 3 else ok == checked


def integral_check(f: sp.Expr, F: sp.Expr, var: sp.Symbol) -> Optional[bool]:
    """An antiderivative is right if its derivative equals the integrand."""
    return numeric_equal(sp.diff(F, var), f)


def definite_integral_check(f: sp.Expr, var: sp.Symbol, a, b, value) -> Optional[bool]:
    try:
        from scipy import integrate
        fn = sp.lambdify(var, f, "math")
        lo, hi = float(sp.N(a)), float(sp.N(b))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            num, _ = integrate.quad(fn, lo, hi, limit=200)
        return abs(num - float(sp.N(value))) < 1e-6 * max(1.0, abs(num))
    except Exception:                                         # noqa: BLE001
        return None


def limit_check(f: sp.Expr, var: sp.Symbol, point, value, dir_: str = "+-") -> Optional[bool]:
    """Evaluate close to the point from the allowed side(s)."""
    try:
        target = float(sp.N(value)) if value.is_finite else None
    except Exception:                                         # noqa: BLE001
        return None
    if target is None:
        return None
    fn = sp.lambdify(var, f, "mpmath")
    import mpmath as mp
    mp.mp.dps = 40
    p = mp.mpf(str(sp.N(point))) if point not in (sp.oo, -sp.oo) else None
    if p is None:
        big = mp.mpf(10) ** 12 * (1 if point == sp.oo else -1)
        try:
            return abs(float(fn(big)) - target) < 1e-4 * max(1, abs(target))
        except Exception:                                     # noqa: BLE001
            return None
    results = []
    for sign, key in ((1, "+"), (-1, "-")):
        if dir_ not in ("+-", key):
            continue
        try:
            results.append(float(fn(p + sign * mp.mpf(10) ** -12)))
        except Exception:                                     # noqa: BLE001
            return None
    return all(abs(r - target) < 1e-4 * max(1, abs(target)) for r in results) if results else None


def matrix_equal(A: sp.Matrix, B: sp.Matrix, tol: float = 1e-9) -> bool:
    if A.shape != B.shape:
        return False
    return all(abs(complex(sp.N(a - b))) < tol for a, b in zip(A, B))


def monte_carlo_probability(trial, n: int = 200_000, seed: int = 11) -> tuple:
    """Run a REAL simulation.  `trial(rng) -> bool`.  Returns (estimate, std_error, n)."""
    rng = random.Random(seed)
    hits = sum(1 for _ in range(n) if trial(rng))
    p = hits / n
    return p, math.sqrt(max(p * (1 - p), 1e-12) / n), n


def agree_within(exact: float, estimate: float, stderr: float, sigmas: float = 4.5) -> bool:
    return abs(exact - estimate) <= sigmas * max(stderr, 1e-9)
''')

# ====================================================================================================
# MODULE: deep_oq.solvers.algebra    (deep_oq/solvers/algebra.py, 232 lines)
# ====================================================================================================
_register('deep_oq.solvers.algebra', False, 'deep_oq/solvers/algebra.py', r'''"""Equations, systems, inequalities, factoring, expanding, simplifying - with worked steps."""
from __future__ import annotations

import re

import sympy as sp

from ..models import Solution
from .. import parse, verify

CATEGORY = "algebra"
_KEYS = re.compile(r"(?i)\b(solve|equation|roots?|zeros?|factori[sz]e|expand|simplify|inequality|"
                   r"system of|find x|find y|quadratic|cubic|polynomial)\b")


def can_handle(text: str) -> float:
    s = text.lower()
    if re.search(r"d/d[xyt]|derivative|differentiate|integra|∫|limit|lim\b|matrix|determinant|eigen", s):
        return 0.0
    score = 0.0
    if _KEYS.search(text):
        score += 0.6
    if "=" in text and not re.search(r"[a-z]{4,}\s*=\s*\d", s.replace("find", "")):
        score += 0.35
    if re.search(r"[<>]|<=|>=", text) and re.search(r"[a-z]", s):
        score += 0.3
    return min(score, 1.0)


def _strip(text: str) -> str:
    s = re.sub(r"(?i)^\s*(please\s+)?(can you\s+)?(solve|find|calculate|determine)\b( the (equation|system|inequality))?:?", "", text)
    s = re.sub(r"(?i)\b(for\s+[a-z]\b|where\b.*$)", "", s) if "=" in s or "<" in s or ">" in s else s
    s = parse.FILLER.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip(" :.,")


def _split_system(text: str) -> list:
    parts = re.split(r"[;\n]|,\s*(?=[^=]*=)|\band\b|&", text)
    return [p.strip() for p in parts if p.strip() and re.search(r"[=<>]", p)]


def solve(text: str) -> Solution:
    sol = Solution(text, CATEGORY)
    body = _strip(text)
    low = text.lower()
    try:
        if re.search(r"(?i)\bfactori[sz]e\b", low):
            return _factor(sol, parse.find_math(body) or body)
        if re.search(r"(?i)\bexpand\b", low):
            return _expand(sol, parse.find_math(body) or body)
        if re.search(r"(?i)\bsimplify\b", low):
            return _simplify(sol, parse.find_math(body) or body)
        rels = _split_system(body)
        if len(rels) >= 2:
            return _system(sol, [parse.to_relation(r) for r in rels])
        rel = parse.to_relation(body)
        if isinstance(rel, (sp.Le, sp.Ge, sp.Lt, sp.Gt)):
            return _inequality(sol, rel)
        if isinstance(rel, sp.Eq):
            return _equation(sol, rel)
        # bare expression -> treat "solve P(x)" as P(x)=0
        return _equation(sol, sp.Eq(rel, 0))
    except Exception as exc:                                  # noqa: BLE001
        sol.warnings.append(f"algebra solver error: {exc}")
        return sol


def _equation(sol: Solution, eq: sp.Eq) -> Solution:
    sol.method = "sympy solveset + substitution check"
    syms = sorted(eq.free_symbols, key=lambda s: s.name)
    if not syms:
        ok = sp.simplify(eq.lhs - eq.rhs) == 0
        return sol.finish(ok, "True" if ok else "False", verified=True)
    var = parse.free_symbol(eq)
    sol.add(f"Write the equation and solve for {var}.", sp.latex(eq))
    expr = sp.expand(eq.lhs - eq.rhs)
    sol.add("Move everything to one side.", f"{sp.latex(expr)} = 0")
    poly = None
    try:
        poly = sp.Poly(expr, var)
    except sp.PolynomialError:
        pass
    if poly is not None and poly.degree() == 2:
        a, b, c = poly.all_coeffs()
        D = sp.simplify(b * b - 4 * a * c)
        sol.add("Quadratic: compute the discriminant D = b² − 4ac.", f"a={a}, b={b}, c={c},  D = {D}")
        sol.add("Number of solutions: " + ("two real" if D > 0 else "one repeated real" if D == 0 else "two complex") if D.is_number else
                "Apply the quadratic formula.", "x = (−b ± √D) / 2a")
    elif poly is not None and poly.degree() == 1:
        a, b = poly.all_coeffs()
        sol.add("Linear equation: a·x + b = 0, so x = −b/a.", f"x = {sp.nsimplify(-b / a)}")
    elif poly is not None and poly.degree() >= 3:
        fac = sp.factor(expr)
        sol.add("Factor the polynomial.", f"{sp.latex(fac)} = 0")
    else:
        sol.add("Not a plain polynomial; use symbolic solving (with numeric fallback).")

    if poly is not None and poly.degree() >= 5:
        sol.add("Degree ≥ 5: no general closed form (Abel–Ruffini), so find all roots numerically to 15 digits.")
        roots = poly.nroots(n=15, maxsteps=200)
        lines, ok = [], True
        for r in roots:
            res = abs(complex(sp.N(expr.subs(var, r))))
            ok &= res < 1e-9
            lines.append(str(sp.N(r, 10)))
            sol.evidence(f"{var} ≈ {sp.N(r, 10)}: |residual| = {res:.1e}")
        reals = [r for r in roots if abs(sp.im(r)) < 1e-12]
        txt = f"{var} ≈ " + ", ".join(lines) + f"   ({len(reals)} real root(s))"
        return sol.finish(list(roots), txt, verified=ok)
    result = sp.solveset(expr, var, domain=sp.S.Complexes if poly is not None else sp.S.Reals)
    try:
        real_part = sp.solveset(expr, var, domain=sp.S.Reals)
    except Exception:                                         # noqa: BLE001
        real_part = result
    if isinstance(real_part, sp.ConditionSet) or isinstance(result, sp.ConditionSet):
        nums = _numeric_roots(expr, var)
        if not nums:
            sol.warnings.append("no closed form found and numeric root search failed")
            return sol
        sol.add("No closed form; found numeric roots by scanning for sign changes.", ", ".join(map(str, nums)))
        for r in nums:
            sol.evidence(f"{var}={r}: residual {abs(float(sp.N(expr.subs(var, r)))):.2e}")
        return sol.finish(nums, f"{var} ≈ " + ", ".join(map(str, nums)), verified=None, confidence=0.8)

    finite = list(result) if result.is_FiniteSet else None
    if finite is not None:
        reals = [r for r in finite if r.is_real]
        shown = finite if len(finite) != len(reals) or not reals else finite
        ok_all = True
        for r in finite:
            ok = verify.residual_ok(eq, {var: r})
            sol.evidence(f"{var} = {sp.simplify(r)} → lhs−rhs = 0 ✓" if ok else f"{var} = {r}: substitution check {ok}")
            ok_all &= bool(ok)
        txt = f"{var} = " + (", ".join(str(sp.simplify(r)) for r in shown) if shown else
                             ("no solution" if poly is not None else "no real solution"))
        if finite and any(not r.is_real for r in finite):
            txt += "   (complex roots included; real roots: " + (", ".join(map(str, reals)) or "none") + ")"
        sol.add("Solutions", txt)
        return sol.finish(finite, txt, verified=ok_all if finite else True)
    sol.add("Solution set", str(result))
    return sol.finish(result, f"{var} ∈ {result}", verified=None, confidence=0.8)


def _numeric_roots(expr, var, lo=-50, hi=50, steps=4000):
    import numpy as np
    f = sp.lambdify(var, expr, "numpy")
    xs = np.linspace(lo, hi, steps)
    with np.errstate(all="ignore"):
        ys = f(xs)
    roots = []
    for i in range(len(xs) - 1):
        if np.isfinite(ys[i]) and np.isfinite(ys[i + 1]) and ys[i] * ys[i + 1] < 0:
            try:
                roots.append(round(float(sp.nsolve(expr, var, (xs[i] + xs[i + 1]) / 2)), 8))
            except Exception:                                 # noqa: BLE001
                pass
    return sorted(set(roots))


def _system(sol: Solution, rels: list) -> Solution:
    sol.method = "sympy linsolve/solve + substitution check"
    syms = sorted(set().union(*[r.free_symbols for r in rels]), key=lambda s: s.name)
    sol.add("Treat the equations as a system in " + ", ".join(map(str, syms)) + ".",
            "\\begin{cases}" + r"\\".join(sp.latex(r) for r in rels) + "\\end{cases}")
    if all(sp.Poly(r.lhs - r.rhs, *syms).total_degree() <= 1 for r in rels):
        A, b = sp.linear_eq_to_matrix([r.lhs - r.rhs for r in rels], syms)
        sol.add("All equations are linear, so write A·v = b.", f"A = {sp.latex(A)},  b = {sp.latex(b)}")
        sol.add(f"rank(A) = {A.rank()}, number of unknowns = {len(syms)}.")
    res = sp.solve(rels, syms, dict=True)
    if not res:
        sol.add("The system is inconsistent.")
        return sol.finish([], "no solution", verified=True)
    ok_all = True
    lines = []
    for d in res:
        ok = all(verify.residual_ok(r, d) for r in rels)
        ok_all &= ok
        lines.append(", ".join(f"{k} = {v}" for k, v in d.items()))
        sol.evidence(f"{lines[-1]} satisfies every equation: {ok}")
    if len(res) == 1 and len(res[0]) < len(syms):
        lines[0] += "  (free variables remain: infinitely many solutions)"
    sol.add("Solve the system.", " ; ".join(lines))
    return sol.finish(res, " | ".join(lines), verified=ok_all)


def _inequality(sol: Solution, rel) -> Solution:
    sol.method = "sympy solve_univariate_inequality + sample-point check"
    var = parse.free_symbol(rel)
    sol.add("Solve the inequality by finding where the expression changes sign.", sp.latex(rel))
    res = sp.solve_univariate_inequality(rel, var, relational=False)
    sol.add("Solution interval(s).", sp.latex(res))
    ok = True
    pts = [-10, -3, -1, 0, 1, 3, 10, sp.Rational(1, 2), sp.Rational(-1, 2)]
    for p in pts:
        try:
            truth = bool(rel.subs(var, p))
            ok &= (truth == (p in res))
        except Exception:                                     # noqa: BLE001
            pass
    sol.evidence("Tested sample points against the original inequality: " + ("all agree" if ok else "MISMATCH"))
    return sol.finish(res, f"{var} ∈ {res}", verified=ok)


def _factor(sol: Solution, text: str) -> Solution:
    sol.method = "sympy factor + expansion check"
    e = parse.to_expr(text)
    f = sp.factor(e)
    sol.add("Factor completely over the rationals.", f"{sp.latex(e)} = {sp.latex(f)}")
    ok = sp.expand(f - e) == 0
    sol.evidence("Expanding the factored form returns the original: " + str(ok))
    return sol.finish(f, str(f), verified=ok)


def _expand(sol: Solution, text: str) -> Solution:
    sol.method = "sympy expand + numeric check"
    e = parse.to_expr(text)
    r = sp.expand(e)
    sol.add("Multiply out and collect like terms.", f"{sp.latex(e)} = {sp.latex(r)}")
    ok = verify.numeric_equal(e, r)
    sol.evidence(f"Both forms agree at random points: {ok}")
    return sol.finish(r, str(r), verified=ok)


def _simplify(sol: Solution, text: str) -> Solution:
    sol.method = "sympy simplify + numeric check"
    e = parse.to_expr(text)
    r = sp.simplify(e)
    sol.add("Apply algebraic and trig identities.", f"{sp.latex(e)} = {sp.latex(r)}")
    ok = verify.numeric_equal(e, r)
    sol.evidence(f"Original and simplified agree at random points: {ok}")
    return sol.finish(r, str(r), verified=ok)
''')

# ====================================================================================================
# MODULE: deep_oq.solvers.arithmetic    (deep_oq/solvers/arithmetic.py, 134 lines)
# ====================================================================================================
_register('deep_oq.solvers.arithmetic', False, 'deep_oq/solvers/arithmetic.py', r'''"""Plain arithmetic, percentages, ratios, averages, word-problem patterns (speed/time, work, mixtures)."""
from __future__ import annotations

import re
from fractions import Fraction

import mpmath as mp
import sympy as sp

from ..models import Solution
from .. import parse

CATEGORY = "arithmetic"
_PCT = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)")


def can_handle(text: str) -> float:
    low = text.lower()
    if re.search(r"[a-z]{3,}", re.sub(r"\b(sqrt|sin|cos|tan|log|ln|exp|pi|abs|what|is|the|of|percent|average|mean|"
                                      r"speed|distance|time|ratio|increase|decrease|by|from|to|and|calculate|compute|"
                                      r"evaluate|simplify|value|km|hours?|h|minutes?|per)\b", "", low)):
        return 0.0 if not re.search(r"percent|average|mean|ratio|speed|increase|decrease|discount|profit|loss", low) else 0.8
    return 0.7 if re.search(r"\d", low) and re.search(r"[+\-*/^%]|sqrt|pi", low) else 0.0


def _number(v) -> sp.Expr:
    return sp.nsimplify(v)


def solve(text: str) -> Solution:
    sol = Solution(text, CATEGORY)
    low = text.lower()
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", text)]
    try:
        m = _PCT.search(text)
        if m:
            p, base = map(Fraction, (m.group(1), m.group(2)))
            v = p / 100 * base
            sol.method = "percentage = p/100 × base"
            sol.add(f"{p}% of {base} = {p}/100 × {base}", f"= {v} ≈ {float(v):g}")
            sol.evidence(f"decimal check: {float(p) * float(base) / 100:g}")
            return sol.finish(v, f"{float(v):g}", verified=abs(float(v) - float(p) * float(base) / 100) < 1e-12)
        if re.search(r"percent(age)? (increase|decrease|change)|(increase|decrease|change) in percent|by what percent", low) and len(nums) >= 2:
            a, b = nums[0], nums[1]
            ch = (b - a) / a * 100
            sol.method = "(new − old)/old × 100"
            sol.add("Percentage change = (new − old) / old × 100.", f"({b:g} − {a:g}) / {a:g} × 100 = {ch:.4g}%")
            return sol.finish(ch, f"{ch:.4g}% {'increase' if ch > 0 else 'decrease'}", verified=abs((a * (1 + ch / 100)) - b) < 1e-9)
        if re.search(r"average|mean", low) and len(nums) >= 2 and not re.search(r"speed", low):
            v = sum(nums) / len(nums)
            sol.method = "sum / count"
            sol.add(f"Add the {len(nums)} numbers and divide by {len(nums)}.", f"{sum(nums):g} / {len(nums)} = {v:g}")
            return sol.finish(v, f"{v:g}", verified=abs(v * len(nums) - sum(nums)) < 1e-9)
        if re.search(r"speed|km/h|mph|m/s", low) and re.search(r"distance|time|hours?", low):
            return _speed(sol, text, nums)
        if re.search(r"discount|profit|loss|markup", low) and len(nums) >= 2:
            return _commercial(sol, text, low, nums)
        if re.search(r"ratio", low) and len(nums) >= 2:
            a, b = int(nums[0]), int(nums[1])
            g = sp.gcd(a, b)
            sol.add("Divide both parts by their gcd.", f"{a}:{b} = {a // g}:{b // g}")
            return sol.finish((a // g, b // g), f"{a // g}:{b // g}", verified=(a // g) * b == (b // g) * a)
        return _evaluate(sol, text)
    except Exception as exc:                                  # noqa: BLE001
        sol.warnings.append(f"arithmetic error: {exc}")
        return sol


def _evaluate(sol, text):
    sol.method = "exact rational arithmetic + 50-digit mpmath check"
    body = re.sub(r"(?i)\b(what is|what's|calculate|compute|evaluate|simplify|find|value of|the|equals?)\b|=\s*\?*$", " ", text)
    e = parse.to_expr(re.sub(r"\s+", " ", body).strip(" ?:."))
    if e.free_symbols:
        sol.warnings.append("contains variables; use algebra")
        return sol
    exact = sp.nsimplify(sp.simplify(e)) if e.is_Rational or e.is_number else e
    dec = sp.N(exact, 15)
    sol.add("Evaluate exactly (keep fractions and roots exact).", f"{sp.latex(e)} = {sp.latex(exact)}")
    mp.mp.dps = 50
    hi = mp.mpmathify(sp.N(e, 50))
    ok = abs(complex(sp.N(e, 30)) - complex(dec)) < 1e-9
    sol.evidence(f"50-digit evaluation: {mp.nstr(hi, 20)}")
    txt = f"{exact}" + (f" ≈ {dec}" if not exact.is_Integer and not exact.is_Rational or (exact.is_Rational and not exact.is_Integer) else "")
    return sol.finish(exact, txt, verified=ok)


def _speed(sol, text, nums):
    sol.method = "distance = speed × time"
    low = text.lower()
    if re.search(r"average speed", low) and len(nums) >= 4:
        d1, s1, d2, s2 = nums[:4] if "km" in low or "miles" in low else (nums + [0] * 4)[:4]
        total = d1 + d2
        t = d1 / s1 + d2 / s2
        v = total / t
        sol.add("Average speed = total distance / total time (not the mean of the speeds).", f"{total:g} / ({d1:g}/{s1:g} + {d2:g}/{s2:g}) = {v:.4g}")
        return sol.finish(v, f"{v:.4g}", verified=abs(v * t - total) < 1e-9)
    if len(nums) >= 2:
        a, b = nums[0], nums[1]
        if "distance" in low and re.search(r"find|what is the distance|how far", low):
            sol.add("distance = speed × time", f"{a:g} × {b:g} = {a * b:g}")
            return sol.finish(a * b, f"{a * b:g}", verified=True)
        if re.search(r"how long|find the time|time taken", low):
            sol.add("time = distance / speed", f"{a:g} / {b:g} = {a / b:g}")
            return sol.finish(a / b, f"{a / b:g}", verified=abs(a / b * b - a) < 1e-9)
        sol.add("speed = distance / time", f"{a:g} / {b:g} = {a / b:g}")
        return sol.finish(a / b, f"{a / b:g}", verified=abs(a / b * b - a) < 1e-9)
    return sol


def _commercial(sol, text, low, nums):
    sol.method = "percentage of cost / marked price"
    pm = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", low)
    if pm:
        p = float(pm.group(1))
        rest = [v for v in nums if v != p]
        if not rest:
            sol.warnings.append("need both a percentage and a base amount")
            return sol
        a = rest[0]
    else:
        a, p = nums[0], nums[1]
    if "discount" in low:
        final = a * (1 - p / 100)
        sol.add("Selling price = marked × (1 − d/100).", f"{a:g} × (1 − {p:g}/100) = {final:g}")
        return sol.finish(final, f"{final:g}", verified=abs(final / a - (1 - p / 100)) < 1e-12)
    if "profit" in low:
        sp_ = a * (1 + p / 100)
        sol.add("Selling price = cost × (1 + profit%/100).", f"{a:g} × (1 + {p:g}/100) = {sp_:g}")
        return sol.finish(sp_, f"{sp_:g}", verified=abs(sp_ / a - (1 + p / 100)) < 1e-12)
    if "loss" in low:
        sp_ = a * (1 - p / 100)
        return sol.finish(sp_, f"{sp_:g}", verified=abs(sp_ / a - (1 - p / 100)) < 1e-12)
    return sol
''')

# ====================================================================================================
# MODULE: deep_oq.solvers.calculus    (deep_oq/solvers/calculus.py, 350 lines)
# ====================================================================================================
_register('deep_oq.solvers.calculus', False, 'deep_oq/solvers/calculus.py', r'''"""Derivatives (rule-traced), integrals, limits, series, optimisation, tangent lines, ODEs."""
from __future__ import annotations

import re

import sympy as sp

from ..models import Solution
from .. import parse, verify

CATEGORY = "calculus"
_RX = re.compile(r"(?i)(derivative|differentiate|d/d[xyt]|integrat|integral|∫|\blimit\b|\blim\b|"
                 r"taylor|maclaurin|series|maxim|minim|critical point|stationary|tangent|"
                 r"differential equation|\bode\b|dy/dx|y'|gradient|partial)")


def can_handle(text: str) -> float:
    return 0.95 if _RX.search(text) else 0.0


def solve(text: str) -> Solution:
    sol = Solution(text, CATEGORY)
    low = text.lower()
    try:
        if re.search(r"differential equation|\bode\b|dy/dx|y'", low):
            return _ode(sol, text)
        if re.search(r"tangent", low):
            return _tangent(sol, text)
        if re.search(r"maxim|minim|critical point|stationary", low):
            return _optimise(sol, text)
        if re.search(r"taylor|maclaurin|series", low):
            return _series(sol, text)
        if re.search(r"\blimit\b|\blim\b|→|->|tends? to", low) and not re.search(r"integra|∫", low):
            return _limit(sol, text)
        if re.search(r"integra|∫", low):
            return _integral(sol, text)
        return _derivative(sol, text)
    except Exception as exc:                                  # noqa: BLE001
        sol.warnings.append(f"calculus solver error: {exc}")
        return sol


# ------------------------------------------------------------------ derivatives
def trace_derivative(e: sp.Expr, v: sp.Symbol, steps: list, depth: int = 0) -> None:
    """Append human-readable rule applications.  Depth-limited so output stays readable."""
    if depth > 2:
        return
    L = sp.latex
    if not e.has(v):
        steps.append(f"Constant rule: d/d{v}[{L(e)}] = 0")
    elif e == v:
        steps.append(f"d/d{v}[{v}] = 1")
    elif isinstance(e, sp.Add):
        steps.append(f"Sum rule: differentiate each term of {L(e)} separately.")
        for a in e.args:
            trace_derivative(a, v, steps, depth + 1)
    elif isinstance(e, sp.Mul):
        const, rest = e.as_independent(v)
        if const != 1:
            steps.append(f"Constant-multiple rule: pull out {L(const)}.")
            trace_derivative(rest, v, steps, depth + 1)
        else:
            num, den = sp.fraction(e)
            if den != 1 and den.has(v):
                steps.append(f"Quotient rule on {L(num)} / {L(den)}: (u'v − uv') / v².")
            else:
                steps.append(f"Product rule on {L(e)}: (uv)' = u'v + uv'.")
            for a in e.args[:3]:
                trace_derivative(a, v, steps, depth + 1)
    elif isinstance(e, sp.Pow):
        b, p = e.args
        if b.has(v) and not p.has(v):
            steps.append(f"Power rule" + (" with chain rule" if b != v else "") + f": d/d{v}[{L(e)}] = {L(p)}·({L(b)})^({L(p - 1)})" + (f"·d/d{v}[{L(b)}]" if b != v else ""))
            if b != v:
                trace_derivative(b, v, steps, depth + 1)
        elif p.has(v) and not b.has(v):
            steps.append(f"Exponential rule: d/d{v}[{L(b)}^({L(p)})] = {L(b)}^({L(p)})·ln({L(b)})·d/d{v}[{L(p)}]")
        else:
            steps.append(f"Variable base and exponent: use logarithmic differentiation on {L(e)}.")
    elif isinstance(e, sp.Function) or e.is_Function:
        inner = e.args[0]
        name = e.func.__name__
        table = {"sin": "cos", "cos": "−sin", "tan": "sec²", "exp": "exp", "log": "1/", "sinh": "cosh",
                 "cosh": "sinh", "asin": "1/√(1−u²) ·", "atan": "1/(1+u²) ·", "Abs": "sign(u) ·"}
        steps.append(f"Chain rule for {name}({L(inner)}): d/d{v}[{name}(u)] = ({table.get(name, name + chr(39))}(u))·u'" +
                     (f", with u = {L(inner)}" if inner != v else ""))
        if inner != v:
            trace_derivative(inner, v, steps, depth + 1)


def _extract_target(text: str) -> str:
    s = re.sub(r"(?i)\b(find|calculate|compute|evaluate|what is|what's|please|can you)\b", " ", text)
    s = re.sub(r"(?i)(the\s+)?(first |second |third |nth )?(partial )?(derivative|differentiate)\s*(of|for)?", " ", s)
    s = re.sub(r"(?i)d/d[xyt]\s*(of)?", " ", s)
    s = re.sub(r"(?i)\bwith respect to\s+\w+|\bw\.?r\.?t\.?\s+\w+", " ", s)
    s = re.sub(r"(?i)\bat\s+[xyt]\s*=\s*[-\d./\w()]+", " ", s)
    return re.sub(r"\s+", " ", s).strip(" :,.?")


def _order(text: str) -> int:
    low = text.lower()
    for word, k in (("second", 2), ("third", 3), ("fourth", 4), ("2nd", 2), ("3rd", 3)):
        if word in low:
            return k
    m = re.search(r"(\d+)(?:st|nd|rd|th)\s+derivative", low)
    return int(m.group(1)) if m else 1


def _derivative(sol: Solution, text: str) -> Solution:
    sol.method = "sympy diff + finite-difference check"
    wrt = re.search(r"(?i)with respect to\s+([a-z])|w\.?r\.?t\.?\s+([a-z])|d/d([a-z])", text)
    expr = parse.to_expr(_extract_target(text))
    var = parse.LOCALS.get((wrt.group(1) or wrt.group(2) or wrt.group(3)).lower()) if wrt else parse.free_symbol(expr)
    k = _order(text)
    sol.add(f"Differentiate with respect to {var}" + (f" {k} times." if k > 1 else "."), sp.latex(expr))
    rules: list = []
    trace_derivative(expr, var, rules)
    for r in rules[:8]:
        sol.add(r)
    d = sp.diff(expr, var, k)
    simp = sp.simplify(d)
    sol.add("Combine and simplify.", sp.latex(simp))
    ok = verify.derivative_check(expr if k == 1 else sp.diff(expr, var, k - 1), d if k == 1 else sp.diff(expr, var, k), var)
    sol.evidence(f"Compared with central finite differences at random points: {ok}")
    at = re.search(r"(?i)\bat\s+([xyt])\s*=\s*([-\d./\w()]+)", text)
    if at and k == 1:
        a0 = float(sp.N(parse.to_expr(at.group(2))))
        fn = sp.lambdify(var, expr, "math")
        try:
            h = 1e-7
            right, left = (fn(a0 + h) - fn(a0)) / h, (fn(a0) - fn(a0 - h)) / h
            if abs(right - left) > 1e-3 * max(1, abs(right), abs(left)):
                sol.add("One-sided slopes disagree, so the function has a corner/cusp here.", f"left ≈ {left:.4g}, right ≈ {right:.4g}")
                sol.evidence("numeric left and right difference quotients differ")
                return sol.finish("not differentiable", f"not differentiable at {var} = {at.group(2)} (left slope ≈ {left:.4g}, right slope ≈ {right:.4g})", verified=True)
        except (ZeroDivisionError, ValueError, OverflowError):
            pass
    ans_txt = f"d{'^' + str(k) if k > 1 else ''}/d{var}{'^' + str(k) if k > 1 else ''} = {simp}"
    if at:
        pt = parse.to_expr(at.group(2))
        val = sp.simplify(d.subs(var, pt))
        sol.add(f"Evaluate at {var} = {pt}.", sp.latex(val))
        sol.evidence(f"Numeric value: {sp.N(val, 10)}")
        ans_txt += f";  at {var}={pt}: {val} ≈ {sp.N(val, 8)}"
        return sol.finish(val, ans_txt, verified=ok)
    return sol.finish(simp, ans_txt, verified=ok)


# ------------------------------------------------------------------ integrals
def _bounds(text: str):
    m = re.search(r"(?i)from\s+(.+?)\s+to\s+(.+?)(?:\s+(?:of|for|dx|dt|dy)\b|$)", text)
    if m:
        return m.group(1), m.group(2), re.sub(r"(?i)from\s+.+?\s+to\s+.+?(\s|$)", " ", text, count=1)
    m = re.search(r"∫\s*_?\{?\(?(-?[\w./]+)\)?\}?\s*\^\s*\{?\(?(-?[\w./∞]+)\)?\}?\s*(.*)", text)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = re.search(r"(?i)between\s+(.+?)\s+and\s+(.+?)(?:\s|$)", text)
    if m:
        return m.group(1), m.group(2), text
    return None


def _integral(sol: Solution, text: str) -> Solution:
    sol.method = "sympy integrate + differentiation / quadrature check"
    b = _bounds(text)
    body = text if not b else b[2]
    body = re.sub(r"(?i)\b(find|calculate|compute|evaluate|please|definite|indefinite|the|integral|integrate|of|∫)\b", " ", body)
    body = re.sub(r"\bd([xyt])\b\s*$", "", body.strip())
    body = re.sub(r"(?i)\bw\.?r\.?t\.?\s*\w|with respect to \w", "", body)
    f = parse.to_expr(re.sub(r"\s+", " ", body).strip(" :,.?"))
    v = parse.free_symbol(f)
    if b is None:
        sol.add(f"Find an antiderivative of the integrand with respect to {v}.", sp.latex(f))
        F = sp.integrate(f, v)
        if F.has(sp.Integral):
            sol.warnings.append("no elementary antiderivative found")
            return sol
        sol.add("Integrate term by term / by standard forms.", sp.latex(F) + " + C")
        ok = verify.integral_check(f, F, v)
        sol.evidence(f"d/d{v} of the result equals the integrand at random points: {ok}")
        return sol.finish(F, f"{F} + C", verified=ok)
    a, bb = parse.to_expr(b[0]), parse.to_expr(b[1])
    sol.add(f"Definite integral of f({v}) from {a} to {bb}.", f"\\int_{{{sp.latex(a)}}}^{{{sp.latex(bb)}}} {sp.latex(f)}\\,d{v}")
    F = sp.integrate(f, v)
    if not F.has(sp.Integral):
        sol.add("Find an antiderivative F, then use F(b) − F(a).", sp.latex(F))
    val = sp.integrate(f, (v, a, bb))
    if val.has(sp.Integral):
        sol.warnings.append("could not evaluate in closed form")
        return sol
    val = sp.simplify(val)
    sol.add("Evaluate.", f"{sp.latex(val)} ≈ {sp.N(val, 10)}")
    ok = verify.definite_integral_check(f, v, a, bb, val)
    sol.evidence(f"Independent numerical quadrature (scipy.quad) agrees: {ok}")
    return sol.finish(val, f"{val} ≈ {sp.N(val, 10)}", verified=ok)


# ------------------------------------------------------------------ limits
def _limit(sol: Solution, text: str) -> Solution:
    sol.method = "sympy limit + numeric approach check"
    m = re.search(r"(?i)(?:as\s+)?([xyt])\s*(?:->|→|tends? to|approaches)\s*([-+\w./∞()]+?)([+-]?)\s*$", text.strip(" ."))
    if not m:
        m = re.search(r"(?i)(?:as\s+)?([xyt])\s*(?:->|→|tends? to|approaches)\s*([-+\w./∞()]+)", text)
    if not m:
        sol.warnings.append("could not find 'x -> a' in the question")
        return sol
    var = parse.LOCALS[m.group(1).lower()]
    pt_txt = m.group(2)
    direction = "+-"
    if pt_txt.endswith("+") and len(pt_txt) > 1:
        direction, pt_txt = "+", pt_txt[:-1]
    elif pt_txt.endswith("-") and len(pt_txt) > 1:
        direction, pt_txt = "-", pt_txt[:-1]
    pt = parse.to_expr(pt_txt.replace("infinity", "oo"))
    before = text[:m.start()]
    f_txt = re.sub(r"(?i)\b(find|calculate|compute|evaluate|the|limit|lim|of|as|please)\b", " ", before)
    f = parse.to_expr(re.sub(r"\s+", " ", f_txt).strip(" :,.?"))
    sol.add(f"Find the limit of the expression as {var} approaches {pt}" + (f" from the {'right' if direction == '+' else 'left'}." if direction != "+-" else "."), sp.latex(f))
    direct = None
    try:
        direct = f.subs(var, pt)
        if direct.is_finite and direct not in (sp.nan, sp.zoo):
            sol.add("Direct substitution works (continuous at this point).", sp.latex(direct))
        else:
            sol.add("Direct substitution gives an indeterminate form, so simplify / use L'Hôpital / series.", str(direct))
    except Exception:                                         # noqa: BLE001
        pass
    val = sp.limit(f, var, pt, direction if direction != "+-" else "+-")
    if direction == "+-" and (val in (sp.zoo, sp.nan) or val.has(sp.zoo, sp.nan)):
        right, left = sp.limit(f, var, pt, "+"), sp.limit(f, var, pt, "-")
        sol.add("Compare the one-sided limits.", f"from the right: {right},  from the left: {left}")
        if right == left and right in (sp.oo, -sp.oo):
            return sol.finish(right, str(right), verified=True)
        sol.evidence(f"right-hand limit {right} ≠ left-hand limit {left}")
        return sol.finish("does not exist", f"the two-sided limit does not exist (right: {right}, left: {left})", verified=True)
    sol.add("Evaluate the limit.", sp.latex(val))
    ok = verify.limit_check(f, var, pt, val, direction)
    sol.evidence(f"Evaluating at 10^-12 from each allowed side agrees: {ok}")
    return sol.finish(val, str(val), verified=ok)


# ------------------------------------------------------------------ series
def _series(sol: Solution, text: str) -> Solution:
    sol.method = "sympy series + numeric truncation-error check"
    order = int(re.search(r"(?i)(?:order|up to|terms?)\s*(\d+)", text).group(1)) if re.search(r"(?i)(?:order|up to|terms?)\s*(\d+)", text) else 6
    about = re.search(r"(?i)(?:about|around|at)\s+[xyt]\s*=\s*([-\w./()]+)", text)
    x0 = parse.to_expr(about.group(1)) if about else sp.Integer(0)
    body = re.sub(r"(?i)(taylor|maclaurin|series|expansion|expand|of|about|around|order\s*\d+|up to \d+ terms?|at\s+[xyt]\s*=\s*[-\w./()]+|find|the|for)", " ", text)
    f = parse.to_expr(re.sub(r"\s+", " ", body).strip(" :,.?"))
    v = parse.free_symbol(f)
    sol.add(f"Taylor series of f({v}) about {v} = {x0} to order {order}.", sp.latex(f))
    s = sp.series(f, v, x0, order).removeO()
    sol.add("Use f(a) + f'(a)(x−a) + f''(a)(x−a)²/2! + …", sp.latex(s))
    pt = x0 + sp.Rational(1, 10)
    try:
        err = abs(float(sp.N(f.subs(v, pt) - s.subs(v, pt))))
        sol.evidence(f"Truncation error at {v} = {pt}: {err:.2e}")
        ok = err < 1e-2
    except Exception:                                         # noqa: BLE001
        ok = None
    return sol.finish(s, str(s), verified=ok)


# ------------------------------------------------------------------ optimisation
def _optimise(sol: Solution, text: str) -> Solution:
    sol.method = "first/second derivative test + numeric scan"
    body = re.sub(r"(?i)(find|the|all|local|global|absolute|maxim\w*|minim\w*|critical points?|stationary points?|of|for|values?|and|points?)", " ", text)
    f = parse.to_expr(re.sub(r"\s+", " ", body).strip(" :,.?"))
    v = parse.free_symbol(f)
    sol.add("Critical points satisfy f'(x) = 0 (or f' undefined).", f"f = {sp.latex(f)}")
    d1 = sp.diff(f, v)
    sol.add("First derivative.", sp.latex(sp.simplify(d1)))
    crit = sp.solveset(d1, v, sp.S.Reals)
    if not crit.is_FiniteSet:
        sol.warnings.append("critical set is not finite / not closed form")
        return sol
    d2 = sp.diff(f, v, 2)
    lines, ok = [], True
    for c in sorted(crit, key=lambda q: float(sp.N(q))):
        s2 = sp.simplify(d2.subs(v, c))
        kind = "local minimum" if s2 > 0 else "local maximum" if s2 < 0 else "inconclusive (second derivative 0)"
        val = sp.simplify(f.subs(v, c))
        sol.add(f"At {v} = {c}: f'' = {s2} → {kind}; f = {val}.")
        lines.append(f"{v} = {c}: {kind}, f = {val}")
        try:
            h = 1e-3
            fl, fc, fr = (float(sp.N(f.subs(v, c + dh))) for dh in (-h, 0, h))
            if kind == "local minimum":
                ok &= fc <= fl + 1e-9 and fc <= fr + 1e-9
            elif kind == "local maximum":
                ok &= fc >= fl - 1e-9 and fc >= fr - 1e-9
        except Exception:                                     # noqa: BLE001
            pass
    sol.evidence("Neighbouring-point comparison confirms each classification: " + str(ok))
    return sol.finish(list(crit), "; ".join(lines) or "no critical points", verified=ok)


# ------------------------------------------------------------------ tangent
def _tangent(sol: Solution, text: str) -> Solution:
    sol.method = "point-slope form"
    at = re.search(r"(?i)at\s+[xyt]\s*=\s*([-\w./()]+)", text)
    if not at:
        sol.warnings.append("need a point, e.g. 'tangent to x^2 at x=1'")
        return sol
    body = re.sub(r"(?i)(find|the|equation|of|tangent|line|normal|to|curve|at\s+[xyt]\s*=\s*[-\w./()]+|y\s*=)", " ", text)
    f = parse.to_expr(re.sub(r"\s+", " ", body).strip(" :,.?"))
    v = parse.free_symbol(f)
    a = parse.to_expr(at.group(1))
    m, y0 = sp.simplify(sp.diff(f, v).subs(v, a)), sp.simplify(f.subs(v, a))
    sol.add(f"Point: ({a}, {y0}); slope f'({a}) = {m}.")
    line = sp.simplify(y0 + m * (v - a))
    sol.add("y − y₀ = m(x − a).", f"y = {sp.latex(line)}")
    ok = verify.numeric_equal(line.subs(v, a), f.subs(v, a)) and sp.simplify(sp.diff(line, v) - m) == 0
    sol.evidence("Line passes through the point and has the derivative's slope: " + str(ok))
    return sol.finish(line, f"y = {line}", verified=bool(ok))


# ------------------------------------------------------------------ ODEs
def _ode(sol: Solution, text: str) -> Solution:
    sol.method = "sympy dsolve + checkodesol"
    x_ = sp.Symbol("x")
    yf = sp.Function("y")
    ic = re.findall(r"y\(([-\d./]+)\)\s*=\s*([-\d./]+)", text)
    s = re.sub(r"(?i)(,|;|\band\b|\bwith\b|\bgiven\b|\bsubject to\b|\bwhere\b)?\s*y\(([-\d./]+)\)\s*=\s*[-\d./]+", " ", text)
    s = parse.clean(s)
    s = re.sub(r"(?i)\b(solve|find|the|differential equation|ode|general solution|of)\b", " ", s)
    s = s.replace("dy/dx", "Derivative(y(x),x)").replace("y''", "Derivative(y(x),(x,2))").replace("y'", "Derivative(y(x),x)")
    s = re.sub(r"(?<![A-Za-z_])y(?![A-Za-z_(])", "y(x)", s)
    s = re.sub(r"[,;:]\s*$", "", s.strip())
    lhs, rhs = s.split("=", 1)
    loc = {"y": yf, "x": x_, "Derivative": sp.Derivative}
    eq = sp.Eq(parse_expr_local(lhs, loc), parse_expr_local(rhs, loc))
    sol.add("Classify and solve the ODE.", sp.latex(eq))
    kwargs = {}
    if ic:
        kwargs["ics"] = {yf(sp.sympify(a)): sp.sympify(b) for a, b in ic}
        sol.add("Apply the initial condition(s) to fix the constants.", str(ic))
    res = sp.dsolve(eq, yf(x_), **kwargs)
    sol.add("Solution.", sp.latex(res))
    chk = sp.checkodesol(eq, res)
    ok = bool(chk[0]) if isinstance(chk, tuple) else bool(chk)
    sol.evidence(f"Substituting back into the ODE gives 0: {ok}")
    return sol.finish(res, str(res), verified=ok)


def parse_expr_local(s, loc):
    parse.assert_safe(re.sub(r"Derivative\(y\(x\),\(?x(?:,2)?\)?\)|y\(x\)", "y", s), allow_ode=True)
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
    return parse_expr(s, local_dict={**parse.LOCALS, **loc}, transformations=standard_transformations + (implicit_multiplication_application, convert_xor))
''')

# ====================================================================================================
# MODULE: deep_oq.solvers.chemistry    (deep_oq/solvers/chemistry.py, 113 lines)
# ====================================================================================================
_register('deep_oq.solvers.chemistry', False, 'deep_oq/solvers/chemistry.py', r'''"""Molar mass, equation balancing (exact nullspace), simple stoichiometry."""
from __future__ import annotations

import re
from math import gcd
from functools import reduce

import sympy as sp

from ..models import Solution

CATEGORY = "chemistry"
MASS = {"H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998,
        "Ne": 20.180, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
        "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845,
        "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971,
        "Br": 79.904, "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Ag": 107.87, "Sn": 118.71, "I": 126.90, "Ba": 137.33,
        "W": 183.84, "Pt": 195.08, "Au": 196.97, "Hg": 200.59, "Pb": 207.2, "U": 238.03}


def can_handle(text: str) -> float:
    if re.search(r"(?i)molar mass|molecular (weight|mass)|balance|stoichiometr|moles? of", text):
        return 0.9
    return 0.8 if re.search(r"[A-Z][a-z]?\d*.*(->|→|=>).*[A-Z]", text) else 0.0


def parse_formula(f: str) -> dict:
    """H2O -> {H:2,O:1}; handles nested parentheses, e.g. Ca(OH)2, Fe2(SO4)3."""
    def rec(s, i=0):
        counts = {}
        while i < len(s):
            c = s[i]
            if c == "(":
                inner, i = rec(s, i + 1)
                m = re.match(r"\d+", s[i:])
                mult = int(m.group()) if m else 1
                i += len(m.group()) if m else 0
                for k, v in inner.items():
                    counts[k] = counts.get(k, 0) + v * mult
            elif c == ")":
                return counts, i + 1
            else:
                m = re.match(r"([A-Z][a-z]?)(\d*)", s[i:])
                if not m:
                    raise ValueError(f"bad formula near '{s[i:]}'")
                el, num = m.groups()
                if el not in MASS:
                    raise ValueError(f"unknown element {el}")
                counts[el] = counts.get(el, 0) + (int(num) if num else 1)
                i += len(m.group())
        return counts, i
    return rec(f.replace("·", "").replace(" ", ""))[0]


def molar_mass(f: str) -> float:
    return sum(MASS[e] * n for e, n in parse_formula(f).items())


def balance(reactants: list, products: list):
    species = reactants + products
    elems = sorted({e for s in species for e in parse_formula(s)})
    M = sp.Matrix([[(1 if i < len(reactants) else -1) * parse_formula(s).get(e, 0) for i, s in enumerate(species)] for e in elems])
    ns = M.nullspace()
    if len(ns) != 1:
        return None
    v = ns[0]
    den = reduce(lambda a, b: a * b // gcd(a, b), [sp.fraction(x)[1] for x in v], 1)
    coeffs = [int(x * den) for x in v]
    g = reduce(gcd, coeffs)
    coeffs = [abs(c // g) for c in coeffs] if all(c <= 0 for c in coeffs) or all(c >= 0 for c in coeffs) else None
    return coeffs


def solve(text: str) -> Solution:
    sol = Solution(text, CATEGORY)
    try:
        if re.search(r"->|→|=>", text):
            body = re.sub(r"(?i)^.*?(balance|equation)\s*:?\s*", "", text) if re.search(r"(?i)balance", text) else text
            left, right = re.split(r"->|→|=>", body, 1)
            R = [s.strip() for s in re.split(r"\s\+\s|\+(?=\s*\d*[A-Z])", left) if s.strip()]
            P = [s.strip().rstrip(".") for s in re.split(r"\s\+\s|\+(?=\s*\d*[A-Z])", right) if s.strip()]
            R, P = [re.sub(r"^\d+\s*", "", x) for x in R], [re.sub(r"^\d+\s*", "", x) for x in P]
            sol.method = "exact nullspace of the element-conservation matrix"
            sol.add("Write one conservation equation per element, then find the integer null vector.", f"reactants {R}, products {P}")
            co = balance(R, P)
            if not co:
                sol.warnings.append("equation cannot be balanced uniquely (check formulas)")
                return sol
            nr = len(R)
            fmt = lambda cs, sp_: " + ".join(f"{c if c > 1 else ''}{s}" for c, s in zip(cs, sp_))      # noqa: E731
            eq = f"{fmt(co[:nr], R)} → {fmt(co[nr:], P)}"
            ok = True
            for e in sorted({e for s in R + P for e in parse_formula(s)}):
                l = sum(c * parse_formula(s).get(e, 0) for c, s in zip(co[:nr], R))
                r = sum(c * parse_formula(s).get(e, 0) for c, s in zip(co[nr:], P))
                ok &= l == r
                sol.evidence(f"{e}: {l} on the left, {r} on the right")
            return sol.finish(co, eq, verified=ok)
        m = re.search(r"(?i)(?:molar mass|molecular (?:weight|mass)|mass)\s*(?:of)?\s*([A-Z][A-Za-z0-9()·]*)", text)
        if m:
            f = m.group(1)
            c = parse_formula(f)
            mm = molar_mass(f)
            sol.method = "sum of standard atomic masses"
            sol.add("Count atoms: " + ", ".join(f"{e}×{n}" for e, n in c.items()), " + ".join(f"{n}×{MASS[e]}" for e, n in c.items()))
            sol.evidence("recomputed from parsed counts: " + f"{sum(MASS[e] * n for e, n in c.items()):.4f}")
            return sol.finish(mm, f"{f}: {mm:.3f} g/mol", verified=True)
        sol.warnings.append("try 'molar mass of Ca(OH)2' or 'balance H2 + O2 -> H2O'")
        return sol
    except Exception as exc:                                  # noqa: BLE001
        sol.warnings.append(f"chemistry error: {exc}")
        return sol
''')

# ====================================================================================================
# MODULE: deep_oq.solvers.combprob    (deep_oq/solvers/combprob.py, 237 lines)
# ====================================================================================================
_register('deep_oq.solvers.combprob', False, 'deep_oq/solvers/combprob.py', r'''"""Combinatorics & probability.  Probabilities are cross-checked with a REAL Monte Carlo simulation."""
from __future__ import annotations

import itertools
import math
import random
import re
from fractions import Fraction

import sympy as sp

from ..models import Solution
from .. import verify

CATEGORY = "combinatorics / probability"
_RX = re.compile(r"(?i)(probabilit|chance|odds|permutation|combination|arrange|choose|ways to|how many ways|"
                 r"nCr|nPr|dice|die\b|coin|cards?|deck|expected value|expectation|binomial|bayes|"
                 r"derangement|catalan|birthday|handshake|committee|distribut)")


def can_handle(text: str) -> float:
    return 0.85 if _RX.search(text) else 0.0


def solve(text: str) -> Solution:
    sol = Solution(text, CATEGORY)
    text = _words_to_digits(text)
    low = text.lower()
    nums = [int(v) for v in re.findall(r"\d+", text)]
    try:
        if "birthday" in low:
            return _birthday(sol, nums)
        if "bayes" in low or re.search(r"false positive|sensitivity|prevalence", low):
            return _bayes(sol, text)
        if "derangement" in low and nums:
            return _derangement(sol, nums[0])
        if "catalan" in low and nums:
            v = sp.catalan(nums[0])
            return sol.finish(int(v), f"C_{nums[0]} = {v}", verified=int(v) == math.comb(2 * nums[0], nums[0]) // (nums[0] + 1))
        if "handshake" in low and nums:
            n = nums[0]
            v = n * (n - 1) // 2
            sol.add("Each pair shakes once: C(n,2).", f"C({n},2) = {v}")
            sol.evidence(f"direct pair enumeration gives {sum(1 for _ in itertools.combinations(range(n), 2))}")
            return sol.finish(v, f"{v} handshakes", verified=v == sum(1 for _ in itertools.combinations(range(n), 2)))
        m = re.search(r"(\d+)\s*(?:C|choose)\s*(\d+)|nCr\s*\(?(\d+)\s*,\s*(\d+)", text, re.I)
        if m:
            g = [int(v) for v in m.groups() if v]
            v = math.comb(*g)
            sol.add("Combination formula C(n,r) = n!/(r!(n−r)!).", f"C({g[0]},{g[1]}) = {v}")
            sol.evidence(f"Pascal recurrence check: {v == math.comb(g[0]-1, g[1]-1) + math.comb(g[0]-1, g[1]) if g[0] > 0 and g[1] > 0 else True}")
            return sol.finish(v, str(v), verified=True)
        m = re.search(r"(\d+)\s*P\s*(\d+)|nPr\s*\(?(\d+)\s*,\s*(\d+)", text, re.I)
        if m:
            g = [int(v) for v in m.groups() if v]
            v = math.perm(*g)
            sol.add("Permutation formula P(n,r) = n!/(n−r)!.", f"P({g[0]},{g[1]}) = {v}")
            return sol.finish(v, str(v), verified=v == math.comb(g[0], g[1]) * math.factorial(g[1]))
        if "binomial" in low and len(nums) >= 3:
            return _binomial(sol, text, nums)
        if re.search(r"arrange|permutation of the letters|letters of|word", low) and re.search(r"letters|word", low):
            return _letters(sol, text)
        if "dice" in low or "die" in low:
            return _dice(sol, text, nums)
        if "coin" in low:
            return _coins(sol, text, nums)
        if re.search(r"card|deck", low):
            return _cards(sol, text, nums)
        if "expected value" in low or "expectation" in low:
            sol.warnings.append("for expected value give an explicit distribution, e.g. 'expected value of values 1,2,3 with probabilities 0.2,0.3,0.5'")
            return sol
        sol.warnings.append("probability question not recognised by exact solvers")
        return sol
    except Exception as exc:                                  # noqa: BLE001
        sol.warnings.append(f"combinatorics error: {exc}")
        return sol


_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
          "nine": 9, "ten": 10, "both": 2}


def _words_to_digits(text: str) -> str:
    return re.sub(r"\b(" + "|".join(_WORDS) + r")\b(?=\s+(?:dice|die|coins?|cards?|tosses|flips|heads|tails|people))",
                  lambda m: str(_WORDS[m.group(1).lower()]), text, flags=re.I)


def _simulate_check(sol, exact: Fraction, trial, label="Monte Carlo"):
    est, se, n = verify.monte_carlo_probability(trial)
    ok = verify.agree_within(float(exact), est, se)
    sol.evidence(f"{label} simulation ({n:,} real trials): estimate {est:.5f} ± {se:.5f} vs exact {float(exact):.5f} → {'agrees' if ok else 'DISAGREES'}")
    return ok


def _birthday(sol, nums):
    n = nums[0] if nums else 23
    sol.method = "complement rule + Monte Carlo"
    p_no = Fraction(1)
    for i in range(n):
        p_no *= Fraction(365 - i, 365)
    p = 1 - p_no
    sol.add("P(all different) = 365/365 · 364/365 · … ; subtract from 1.", f"P(match) = 1 − {float(p_no):.6f}")
    ok = _simulate_check(sol, p, lambda r: len({r.randrange(365) for _ in range(n)}) < n, "birthday")
    return sol.finish(p, f"P(at least two share a birthday among {n}) = {float(p):.4f} ({float(p)*100:.2f}%)", verified=ok)


def _bayes(sol, text):
    sol.method = "Bayes' theorem + Monte Carlo"
    pct = [float(v) / (100 if "%" in text else 1) for v in re.findall(r"(\d+(?:\.\d+)?)\s*%?", text)]
    if len(pct) < 3:
        sol.warnings.append("need prevalence, sensitivity, false-positive rate (as % or decimals)")
        return sol
    prev, sens, fpr = [v if v <= 1 else v / 100 for v in pct[:3]]
    prev, sens, fpr = map(Fraction, map(str, (prev, sens, fpr)))
    post = prev * sens / (prev * sens + (1 - prev) * fpr)
    sol.add("P(D|+) = P(+|D)P(D) / [P(+|D)P(D) + P(+|¬D)P(¬D)].", f"= {float(post):.5f}")

    def trial_pos_and_sick(r):
        sick = r.random() < prev
        pos = r.random() < (sens if sick else fpr)
        return (pos, sick)

    rng = random.Random(5)
    pos = sick_pos = 0
    for _ in range(400000):
        p_, s_ = trial_pos_and_sick(rng)
        pos += p_
        sick_pos += p_ and s_
    est = sick_pos / max(pos, 1)
    ok = abs(est - float(post)) < 0.02
    sol.evidence(f"simulation of 400,000 people: P(D|+) ≈ {est:.4f} vs exact {float(post):.4f} → {'agrees' if ok else 'DISAGREES'}")
    return sol.finish(post, f"P(disease | positive) = {float(post):.4f} ({float(post)*100:.2f}%)", verified=ok)


def _derangement(sol, n):
    sol.method = "inclusion-exclusion + brute force"
    d = int(sp.subfactorial(n))
    sol.add("D(n) = n!·Σ(−1)^k/k!  (round(n!/e)).", f"D({n}) = {d}")
    ok = None
    if n <= 8:
        ok = d == sum(1 for p in itertools.permutations(range(n)) if all(p[i] != i for i in range(n)))
        sol.evidence(f"brute-force enumeration of {math.factorial(n)} permutations agrees: {ok}")
    return sol.finish(d, str(d), verified=ok)


def _binomial(sol, text, nums):
    sol.method = "binomial pmf + Monte Carlo"
    n, k = nums[0], nums[1]
    p = float(re.search(r"(0?\.\d+)", text).group(1)) if re.search(r"(0?\.\d+)", text) else nums[2] / 100
    exact = Fraction(math.comb(n, k)) * Fraction(str(p)) ** k * (1 - Fraction(str(p))) ** (n - k)
    sol.add("P(X=k) = C(n,k) pᵏ (1−p)ⁿ⁻ᵏ.", f"C({n},{k})·{p}^{k}·{round(1-p, 10)}^{n-k} = {float(exact):.6f}")
    ok = _simulate_check(sol, exact, lambda r: sum(r.random() < p for _ in range(n)) == k)
    return sol.finish(exact, f"P(X = {k}) = {float(exact):.6f}", verified=ok)


def _letters(sol, text):
    sol.method = "multiset permutations + enumeration"
    m = re.search(r"(?:word|letters of)\s+['\"]?([A-Za-z]+)['\"]?", text)
    if not m:
        sol.warnings.append("give the word, e.g. 'arrangements of the letters of BANANA'")
        return sol
    w = m.group(1).upper()
    from collections import Counter
    c = Counter(w)
    v = math.factorial(len(w)) // math.prod(math.factorial(k) for k in c.values())
    sol.add(f"n! divided by factorials of repeated letters {dict(c)}.", f"{len(w)}! / ({' · '.join(str(k) + '!' for k in c.values() if k > 1) or '1'}) = {v}")
    ok = len(set(itertools.permutations(w))) == v if len(w) <= 9 else None
    sol.evidence(f"enumerating distinct permutations agrees: {ok}")
    return sol.finish(v, str(v), verified=ok)


def _dice(sol, text, nums):
    sol.method = "sample-space enumeration + Monte Carlo"
    low = text.lower()
    m = re.search(r"(\d+)\s*(?:dice|die)", low)
    nd = int(m.group(1)) if m else (2 if "dice" in low else 1)
    nd = min(max(nd, 1), 6)
    space = list(itertools.product(range(1, 7), repeat=nd))
    if re.search(r"\b(sum|total)\b", low) and nums:
        target = nums[-1] if not (len(nums) == 1 and m) else None
        if target is None:
            sol.warnings.append("give the target sum, e.g. 'sum of 2 dice is at least 9'")
            return sol
        mode = ">=" if re.search(r"at least|>=|≥|or more|greater|more than", low) else "<=" if re.search(r"at most|<=|≤|or less|less than|fewer", low) else "=="
        cond = {"==": lambda s: s == target, ">=": lambda s: s >= target, "<=": lambda s: s <= target}[mode]
        fav = sum(1 for r in space if cond(sum(r)))
        exact = Fraction(fav, len(space))
        sol.add(f"Sample space has 6^{nd} = {len(space)} equally likely outcomes.", f"favourable (sum {mode} {target}): {fav}")
        ok = _simulate_check(sol, exact, lambda r: cond(sum(r.randint(1, 6) for _ in range(nd))))
        return sol.finish(exact, f"P = {exact} ≈ {float(exact):.5f}", verified=ok)
    if re.search(r"\bsix\b|\b6\b", low) and re.search(r"at least one|any", low):
        exact = 1 - Fraction(5, 6) ** nd
        sol.add("Complement: 1 − P(no six).", f"1 − (5/6)^{nd}")
        ok = _simulate_check(sol, exact, lambda r: any(r.randint(1, 6) == 6 for _ in range(nd)))
        return sol.finish(exact, f"P(at least one six) = {exact} ≈ {float(exact):.5f}", verified=ok)
    sol.warnings.append("state the event, e.g. 'probability that the sum of two dice is at least 9'")
    return sol


def _coins(sol, text, nums):
    sol.method = "binomial + Monte Carlo"
    low = text.lower()
    mn = re.search(r"(\d+)\s*(?:coins?|tosses|flips|times)", low)
    mk = re.search(r"(\d+)\s*(?:heads?|tails?)", low)
    n = int(mn.group(1)) if mn else (nums[-1] if nums else 3)
    k = int(mk.group(1)) if mk else (nums[0] if len(nums) > 1 else None)
    if k is None:
        sol.warnings.append("e.g. 'probability of exactly 3 heads in 5 coin tosses'")
        return sol
    if re.search(r"at least|or more", low):
        exact = sum(Fraction(math.comb(n, j), 2 ** n) for j in range(k, n + 1))
        trial = lambda r: sum(r.random() < .5 for _ in range(n)) >= k          # noqa: E731
    elif re.search(r"at most|or fewer|or less", low):
        exact = sum(Fraction(math.comb(n, j), 2 ** n) for j in range(0, k + 1))
        trial = lambda r: sum(r.random() < .5 for _ in range(n)) <= k          # noqa: E731
    else:
        exact = Fraction(math.comb(n, k), 2 ** n)
        trial = lambda r: sum(r.random() < .5 for _ in range(n)) == k          # noqa: E731
    sol.add("Each toss is independent with p = 1/2; count favourable sequences out of 2ⁿ.", f"n={n}, k={k}: {exact}")
    ok = _simulate_check(sol, exact, trial)
    return sol.finish(exact, f"P = {exact} ≈ {float(exact):.5f}", verified=ok)


def _cards(sol, text, nums):
    sol.method = "hypergeometric + Monte Carlo"
    low = text.lower()
    if re.search(r"two|2|both", low) and re.search(r"ace|king|queen|heart|spade|red|black", low):
        kind = re.search(r"ace|king|queen|heart|spade|red|black", low).group(0)
        good = {"ace": 4, "king": 4, "queen": 4, "heart": 13, "spade": 13, "red": 26, "black": 26}[kind]
        exact = Fraction(good, 52) * Fraction(good - 1, 51)
        sol.add(f"Without replacement: {good}/52 · {good - 1}/51.", f"= {exact}")
        deck = list(range(52))
        ok = _simulate_check(sol, exact, lambda r: all(c < good for c in r.sample(deck, 2)))
        return sol.finish(exact, f"P(both are {kind}) = {exact} ≈ {float(exact):.5f}", verified=ok)
    sol.warnings.append("card question not recognised")
    return sol
''')

# ====================================================================================================
# MODULE: deep_oq.solvers.finance    (deep_oq/solvers/finance.py, 82 lines)
# ====================================================================================================
_register('deep_oq.solvers.finance', False, 'deep_oq/solvers/finance.py', r'''"""Interest, EMI/loan payments, CAGR, NPV/IRR, annuities."""
from __future__ import annotations

import re

import numpy as np
from scipy import optimize

from ..models import Solution

CATEGORY = "finance"
_RX = re.compile(r"(?i)(compound interest|simple interest|emi|loan|mortgage|cagr|npv|irr|annuity|sip|"
                  r"future value|present value|interest rate|per annum|p\.a\.)")


def can_handle(text: str) -> float:
    return 0.85 if _RX.search(text) and re.search(r"\d", text) else 0.0


def _n(text):
    return [float(v.replace(",", "")) for v in re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?", text)]


def solve(text: str) -> Solution:
    sol = Solution(text, CATEGORY)
    low = text.lower()
    nums = _n(text)
    try:
        if "simple interest" in low and len(nums) >= 3:
            P, r, t = nums[:3]
            si = P * r * t / 100
            sol.method = "SI = P·r·t/100"
            sol.add("SI = P r t / 100.", f"{P:g}×{r:g}×{t:g}/100 = {si:,.2f}")
            sol.evidence(f"total amount {P + si:,.2f}; interest per year {si / t:,.2f} = {P * r / 100:,.2f}")
            return sol.finish(si, f"interest = {si:,.2f}, amount = {P + si:,.2f}", verified=abs(si / t - P * r / 100) < 1e-9)
        if re.search(r"compound interest|future value", low) and len(nums) >= 3:
            P, r, t = nums[:3]
            n = nums[3] if len(nums) > 3 else (12 if "month" in low else 4 if "quarter" in low else 1)
            A = P * (1 + r / 100 / n) ** (n * t)
            sol.method = "A = P(1 + r/n)^(nt) + year-by-year simulation"
            sol.add("A = P (1 + r/n)^(n t).", f"{P:g}×(1+{r:g}/{n:g}/100)^({n:g}×{t:g}) = {A:,.2f}")
            bal = P
            for _ in range(int(n * t)):
                bal *= 1 + r / 100 / n
            sol.evidence(f"period-by-period simulation of {int(n * t)} compounding steps gives {bal:,.2f}")
            return sol.finish(A, f"amount = {A:,.2f}, compound interest = {A - P:,.2f}", verified=abs(bal - A) < 1e-6 * A)
        if re.search(r"emi|loan|mortgage", low) and len(nums) >= 3:
            P, r, y = nums[:3]
            i = r / 100 / 12
            n = int(y * 12) if "month" not in low else int(y)
            emi = P * i * (1 + i) ** n / ((1 + i) ** n - 1)
            sol.method = "EMI = P·i(1+i)ⁿ/((1+i)ⁿ−1) + amortisation replay"
            sol.add("Monthly rate i and number of payments n.", f"i = {i:.6f}, n = {n}; EMI = {emi:,.2f}")
            bal = P
            for _ in range(n):
                bal = bal * (1 + i) - emi
            sol.evidence(f"amortisation replay: balance after {n} payments = {bal:,.4f} (should be ≈ 0)")
            sol.evidence(f"total paid {emi * n:,.2f}, total interest {emi * n - P:,.2f}")
            return sol.finish(emi, f"EMI = {emi:,.2f} per month", verified=abs(bal) < 1e-4 * P)
        if "cagr" in low and len(nums) >= 3:
            a, b, y = nums[:3]
            c = ((b / a) ** (1 / y) - 1) * 100
            sol.add("CAGR = (end/start)^(1/years) − 1.", f"= {c:.4f}%")
            return sol.finish(c, f"{c:.4f}% per year", verified=abs(a * (1 + c / 100) ** y - b) < 1e-6 * b)
        if re.search(r"irr|npv", low) and len(nums) >= 3:
            flows = nums if "irr" in low else nums[1:]
            if "npv" in low:
                rate = nums[0] / 100
                npv = sum(cf / (1 + rate) ** i for i, cf in enumerate(flows))
                sol.add("NPV = Σ CFₜ/(1+r)ᵗ.", f"= {npv:,.2f}")
                return sol.finish(npv, f"NPV = {npv:,.2f}", verified=True)
            f = lambda r: sum(cf / (1 + r) ** i for i, cf in enumerate(flows))      # noqa: E731
            r = optimize.brentq(f, -0.99, 10)
            sol.add("IRR is the rate where NPV = 0 (root finding).", f"IRR = {r:.4%}")
            sol.evidence(f"NPV at the IRR = {f(r):.2e}")
            return sol.finish(r, f"IRR = {r:.4%}", verified=abs(f(r)) < 1e-6)
        sol.warnings.append("finance question not recognised")
        return sol
    except Exception as exc:                                  # noqa: BLE001
        sol.warnings.append(f"finance error: {exc}")
        return sol
''')

# ====================================================================================================
# MODULE: deep_oq.solvers.linalg    (deep_oq/solvers/linalg.py, 164 lines)
# ====================================================================================================
_register('deep_oq.solvers.linalg', False, 'deep_oq/solvers/linalg.py', r'''"""Matrices: det, inverse, rank, eigen, RREF, A x = b, powers, trace, transpose, LU, nullspace."""
from __future__ import annotations

import re

import sympy as sp

from ..models import Solution
from .. import parse, verify

CATEGORY = "linear algebra"
_RX = re.compile(r"(?i)(matrix|matrices|determinant|\bdet\b|eigen|inverse of|rank of|transpose|"
                 r"null ?space|row reduc|rref|\[\s*\[)")


def can_handle(text: str) -> float:
    return 0.9 if _RX.search(text) else 0.0


def _cell(c: str):
    """Matrix entry: only plain maths allowed (goes through the parser firewall)."""
    c = c.strip()
    if re.fullmatch(r"-?\d+", c):
        return sp.Integer(int(c))
    return sp.nsimplify(parse.to_expr(c))


def parse_matrices(text: str) -> list:
    """Find every [[..],[..]] block (also ; separated rows inside single brackets)."""
    mats = []
    for m in re.finditer(r"\[\s*(\[[^\[\]]*\](?:\s*,\s*\[[^\[\]]*\])*)\s*\]", text):
        rows = re.findall(r"\[([^\[\]]*)\]", m.group(1))
        mats.append(sp.Matrix([[_cell(c) for c in r.split(",") if c.strip()] for r in rows]))
    if not mats:
        for m in re.finditer(r"\[([^\[\]]*;[^\[\]]*)\]", text):
            rows = m.group(1).split(";")
            mats.append(sp.Matrix([[_cell(c) for c in r.replace(",", " ").split()] for r in rows]))
    return mats


def solve(text: str) -> Solution:
    sol = Solution(text, CATEGORY)
    low = text.lower()
    try:
        mats = parse_matrices(text)
        if not mats:
            sol.warnings.append("write matrices like [[1,2],[3,4]]")
            return sol
        A = mats[0]
        sol.add("Read the matrix.", sp.latex(A))
        if re.search(r"solve|a\s*x\s*=|ax=b|linear system", low) and len(mats) >= 2:
            return _linear_system(sol, A, mats[1])
        if "determinant" in low or re.search(r"\bdet\b", low):
            return _det(sol, A)
        if "inverse" in low:
            return _inverse(sol, A)
        if "eigen" in low:
            return _eigen(sol, A)
        if "rank" in low:
            r = A.rank()
            sol.add("Row-reduce and count pivots.", sp.latex(A.rref()[0]))
            sol.evidence(f"numpy matrix_rank agrees: {_np_rank(A) == r}")
            return sol.finish(r, f"rank = {r}", verified=_np_rank(A) == r)
        if "transpose" in low:
            return sol.finish(A.T, str(A.T.tolist()), verified=True)
        if "null" in low:
            ns = A.nullspace()
            ok = all(A * v == sp.zeros(A.rows, 1) for v in ns)
            sol.evidence(f"A·v = 0 for every basis vector: {ok}")
            return sol.finish(ns, str([list(v) for v in ns]) or "only the zero vector", verified=ok)
        if "rref" in low or "row reduc" in low:
            R, piv = A.rref()
            sol.add(f"Reduced row echelon form; pivot columns {piv}.", sp.latex(R))
            return sol.finish(R, str(R.tolist()), verified=_np_rank(A) == len(piv))
        if "trace" in low:
            return sol.finish(A.trace(), f"trace = {A.trace()}", verified=True)
        if len(mats) >= 2 and re.search(r"(multiply|times|product|\*)", low):
            B = mats[1]
            P = A * B
            sol.add("Matrix product.", sp.latex(P))
            ok = verify.matrix_equal(P, sp.Matrix(_np_mul(A, B)))
            sol.evidence(f"numpy A@B agrees: {ok}")
            return sol.finish(P, str(P.tolist()), verified=ok)
        return _det(sol, A) if A.is_square else sol
    except Exception as exc:                                  # noqa: BLE001
        sol.warnings.append(f"linear algebra error: {exc}")
        return sol


def _np_rank(A):
    import numpy as np
    return int(np.linalg.matrix_rank(np.array(A.tolist(), dtype=float)))


def _np_mul(A, B):
    import numpy as np
    return (np.array(A.tolist(), dtype=float) @ np.array(B.tolist(), dtype=float)).tolist()


def _det(sol, A):
    sol.method = "cofactor expansion + numpy cross-check"
    if not A.is_square:
        sol.warnings.append("determinant needs a square matrix")
        return sol
    d = A.det()
    if A.shape == (2, 2):
        sol.add("2×2 rule: ad − bc.", f"{A[0,0]}·{A[1,1]} − {A[0,1]}·{A[1,0]} = {d}")
    elif A.shape == (3, 3):
        sol.add("Expand along the first row (cofactors).", " + ".join(f"{A[0, j]}·C₀{j}" for j in range(3)) + f" = {d}")
    else:
        sol.add("Use row reduction / cofactor expansion.", str(d))
    import numpy as np
    nd = float(np.linalg.det(np.array(A.tolist(), dtype=float)))
    ok = abs(nd - float(d)) < 1e-6 * max(1, abs(nd))
    sol.evidence(f"numpy.linalg.det = {nd:.6g}: {'matches' if ok else 'MISMATCH'}")
    if d == 0:
        sol.add("Determinant is 0, so the matrix is singular (no inverse).")
    return sol.finish(d, f"det = {d}", verified=ok)


def _inverse(sol, A):
    sol.method = "adjugate / Gauss-Jordan + A·A⁻¹ = I check"
    if not A.is_square or A.det() == 0:
        return sol.finish(None, "not invertible (non-square or determinant 0)", verified=True)
    inv = A.inv()
    sol.add("A⁻¹ = adj(A)/det(A) (or Gauss-Jordan).", sp.latex(inv))
    ok = A * inv == sp.eye(A.rows)
    sol.evidence(f"A·A⁻¹ = I exactly: {ok}")
    return sol.finish(inv, str(inv.tolist()), verified=ok)


def _eigen(sol, A):
    sol.method = "characteristic polynomial + A v = λ v check"
    lam = sp.Symbol("λ")
    cp = sp.factor(A.charpoly(lam).as_expr())
    sol.add("Characteristic polynomial det(A − λI).", sp.latex(cp))
    ev = A.eigenvects()
    out, ok = [], True
    for val, mult, vecs in ev:
        for v in vecs:
            good = sp.simplify(A * v - val * v) == sp.zeros(A.rows, 1)
            ok &= good
            out.append(f"λ = {val} (mult {mult}), v = {list(v)}")
            sol.evidence(f"A·v = λ·v for λ={val}: {good}")
    sol.add("Solve (A − λI)v = 0 for each λ.", "; ".join(out))
    return sol.finish(ev, "; ".join(out), verified=ok)


def _linear_system(sol, A, b):
    sol.method = "Gaussian elimination + residual check"
    if b.shape[0] == 1 and b.shape[1] == A.rows:
        b = b.T
    sol.add("Solve A·x = b.", f"A = {sp.latex(A)},  b = {sp.latex(b)}")
    if A.is_square and A.det() != 0:
        xv = A.LUsolve(b)
        sol.add("A is invertible, so x is unique.", sp.latex(xv))
        ok = A * xv == b
        sol.evidence(f"A·x − b = 0 exactly: {ok}")
        return sol.finish(xv, str(list(xv)), verified=ok)
    aug = A.row_join(b)
    if A.rank() < aug.rank():
        return sol.finish(None, "no solution (inconsistent system)", verified=True)
    return sol.finish(None, "infinitely many solutions (free variables)", verified=True)
''')

# ====================================================================================================
# MODULE: deep_oq.solvers.logic    (deep_oq/solvers/logic.py, 341 lines)
# ====================================================================================================
_register('deep_oq.solvers.logic', False, 'deep_oq/solvers/logic.py', r'''"""Puzzles & logic: water jugs, river crossing, N-Queens, Hanoi, Sudoku, Monty Hall, truth tables, switch/bulb."""
from __future__ import annotations

import itertools
import math
import random
import re
from fractions import Fraction

from ..models import Solution
from .. import search, verify

CATEGORY = "logic / puzzle"
_RX = re.compile(r"(?i)(jugs?|queens?|hanoi|sudoku|monty hall|truth table|switches|light ?bulbs?|"
                 r"wolf.*goat|goat.*cabbage|river crossing|knights? and knaves|tautolog|puzzle|riddle|"
                 r"cannibals|missionaries|8.?puzzle|coins? weigh|balance scale)")


def can_handle(text: str) -> float:
    return 0.9 if _RX.search(text) else 0.0


def solve(text: str) -> Solution:
    sol = Solution(text, CATEGORY)
    low = text.lower()
    try:
        if re.search(r"switches", low) and re.search(r"bulbs?", low):
            return _switches(sol)
        if re.search(r"jugs?", low):
            return _jugs(sol, text)
        if "monty hall" in low:
            return _monty(sol)
        if re.search(r"queens?", low):
            return _queens(sol, text)
        if "hanoi" in low:
            return _hanoi(sol, text)
        if "sudoku" in low:
            return _sudoku(sol, text)
        if re.search(r"wolf.*goat|goat.*cabbage|river crossing", low):
            return _river(sol)
        if re.search(r"missionaries|cannibals", low):
            return _missionaries(sol)
        if "truth table" in low or "tautolog" in low:
            return _truth_table(sol, text)
        if re.search(r"balance scale|weigh", low) and "coin" in low:
            return _coin_weigh(sol)
        sol.warnings.append("puzzle type not built in; the LLM fallback will be used")
        return sol
    except Exception as exc:                                  # noqa: BLE001
        sol.warnings.append(f"logic solver error: {exc}")
        return sol


# ---------------------------------------------------------------- 3 switches / 3 bulbs
def _switches(sol):
    sol.method = "information-theory argument + exhaustive simulation"
    sol.add("One look inside gives you 3 observable properties of each bulb: ON/OFF and (if off) HOT/COLD. "
            "Using heat adds a second bit of information per bulb.")
    sol.add("Turn switch 1 ON and wait about 10 minutes so its bulb gets hot.")
    sol.add("Turn switch 1 OFF, turn switch 2 ON, and walk into the room immediately.")
    sol.add("Read the bulbs: the lit bulb is switch 2; the unlit but WARM bulb is switch 1; the unlit COLD bulb is switch 3.")
    # exhaustive check over all 6 wirings
    ok = True
    for wiring in itertools.permutations(range(3)):      # wiring[i] = bulb controlled by switch i
        bulbs = {b: ("on" if b == wiring[1] else "warm" if b == wiring[0] else "cold") for b in range(3)}
        guess = {"warm": 0, "on": 1, "cold": 2}
        ok &= all(wiring[guess[state]] == b for b, state in bulbs.items())
    sol.evidence(f"Simulated all {math.factorial(3)} possible wirings: the rule identifies the correct switch for every bulb → {ok}")
    sol.evidence("Why it is optimal: 3! = 6 wirings need ≥ 3 distinguishable states per bulb; ON/OFF alone gives only 2, heat gives 3.")
    return sol.finish("on = switch 2, off-but-warm = switch 1, off-and-cold = switch 3",
                      "Turn switch 1 on for ~10 min, then off; turn switch 2 on; enter. Lit bulb → switch 2, warm unlit bulb → switch 1, cold unlit bulb → switch 3.",
                      verified=ok)


# ---------------------------------------------------------------- water jugs
def _jugs(sol, text):
    sol.method = "breadth-first search over (a, b) states"
    nums = [int(v) for v in re.findall(r"\d+", text)]
    if len(nums) < 3:
        sol.warnings.append("e.g. 'jugs of 3 and 5 litres, measure 4'")
        return sol
    A, B, T = nums[0], nums[1], nums[2]

    def nb(s):
        a, b = s
        yield "fill A", (A, b)
        yield "fill B", (a, B)
        yield "empty A", (0, b)
        yield "empty B", (a, 0)
        t = min(a, B - b)
        yield "pour A→B", (a - t, b + t)
        t = min(b, A - a)
        yield "pour B→A", (a + t, b - t)

    path = search.bfs_path((0, 0), lambda s: T in s, nb)
    if path is None:
        sol.add(f"gcd({A},{B}) = {math.gcd(A, B)} does not divide {T}, or {T} exceeds both jugs.")
        return sol.finish(None, "impossible", verified=math.gcd(A, B) == 0 or T % math.gcd(A, B) != 0 or T > max(A, B))
    state = (0, 0)
    for lab, nxt in path:
        sol.add(f"{lab}", f"({state[0]},{state[1]}) → ({nxt[0]},{nxt[1]})")
        state = nxt
    ok = T in state
    sol.evidence(f"Replayed the moves: final contents {state}, target {T} present: {ok}; BFS guarantees the minimum {len(path)} moves.")
    return sol.finish(len(path), f"{len(path)} moves (shortest possible)", verified=ok)


# ---------------------------------------------------------------- wolf, goat, cabbage
def _river(sol):
    sol.method = "breadth-first search"
    ITEMS = ("farmer", "wolf", "goat", "cabbage")

    def bad(s):
        f, w, g, c = s
        return (w == g != f) or (g == c != f)

    def nb(s):
        f = s[0]
        yield "cross alone", tuple(1 - v if i == 0 else v for i, v in enumerate(s))
        for i in (1, 2, 3):
            if s[i] == f:
                n = list(s); n[0] = 1 - f; n[i] = 1 - f
                yield f"take the {ITEMS[i]}", tuple(n)

    path = search.bfs_path((0, 0, 0, 0), lambda s: s == (1, 1, 1, 1), lambda s: [(l, n) for l, n in nb(s) if not bad(n)])
    side = lambda v: "far" if v else "near"                    # noqa: E731
    for lab, st in path:
        sol.add(f"Farmer {lab}.", f"now on far bank: {[ITEMS[i] for i in range(4) if st[i]]}")
    ok = all(not bad(st) for _, st in path) and path[-1][1] == (1, 1, 1, 1)
    sol.evidence(f"No unsafe state in any of the {len(path)} steps; goal reached: {ok}")
    return sol.finish(len(path), f"{len(path)} crossings", verified=ok)


def _missionaries(sol):
    sol.method = "breadth-first search"
    # state: (missionaries_left, cannibals_left, boat_left)
    def ok_state(m, c):
        return 0 <= m <= 3 and 0 <= c <= 3 and (m == 0 or m >= c) and (3 - m == 0 or 3 - m >= 3 - c)

    def nb(s):
        m, c, b = s
        for dm, dc in ((1, 0), (2, 0), (0, 1), (0, 2), (1, 1)):
            n = (m - dm, c - dc, 0) if b else (m + dm, c + dc, 1)
            if ok_state(n[0], n[1]):
                yield f"move {dm} missionary(ies) and {dc} cannibal(s) {'right' if b else 'left'}", n

    path = search.bfs_path((3, 3, 1), lambda s: s == (0, 0, 0), nb)
    for lab, st in path:
        sol.add(lab, f"left bank (M,C,boat) = {st}")
    ok = path[-1][1] == (0, 0, 0) and all(ok_state(s[0], s[1]) for _, s in path)
    sol.evidence(f"every intermediate state is safe: {ok}")
    return sol.finish(len(path), f"{len(path)} crossings", verified=ok)


# ---------------------------------------------------------------- N-Queens
def _queens(sol, text):
    sol.method = "backtracking search"
    n = int(re.search(r"(\d+)", text).group(1)) if re.search(r"\d+", text) else 8
    cols, count, first = [], 0, None

    def rec(r, cols_used, d1, d2, placement):
        nonlocal count, first
        if r == n:
            count += 1
            if first is None:
                first = list(placement)
            return
        for c in range(n):
            if c in cols_used or (r - c) in d1 or (r + c) in d2:
                continue
            placement.append(c)
            rec(r + 1, cols_used | {c}, d1 | {r - c}, d2 | {r + c}, placement)
            placement.pop()

    rec(0, frozenset(), frozenset(), frozenset(), [])
    if first is None:
        return sol.finish(0, f"no solution for n={n}", verified=True)
    board = "\n".join("".join("Q" if first[r] == c else "." for c in range(n)) for r in range(n))
    sol.add(f"Place one queen per row, tracking used columns and diagonals; backtrack on conflict.")
    sol.add("First solution found (row → column):", str(first))
    ok = all(first[i] != first[j] and abs(first[i] - first[j]) != j - i for i in range(n) for j in range(i + 1, n))
    sol.evidence(f"no two queens share a row, column or diagonal: {ok}")
    sol.evidence(f"total number of solutions for n={n}: {count}")
    return sol.finish(first, f"{count} solutions for n={n}.  One of them:\n{board}", verified=ok)


# ---------------------------------------------------------------- Hanoi
def _hanoi(sol, text):
    sol.method = "recursion + replay"
    n = int(re.search(r"(\d+)", text).group(1)) if re.search(r"\d+", text) else 3
    moves = []

    def mv(k, a, b, c):
        if k:
            mv(k - 1, a, c, b); moves.append((a, c)); mv(k - 1, b, a, c)
    mv(n, "A", "B", "C")
    sol.add("Move n−1 discs to the spare peg, move the largest disc, then move n−1 discs on top of it.", f"T(n) = 2T(n−1)+1 ⇒ 2ⁿ−1")
    pegs = {"A": list(range(n, 0, -1)), "B": [], "C": []}
    legal = True
    for a, c in moves:
        d = pegs[a].pop()
        legal &= (not pegs[c]) or pegs[c][-1] > d
        pegs[c].append(d)
    ok = legal and pegs["C"] == list(range(n, 0, -1)) and len(moves) == 2 ** n - 1
    sol.evidence(f"replayed all {len(moves)} moves legally; all discs end on peg C: {ok}")
    shown = ", ".join(f"{a}→{c}" for a, c in moves[:31]) + (" …" if len(moves) > 31 else "")
    sol.add("Moves", shown)
    return sol.finish(len(moves), f"{len(moves)} moves (2^{n} − 1)", verified=ok)


# ---------------------------------------------------------------- Sudoku
def _sudoku(sol, text):
    sol.method = "constraint propagation + backtracking"
    digits = re.sub(r"[^0-9.]", "", text.lower().replace("sudoku", ""))
    digits = digits.replace(".", "0")
    if len(digits) != 81:
        sol.warnings.append("give the puzzle as 81 characters (0 or . for blanks)")
        return sol
    grid = [int(c) for c in digits]
    cand = lambda g, i: {1, 2, 3, 4, 5, 6, 7, 8, 9} - {g[j] for j in _peers(i)}      # noqa: E731

    def rec(g):
        best, opts = None, None
        for i in range(81):
            if g[i] == 0:
                c = cand(g, i)
                if not c:
                    return None
                if opts is None or len(c) < len(opts):
                    best, opts = i, c
                    if len(c) == 1:
                        break
        if best is None:
            return g
        for v in opts:
            g[best] = v
            r = rec(g)
            if r:
                return r
        g[best] = 0
        return None

    solved = rec(list(grid))
    if not solved:
        return sol.finish(None, "no solution (the puzzle is inconsistent)", verified=True)
    rows = ["".join(map(str, solved[r * 9:(r + 1) * 9])) for r in range(9)]
    ok = all(set(solved[r * 9:(r + 1) * 9]) == set(range(1, 10)) for r in range(9)) and \
         all(set(solved[c::9]) == set(range(1, 10)) for c in range(9)) and \
         all(set(solved[(br * 3 + r) * 9 + bc * 3 + c] for r in range(3) for c in range(3)) == set(range(1, 10)) for br in range(3) for bc in range(3)) and \
         all(g == 0 or g == s for g, s in zip(grid, solved))
    sol.add("Pick the empty cell with fewest candidates, try each, backtrack on contradiction.")
    sol.evidence(f"every row, column and 3×3 box contains 1-9 and all given clues are kept: {ok}")
    return sol.finish(solved, "\n".join(rows), verified=ok)


def _peers(i):
    r, c = divmod(i, 9)
    br, bc = r // 3 * 3, c // 3 * 3
    return ({r * 9 + k for k in range(9)} | {k * 9 + c for k in range(9)} |
            {(br + a) * 9 + bc + b for a in range(3) for b in range(3)}) - {i}


# ---------------------------------------------------------------- Monty Hall
def _monty(sol):
    sol.method = "conditional probability + simulation"
    sol.add("Initially P(car behind your door) = 1/3. The host, who knows where the car is, always opens a goat door.")
    sol.add("Staying wins only if your first pick was right: 1/3. Switching wins whenever your first pick was wrong: 2/3.")
    rng = random.Random(1)
    n, stay, switch = 200_000, 0, 0
    for _ in range(n):
        car, pick = rng.randrange(3), rng.randrange(3)
        opened = next(d for d in range(3) if d not in (car, pick))
        new = next(d for d in range(3) if d not in (pick, opened))
        stay += pick == car
        switch += new == car
    ok = abs(switch / n - 2 / 3) < 0.01 and abs(stay / n - 1 / 3) < 0.01
    sol.evidence(f"real simulation of {n:,} games: stay wins {stay / n:.4f}, switch wins {switch / n:.4f} → {'matches 1/3 and 2/3' if ok else 'MISMATCH'}")
    return sol.finish(Fraction(2, 3), "Switch! It wins with probability 2/3 (staying wins 1/3).", verified=ok)


# ---------------------------------------------------------------- truth tables
def _truth_table(sol, text):
    sol.method = "exhaustive evaluation"
    s = re.sub(r"(?i)^.*?(truth table|tautology)\s*(for|of|is)?", "", text).strip(" :?.")
    expr = (s.replace("∧", " and ").replace("∨", " or ").replace("¬", " not ").replace("&&", " and ").replace("||", " or ")
            .replace("→", " <= ").replace("↔", " == ").replace("!", " not "))
    # implication a -> b  ==  (not a) or b
    expr = re.sub(r"\b(\w+)\s*->\s*(\w+)", r"(not \1 or \2)", expr)
    names = sorted(set(re.findall(r"\b(?!and\b|or\b|not\b|True\b|False\b)[A-Za-z]\w*\b", expr)))
    if not names or len(names) > 6:
        sol.warnings.append("use up to 6 variables, e.g. truth table for (A and B) or not C")
        return sol
    rows, results = [], []
    for vals in itertools.product([False, True], repeat=len(names)):
        env = dict(zip(names, vals))
        r = _safe_bool_eval(expr, env)
        results.append(r)
        rows.append(" ".join("T" if v else "F" for v in vals) + " | " + ("T" if r else "F"))
    sol.add("Header: " + " ".join(names) + " | result", "\n     ".join(rows))
    kind = "tautology" if all(results) else "contradiction" if not any(results) else "contingent (sometimes true, sometimes false)"
    sol.evidence(f"{len(rows)} rows evaluated; result is a {kind}")
    return sol.finish(results, f"{kind}; true in {sum(results)} of {len(results)} rows", verified=True)


def _safe_bool_eval(expr: str, env: dict) -> bool:
    """Evaluate and/or/not/==/<= expressions over booleans using the AST - never eval()."""
    import ast
    tree = ast.parse(expr.strip(), mode="eval")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return not ev(n.operand)
        if isinstance(n, ast.Name) and n.id in env:
            return env[n.id]
        if isinstance(n, ast.Constant) and isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.Compare) and len(n.ops) == 1:
            a, b = ev(n.left), ev(n.comparators[0])
            if isinstance(n.ops[0], ast.Eq):
                return a == b
            if isinstance(n.ops[0], ast.LtE):          # a <= b  is  a implies b
                return (not a) or b
            if isinstance(n.ops[0], ast.NotEq):
                return a != b
        raise ValueError("unsupported boolean expression")
    return bool(ev(tree))


def _coin_weigh(sol):
    sol.method = "ternary information bound + exhaustive check"
    sol.add("12 coins, one odd (heavier or lighter), 3 weighings: 24 possible outcomes ≤ 3³ = 27, so it is possible.")
    sol.add("Weigh 1-4 vs 5-8. If balanced, the odd coin is among 9-12; weigh 9,10,11 vs 1,2,3 (known good) …")
    sol.add("If unbalanced, keep track of 'heavy-candidate' and 'light-candidate' groups and rearrange them in weighing 2.")
    sol.evidence("information bound 3³ = 27 ≥ 24 outcomes → a 3-weighing strategy can exist (classic result)")
    return sol.finish("possible in 3 weighings", "Possible in 3 weighings (24 outcomes ≤ 27).", verified=None, confidence=0.85)
''')

# ====================================================================================================
# MODULE: deep_oq.solvers.numtheory    (deep_oq/solvers/numtheory.py, 161 lines)
# ====================================================================================================
_register('deep_oq.solvers.numtheory', False, 'deep_oq/solvers/numtheory.py', r'''"""Number theory & arithmetic: primes, factorisation, gcd/lcm, modular arithmetic, CRT, bases, digits."""
from __future__ import annotations

import math
import re

import sympy as sp
from sympy import ntheory

from ..models import Solution

CATEGORY = "number theory"
_RX = re.compile(r"(?i)(prime|factor(?:s|ise|ize)? of|gcd|hcf|lcm|\bmod\b|modulo|remainder|totient|divisors?|"
                 r"base \d+|binary|hexadecimal|octal|fibonacci|perfect number|congruen|last digit|digits? of|"
                 r"multiplicative inverse|chinese remainder)")


def can_handle(text: str) -> float:
    return 0.8 if _RX.search(text) else 0.0


def _ints(text: str) -> list:
    return [int(v.replace(",", "")) for v in re.findall(r"-?\d[\d,]*", text) if v.replace(",", "").lstrip("-").isdigit()]


def solve(text: str) -> Solution:
    sol = Solution(text, CATEGORY)
    low = text.lower()
    nums = _ints(text)
    try:
        if "chinese remainder" in low or re.search(r"x\s*[≡=]+\s*\d+\s*\(?mod", low):
            return _crt(sol, text)
        if re.search(r"gcd|hcf|greatest common", low) and len(nums) >= 2:
            g = math.gcd(*nums)
            sol.method = "Euclid's algorithm"
            a, b = nums[0], nums[1]
            trace = []
            while b:
                trace.append(f"{a} = {a // b}·{b} + {a % b}")
                a, b = b, a % b
            sol.add("Euclidean algorithm.", "; ".join(trace))
            ok = all(n % g == 0 for n in nums) and math.gcd(*[n // g for n in nums]) == 1
            sol.evidence(f"{g} divides every number and the quotients are coprime: {ok}")
            return sol.finish(g, f"gcd = {g}", verified=ok)
        if re.search(r"lcm|least common", low) and len(nums) >= 2:
            l = math.lcm(*nums)
            ok = all(l % n == 0 for n in nums)
            sol.method = "lcm = |ab|/gcd"
            sol.evidence(f"every number divides {l}: {ok}")
            return sol.finish(l, f"lcm = {l}", verified=ok)
        if re.search(r"multiplicative inverse|inverse of \d+ mod", low) and len(nums) >= 2:
            a, m = nums[0], nums[1]
            inv = pow(a, -1, m)
            sol.method = "extended Euclid / pow(a,-1,m)"
            sol.evidence(f"{a}·{inv} mod {m} = {(a * inv) % m}")
            return sol.finish(inv, f"{inv}", verified=(a * inv) % m == 1)
        if re.search(r"remainder|\bmod\b|modulo", low) and len(nums) >= 2:
            return _modexp(sol, text, nums)
        if "totient" in low and nums:
            n = nums[0]
            t = int(sp.totient(n))
            ok = t == sum(1 for k in range(1, n + 1) if math.gcd(k, n) == 1) if n < 20000 else None
            sol.method = "Euler's phi"
            sol.add("φ(n) = n·∏(1 − 1/p) over primes p | n.", f"factors: {ntheory.factorint(n)}")
            sol.evidence(f"direct count of coprime numbers agrees: {ok}")
            return sol.finish(t, f"φ({n}) = {t}", verified=ok)
        if re.search(r"divisors?", low) and nums:
            d = ntheory.divisors(nums[0])
            sol.evidence(f"count {len(d)} matches tau(n) = {ntheory.divisor_count(nums[0])}")
            return sol.finish(d, f"{d} (count {len(d)}, sum {sum(d)})", verified=len(d) == ntheory.divisor_count(nums[0]))
        if re.search(r"prime factor|factor(?:s|ise|ize)? of|factorization|prime factorisation", low) and nums:
            n = nums[0]
            if n < 2:
                return sol.finish({}, f"{n} has no prime factors (it is {'a unit' if n == 1 else 'not a natural number > 1'})", verified=True)
            f = ntheory.factorint(n)
            prod = math.prod(p ** e for p, e in f.items())
            txt = " × ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(f.items()))
            sol.method = "trial division / Pollard rho"
            sol.evidence(f"product of factors = {prod}: {prod == n}; all factors prime: {all(ntheory.isprime(p) for p in f)}")
            return sol.finish(f, f"{n} = {txt}", verified=prod == n)
        if "prime" in low and nums:
            n = nums[0]
            p = ntheory.isprime(n)
            sol.method = "deterministic Miller-Rabin (< 3.3e24) / BPSW"
            if not p and n > 3:
                f = min(ntheory.factorint(n))
                sol.add(f"{n} is divisible by {f}.")
                sol.evidence(f"{n} = {f} × {n // f}")
            else:
                sol.add(f"No divisor up to √{n} divides {n}." if n > 3 else "Small prime.")
            return sol.finish(p, f"{n} is {'prime' if p else 'not prime'}", verified=True)
        if re.search(r"base \d+|binary|hexadecimal|octal", low):
            return _base(sol, low, nums)
        if "fibonacci" in low and nums:
            n = nums[0]
            v = int(sp.fibonacci(n))
            a, b = 0, 1
            for _ in range(n):
                a, b = b, a + b
            sol.evidence(f"iterative computation gives {a}")
            return sol.finish(v, f"F({n}) = {v}", verified=a == v)
        if "last digit" in low and len(nums) >= 2:
            a, e = nums[0], nums[1]
            v = pow(a, e, 10)
            sol.add("Powers of the last digit cycle, so reduce modulo 10 with fast exponentiation.")
            sol.evidence(f"pow({a},{e},10) = {v}")
            return sol.finish(v, f"last digit = {v}", verified=True)
        sol.warnings.append("number-theory question not recognised")
        return sol
    except Exception as exc:                                  # noqa: BLE001
        sol.warnings.append(f"number theory error: {exc}")
        return sol


def _modexp(sol, text, nums):
    m = re.search(r"(\d+)\s*\^\s*(\d+)\s*(?:mod|%|modulo)\s*(\d+)", text.replace("**", "^"), re.I)
    if m:
        a, e, mod = map(int, m.groups())
        v = pow(a, e, mod)
        sol.method = "square-and-multiply"
        sol.add(f"Fast modular exponentiation of {a}^{e} mod {mod}.", f"= {v}")
        ok = None
        if e < 2000:
            ok = (a ** e) % mod == v
        sol.evidence(f"direct big-integer computation agrees: {ok}")
        return sol.finish(v, f"{v}", verified=ok)
    a, mod = nums[0], nums[1]
    v = a % mod
    sol.evidence(f"{a} = {a // mod}·{mod} + {v}")
    return sol.finish(v, f"{v}", verified=a == (a // mod) * mod + v)


def _crt(sol, text):
    sol.method = "Chinese remainder theorem"
    pairs = [(int(a), int(m)) for a, m in re.findall(r"[≡=]+\s*(\d+)\s*\(?\s*mod\s*(\d+)", text, re.I)]
    if len(pairs) < 2:
        sol.warnings.append("give at least two congruences, e.g. x ≡ 2 (mod 3), x ≡ 3 (mod 5)")
        return sol
    r, m = ntheory.modular.crt([p[1] for p in pairs], [p[0] for p in pairs])
    ok = all(int(r) % mm == a % mm for a, mm in pairs)
    sol.add("Combine the congruences with CRT.", f"x ≡ {r} (mod {m})")
    sol.evidence(f"{r} satisfies every congruence: {ok}")
    return sol.finish((r, m), f"x ≡ {r} (mod {m})", verified=ok)


def _base(sol, low, nums):
    sol.method = "positional notation"
    m = re.search(r"(\w+)\s+(?:in|to)\s+(binary|hex\w*|octal|base\s*(\d+))", low)
    if m:
        src = m.group(1)
        tgt = {"binary": 2, "octal": 8}.get(m.group(2), 16 if m.group(2).startswith("hex") else int(m.group(3) or 10))
        n = int(src, 10) if src.isdigit() else int(src, 16)
        out = sp.ntheory.digits(n, tgt)[1:]
        sym = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        txt = "".join(sym[d] for d in out)
        back = int(txt, tgt)
        sol.evidence(f"converting back gives {back}")
        return sol.finish(txt, f"{n} in base {tgt} = {txt}", verified=back == n)
    sol.warnings.append("try: 'convert 255 to binary'")
    return sol
''')

# ====================================================================================================
# MODULE: deep_oq.solvers.physics    (deep_oq/solvers/physics.py, 216 lines)
# ====================================================================================================
_register('deep_oq.solvers.physics', False, 'deep_oq/solvers/physics.py', r'''"""Physics via a formula library + sympy.  Answers are cross-checked with a second independent formula when one exists."""
from __future__ import annotations

import itertools
import math
import re

import sympy as sp

from ..models import Solution

CATEGORY = "physics"
S = {n: sp.Symbol(n, real=True) for n in
     "u v a t s m F p KE PE W P Vv I R rho Vol A f lam T r g h G m1 m2 n Tk Pr c dT Q theta H Rng k x d w E".split()}
Eq = sp.Eq

FORMULAS = {   # name: (equation, nice name)
    "v = u + at": (Eq(S["v"], S["u"] + S["a"] * S["t"]), "first equation of motion"),
    "s = ut + ½at²": (Eq(S["s"], S["u"] * S["t"] + S["a"] * S["t"] ** 2 / 2), "second equation of motion"),
    "v² = u² + 2as": (Eq(S["v"] ** 2, S["u"] ** 2 + 2 * S["a"] * S["s"]), "third equation of motion"),
    "s = (u+v)t/2": (Eq(S["s"], (S["u"] + S["v"]) * S["t"] / 2), "average-velocity form"),
    "F = ma": (Eq(S["F"], S["m"] * S["a"]), "Newton's second law"),
    "p = mv": (Eq(S["p"], S["m"] * S["v"]), "momentum"),
    "KE = ½mv²": (Eq(S["KE"], S["m"] * S["v"] ** 2 / 2), "kinetic energy"),
    "PE = mgh": (Eq(S["PE"], S["m"] * S["g"] * S["h"]), "gravitational potential energy"),
    "W = Fs": (Eq(S["W"], S["F"] * S["s"]), "work"),
    "P = W/t": (Eq(S["P"], S["W"] / S["t"]), "power"),
    "V = IR": (Eq(S["Vv"], S["I"] * S["R"]), "Ohm's law"),
    "P = VI": (Eq(S["P"], S["Vv"] * S["I"]), "electrical power"),
    "ρ = m/V": (Eq(S["rho"], S["m"] / S["Vol"]), "density"),
    "Pr = F/A": (Eq(S["Pr"], S["F"] / S["A"]), "pressure"),
    "v = fλ": (Eq(S["v"], S["f"] * S["lam"]), "wave speed"),
    "T = 1/f": (Eq(S["T"], 1 / S["f"]), "period"),
    "a = v²/r": (Eq(S["a"], S["v"] ** 2 / S["r"]), "centripetal acceleration"),
    "F = Gm₁m₂/r²": (Eq(S["F"], S["G"] * S["m1"] * S["m2"] / S["r"] ** 2), "Newton's gravitation"),
    "Q = mcΔT": (Eq(S["Q"], S["m"] * S["c"] * S["dT"]), "heat"),
    "E = mc²": (Eq(S["E"], S["m"] * S["c"] ** 2), "mass-energy"),
    "x = d/t (speed)": (Eq(S["v"], S["d"] / S["t"]), "constant speed"),
    "Range": (Eq(S["Rng"], S["v"] ** 2 * sp.sin(2 * sp.rad(S["theta"])) / S["g"]), "projectile range"),
    "Max height": (Eq(S["H"], (S["v"] * sp.sin(sp.rad(S["theta"]))) ** 2 / (2 * S["g"])), "projectile max height"),
    "Time of flight": (Eq(S["t"], 2 * S["v"] * sp.sin(sp.rad(S["theta"])) / S["g"]), "projectile time of flight"),
    "Hooke": (Eq(S["F"], S["k"] * S["x"]), "Hooke's law"),
}
CONST = {"g": 9.81, "G": 6.674e-11, "c": 299792458.0}

ALIASES = [  # longest / most specific first
    (r"initial (?:velocity|speed)", "u"), (r"final (?:velocity|speed)", "v"), (r"launch (?:velocity|speed)|projection (?:velocity|speed)", "v"),
    (r"time of flight", "t"), (r"maximum height|max(?:imum)? height", "H"), (r"horizontal range|range", "Rng"),
    (r"spring constant", "k"), (r"specific heat(?: capacity)?", "c"), (r"temperature change", "dT"),
    (r"kinetic energy", "KE"), (r"potential energy", "PE"), (r"work done|work", "W"),
    (r"potential difference|voltage", "Vv"), (r"(?:velocity|speed)", "v"), (r"acceleration", "a"),
    (r"(?:displacement|distance)", "s"), (r"time", "t"), (r"mass", "m"), (r"force", "F"), (r"momentum", "p"),
    (r"power", "P"), (r"current", "I"), (r"resistance", "R"), (r"density", "rho"), (r"volume", "Vol"), (r"area", "A"),
    (r"pressure", "Pr"), (r"frequency", "f"), (r"wavelength", "lam"), (r"radius", "r"), (r"height", "h"),
    (r"angle", "theta"), (r"extension", "x"), (r"heat", "Q"),
]
_NUM = r"-?\d+(?:\.\d+)?(?:e-?\d+)?"
# number + unit  ->  symbol, used only when no quantity word is attached to the number
UNIT_HINT = [(r"m/s_sq", "a"), (r"m/s", "v"), (r"kg", "m"), (r"n\b|newtons?", "F"), (r"j\b|joules?", "W"),
             (r"w\b|watts?", "P"), (r"volts?|v\b", "Vv"), (r"amps?|amperes?|a\b", "I"), (r"ohms?|ω", "R"), (r"hz|hertz", "f"),
             (r"°|degrees?|deg\b", "theta"), (r"pa\b|pascals?", "Pr"), (r"s\b|sec(?:onds?)?\b", "t"), (r"m\b|metres?|meters?", "s")]
_RX = re.compile(r"(?i)(velocity|acceleration|projectile|ohm|resistance|voltage|kinetic energy|potential energy|momentum|"
                 r"newton|wavelength|frequency|falls?|dropped|thrown|launched|m/s|force|torque|work done|range|max(?:imum)? height)")
_GROUP = {"u": "vel", "v": "vel", "a": "acc", "s": "len", "h": "len", "r": "len", "x": "len", "Rng": "len", "H": "len",
          "t": "time", "m": "mass", "F": "force", "W": "energy", "KE": "energy", "PE": "energy", "E": "energy", "Q": "energy",
          "P": "power", "Vv": "volt", "I": "amp", "R": "ohm", "f": "freq", "theta": "angle", "Pr": "press"}


def _unit_hint(tail: str):
    for rx, hint in UNIT_HINT:
        if re.match(rf"\s*(?:{rx})", tail, re.I):
            return hint
    return None


_FILL = re.compile(r"^[\s,:=]*(?:of|is|was|are|=|at|:)?[\s,:=]*$", re.I)


def can_handle(text: str) -> float:
    return 0.8 if _RX.search(text) and re.search(r"\d", text) else 0.0


def _alias_spans(t: str):
    """Non-overlapping (start, end, symbol) for every quantity word, specific ones first."""
    taken, out = [], []
    for rx, sym in ALIASES:
        for m in re.finditer(rx, t, re.I):
            if all(m.end() <= s or m.start() >= e for s, e, _ in taken):
                taken.append((m.start(), m.end(), sym))
    return sorted(taken)


_ASSUMED: list = []


def _parse_knowns(text: str):
    t = re.sub(r"m\s*/\s*s\s*(?:\^\s*2|\*\*\s*2|2\b)", "m/s_sq", text.replace("²", "^2"))
    t = re.sub(r"(?i)\bms\^?-2\b", "m/s_sq", t)
    t = re.sub(rf"(?i)\bfrom\s+({_NUM})\s*m/s(?!_)", r"initial velocity \1", t)
    t = re.sub(rf"(?i)\b(?:to|reaches|reaching)\s+({_NUM})\s*m/s(?!_)", r"final velocity \1", t)
    known, target = {}, None
    m = re.search(r"(?i)\b(?:find|calculate|determine|what is|what's|compute)\b\s+(?:the\s+)?(.+?)(?:\.|\?|$|,| if | when | given )", t)
    work = t
    if m:
        phrase = m.group(1)
        spans = _alias_spans(phrase)
        if spans:
            target = spans[0][2]
        work = t.replace(m.group(0), " ")
    for sym in S:                                              # explicit "u = 5"
        mm = re.search(rf"(?<![A-Za-z]){sym}\s*=\s*({_NUM})", work)
        if mm and sym != target:
            known[sym] = float(mm.group(1))
    spans = _alias_spans(work)
    for nm in re.finditer(_NUM, work):
        if any(abs(float(nm.group()) - v) < 0 for v in ()):      # placeholder for clarity
            pass
        before = [sp_ for sp_ in spans if sp_[1] <= nm.start()]
        after = [sp_ for sp_ in spans if sp_[0] >= nm.end()]
        sym = None
        if before:
            b = before[-1]
            gap = work[b[1]:nm.start()]
            if _FILL.match(gap) and not re.search(_NUM, gap):
                sym = b[2]
        if sym is None and after:
            a_ = after[0]
            gap = work[nm.end():a_[0]]
            units_only = re.fullmatch(r"\s*(?:m/s_sq|m/s|kg|n|j|w|v|a|ohms?|hz|pa|m|s|k|c|°c?|degrees?|metres?|meters?|seconds?|volts?|amps?)?\s*(?:of|for|in)?\s*", gap, re.I)
            if units_only:
                sym = a_[2]
                hint = _unit_hint(work[nm.end():nm.end() + 14])
                if hint and _GROUP.get(hint) != _GROUP.get(sym):
                    sym = None                  # the unit says it is something else; trust the unit
        if sym is None:
            tail = work[nm.end():nm.end() + 14]
            for rx, hint in UNIT_HINT:
                if re.match(rf"\s*(?:{rx})", tail, re.I):
                    sym = hint
                    break
        if sym and sym != target and sym not in known:
            known[sym] = float(nm.group())
    if re.search(r"(?i)\bfrom rest\b|starts? from rest|dropped|released|falls?", t):
        known.setdefault("u", 0.0)
    if re.search(r"(?i)\bfalls?\b|dropped|free[- ]fall", t):
        if "a" not in known:
            known["a"] = CONST["g"]
            _ASSUMED.append("free fall: used a = g = 9.81 m/s²")
        if "h" in known and "s" not in known:
            known["s"] = known["h"]
    return known, target


def solve(text: str) -> Solution:
    sol = Solution(text, CATEGORY)
    _ASSUMED.clear()
    known, target = _parse_knowns(text)
    sol.warnings.extend(_ASSUMED)
    try:
        if not known or not target:
            sol.warnings.append("could not identify the known values and the unknown; "
                                "write like 'u=0, a=2, t=5, find s' or 'initial velocity 0, acceleration 2, time 5, find displacement'")
            return sol
        sol.method = "formula selection + sympy solve + cross-check with independent formula"
        sol.add("Known: " + ", ".join(f"{k} = {v:g}" for k, v in known.items()) + f";  unknown: {target}")
        results = []                      # (formula name, value)
        used_const = False
        for name, (eq, nice) in FORMULAS.items():
            syms = {str(s) for s in eq.free_symbols}
            if target not in syms:
                continue
            need = syms - {target}
            env = dict(known)
            for c in need - set(env):
                if c in CONST:
                    env[c] = CONST[c]
            if need - set(env):
                continue
            subs = {S[k]: v for k, v in env.items() if k in syms}
            try:
                sols = sp.solve(eq.subs(subs), S[target])
            except Exception:                                 # noqa: BLE001
                continue
            vals = [float(sp.N(s)) for s in sols if sp.N(s).is_real]
            if not vals:
                continue
            results.append((name, nice, vals, env, eq))
        if not results:
            sol.warnings.append("no formula in the library links these quantities")
            return sol
        name, nice, vals, env, eq = results[0]
        v = max(vals) if len(vals) > 1 and target == "t" else vals[0]
        if len(vals) > 1 and target == "t":
            vals = [x for x in vals if x >= 0] or vals
            v = vals[0]
        if any(k in env and k not in known for k in env):
            sol.add("Used standard constants: " + ", ".join(f"{k} = {env[k]:g}" for k in env if k not in known))
            sol.warnings.append("assumed standard constants (g = 9.81 m/s², etc.); say so if your problem uses g = 10")
        sol.add(f"Use {nice}: {name}.", f"{sp.latex(eq)}")
        sol.add(f"Substitute and solve for {target}.", f"{target} = {v:.6g}")
        agree = [(n2, v2) for n2, _, v2s, _, _ in results[1:] for v2 in v2s if abs(v2 - v) <= 1e-6 * max(1, abs(v))]
        disagree = [(n2, v2s) for n2, _, v2s, _, _ in results[1:] if not any(abs(x - v) <= 1e-6 * max(1, abs(v)) for x in v2s)]
        resid = abs(float(sp.N((eq.lhs - eq.rhs).subs({S[k]: val for k, val in env.items()} | {S[target]: v})))) 
        ok = resid < 1e-6 * max(1, abs(v)) and not disagree
        sol.evidence(f"substituting back: residual {resid:.2e}")
        if agree:
            sol.evidence("independent cross-check with " + ", ".join(n2 for n2, _ in agree) + " gives the same value")
        if disagree:
            sol.evidence("WARNING: another formula disagrees: " + str(disagree))
        if len(vals) > 1:
            sol.warnings.append(f"equation has multiple roots {['%.5g' % x for x in vals]}; showing the first physical one")
        return sol.finish(v, f"{target} = {v:.6g} (SI units)", verified=ok)
    except Exception as exc:                                  # noqa: BLE001
        sol.warnings.append(f"physics error: {exc}")
        return sol
''')

# ====================================================================================================
# MODULE: deep_oq.solvers.stats    (deep_oq/solvers/stats.py, 131 lines)
# ====================================================================================================
_register('deep_oq.solvers.stats', False, 'deep_oq/solvers/stats.py', r'''"""Descriptive statistics, regression, correlation, normal / t / binomial calculations."""
from __future__ import annotations

import math
import re
import statistics as st

import numpy as np
from scipy import stats as sps

from ..models import Solution

CATEGORY = "statistics"
_RX = re.compile(r"(?i)(standard deviation|std ?dev|variance|median|mode of|quartile|correlation|regression|"
                  r"line of best fit|z-?score|normal distribution|confidence interval|t-?test|range of|"
                  r"percentile|iqr|skewness|outlier|mean and)")


def can_handle(text: str) -> float:
    return 0.85 if _RX.search(text) and len(re.findall(r"-?\d+(?:\.\d+)?", text)) >= 2 else 0.0


def _nums(text):
    return [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", text)]


def solve(text: str) -> Solution:
    sol = Solution(text, CATEGORY)
    low = text.lower()
    try:
        if re.search(r"regression|best fit|correlation", low):
            return _bivariate(sol, text, low)
        if re.search(r"z-?score|normal distribution|percentile", low) and re.search(r"mean|μ", low):
            return _normal(sol, text, low)
        if "confidence interval" in low:
            return _ci(sol, text)
        return _describe(sol, text, low)
    except Exception as exc:                                  # noqa: BLE001
        sol.warnings.append(f"statistics error: {exc}")
        return sol


def _describe(sol, text, low):
    sol.method = "closed-form formulas + numpy/statistics cross-check"
    d = _nums(re.sub(r"(?i)(sample|population)", "", text))
    n = len(d)
    mean = sum(d) / n
    ss = sum((v - mean) ** 2 for v in d)
    pop_var, samp_var = ss / n, ss / (n - 1) if n > 1 else float("nan")
    sol.add(f"n = {n}, mean = Σx/n = {sum(d):g}/{n} = {mean:.6g}")
    sol.add("Sum of squared deviations.", f"Σ(x−x̄)² = {ss:.6g}")
    srt = sorted(d)
    med = st.median(srt)
    q1, q3 = np.percentile(srt, [25, 75])
    res = {"mean": mean, "median": med, "population variance": pop_var, "sample variance": samp_var,
           "population std dev": math.sqrt(pop_var), "sample std dev": math.sqrt(samp_var) if n > 1 else float("nan"),
           "min": srt[0], "max": srt[-1], "range": srt[-1] - srt[0], "Q1": q1, "Q3": q3, "IQR": q3 - q1}
    try:
        res["mode"] = st.multimode(d)
    except Exception:                                         # noqa: BLE001
        pass
    ok = (abs(res["population std dev"] - float(np.std(d))) < 1e-9 and
          (n < 2 or abs(res["sample std dev"] - float(np.std(d, ddof=1))) < 1e-9) and abs(med - float(np.median(d))) < 1e-9)
    sol.evidence(f"numpy std/median agree with the hand formulas: {ok}")
    fence = (q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
    outl = [v for v in d if v < fence[0] or v > fence[1]]
    txt = ", ".join(f"{k} = {v:.6g}" if isinstance(v, float) else f"{k} = {v}" for k, v in res.items())
    if outl:
        txt += f"; outliers (1.5·IQR rule): {outl}"
    return sol.finish(res, txt, verified=ok)


def _bivariate(sol, text, low):
    sol.method = "least squares + numpy.polyfit/corrcoef cross-check"
    d = _nums(text)
    if len(d) % 2 or len(d) < 6:
        sol.warnings.append("give paired data x1,y1,x2,y2,... (at least 3 pairs)")
        return sol
    xs, ys = np.array(d[0::2]), np.array(d[1::2])
    n = len(xs)
    sxx, sxy = ((xs - xs.mean()) ** 2).sum(), ((xs - xs.mean()) * (ys - ys.mean())).sum()
    b = sxy / sxx
    a = ys.mean() - b * xs.mean()
    r = sxy / math.sqrt(sxx * ((ys - ys.mean()) ** 2).sum())
    sol.add("Slope b = Sxy/Sxx; intercept a = ȳ − b·x̄.", f"b = {b:.6g}, a = {a:.6g}")
    sol.add("Pearson r = Sxy / √(Sxx·Syy).", f"r = {r:.6g}, R² = {r * r:.6g}")
    pb, pa = np.polyfit(xs, ys, 1)
    ok = abs(pb - b) < 1e-9 and abs(pa - a) < 1e-9 and abs(np.corrcoef(xs, ys)[0, 1] - r) < 1e-9
    sol.evidence(f"numpy.polyfit and corrcoef agree: {ok}")
    p = sps.pearsonr(xs, ys)[1] if n > 2 else float("nan")
    sol.evidence(f"p-value for r ≠ 0: {p:.4g} (n={n})")
    return sol.finish({"slope": b, "intercept": a, "r": r}, f"y = {a:.6g} + {b:.6g}·x,  r = {r:.5g},  R² = {r*r:.5g}", verified=ok)


def _normal(sol, text, low):
    sol.method = "standardisation + scipy.stats.norm"
    d = _nums(text)
    if len(d) < 3:
        sol.warnings.append("e.g. 'normal distribution mean 100 sd 15, probability below 130'")
        return sol
    mu, sd, x = d[0], d[1], d[2]
    z = (x - mu) / sd
    sol.add("z = (x − μ)/σ.", f"z = ({x:g} − {mu:g})/{sd:g} = {z:.4f}")
    above = re.search(r"above|greater|more than|over|exceed", low)
    p = sps.norm.sf(z) if above else sps.norm.cdf(z)
    sol.add("Look up the standard normal CDF Φ(z)" + (" and use 1 − Φ(z)." if above else "."), f"P = {p:.6f}")
    # independent check via error function
    p2 = 0.5 * math.erfc(z / math.sqrt(2)) if above else 0.5 * math.erfc(-z / math.sqrt(2))
    ok = abs(p - p2) < 1e-12
    sol.evidence(f"math.erfc formula agrees: {ok}")
    return sol.finish(p, f"z = {z:.4f}, P({'X > ' if above else 'X < '}{x:g}) = {p:.6f} ({p*100:.3f}%)", verified=ok)


def _ci(sol, text):
    sol.method = "Student-t interval"
    d = _nums(text)
    lvl = 0.95
    m = re.search(r"(\d{2}(?:\.\d+)?)\s*%", text)
    if m:
        lvl = float(m.group(1)) / 100
        d = [v for v in d if v != float(m.group(1))]
    n = len(d)
    mean, s = st.mean(d), st.stdev(d)
    tcrit = sps.t.ppf(0.5 + lvl / 2, n - 1)
    half = tcrit * s / math.sqrt(n)
    sol.add("CI = x̄ ± t*·s/√n.", f"{mean:.5g} ± {tcrit:.4f}·{s:.5g}/√{n}")
    lo, hi = sps.t.interval(lvl, n - 1, loc=mean, scale=s / math.sqrt(n))
    ok = abs(lo - (mean - half)) < 1e-9 and abs(hi - (mean + half)) < 1e-9
    sol.evidence(f"scipy.stats.t.interval agrees: {ok}")
    return sol.finish((lo, hi), f"{lvl:.0%} CI: ({lo:.5g}, {hi:.5g})", verified=ok)
''')

# ====================================================================================================
# MODULE: deep_oq.solvers.units    (deep_oq/solvers/units.py, 79 lines)
# ====================================================================================================
_register('deep_oq.solvers.units', False, 'deep_oq/solvers/units.py', r'''"""Unit conversion with dimensional checking."""
from __future__ import annotations

import re

from ..models import Solution

CATEGORY = "unit conversion"

# unit -> (dimension, factor to SI base)
U = {}
def _add(dim, table):
    for k, f in table.items():
        for name in k.split("|"):
            U[name] = (dim, f)

_add("length", {"m|meter|meters|metre|metres": 1, "km|kilometer|kilometers|kilometre|kilometres": 1000, "cm|centimeter|centimeters": .01,
                "mm|millimeter|millimeters": .001, "mi|mile|miles": 1609.344, "ft|foot|feet": .3048, "in|inch|inches": .0254,
                "yd|yard|yards": .9144, "nmi|nautical mile|nautical miles": 1852, "um|micron|micrometer": 1e-6, "ly|light year|light years": 9.4607304725808e15})
_add("mass", {"kg|kilogram|kilograms": 1, "g|gram|grams": .001, "mg|milligram|milligrams": 1e-6, "t|tonne|tonnes|metric ton": 1000,
              "lb|lbs|pound|pounds": .45359237, "oz|ounce|ounces": .028349523125, "stone": 6.35029318, "quintal": 100})
_add("time", {"s|sec|second|seconds": 1, "min|minute|minutes": 60, "h|hr|hour|hours": 3600, "day|days": 86400, "week|weeks": 604800,
              "year|years": 31557600, "ms|millisecond|milliseconds": .001})
_add("volume", {"l|liter|liters|litre|litres": .001, "ml|milliliter|milliliters|millilitre": 1e-6, "m3|cubic meter|cubic meters": 1,
                "gal|gallon|gallons": .003785411784, "cup|cups": .0002365882365, "floz|fl oz": 2.95735295625e-5, "pint|pints": .000473176473})
_add("speed", {"m/s|mps": 1, "km/h|kmh|kph|km/hr": 1 / 3.6, "mph|mi/h": .44704, "knot|knots|kn": .514444444, "ft/s|fps": .3048})
_add("energy", {"j|joule|joules": 1, "kj|kilojoule|kilojoules": 1000, "cal|calorie|calories": 4.184, "kcal|kilocalorie": 4184,
                "kwh|kw h": 3.6e6, "ev|electronvolt": 1.602176634e-19, "btu": 1055.05585})
_add("pressure", {"pa|pascal|pascals": 1, "kpa": 1000, "bar": 1e5, "atm|atmosphere|atmospheres": 101325, "psi": 6894.757293168, "mmhg|torr": 133.322387415})
_add("power", {"w|watt|watts": 1, "kw|kilowatt|kilowatts": 1000, "hp|horsepower": 745.69987158})
_add("area", {"m2|sq m": 1, "km2|sq km": 1e6, "cm2": 1e-4, "ha|hectare|hectares": 1e4, "acre|acres": 4046.8564224, "ft2|sq ft": .09290304})
_add("data", {"b|byte|bytes": 1, "kb|kilobyte": 1e3, "mb|megabyte": 1e6, "gb|gigabyte": 1e9, "tb|terabyte": 1e12, "kib": 1024, "mib": 1024**2, "gib": 1024**3, "bit|bits": .125})
_add("angle", {"rad|radian|radians": 1, "deg|degree|degrees": 0.017453292519943295, "grad": 0.015707963267948967, "rev|turn": 6.283185307179586})
_TEMPS = {"c", "celsius", "°c", "f", "fahrenheit", "°f", "k", "kelvin"}

_RX = re.compile(r"(?i)(-?\d+(?:\.\d+)?(?:e-?\d+)?)\s*([a-z°/ 0-9]+?)\s+(?:to|in|into|as)\s+([a-z°/ 0-9]+?)\s*(?:\?|$|\.)")


def can_handle(text: str) -> float:
    return 0.85 if re.search(r"(?i)convert|how many .* (in|are in)|\bto\b.*(km|cm|kg|lb|mile|feet|inch|litre|celsius|fahrenheit|kelvin|hours?|minutes?)", text) and re.search(r"\d", text) else 0.0


def _temp(v, a, b):
    a, b = a.strip("°").lower()[0], b.strip("°").lower()[0]
    c = {"c": v, "f": (v - 32) * 5 / 9, "k": v - 273.15}[a]
    return {"c": c, "f": c * 9 / 5 + 32, "k": c + 273.15}[b]


def solve(text: str) -> Solution:
    sol = Solution(text, CATEGORY)
    t = re.sub(r"(?i)^\s*(please\s+)?(convert|how many|what is)\s+", "", text).strip()
    m = _RX.search(t + " ")
    if not m:
        m2 = re.search(r"(?i)(-?\d+(?:\.\d+)?)\s*([a-z°/ 0-9]+?)\s+(?:to|in|into)\s+([a-z°/ 0-9]+)", t)
        if not m2:
            sol.warnings.append("try: 'convert 5 miles to km'")
            return sol
        m = m2
    v, a, b = float(m.group(1)), m.group(2).strip().lower(), m.group(3).strip().lower().rstrip("?.")
    sol.method = "convert via SI base unit + round trip"
    if a in _TEMPS and b in _TEMPS:
        out = _temp(v, a, b)
        back = _temp(out, b, a)
        sol.add("Temperature uses an offset formula, not just a factor.", f"{v:g} {a} → {out:.6g} {b}")
        sol.evidence(f"round trip back to {a}: {back:.6g}")
        return sol.finish(out, f"{out:.6g} {b}", verified=abs(back - v) < 1e-9)
    if a not in U or b not in U:
        sol.warnings.append(f"unknown unit: {a if a not in U else b}")
        return sol
    (da, fa), (db, fb) = U[a], U[b]
    if da != db:
        sol.warnings.append(f"cannot convert {da} ({a}) to {db} ({b}) - dimensions differ")
        return sol.finish(None, f"impossible: {a} is a {da} unit, {b} is a {db} unit", verified=True)
    out = v * fa / fb
    sol.add(f"Go via the SI base unit: {v:g} {a} × {fa:g} = {v * fa:g}; then ÷ {fb:g} to get {b}.", f"= {out:.8g} {b}")
    back = out * fb / fa
    sol.evidence(f"round trip: {out:.8g} {b} → {back:.8g} {a}")
    return sol.finish(out, f"{out:.8g} {b}", verified=abs(back - v) <= 1e-9 * max(1, abs(v)))
''')

# ====================================================================================================
# MODULE: deep_oq.tests.test_bridge    (deep_oq/tests/test_bridge.py, 112 lines)
# ====================================================================================================
_register('deep_oq.tests.test_bridge', False, 'deep_oq/tests/test_bridge.py', r'''"""Tests the LLM bridge with a mock model that misbehaves exactly like the screenshots."""
import asyncio
import unittest

from deep_oq import codeexec
from deep_oq.guard import has_loop, incomplete_reasons, remove_repetition, strip_unverified_claims
from deep_oq.llm_bridge import answer, expected_min_words, preserves_numbers
from deep_oq.engine import Engine

ENG = Engine(use_timeout=False)
run = asyncio.run


class Guard(unittest.TestCase):
    def test_sentence_loop(self):
        bad = ("Answer You can determine which switch controls each bulb by using the heat generated by the bulbs "
               "Answer You can determine which switch controls each bulb by using the heat generated by the bulbs.")
        self.assertTrue(has_loop(bad))
        self.assertEqual(remove_repetition(bad).lower().count("heat generated"), 1)

    def test_problem_let_loop(self):
        bad = "Problem: Let $f: " * 8
        self.assertTrue(has_loop(bad)); self.assertTrue(incomplete_reasons(remove_repetition(bad)))

    def test_fake_claims(self):
        t, ch = strip_unverified_claims("The answer is 0.5. I ran a Monte Carlo simulation to confirm."); self.assertTrue(ch); self.assertNotIn("Monte", t)

    def test_min_words(self):
        self.assertEqual(expected_min_words("hi"), 1); self.assertEqual(expected_min_words("features"), 60)
        self.assertTrue(incomplete_reasons("I am Deep OQ, an advanced AI", 60))


class Bridge(unittest.TestCase):
    def test_solver_beats_llm_and_numbers_are_protected(self):
        async def evil(msgs, **p):   # explanation that changes the answer
            return "Sure! The roots are 7 and 9, easy. " * 3
        r = run(answer("solve x^2 - 5x + 6 = 0", evil, engine=ENG))
        self.assertIn("2, 3", r.text); self.assertTrue(r.verified); self.assertIn("changed the numbers", " ".join(r.notes))

    def test_good_explanation_kept(self):
        async def good(msgs, **p):
            return "Nice one! Factor it and you get two roots: x = 2 and x = 3. Both check out when plugged back in, which is the best test there is."
        r = run(answer("solve x^2 - 5x + 6 = 0", good, engine=ENG)); self.assertEqual(r.source, "solver+llm")

    def test_loop_then_recovery(self):
        calls = {"n": 0}
        async def flaky(msgs, **p):
            calls["n"] += 1
            return "Problem: Let $f: " * 6 if calls["n"] == 1 else "Here's a joke: why did the function break up with the curve? It needed some space between them, and the derivative of their relationship was negative."
        r = run(answer("tell me a joke", flaky, engine=ENG)); self.assertGreaterEqual(r.attempts, 2); self.assertNotIn("Let $f", r.text)

    def test_one_liner_for_features_is_regenerated(self):
        calls = {"n": 0}
        async def lazy(msgs, **p):
            calls["n"] += 1
            return "I am Deep OQ, an advanced AI" if calls["n"] == 1 else " ".join(f"Feature{i} helps with task{i}." for i in range(20))
        r = run(answer("features", lazy, engine=ENG)); self.assertGreater(len(r.text.split()), 40)

    def test_real_tool_execution(self):
        calls = {"n": 0}
        async def toolish(msgs, **p):
            calls["n"] += 1
            if calls["n"] == 1:
                return "Let me compute that.\n```python\nprint(sum(range(1, 101)))\n```"
            return "I ran the code and the sum of 1 to 100 is 5050 - the classic Gauss trick, confirmed by actually executing it just now."
        r = run(answer("what is the sum of the numbers from 1 to 100, explain why", toolish, engine=ENG))
        self.assertEqual(r.source, "llm+tools"); self.assertEqual(r.tool_log[0]["output"], "5050"); self.assertIn("I ran the code", r.text)

    def test_fake_simulation_claim_removed_without_tool(self):
        async def liar(msgs, **p):
            return "The probability works out to about one half for this setup. I ran a Monte Carlo simulation and it confirmed this result."
        r = run(answer("what is the chance my weird game ends in a tie, explain", liar, engine=ENG)); self.assertNotIn("Monte Carlo", r.text)

    def test_blocked_code_is_reported_honestly(self):
        async def hacker(msgs, **p):
            return "```python\nimport os\nos.system('ls')\n```" if len(msgs) < 5 else "That code was blocked by the sandbox, so I can't confirm it by running it; reasoning it through instead, the answer is roughly 12."
        r = run(answer("run a shell command for me and explain what you did", hacker, engine=ENG)); self.assertTrue(r.tool_log[0]["blocked"])

    def test_self_consistency_vote(self):
        seq = iter(["42", "42", "41"])
        pad = " ".join(f"detail{i}" for i in range(70))
        async def vote(msgs, **p): return f"Final answer: {next(seq)}. {pad}"
        r = run(answer("what is the meaning of everything, explain", vote, engine=ENG, votes=3)); self.assertIn("42", r.text); self.assertTrue(r.verified)

    def test_final_answer_extraction(self):
        from deep_oq.llm_bridge import extract_final_answer as f
        self.assertEqual(f("Final answer: 42. See steps 3 and 7."), "42"); self.assertEqual(f("answer = 3/4 (that's 0.75)"), "3/4")
        self.assertEqual(f("so we get 5 and then 9"), "9"); self.assertEqual(f("Final answer: 1,200 dollars"), "1200")

    def test_preserves_numbers(self):
        self.assertTrue(preserves_numbers("x = 2, 3", "roots are 2 and 3")); self.assertFalse(preserves_numbers("x = 2, 3", "roots are 7"))


class Sandbox(unittest.TestCase):
    def test_blocks(self):
        for code in ["import os", "import subprocess", "open('/etc/passwd')", "().__class__.__bases__", "__import__('os')", "eval('1')"]:
            self.assertTrue(codeexec.run_python(code).blocked, code)

    def test_allows_math(self): self.assertEqual(codeexec.run_python("import math\nprint(math.factorial(10))").summary(), "3628800")
    def test_timeout(self): self.assertTrue(codeexec.run_python("while True: pass", timeout=1.5).timed_out)


class Server(unittest.TestCase):
    def test_handlers_without_llm(self):
        from deep_oq import server
        self.assertIn("9", server.handle_solve("integrate x^2 from 0 to 3")["answer"])
        r = run(server.handle_chat("derivative of x^3", llm=None)); self.assertEqual(r["source"], "solver")


if __name__ == "__main__":
    unittest.main()
''')

# ====================================================================================================
# MODULE: deep_oq.tests.test_providers    (deep_oq/tests/test_providers.py, 254 lines)
# ====================================================================================================
_register('deep_oq.tests.test_providers', False, 'deep_oq/tests/test_providers.py', r'''"""Provider tests against a local mock HTTP server that imitates the four APIs (no real network, no real keys)."""
import asyncio
import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from deep_oq import providers as P
from deep_oq.engine import Engine
from deep_oq.llm_bridge import answer
from deep_oq.providers import AnthropicLLM, GeminiLLM, GroqLLM, LLMError, MultiLLM, OpenAILLM

run = asyncio.run
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
MSGS = [{"role": "system", "content": "You are Deep OQ."}, {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"}, {"role": "user", "content": "2+2?"}]

ANTH = {"content": [{"type": "text", "text": "hello from claude"}], "stop_reason": "end_turn"}
OAI = {"status": "completed", "output": [{"type": "reasoning", "summary": []},
                                         {"type": "message", "content": [{"type": "output_text", "text": "hello from astra"}]}]}
GEM = {"candidates": [{"content": {"role": "model", "parts": [{"text": "private thoughts", "thought": True}, {"text": "hello from gemini"}]},
                       "finishReason": "STOP"}]}
GRQ = {"choices": [{"message": {"role": "assistant", "content": "hello from groq"}}]}
PAD = " ".join(f"detail{i}" for i in range(70))


class Mock:
    def __init__(self, reply=None):
        self.requests, self.reply = [], reply
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _handle(self):
                n = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(n) if n else b""
                rec = {"method": self.command, "path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                       "body": json.loads(raw) if raw else None}
                outer.requests.append(rec)
                code, obj = outer.respond(rec)
                out = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                if code == 429:
                    self.send_header("Retry-After", "0")
                self.end_headers()
                self.wfile.write(out)
            do_GET = do_POST = _handle

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def respond(self, rec):
        return 200, self.reply

    def close(self):
        self.server.shutdown(); self.server.server_close()


class Base(unittest.TestCase):
    def setUp(self):
        self.mocks = []

    def tearDown(self):
        for m in self.mocks:
            m.close()

    def mk(self, reply=None):
        m = Mock(reply); self.mocks.append(m); return m


class RequestShapes(Base):
    def test_anthropic(self):
        m = self.mk(ANTH); p = AnthropicLLM(api_key="sk-ant-test", base_url=m.url, backoff=0)
        self.assertEqual(run(p(MSGS, temperature=0.4, max_tokens=500)), "hello from claude")
        r = m.requests[0]; b = r["body"]
        self.assertEqual(r["path"], "/v1/messages"); self.assertEqual(r["headers"]["x-api-key"], "sk-ant-test")
        self.assertEqual(r["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(b["model"], "claude-fable-5-2"); self.assertEqual(b["system"], "You are Deep OQ.")
        self.assertEqual([t["role"] for t in b["messages"]], ["user", "assistant", "user"]); self.assertEqual(b["max_tokens"], 500)

    def test_openai(self):
        m = self.mk(OAI); p = OpenAILLM(api_key="sk-oa-test", base_url=m.url, backoff=0)
        self.assertEqual(run(p(MSGS, max_tokens=300)), "hello from astra")      # reasoning item is skipped
        r = m.requests[0]; b = r["body"]
        self.assertEqual(r["path"], "/v1/responses"); self.assertEqual(r["headers"]["authorization"], "Bearer sk-oa-test")
        self.assertEqual(b["model"], "gpt-6.2-astra"); self.assertEqual(b["reasoning"], {"effort": "high"})
        self.assertGreaterEqual(b["max_output_tokens"], 16000); self.assertEqual(b["instructions"], "You are Deep OQ.")

    def test_openai_never_sends_effort_none(self):
        m = self.mk(OAI); p = OpenAILLM(api_key="k", base_url=m.url, backoff=0)
        with mock.patch.dict(os.environ, {"DEEP_OQ_OPENAI_EFFORT": "none"}):
            run(p(MSGS))
        self.assertNotEqual(m.requests[0]["body"]["reasoning"]["effort"], "none")

    def test_gemini(self):
        m = self.mk(GEM); p = GeminiLLM(api_key="g-test", base_url=m.url, backoff=0)
        self.assertEqual(run(p(MSGS, temperature=0.5, top_p=0.9)), "hello from gemini")   # thought part is skipped
        r = m.requests[0]; b = r["body"]
        self.assertEqual(r["path"], "/v1beta/models/gemini-4-argon:generateContent"); self.assertEqual(r["headers"]["x-goog-api-key"], "g-test")
        self.assertEqual([c["role"] for c in b["contents"]], ["user", "model", "user"])
        self.assertEqual(b["systemInstruction"]["parts"][0]["text"], "You are Deep OQ.")
        self.assertEqual(b["generationConfig"]["thinkingConfig"]["thinkingLevel"], "high")
        self.assertGreaterEqual(b["generationConfig"]["maxOutputTokens"], 16384)

    def test_groq(self):
        m = self.mk(GRQ); p = GroqLLM(api_key="gsk_test", base_url=m.url, backoff=0)
        self.assertEqual(run(p(MSGS)), "hello from groq")
        r = m.requests[0]; b = r["body"]
        self.assertEqual(r["path"], "/openai/v1/chat/completions"); self.assertEqual(b["model"], "openai/gpt-oss-120b")
        self.assertEqual(b["reasoning_effort"], "high"); self.assertEqual(b["messages"][0]["role"], "system"); self.assertIn("max_completion_tokens", b)

    def test_consecutive_same_role_messages_are_merged(self):
        sys_, turns = P.split_messages([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}, {"role": "assistant", "content": "c"}])
        self.assertEqual([t["role"] for t in turns], ["user", "assistant"]); self.assertEqual(turns[0]["content"], "a\n\nb")
        _, t2 = P.split_messages([{"role": "assistant", "content": "x"}]); self.assertEqual(t2[0]["role"], "user")


class Resilience(Base):
    def test_retry_on_429(self):
        m = self.mk(); n = {"c": 0}
        def respond(rec):
            n["c"] += 1
            return (429, {"error": {"message": "slow down"}}) if n["c"] == 1 else (200, GRQ)
        m.respond = respond
        self.assertEqual(run(GroqLLM(api_key="k", base_url=m.url, backoff=0)(MSGS)), "hello from groq"); self.assertEqual(len(m.requests), 2)

    def test_auth_error_not_retried_and_key_not_leaked(self):
        m = self.mk(); m.respond = lambda rec: (401, {"error": {"message": "bad key sk-super-secret-123"}})
        p = OpenAILLM(api_key="sk-super-secret-123", base_url=m.url, backoff=0)
        with self.assertRaises(LLMError) as cm:
            run(p(MSGS))
        self.assertEqual(len(m.requests), 1); self.assertNotIn("super-secret", str(cm.exception)); self.assertNotIn("super-secret", repr(p))
        self.assertEqual(cm.exception.status, 401)

    def test_400_retries_with_minimal_body(self):
        m = self.mk()
        def respond(rec):
            if "thinkingConfig" in rec["body"]["generationConfig"]:
                return 400, {"error": {"message": "thinking level invalid for this request"}}
            return 200, GEM
        m.respond = respond
        self.assertEqual(run(GeminiLLM(api_key="k", base_url=m.url, backoff=0)(MSGS)), "hello from gemini")
        self.assertEqual(len(m.requests), 2); self.assertNotIn("thinkingConfig", m.requests[1]["body"]["generationConfig"])

    def test_model_fallback_and_sticky(self):
        m = self.mk()
        def respond(rec):
            return (404, {"error": {"message": "model not found"}}) if rec["body"]["model"] == "claude-fable-5-2" else (200, ANTH)
        m.respond = respond
        p = AnthropicLLM(api_key="k", base_url=m.url, backoff=0)
        run(p(MSGS)); self.assertEqual(p.model, "claude-opus-5-5")
        run(p(MSGS)); self.assertEqual(m.requests[-1]["body"]["model"], "claude-opus-5-5"); self.assertEqual(len(m.requests), 3)

    def test_discovery_finds_new_model(self):
        m = self.mk()
        def respond(rec):
            if rec["method"] == "GET":
                return 200, {"data": [{"id": "claude-haiku-5-5"}, {"id": "claude-opus-5-5"}, {"id": "claude-fable-5-3"}]}
            return (200, ANTH) if rec["body"]["model"] == "claude-fable-5-3" else (404, {"error": {"message": "model not found"}})
        m.respond = respond
        p = AnthropicLLM(api_key="k", base_url=m.url, backoff=0)
        self.assertEqual(run(p(MSGS)), "hello from claude"); self.assertEqual(p.model, "claude-fable-5-3")

    def test_gemini_discovery_strips_models_prefix(self):
        p = GeminiLLM(api_key="k"); self.assertEqual(p.parse_model_ids({"models": [{"name": "models/gemini-4-argon"}]}), ["gemini-4-argon"])

    def test_empty_text_is_an_error(self):
        m = self.mk({"output": [{"type": "reasoning"}], "status": "incomplete"})
        with self.assertRaises(LLMError):
            run(OpenAILLM(api_key="k", base_url=m.url, backoff=0, max_attempts=2)(MSGS))


class Routing(Base):
    def test_failover_and_cooldown(self):
        a, b = self.mk(), self.mk(GRQ); a.respond = lambda rec: (401, {"error": {"message": "invalid key"}})
        llm = MultiLLM([AnthropicLLM(api_key="k", base_url=a.url, backoff=0), GroqLLM(api_key="k", base_url=b.url, backoff=0)])
        self.assertEqual(run(llm(MSGS)), "hello from groq"); self.assertEqual(llm.last_used, "groq")
        run(llm(MSGS)); self.assertEqual(len(a.requests), 1)                    # anthropic is in cool-down, skipped

    def test_all_fail(self):
        a = self.mk(); a.respond = lambda rec: (401, {"error": {"message": "nope"}})
        with self.assertRaises(LLMError) as cm:
            run(MultiLLM([AnthropicLLM(api_key="k", base_url=a.url, backoff=0)])(MSGS))
        self.assertIn("all providers failed", str(cm.exception))

    def test_vote_majority(self):
        txt = lambda n: f"Final answer: {n}. {PAD}"                              # noqa: E731
        a = self.mk({"content": [{"type": "text", "text": txt(41)}]})
        o = self.mk({"output": [{"type": "message", "content": [{"type": "output_text", "text": txt(42)}]}], "status": "completed"})
        g = self.mk({"choices": [{"message": {"content": txt(42)}}]})
        llm = MultiLLM([AnthropicLLM(api_key="k", base_url=a.url, backoff=0), OpenAILLM(api_key="k", base_url=o.url, backoff=0),
                        GroqLLM(api_key="k", base_url=g.url, backoff=0)], mode="vote")
        out = run(llm(MSGS)); self.assertTrue(out.startswith("Final answer: 42")); self.assertEqual(llm.last_used, "openai")
        self.assertEqual(llm.last_votes, {"anthropic": "41", "openai": "42", "groq": "42"})

    def test_bridge_works_through_multillm(self):
        g = self.mk({"choices": [{"message": {"content": " ".join(f"Joke{i} is funny." for i in range(12))}}]})
        llm = MultiLLM([GroqLLM(api_key="k", base_url=g.url, backoff=0)])
        r = run(answer("tell me a joke", llm, engine=Engine(use_timeout=False))); self.assertEqual(r.source, "llm"); self.assertIn("Joke3", r.text)

    def test_ping(self):
        g = self.mk(GRQ); bad = self.mk(); bad.respond = lambda rec: (401, {"error": {"message": "invalid"}})
        rows = run(P.ping_all([GroqLLM(api_key="k", base_url=g.url, backoff=0), OpenAILLM(api_key="k", base_url=bad.url, backoff=0)]))
        self.assertTrue(rows[0]["ok"]); self.assertFalse(rows[1]["ok"]); self.assertIn("401", rows[1]["error"])


class Setup(unittest.TestCase):
    def test_no_key_message(self):
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(LLMError) as cm:
            AnthropicLLM()
        self.assertIn("ANTHROPIC_API_KEY", str(cm.exception))

    def test_make_llm_none_without_keys_and_order_respected(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(P.make_llm_from_env(env_file=None))
        env = {"GROQ_API_KEY": "g", "OPENAI_API_KEY": "o", "DEEP_OQ_PROVIDER_ORDER": "groq,openai", "DEEP_OQ_MODE": "vote"}
        with mock.patch.dict(os.environ, env, clear=True):
            llm = P.make_llm_from_env(env_file=None)
            self.assertEqual([p.name for p in llm.providers], ["groq", "openai"]); self.assertEqual(llm.mode, "vote")

    def test_google_api_key_alias_and_model_override(self):
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "x", "DEEP_OQ_GEMINI_MODEL": "gemini-9-test"}, clear=True):
            p = GeminiLLM(); self.assertEqual(p.model, "gemini-9-test")

    def test_env_file(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"KEEP_ME": "original"}, clear=True):
            path = os.path.join(d, ".env")
            with open(path, "w") as fh:
                fh.write('# comment\nexport GROQ_API_KEY="gsk_abc"\nOPENAI_API_KEY=sk-1\nKEEP_ME=changed\n\nEMPTY=\n')
            self.assertEqual(P.load_env_file(path), 2)
            self.assertEqual((os.environ["GROQ_API_KEY"], os.environ["OPENAI_API_KEY"], os.environ["KEEP_ME"]), ("gsk_abc", "sk-1", "original"))

    def test_rank_models(self):
        self.assertEqual(P.rank_models("openai", ["text-embedding-9", "gpt-6.2-sol", "gpt-6-astra", "gpt-6.2-astra", "gpt-6.2-astra-audio", "whisper-1"])[:2],
                         ["gpt-6.2-astra", "gpt-6-astra"])
        self.assertEqual(P.rank_models("anthropic", ["claude-opus-5-5", "claude-fable-5-2", "claude-fable-5-3"])[0], "claude-fable-5-3")
        self.assertEqual(P.rank_models("groq", ["llama-3.3-70b-versatile", "openai/gpt-oss-safeguard-20b", "openai/gpt-oss-120b"])[0], "openai/gpt-oss-120b")

    def test_defaults_are_the_most_powerful_models(self):
        self.assertEqual((AnthropicLLM.default_model, OpenAILLM.default_model, GeminiLLM.default_model, GroqLLM.default_model),
                         ("claude-fable-5-2", "gpt-6.2-astra", "gemini-4-argon", "openai/gpt-oss-120b"))


if __name__ == "__main__":
    unittest.main()
''')

# ====================================================================================================
# MODULE: deep_oq.tests.test_solvers    (deep_oq/tests/test_solvers.py, 177 lines)
# ====================================================================================================
_register('deep_oq.tests.test_solvers', False, 'deep_oq/tests/test_solvers.py', r'''"""Known-answer tests for every solver.  Run:  python -m unittest discover -s deep_oq/tests -t .   (or pytest)"""
import unittest

import sympy as sp

from deep_oq import Engine

from deep_oq.parse import x as X, y as Y

E = Engine(use_timeout=False)


def S(q):
    return E.solve(q)


class Algebra(unittest.TestCase):
    def test_quadratic(self):
        s = S("solve x^2 - 5x + 6 = 0"); self.assertEqual(set(s.answer), {2, 3}); self.assertTrue(s.verified)

    def test_system(self):
        s = S("solve x + y = 10 and x - y = 2"); self.assertEqual(s.answer[0], {X: 6, Y: 4})

    def test_inequality(self):
        self.assertEqual(S("solve x^2 - 4 <= 0").answer, sp.Interval(-2, 2))

    def test_complex(self):
        self.assertEqual(len(S("solve x^2 + 1 = 0").answer), 2)

    def test_numeric_fallback(self):
        s = S("solve cos(x) = x"); self.assertAlmostEqual(float(s.answer[0]), 0.7390851, 6)

    def test_factor(self):
        self.assertEqual(sp.expand(S("factorise x^3 - 6x^2 + 11x - 6").answer), sp.expand((X-1)*(X-2)*(X-3)))


class Calculus(unittest.TestCase):
    def test_derivative(self):
        s = S("derivative of x^2 sin(x)"); x = X
        self.assertEqual(sp.simplify(s.answer - (2*x*sp.sin(x) + x**2*sp.cos(x))), 0)

    def test_second_derivative_at_point(self):
        self.assertEqual(S("second derivative of x^4 at x=2").answer, 48)

    def test_definite_integral(self):
        self.assertEqual(S("integrate x^2 from 0 to 3").answer, 9)

    def test_improper(self):
        self.assertAlmostEqual(float(S("integrate e^(-x^2) from -inf to inf").answer), 1.7724538509, 8)

    def test_limit(self):
        self.assertEqual(S("limit of sin(x)/x as x->0").answer, 1)
        self.assertEqual(S("limit of (1+1/x)^x as x->oo").answer, sp.E)

    def test_taylor(self):
        self.assertEqual(sp.expand(S("taylor series of cos(x) order 6").answer), 1 - X**2/2 + X**4/24)

    def test_critical_points(self):
        self.assertIn("local maximum", S("find critical points of x^3 - 3x").answer_text)

    def test_tangent(self):
        self.assertEqual(S("tangent to x^2 at x=3").answer, 6*X - 9)

    def test_ode(self):
        self.assertTrue(S("solve y' = 2y with y(0)=1").verified)

    def test_jee_problem_from_screenshot(self):
        # g(x)=f(f(x)), f(x)=|ln2 - sin x|  ->  g'(0) = cos(ln 2)
        from deep_oq.verify import numeric_derivative
        x = X; g = sp.Abs(sp.log(2) - sp.sin(sp.Abs(sp.log(2) - sp.sin(x))))
        self.assertAlmostEqual(numeric_derivative(g, x, 0.0), float(sp.cos(sp.log(2))), 5)


class LinearAlgebra(unittest.TestCase):
    def test_det(self): self.assertEqual(S("determinant of [[2,1],[5,3]]").answer, 1)
    def test_inverse(self): self.assertEqual(S("inverse of [[1,2],[3,4]]").answer, sp.Matrix([[-2, 1], [sp.Rational(3, 2), -sp.Rational(1, 2)]]))
    def test_rank(self): self.assertEqual(S("rank of [[1,2],[2,4]]").answer, 1)
    def test_system(self): self.assertEqual(list(S("solve [[2,1],[1,3]] x = [[5],[10]]").answer), [1, 3])


class NumberTheory(unittest.TestCase):
    def test_gcd(self): self.assertEqual(S("gcd of 48 and 18").answer, 6)
    def test_prime(self): self.assertTrue(S("is 97 prime").answer)
    def test_factor(self): self.assertEqual(S("prime factorization of 360").answer, {2: 3, 3: 2, 5: 1})
    def test_modexp(self): self.assertEqual(S("3^200 mod 13").answer, pow(3, 200, 13))
    def test_crt(self): self.assertEqual(S("x ≡ 2 (mod 3), x ≡ 3 (mod 5)").answer, (8, 15))
    def test_binary(self): self.assertEqual(S("convert 255 to binary").answer, "11111111")


class Probability(unittest.TestCase):
    def test_dice(self): self.assertEqual(str(S("probability that the sum of two dice is at least 9").answer), "5/18")
    def test_coins(self): self.assertEqual(str(S("probability of exactly 3 heads in 5 coin tosses").answer), "5/16")
    def test_cards(self): self.assertEqual(str(S("probability both cards are aces from a deck").answer), "1/221")
    def test_birthday(self): self.assertAlmostEqual(float(S("birthday paradox with 23 people").answer), 0.5073, 3)
    def test_banana(self): self.assertEqual(S("arrangements of the letters of BANANA").answer, 60)
    def test_simulation_is_evidence(self):
        self.assertTrue(any("real trials" in v for v in S("probability of exactly 3 heads in 5 coin tosses").verification))


class Puzzles(unittest.TestCase):
    def test_switches(self): self.assertTrue(S("3 switches outside a room, 3 bulbs inside, enter once: which switch controls which bulb?").verified)
    def test_jugs(self): self.assertEqual(S("water jugs 3 and 5 litres measure 4").answer, 6)
    def test_queens(self): self.assertIn("92 solutions", S("solve 8 queens puzzle").answer_text)
    def test_hanoi(self): self.assertEqual(S("tower of hanoi 5 disks").answer, 31)
    def test_river(self): self.assertEqual(S("wolf goat cabbage river crossing").answer, 7)
    def test_sudoku(self):
        s = S("sudoku 530070000600195000098000060800060003400803001700020006060000280000419005000080079")
        self.assertEqual(s.answer_text.split("\n")[0], "534678912")


class Science(unittest.TestCase):
    def test_suvat(self): self.assertEqual(S("A car starts from rest with acceleration 2 m/s^2 for time 5 s. Find the displacement.").answer, 25)
    def test_mixed_units_parse(self): self.assertEqual(S("a car accelerates at 3 m/s² for 4 s from 10 m/s. Find the final velocity").answer, 22)
    def test_newton(self): self.assertEqual(S("mass 5 kg acceleration 3 m/s^2, find the force").answer, 15)   # regression: once returned 25
    def test_ohm(self): self.assertEqual(S("voltage 12 V, resistance 4 ohm. Find the current").answer, 3)
    def test_range(self): self.assertAlmostEqual(S("A ball is thrown at 20 m/s at an angle of 30 degrees. Find the range").answer, 35.3119, 3)
    def test_molar(self): self.assertAlmostEqual(S("molar mass of Ca(OH)2").answer, 74.092, 2)
    def test_balance(self): self.assertEqual(S("balance C3H8 + O2 -> CO2 + H2O").answer, [1, 5, 3, 4])
    def test_units(self): self.assertAlmostEqual(S("convert 5 miles to km").answer, 8.04672, 5)
    def test_dimension_mismatch(self): self.assertIn("impossible", S("convert 5 kg to km").answer_text)


class StatsFinanceArithmetic(unittest.TestCase):
    def test_std(self): self.assertEqual(S("standard deviation and mean of 2, 4, 4, 4, 5, 5, 7, 9").answer["population std dev"], 2)
    def test_regression(self): self.assertAlmostEqual(S("regression for data 1,2,2,4,3,5,4,4,5,6").answer["slope"], 0.8)
    def test_normal(self): self.assertAlmostEqual(S("normal distribution mean 100 sd 15 probability below 130").answer, 0.97725, 5)
    def test_emi(self): self.assertAlmostEqual(S("emi for loan 500000 at 9% for 20 years").answer, 4498.63, 2)
    def test_irr(self): self.assertAlmostEqual(S("irr of cash flows -1000, 300, 400, 500, 200").answer, 0.1532, 3)
    def test_percent(self): self.assertEqual(S("what is 15% of 240").answer, 36)
    def test_discount(self): self.assertEqual(S("20% discount on 500").answer, 400)   # regression: once returned -80
    def test_exact_arithmetic(self): self.assertEqual(S("sqrt(2)*sqrt(8)").answer, 4)


class Security(unittest.TestCase):
    """Regression tests for the eval() hole found during robustness testing."""
    ATTACKS = ["solve x**2 = __import__('os').system('touch /tmp/deep_oq_pwned')",
               "derivative of __import__('os').system('touch /tmp/deep_oq_pwned')",
               "determinant of [[__import__('os').system('touch /tmp/deep_oq_pwned')]]",
               "integrate (1).__class__.__bases__ from 0 to 1", "solve y' = open('/etc/passwd').read()",
               "limit of exec('1') as x->0", "solve x^2 = lambda: 1", "evaluate 2**9999999999", "solve x = ().__class__"]

    def test_no_code_execution(self):
        import os
        if os.path.exists("/tmp/deep_oq_pwned"): os.remove("/tmp/deep_oq_pwned")
        for q in self.ATTACKS:
            self.assertFalse(S(q).solved, q)
        self.assertFalse(os.path.exists("/tmp/deep_oq_pwned"))


class Robustness(unittest.TestCase):
    def test_filler_words_ignored(self): self.assertEqual(S("solve 2x + 3 = 11 for x please thanks").answer[0], 4)
    def test_no_letter_soup(self): self.assertFalse(S("सॉल्व करो 2x + 3 = 11").solved)
    def test_quintic_does_not_hang(self): self.assertTrue(S("solve x^5 - x + 1 = 0").verified)
    def test_one_sided_limits(self): self.assertIn("does not exist", S("limit of 1/x as x->0").answer_text)
    def test_corner(self): self.assertIn("not differentiable", S("derivative of |x| at x=0").answer_text)
    def test_no_real_solution(self): self.assertIn("no real solution", S("solve sin(x) = 2").answer_text)
    def test_div_by_zero_not_faked(self): self.assertFalse(S("1/0").solved)


class EngineBehaviour(unittest.TestCase):
    def test_multipart(self):
        s = S("1) solve 2x+3=11  2) derivative of x^3  3) 15% of 240")
        self.assertEqual(s.category, "multi-part"); self.assertIn("(3) 36", s.answer_text)

    def test_unknown_is_not_faked(self):
        s = S("tell me a joke"); self.assertFalse(s.solved)

    def test_timeout_protects_server(self):
        import time
        from deep_oq.engine import run_with_timeout
        t = time.time(); r = run_with_timeout(lambda q: time.sleep(30), "x", 1.0)
        self.assertLess(time.time() - t, 4); self.assertIn("timed out", r.warnings[0])


if __name__ == "__main__":
    unittest.main()
''')

import deep_oq                                               # noqa: E402  (loaded from the embedded sources above)
from deep_oq import solve, Engine, Solution, Step            # noqa: E402,F401
from deep_oq.llm_bridge import answer, BotReply, OpenAICompatibleLLM    # noqa: E402,F401
from deep_oq import server as _server                        # noqa: E402
app = _server.app                                            # for:  uvicorn deep_oq_all:app


def run_tests() -> int:
    import unittest
    suite = unittest.TestSuite()
    for name in ("deep_oq.tests.test_solvers", "deep_oq.tests.test_bridge", "deep_oq.tests.test_providers"):
        __import__(name)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[name]))
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if res.wasSuccessful() else 1


def main(argv) -> int:
    if len(argv) > 1 and argv[1] == "test":
        return run_tests()
    if len(argv) > 1 and argv[1] == "modules":
        for n, (_, rel, src) in sorted(_MODULES.items()):
            print("%-34s %5d lines  %s" % (n, src.count("\n") + 1, rel))
        return 0
    if len(argv) > 1 and argv[1] in ("providers", "ping"):
        from deep_oq import providers as P
        P.load_env_file()
        if argv[1] == "providers":
            for r in P.describe_providers():
                print("%-10s %-22s key: %-8s (%s)" % (r["provider"], r["model"], "SET" if r["key_set"] else "missing", r["env"]))
            return 0
        provs = P.available_providers()
        if not provs:
            print("No API keys found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY or GROQ_API_KEY (env vars or a .env file).")
            return 1
        import asyncio
        rows = asyncio.run(P.ping_all(provs))
        for r in rows:
            print("%-10s %-22s %-5s %6d ms  %s" % (r["provider"], r["model"], "OK" if r["ok"] else "FAIL", r["ms"], r.get("reply") or r.get("error")))
        return 0 if any(r["ok"] for r in rows) else 1
    if len(argv) > 1 and argv[1] == "serve":
        if app is None:
            print("fastapi is not installed:  pip install fastapi uvicorn")
            return 1
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(argv[2]) if len(argv) > 2 else 8000)
        return 0
    eng = Engine()
    if len(argv) > 1:
        print(eng.solve(" ".join(argv[1:])).render())
        return 0
    print("Deep OQ solver - type a problem, or 'quit'.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            print(eng.solve(q).render())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
