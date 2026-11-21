#!/usr/bin/env python3
"""
Deep OQ v2  -  a separate, clean backend for your chatbot.

What it fixes (from your screenshots)
-------------------------------------
* "Problem: Let $f: Problem: Let $f: ..."  -> a repetition loop. Fixed by (a) using a real
  frontier model through the Anthropic API instead of a weak generator, (b) a live loop
  detector that cuts any runaway repetition, and (c) a post-processing de-duplicator.
* "Answer ... Answer ... Step-by-Step Execution 1. Answer"  -> a rigid template that made the
  model echo its own headings. There is no forced template any more.
* Incomplete answers that just stop  -> automatic continuation when the model hits its
  output limit, plus a much larger output budget.
* One-line replies ("I am Deep OQ, an advanced AI")  -> a conversation-first system prompt:
  warm, human tone, developed answers, and a natural follow-up question.
* Confident-but-unverified maths  -> real tools (exact calculator / symbolic maths) and a
  separate self-check pass that can append a correction.
* Claimed-but-not-run simulations  -> the model is told to only claim what a tool actually ran.

Run
---
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY="sk-ant-..."          # never hard-code this
    python deep_oq_v2.py serve                      # http://localhost:8000
    python deep_oq_v2.py chat                       # terminal chat

Environment variables (all optional except the key)
---------------------------------------------------
    ANTHROPIC_API_KEY            your key
    DATABASE_URL                 PostgreSQL connection string (postgresql://user:pass@host:5432/dbname).
                                 If set, chats / memory / usage are stored there (survives redeploys, shared by
                                 all workers). If empty, a local SQLite file is used.
    DEEP_OQ_DB_POOL              default 20  (PostgreSQL connections per worker)
    DEEP_OQ_MODEL_FAST           default claude-sonnet-5-5   (everyday chat)
    DEEP_OQ_MODEL_SMART          default claude-opus-5-5     (hard maths / code / logic)
    DEEP_OQ_MODEL_MAX            default claude-fable-5-2    (most powerful; used when mode="max")
    OPENAI_API_KEY               optional: enables the OpenAI backup if Anthropic fails or is unavailable
    DEEP_OQ_OPENAI_MODEL         default gpt-6.2-astra       (OpenAI's strongest model, backup #1)
    DEEP_OQ_OPENAI_MODEL_ALT     default gpt-6.2-sol         (backup #2 if the first is unavailable)
    DEEP_OQ_OPENAI_REASONING     default high                (low | medium | high | xhigh)
    WEB_CONCURRENCY              default 1  (number of worker processes; use 2-4 on a paid server)
    DEEP_OQ_MAX_CONCURRENT       default 100 (answers generated at the same time per worker; others queue)
    DEEP_OQ_MAX_CONCURRENT_MAXMODE default 10 (simultaneous Fable "max" answers per worker; extras use Opus)
    DEEP_OQ_QUEUE_TIMEOUT        default 60  (seconds a user waits in line before "busy, try again")
    DEEP_OQ_DAILY_MAX_LIMIT      default 20  (max-mode messages per user per day, 0 = unlimited)
    DEEP_OQ_MAX_TOKENS           default 64000
    DEEP_OQ_THINKING_BUDGET      default 32000 (extended thinking on hard problems, 0 = off)
    DEEP_OQ_VERIFY               default 1     (self-check pass on hard problems)
    DEEP_OQ_DB                   default deep_oq.sqlite3
    DEEP_OQ_API_TOKEN            if set, clients must send  Authorization: Bearer <token>
    DEEP_OQ_RATE_LIMIT_PER_MIN   default 20 (messages per minute per user/IP, 0 = no limit)
    DEEP_OQ_CORS_ORIGINS         default *
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------------------
# 1. CONFIG
# --------------------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    model_fast: str = field(default_factory=lambda: os.environ.get("DEEP_OQ_MODEL_FAST", "claude-sonnet-5-5"))
    model_smart: str = field(default_factory=lambda: os.environ.get("DEEP_OQ_MODEL_SMART", "claude-opus-5-5"))
    model_max: str = field(default_factory=lambda: os.environ.get("DEEP_OQ_MODEL_MAX", "claude-fable-5-2"))
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.environ.get("DEEP_OQ_OPENAI_MODEL", "gpt-6.2-astra"))
    openai_model_alt: str = field(default_factory=lambda: os.environ.get("DEEP_OQ_OPENAI_MODEL_ALT", "gpt-6.2-sol"))
    openai_reasoning: str = field(default_factory=lambda: os.environ.get("DEEP_OQ_OPENAI_REASONING", "high"))
    max_tokens: int = field(default_factory=lambda: _env_int("DEEP_OQ_MAX_TOKENS", 64000))
    thinking_budget: int = field(default_factory=lambda: _env_int("DEEP_OQ_THINKING_BUDGET", 32000))
    verify: bool = field(default_factory=lambda: _env_int("DEEP_OQ_VERIFY", 1) == 1)
    temperature: float = 0.7
    db_path: str = field(default_factory=lambda: os.environ.get("DEEP_OQ_DB", "deep_oq.sqlite3"))
    database_url: str = field(default_factory=lambda: os.environ.get("DEEP_OQ_DATABASE_URL")
                              or os.environ.get("DATABASE_URL", ""))
    db_pool_size: int = field(default_factory=lambda: _env_int("DEEP_OQ_DB_POOL", 20))
    api_token: str = field(default_factory=lambda: os.environ.get("DEEP_OQ_API_TOKEN", ""))
    rate_limit_per_min: int = field(default_factory=lambda: _env_int("DEEP_OQ_RATE_LIMIT_PER_MIN", 20))
    cors_origins: str = field(default_factory=lambda: os.environ.get("DEEP_OQ_CORS_ORIGINS", "*"))
    max_tool_rounds: int = 12
    max_continuations: int = 8
    history_char_budget: int = 200_000  # roughly 50k tokens of remembered conversation (faster, cheaper)
    max_concurrent: int = field(default_factory=lambda: _env_int("DEEP_OQ_MAX_CONCURRENT", 100))
    max_concurrent_max: int = field(default_factory=lambda: _env_int("DEEP_OQ_MAX_CONCURRENT_MAXMODE", 10))
    queue_timeout: int = field(default_factory=lambda: _env_int("DEEP_OQ_QUEUE_TIMEOUT", 60))
    daily_max_limit: int = field(default_factory=lambda: _env_int("DEEP_OQ_DAILY_MAX_LIMIT", 20))


CFG = Config()

# --------------------------------------------------------------------------------------
# 2. SYSTEM PROMPT  (this is what makes it talk like a person and answer in depth)
# --------------------------------------------------------------------------------------

BASE_SYSTEM_PROMPT = """\
You are Deep OQ, a warm, sharp and genuinely helpful AI assistant. You are built on Anthropic's \
Claude. If someone sincerely asks what powers you, say so plainly. You are an AI, and you never \
pretend to be a human, but you talk the way a thoughtful, knowledgeable friend talks.

HOW YOU TALK
- Be conversational and natural. Use plain, flowing language, contractions, and a bit of personality. \
Never open with filler like "Certainly!" or "Great question!".
- Engage with the person, not just the query. Pick up on what they seem to care about, react to it, \
and keep the conversation moving.
- Do not give one-line answers to real questions. A real question deserves a developed reply: the \
direct answer first, then the reasoning, an example or two, the caveats that matter, and what to \
do next. Most answers should be several paragraphs. Match the depth to the question: a greeting \
gets a warm, short reply, a hard problem gets a full worked answer.
- Finish most replies with one natural follow-up question or a concrete offer, so the conversation \
continues. Only one, and only when it fits.
- Use Markdown sparingly. Prose by default. Use lists, tables or headings only when the content is \
genuinely list-like or step-by-step. Use LaTeX with $...$ for inline maths and $$...$$ for display maths.
- Never repeat yourself. Never echo a heading or sentence you already wrote. Never leave an answer \
unfinished. If the answer is long, structure it, but always end it properly.

ABOUT YOURSELF
When someone asks who you are or what you can do, give a real, friendly introduction of several \
sentences, never a single line: you are Deep OQ, built on Anthropic's Claude; you can chat about almost \
any topic, explain ideas in depth, solve maths and logic problems step by step (using a calculator and \
symbolic maths that you really run), write and debug code, and remember things people tell you across \
conversations. Only mention abilities you actually have in this conversation. End by asking what they \
would like to explore.

HOW YOU SOLVE PROBLEMS
1. Understand first. Restate the problem to yourself, spot hidden assumptions, and notice if the \
premise is wrong or the question is ambiguous. If it is ambiguous in a way that changes the answer, \
state the interpretation you chose (or ask, if you truly cannot proceed).
2. Plan, then solve step by step. Show the key steps so the person can follow and check them.
3. Verify before you answer. Re-derive, substitute back, test edge cases, and sanity-check units and \
magnitudes. For arithmetic and algebra, use the calculator / symbolic tools instead of doing long \
calculations in your head.
4. Be honest about confidence. Say clearly when you are sure, when you are estimating, and when you \
could not verify something. Never invent facts, sources, numbers or results.
5. Only claim to have run a simulation, code or a search if you actually ran it with a tool in this \
conversation. Otherwise say it was a reasoning-based check.
6. Give the final answer clearly at the end of a worked solution (for a multiple-choice question, \
name the correct option and say why each other option fails).

PUZZLES AND TRICKY QUESTIONS
Look for the twist. Test the standard answer against the exact wording. If there is a cleverer or \
more general solution, mention it. If the puzzle has a false premise, point that out kindly.

CODE
Write complete, runnable code, not fragments. Explain the idea briefly, mention edge cases and \
complexity, and show how to run or test it.

SAFETY AND ETHICS
You are free to discuss almost anything openly and in depth. You still decline to help with things \
that would seriously harm people (for example weapons capable of mass casualties, malware, or \
sexual content involving minors), and you do it briefly and kindly, offering a safe alternative \
when there is one.
"""


def build_system_prompt(facts: List[str], now: Optional[str] = None) -> str:
    parts = [BASE_SYSTEM_PROMPT]
    parts.append(f"\nCurrent date/time (server): {now or time.strftime('%A, %d %B %Y, %H:%M %Z')}.")
    if facts:
        bullet = "\n".join(f"- {f}" for f in facts[-40:])
        parts.append(
            "\nTHINGS THE PERSON TOLD YOU IN PAST CONVERSATIONS (use only when relevant, never "
            "recite them for no reason):\n" + bullet
        )
    parts.append(
        "\nYou can call tools: `calculator` (exact arithmetic), `symbolic_math` (algebra, calculus, "
        "equations), `remember_fact` (save something durable the person tells you), `current_time`. "
        "Prefer tools over mental arithmetic for anything non-trivial."
    )
    return "\n".join(parts)


def backup_system_prompt(system: str) -> str:
    """The same persona for the OpenAI backup, but honest about what is powering it and about having no tools."""
    system = system.replace("Anthropic's Claude",
                            "OpenAI's GPT (you are in backup mode because the main model is unavailable)")
    return system + ("\n\nBACKUP MODE: you have no tools in this reply. Do calculations carefully by hand, "
                     "double-check them, and never claim you ran code or a tool.")


# --------------------------------------------------------------------------------------
# 3. ANTI-REPETITION / COMPLETENESS GUARDS
# --------------------------------------------------------------------------------------

_FENCE_RE = re.compile(r"(```.*?```)", re.S)


def detect_loop(text: str, tail_len: int = 60, repeats: int = 3) -> bool:
    """True if the last `tail_len` characters have already appeared `repeats`+ times.

    This is the exact failure in the screenshots ("Problem: Let $f: Problem: Let $f: ...").
    """
    if len(text) < tail_len * repeats:
        return False
    tail = text[-tail_len:]
    if not tail.strip() or len(set(tail)) < 6:  # whitespace / dashes / etc. are not loops
        return False
    return text.count(tail) >= repeats


def trim_loop(text: str, tail_len: int = 60) -> str:
    """Cut a runaway repetition back to its first occurrence."""
    if len(text) < tail_len * 2:
        return text
    tail = text[-tail_len:]
    first = text.find(tail)
    if first == -1 or first >= len(text) - tail_len:
        return text
    return text[: first + tail_len].rstrip()


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def _dedupe_prose(chunk: str) -> str:
    # 1) collapse stutter: "Answer Answer", "You can X. You can X."
    prev = None
    while prev != chunk:
        prev = chunk
        chunk = re.sub(r"(\b[\w$\\{}^_()\-.,:;' ]{12,200}?)(\s+)\1(?=\s|$|[.!?])", r"\1", chunk)
    # 2) drop repeated paragraphs and repeated sentences (consecutive or near-consecutive)
    paras = re.split(r"(\n\s*\n)", chunk)
    seen_paras: List[str] = []
    out: List[str] = []
    for p in paras:
        if not p.strip():
            out.append(p)
            continue
        key = _norm(p)
        if key and key in seen_paras[-6:]:
            continue
        seen_paras.append(key)
        sentences = re.split(r"(?<=[.!?])\s+", p)
        kept: List[str] = []
        recent: List[str] = []
        for s in sentences:
            k = _norm(s)
            if len(k) > 12 and k in recent[-8:]:
                continue
            recent.append(k)
            kept.append(s)
        out.append(" ".join(kept))
    return "".join(out)


def collapse_repeats(text: str) -> str:
    """Remove accidental repetition without touching fenced code blocks."""
    pieces = _FENCE_RE.split(text)
    cleaned = [pc if pc.startswith("```") else _dedupe_prose(pc) for pc in pieces]
    result = "".join(cleaned)
    return re.sub(r"\n{3,}", "\n\n", result).strip()


def looks_unfinished(text: str) -> bool:
    t = text.rstrip()
    if not t:
        return True
    if t.count("```") % 2 == 1:
        return True
    if t.count("$$") % 2 == 1:
        return True
    return t[-1] in ",:;(-\\$" or t.endswith(("and", " the", " of", " to"))


# --------------------------------------------------------------------------------------
# 4. SAFE TOOLS: exact calculator + symbolic maths (no eval, AST whitelist only)
# --------------------------------------------------------------------------------------

_MATH_FUNCS: Dict[str, Callable[..., Any]] = {
    n: getattr(math, n)
    for n in (
        "sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh atanh exp log log2 log10 sqrt "
        "floor ceil factorial comb perm gcd lcm degrees radians hypot fabs isqrt erf gamma lgamma"
    ).split()
}
_MATH_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max, "ln": math.log})
_MATH_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}


class ToolError(ValueError):
    pass


def _prep_expr(expr: str) -> str:
    expr = expr.strip().replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-")
    expr = expr.replace("π", "pi")
    if len(expr) > 2000:
        raise ToolError("expression too long")
    return expr


def safe_calc(expr: str) -> float | int | complex:
    """Evaluate an arithmetic expression using only whitelisted AST nodes."""
    tree = ast.parse(_prep_expr(expr), mode="eval")

    def ev(n: ast.AST):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in _MATH_CONSTS:
                return _MATH_CONSTS[n.id]
            raise ToolError(f"unknown name: {n.id}")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = ev(n.operand)
            return +v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            op = n.op
            if isinstance(op, ast.Add):
                return a + b
            if isinstance(op, ast.Sub):
                return a - b
            if isinstance(op, ast.Mult):
                return a * b
            if isinstance(op, ast.Div):
                return a / b
            if isinstance(op, ast.FloorDiv):
                return a // b
            if isinstance(op, ast.Mod):
                return a % b
            if isinstance(op, ast.Pow):
                if isinstance(a, int) and isinstance(b, int) and abs(a) > 1 and b > 0:
                    if b * math.log2(abs(a)) > 200_000:
                        raise ToolError("result would be astronomically large")
                return a ** b
            raise ToolError("operator not allowed")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _MATH_FUNCS:
            if n.keywords:
                raise ToolError("keyword arguments not allowed")
            args = [ev(x) for x in n.args]
            if n.func.id == "factorial" and args and args[0] > 5000:
                raise ToolError("factorial argument too large")
            return _MATH_FUNCS[n.func.id](*args)
        raise ToolError(f"unsupported syntax: {type(n).__name__}")

    return ev(tree)


_SYM_FUNCS = (
    "sin cos tan cot sec csc asin acos atan sinh cosh tanh exp log ln sqrt Abs factorial "
    "binomial floor ceiling sign gamma"
).split()
_SYM_CONSTS = {"pi": "pi", "E": "E", "e": "E", "I": "I", "i": "I", "oo": "oo", "inf": "oo"}


def symbolic_math(operation: str, expression: str, variable: str = "x",
                  point: str | None = None, lower: str | None = None,
                  upper: str | None = None, order: int = 6) -> str:
    """simplify | expand | factor | diff | integrate | solve | limit | series | evaluate"""
    try:
        import sympy as sp  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ToolError("sympy is not installed on the server (pip install sympy)") from exc

    funcs = {
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
        "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "sinh": sp.sinh, "cosh": sp.cosh,
        "tanh": sp.tanh, "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs,
        "abs": sp.Abs, "factorial": sp.factorial, "binomial": sp.binomial, "floor": sp.floor,
        "ceiling": sp.ceiling, "sign": sp.sign, "gamma": sp.gamma,
    }
    consts = {"pi": sp.pi, "E": sp.E, "e": sp.E, "I": sp.I, "i": sp.I, "oo": sp.oo, "inf": sp.oo}
    symbols: Dict[str, Any] = {}

    def sym(name: str):
        if name not in symbols:
            symbols[name] = sp.Symbol(name)
        return symbols[name]

    def ev(n: ast.AST):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return sp.Integer(n.value) if isinstance(n.value, int) else sp.Rational(str(n.value))
        if isinstance(n, ast.Name):
            if n.id in consts:
                return consts[n.id]
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,15}", n.id) and n.id not in funcs:
                return sym(n.id)
            raise ToolError(f"bad name: {n.id}")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = ev(n.operand)
            return v if isinstance(n.op, ast.UAdd) else -v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
            if isinstance(n.op, ast.Pow):
                return a ** b
            raise ToolError("operator not allowed")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in funcs and not n.keywords:
            return funcs[n.func.id](*[ev(x) for x in n.args])
        raise ToolError(f"unsupported syntax: {type(n).__name__}")

    def parse(s: str):
        s = _prep_expr(s)
        s = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", s)          # 2x -> 2*x
        s = re.sub(r"(?<=[A-Za-z0-9)])\s+(?=[A-Za-z(])", "*", s)  # x sin(x) -> x*sin(x)
        s = re.sub(r"\)\s*\(", ")*(", s)
        return ev(ast.parse(s, mode="eval"))

    op = operation.strip().lower()
    expr_s = expression.strip()
    var = sym(variable)
    if "=" in expr_s and not re.search(r"[<>!=]=|==", expr_s):
        lhs, rhs = expr_s.split("=", 1)
        target = parse(lhs) - parse(rhs)
    else:
        target = parse(expr_s)

    if op == "simplify":
        res = sp.simplify(target)
    elif op == "expand":
        res = sp.expand(target)
    elif op == "factor":
        res = sp.factor(target)
    elif op in ("diff", "derivative"):
        res = sp.diff(target, var)
    elif op == "integrate":
        res = sp.integrate(target, var) if not (lower and upper) else sp.integrate(
            target, (var, parse(lower), parse(upper)))
    elif op == "solve":
        res = sp.solve(target, var)
    elif op == "limit":
        res = sp.limit(target, var, parse(point if point is not None else "0"))
    elif op == "series":
        res = sp.series(target, var, parse(point if point is not None else "0"), int(order))
    elif op in ("evaluate", "numeric", "n"):
        res = sp.N(target, 30)
    else:
        raise ToolError("operation must be one of simplify, expand, factor, diff, integrate, solve, "
                        "limit, series, evaluate")
    return f"{sp.sstr(res)}\nLaTeX: {sp.latex(res)}"


# --------------------------------------------------------------------------------------
# 5. PERSISTENT MEMORY (SQLite, standard library only)
# --------------------------------------------------------------------------------------


class Store:
    kind = "sqlite"

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")  # readers no longer block writers
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS usage(
                    user_id TEXT, day TEXT, kind TEXT, n INTEGER, PRIMARY KEY(user_id, day, kind));
                CREATE TABLE IF NOT EXISTS conversations(
                    id TEXT PRIMARY KEY, user_id TEXT, title TEXT, created REAL, updated REAL);
                CREATE TABLE IF NOT EXISTS messages(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, conv_id TEXT, role TEXT,
                    content TEXT, ts REAL);
                CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conv_id, id);
                CREATE TABLE IF NOT EXISTS facts(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, fact TEXT, ts REAL);
                CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id, id);
                CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated);
                """
            )

    @contextlib.contextmanager
    def _conn(self):
        """Open, commit and ALWAYS close (the old code leaked connections under load)."""
        c = sqlite3.connect(self.path, timeout=30)
        try:
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA busy_timeout=30000")
            yield c
            c.commit()
        except BaseException:
            c.rollback()
            raise
        finally:
            c.close()

    def ensure_conversation(self, conv_id: Optional[str], user_id: str, first_text: str) -> str:
        conv_id = conv_id or uuid.uuid4().hex
        with self._lock, self._conn() as c:
            row = c.execute("SELECT id FROM conversations WHERE id=?", (conv_id,)).fetchone()
            now = time.time()
            if not row:
                title = re.sub(r"\s+", " ", first_text).strip()[:60] or "New chat"
                c.execute("INSERT INTO conversations VALUES(?,?,?,?,?)",
                          (conv_id, user_id, title, now, now))
            else:
                c.execute("UPDATE conversations SET updated=? WHERE id=?", (now, conv_id))
        return conv_id

    def add_message(self, conv_id: str, role: str, content: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO messages(conv_id, role, content, ts) VALUES(?,?,?,?)",
                      (conv_id, role, content, time.time()))

    def history(self, conv_id: str, char_budget: int) -> List[Dict[str, str]]:
        with self._conn() as c:
            rows = c.execute("SELECT role, content FROM messages WHERE conv_id=? ORDER BY id DESC LIMIT 400",
                             (conv_id,)).fetchall()
        rows = list(reversed(rows))  # only the most recent 400 messages are ever needed
        msgs = [{"role": r["role"], "content": r["content"]} for r in rows if r["content"].strip()]
        total, keep = 0, []
        for m in reversed(msgs):
            total += len(m["content"])
            if total > char_budget and keep:
                break
            keep.append(m)
        keep.reverse()
        while keep and keep[0]["role"] != "user":  # API requires the first turn to be the user
            keep.pop(0)
        return keep

    def list_conversations(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT id, title, updated FROM conversations WHERE user_id=? ORDER BY updated DESC LIMIT ?",
                (user_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def messages(self, conv_id: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT role, content, ts FROM messages WHERE conv_id=? ORDER BY id",
                             (conv_id,)).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, conv_id: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM messages WHERE conv_id=?", (conv_id,))
            c.execute("DELETE FROM conversations WHERE id=?", (conv_id,))

    def add_fact(self, user_id: str, fact: str) -> None:
        fact = fact.strip()[:400]
        if not fact:
            return
        with self._lock, self._conn() as c:
            dup = c.execute("SELECT 1 FROM facts WHERE user_id=? AND lower(fact)=lower(?)",
                            (user_id, fact)).fetchone()
            if not dup:
                c.execute("INSERT INTO facts(user_id, fact, ts) VALUES(?,?,?)",
                          (user_id, fact, time.time()))

    def facts(self, user_id: str) -> List[str]:
        with self._conn() as c:
            rows = c.execute("SELECT fact FROM facts WHERE user_id=? ORDER BY id", (user_id,)).fetchall()
        return [r["fact"] for r in rows]

    def bump_usage(self, user_id: str, kind: str = "max") -> int:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO usage VALUES(?,?,?,1) ON CONFLICT(user_id, day, kind) DO UPDATE SET n=n+1",
                      (user_id, day, kind))
            return int(c.execute("SELECT n FROM usage WHERE user_id=? AND day=? AND kind=?",
                                 (user_id, day, kind)).fetchone()[0])

    def clear_facts(self, user_id: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM facts WHERE user_id=?", (user_id,))


# --------------------------------------------------------------------------------------
# 5b. POSTGRESQL STORE (same interface as Store; used automatically when DATABASE_URL is set)
# --------------------------------------------------------------------------------------

_PG_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS oq_conversations(
           id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT, created DOUBLE PRECISION,
           updated DOUBLE PRECISION)""",
    "CREATE INDEX IF NOT EXISTS oq_idx_conv_user ON oq_conversations(user_id, updated DESC)",
    """CREATE TABLE IF NOT EXISTS oq_messages(
           id BIGSERIAL PRIMARY KEY, conv_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
           ts DOUBLE PRECISION)""",
    "CREATE INDEX IF NOT EXISTS oq_idx_msg_conv ON oq_messages(conv_id, id)",
    """CREATE TABLE IF NOT EXISTS oq_facts(
           id BIGSERIAL PRIMARY KEY, user_id TEXT NOT NULL, fact TEXT NOT NULL, ts DOUBLE PRECISION)""",
    "CREATE INDEX IF NOT EXISTS oq_idx_facts_user ON oq_facts(user_id, id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS oq_uq_facts ON oq_facts(user_id, lower(fact))",
    """CREATE TABLE IF NOT EXISTS oq_usage(
           user_id TEXT NOT NULL, day TEXT NOT NULL, kind TEXT NOT NULL, n INTEGER NOT NULL DEFAULT 0,
           PRIMARY KEY(user_id, day, kind))""",
]


class PgStore:
    """PostgreSQL storage with a connection pool. Tables are prefixed oq_ so they never clash with
    tables your website already has in the same database."""

    kind = "postgres"

    def __init__(self, url: str, pool_size: int = 20):
        try:
            from psycopg.rows import dict_row  # type: ignore
            from psycopg_pool import ConnectionPool  # type: ignore
        except ImportError as exc:
            raise RuntimeError('PostgreSQL support needs:  pip install "psycopg[binary,pool]"') from exc
        # prepare_threshold=None keeps it compatible with PgBouncer / Supabase / Neon poolers
        self._pool = ConnectionPool(url, min_size=1, max_size=max(2, pool_size), open=True, timeout=30,
                                    kwargs={"row_factory": dict_row, "prepare_threshold": None})
        with self._pool.connection() as c:
            c.execute("SELECT pg_advisory_xact_lock(727274)")  # several workers starting at once: take turns
            for stmt in _PG_SCHEMA:
                c.execute(stmt)

    def ensure_conversation(self, conv_id: Optional[str], user_id: str, first_text: str) -> str:
        conv_id = conv_id or uuid.uuid4().hex
        title = re.sub(r"\s+", " ", first_text).strip()[:60] or "New chat"
        now = time.time()
        with self._pool.connection() as c:
            c.execute("INSERT INTO oq_conversations(id, user_id, title, created, updated) "
                      "VALUES(%s,%s,%s,%s,%s) ON CONFLICT (id) DO UPDATE SET updated=EXCLUDED.updated",
                      (conv_id, user_id, title, now, now))
        return conv_id

    def add_message(self, conv_id: str, role: str, content: str) -> None:
        with self._pool.connection() as c:
            c.execute("INSERT INTO oq_messages(conv_id, role, content, ts) VALUES(%s,%s,%s,%s)",
                      (conv_id, role, content, time.time()))

    def history(self, conv_id: str, char_budget: int) -> List[Dict[str, str]]:
        with self._pool.connection() as c:
            rows = c.execute("SELECT role, content FROM (SELECT id, role, content FROM oq_messages "
                             "WHERE conv_id=%s ORDER BY id DESC LIMIT 400) t ORDER BY id",
                             (conv_id,)).fetchall()
        msgs = [{"role": r["role"], "content": r["content"]} for r in rows if r["content"].strip()]
        total, keep = 0, []
        for m in reversed(msgs):
            total += len(m["content"])
            if total > char_budget and keep:
                break
            keep.append(m)
        keep.reverse()
        while keep and keep[0]["role"] != "user":  # API requires the first turn to be the user
            keep.pop(0)
        return keep

    def list_conversations(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self._pool.connection() as c:
            rows = c.execute("SELECT id, title, updated FROM oq_conversations WHERE user_id=%s "
                             "ORDER BY updated DESC LIMIT %s", (user_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def messages(self, conv_id: str) -> List[Dict[str, Any]]:
        with self._pool.connection() as c:
            rows = c.execute("SELECT role, content, ts FROM oq_messages WHERE conv_id=%s ORDER BY id",
                             (conv_id,)).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, conv_id: str) -> None:
        with self._pool.connection() as c:
            c.execute("DELETE FROM oq_messages WHERE conv_id=%s", (conv_id,))
            c.execute("DELETE FROM oq_conversations WHERE id=%s", (conv_id,))

    def add_fact(self, user_id: str, fact: str) -> None:
        fact = fact.strip()[:400]
        if not fact:
            return
        with self._pool.connection() as c:
            c.execute("INSERT INTO oq_facts(user_id, fact, ts) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING",
                      (user_id, fact, time.time()))

    def facts(self, user_id: str) -> List[str]:
        with self._pool.connection() as c:
            rows = c.execute("SELECT fact FROM oq_facts WHERE user_id=%s ORDER BY id", (user_id,)).fetchall()
        return [r["fact"] for r in rows]

    def bump_usage(self, user_id: str, kind: str = "max") -> int:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        with self._pool.connection() as c:
            row = c.execute("INSERT INTO oq_usage(user_id, day, kind, n) VALUES(%s,%s,%s,1) "
                            "ON CONFLICT (user_id, day, kind) DO UPDATE SET n=oq_usage.n+1 RETURNING n",
                            (user_id, day, kind)).fetchone()
        return int(row["n"])

    def clear_facts(self, user_id: str) -> None:
        with self._pool.connection() as c:
            c.execute("DELETE FROM oq_facts WHERE user_id=%s", (user_id,))


def make_store(cfg: Config):
    url = (cfg.database_url or "").strip()
    if url.startswith(("postgres://", "postgresql://")):
        return PgStore(url, cfg.db_pool_size)
    if url and not url.startswith("sqlite"):
        raise RuntimeError("DATABASE_URL must be a PostgreSQL URL (postgresql://...). Got a different type.")
    return Store(cfg.db_path)


# --------------------------------------------------------------------------------------
# 6. ROUTER  (cheap, fast model for chat; strongest model + thinking + verification for hard stuff)
# --------------------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"^\s*(hi+|hey+|hello+|yo|hola|namaste|good (morning|evening|afternoon|night)|thanks?( you)?|"
    r"thank u|ok(ay)?|cool|nice|bye|sup)\W*$", re.I)
_MATH_HINTS = re.compile(
    r"(\$|\\frac|\\int|\\sum|∫|∑|√|\bintegral|\bderivative|\bdifferentiat|\blimit\b|\bprove\b|\bproof\b|"
    r"\bsolve\b|\bequation|\bprobability|\bexpected\b|\bmatrix|\beigen|\bfunction f\b|\bjee\b|\bolympiad|"
    r"\bcalculus|\balgebra|\bgeometry|\btrigonometr|\bcombinatoric|\bf\(x\)|\bg\(x\)|\bsin\b|\bcos\b|\blog\b)", re.I)
_LOGIC_HINTS = re.compile(
    r"(puzzle|riddle|how can you determine|switches|bulbs|prisoners|hats|river crossing|weigh|coins|"
    r"paradox|what is the minimum|optimal strategy|which statement is correct|logic)", re.I)
_CODE_HINTS = re.compile(
    r"(```|\bdef \w+\(|\bclass \w+|\bfunction\b|\bbug\b|\bdebug|\balgorithm|\bcomplexity|\bpython\b|"
    r"\bjavascript\b|\btypescript\b|\bc\+\+|\bjava\b|\brust\b|\bsql\b|\bfastapi\b|\bapi\b|\bregex\b|"
    r"\brefactor|\bimplement)", re.I)


@dataclass
class Plan:
    kind: str                 # chat | math | logic | code
    model: str
    thinking_budget: int
    verify: bool
    max_tokens: int
    reason: str


def route(user_text: str, cfg: Config = CFG, mode: str = "auto") -> Plan:
    """mode: auto | fast | smart | max.  "max" = the most powerful model (Claude Fable) with thinking."""
    t = user_text.strip()
    mode = (mode or "auto").strip().lower()
    if mode == "max":
        return Plan("research", cfg.model_max, cfg.thinking_budget, cfg.verify, cfg.max_tokens, "max mode requested")
    if mode == "smart":
        return Plan("chat", cfg.model_smart, cfg.thinking_budget, False, cfg.max_tokens, "smart mode requested")
    if mode == "fast":
        return Plan("chat", cfg.model_fast, 0, False, min(cfg.max_tokens, 32000), "fast mode requested")
    if len(t) < 40 and _GREETING_RE.match(t):
        return Plan("chat", cfg.model_fast, 0, False, 2000, "greeting")

    score = 0
    kind = "chat"
    if _MATH_HINTS.search(t):
        kind, score = "math", score + 2
    if _LOGIC_HINTS.search(t):
        kind, score = ("logic" if kind == "chat" else kind), score + 2
    if _CODE_HINTS.search(t):
        kind, score = ("code" if kind == "chat" else kind), score + 1
    if len(t) > 500:
        score += 1
    if len(t) > 1500:
        score += 1
    if re.search(r"(step[- ]by[- ]step|rigorous|in detail|explain why|prove|derive|optimi[sz]e|compare)", t, re.I):
        score += 1

    if score >= 2:
        return Plan(kind, cfg.model_smart, cfg.thinking_budget, cfg.verify and kind in ("math", "logic", "code"),
                    cfg.max_tokens, f"hard {kind} (score {score})")
    return Plan(kind, cfg.model_fast, 0, False, min(cfg.max_tokens, 32000), f"standard {kind} (score {score})")


# --------------------------------------------------------------------------------------
# 7. TOOL DEFINITIONS FOR THE MODEL
# --------------------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "calculator",
        "description": "Exact, safe arithmetic. Supports + - * / // % ** ^, parentheses, pi, e, and functions "
                       "like sqrt, sin, cos, tan, log, ln, exp, factorial, comb, perm, gcd, lcm. "
                       "Use this for any non-trivial number crunching instead of mental arithmetic.",
        "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}},
                         "required": ["expression"]},
    },
    {
        "name": "symbolic_math",
        "description": "Computer algebra via SymPy. operation is one of: simplify, expand, factor, diff, "
                       "integrate (optionally with lower/upper bounds), solve (use '=' for equations), "
                       "limit (give point), series (give point and order), evaluate (30-digit numeric).",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string"},
                "expression": {"type": "string"},
                "variable": {"type": "string", "description": "default x"},
                "point": {"type": "string"},
                "lower": {"type": "string"},
                "upper": {"type": "string"},
                "order": {"type": "integer"},
            },
            "required": ["operation", "expression"],
        },
    },
    {
        "name": "remember_fact",
        "description": "Save a durable fact the person told you about themselves (name, goals, exam, "
                       "preferences) so you can use it in future conversations. Never save secrets or "
                       "ID/card numbers.",
        "input_schema": {"type": "object", "properties": {"fact": {"type": "string"}}, "required": ["fact"]},
    },
    {
        "name": "current_time",
        "description": "Get the server's current date and time.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


def run_tool(name: str, args: Dict[str, Any], store: Store, user_id: str) -> Tuple[str, bool]:
    """Returns (text_result, is_error)."""
    try:
        if name == "calculator":
            val = safe_calc(str(args.get("expression", "")))
            return (repr(val) if isinstance(val, float) else str(val)), False
        if name == "symbolic_math":
            return symbolic_math(
                str(args.get("operation", "")), str(args.get("expression", "")),
                str(args.get("variable") or "x"), args.get("point"), args.get("lower"),
                args.get("upper"), int(args.get("order") or 6)), False
        if name == "remember_fact":
            store.add_fact(user_id, str(args.get("fact", "")))
            return "saved", False
        if name == "current_time":
            return time.strftime("%A, %d %B %Y, %H:%M:%S %Z"), False
        return f"unknown tool: {name}", True
    except ToolError as e:
        return f"tool error: {e}", True
    except ZeroDivisionError:
        return "tool error: division by zero", True
    except Exception as e:  # noqa: BLE001
        return f"tool error: {type(e).__name__}: {e}", True


# --------------------------------------------------------------------------------------
# 8. THE ENGINE  (Anthropic streaming, tool loop, continuation, loop guard, verification)
# --------------------------------------------------------------------------------------

Emit = Callable[[Dict[str, Any]], Awaitable[None]]

VERIFIER_PROMPT = """\
You are a strict, independent checker. Below is a question and a proposed answer.

Re-solve the problem yourself, independently. Then judge the proposed answer on: (1) is the final \
result correct, (2) is every key step valid, (3) did it ignore a trick, false premise or edge case, \
(4) does it claim anything it could not have verified (e.g. a simulation that was never run).

Reply with ONLY a JSON object, no prose outside it:
{"verdict": "ok" | "error", "problem": "<one sentence, empty if ok>", "corrected_answer": "<full corrected \
answer in Markdown/LaTeX, empty if ok>"}
"""


class Engine:
    def __init__(self, cfg: Config, store: Store):
        self.cfg = cfg
        self.store = store
        self._client = None
        self._slots_sem: Optional[asyncio.Semaphore] = None
        self._max_sem: Optional[asyncio.Semaphore] = None
        self.active = 0
        self.waiting = 0

    def _slots(self) -> asyncio.Semaphore:
        if self._slots_sem is None:
            self._slots_sem = asyncio.Semaphore(max(1, self.cfg.max_concurrent))
        return self._slots_sem

    def _max_slots(self) -> asyncio.Semaphore:
        if self._max_sem is None:
            self._max_sem = asyncio.Semaphore(max(1, self.cfg.max_concurrent_max))
        return self._max_sem

    # ---- client ---------------------------------------------------------------------
    def client(self):
        if self._client is None:
            if not self.cfg.api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set on the server.")
            try:
                import anthropic  # type: ignore
            except ImportError as exc:
                raise RuntimeError("Missing dependency: pip install anthropic") from exc
            self._client = anthropic.AsyncAnthropic(api_key=self.cfg.api_key, max_retries=2, timeout=600.0)
        return self._client

    # ---- one streamed model call with graceful parameter fallback ---------------------
    async def _stream_once(self, plan: Plan, system: str, messages: List[Dict[str, Any]],
                           emit: Emit, collected: List[str], allow_tools: bool = True):
        """Streams text deltas. Returns (final_message | None, looped: bool).

        Falls back to a simpler parameter set if the model rejects an optional parameter, and to the
        next model (smart, then fast) if the chosen model is not available on this API key.
        """
        import anthropic  # type: ignore

        models = [plan.model] + [m for m in (self.cfg.model_smart, self.cfg.model_fast) if m != plan.model]
        last_err: Optional[Exception] = None
        for model in models:
            base: Dict[str, Any] = dict(model=model, max_tokens=plan.max_tokens, system=system,
                                        messages=messages)
            if allow_tools:
                base["tools"] = TOOLS
            attempts: List[Dict[str, Any]] = []
            if plan.thinking_budget and plan.thinking_budget < plan.max_tokens:
                attempts.append({**base, "thinking": {"type": "enabled", "budget_tokens": plan.thinking_budget}})
            attempts.append({**base, "temperature": self.cfg.temperature})
            attempts.append(base)
            attempts.append({**base, "max_tokens": min(plan.max_tokens, 16000)})  # model rejected the big limit

            for kwargs in attempts:
                try:
                    looped = False
                    async with self.client().messages.stream(**kwargs) as stream:
                        async for text in stream.text_stream:
                            collected.append(text)
                            await emit({"type": "delta", "text": text})
                            if len(collected) % 8 == 0 and detect_loop("".join(collected[-400:])):
                                looped = True
                                break
                        if looped:
                            return None, True
                        final = await stream.get_final_message()
                    return final, False
                except anthropic.BadRequestError as e:  # unsupported optional parameter -> simpler call
                    last_err = e
                    if collected:
                        raise
                    continue
                except (anthropic.NotFoundError, anthropic.PermissionDeniedError, anthropic.RateLimitError,
                        anthropic.InternalServerError, anthropic.APIConnectionError) as e:
                    # model unavailable / rate limited / overloaded -> next model (each has its own limits)
                    last_err = e
                    if collected:
                        raise
                    break
        assert last_err is not None
        raise last_err

    # ---- main entry -------------------------------------------------------------------
    async def run_turn(self, conv_id: str, user_id: str, user_text: str, emit: Emit,
                       mode: str = "auto") -> str:
        plan = route(user_text, self.cfg, mode)
        held: List[asyncio.Semaphore] = []
        counted = False
        try:
            # the expensive "max" model has its own, smaller lane; if it is full, use the smart model instead
            if plan.model == self.cfg.model_max:
                if self._max_slots().locked():
                    plan = replace(plan, model=self.cfg.model_smart, verify=False,
                                   reason=plan.reason + "; max lane busy, using smart model")
                else:
                    await self._max_slots().acquire()
                    held.append(self._max_slots())
            slots = self._slots()
            if slots.locked():
                await emit({"type": "queued", "text": "Lots of people are chatting right now. You're in line..."})
            self.waiting += 1
            try:
                await asyncio.wait_for(slots.acquire(), timeout=self.cfg.queue_timeout)
            except asyncio.TimeoutError:
                raise RuntimeError("Deep OQ is very busy right now. Please try again in a few seconds.")
            finally:
                self.waiting -= 1
            held.append(slots)
            self.active += 1
            counted = True
            return await self._run_inner(conv_id, user_id, user_text, emit, plan)
        finally:
            if counted:
                self.active -= 1
            for sem in held:
                sem.release()

    async def _run_inner(self, conv_id: str, user_id: str, user_text: str, emit: Emit, plan: Plan) -> str:
        await emit({"type": "meta", "conversation_id": conv_id, "model": plan.model,
                    "kind": plan.kind, "thinking": bool(plan.thinking_budget), "reason": plan.reason})

        # database and CPU-heavy work run in threads so they never freeze the server for other users
        await asyncio.to_thread(self.store.add_message, conv_id, "user", user_text)
        messages: List[Dict[str, Any]] = await asyncio.to_thread(
            self.store.history, conv_id, self.cfg.history_char_budget)
        system = build_system_prompt(await asyncio.to_thread(self.store.facts, user_id))

        full_parts: List[str] = []
        continuations = 0

        for _round in range(self.cfg.max_tool_rounds + self.cfg.max_continuations + 1):
            seg: List[str] = []
            try:
                final, looped = await self._stream_once(plan, system, messages, emit, seg)
            except Exception as primary_err:  # noqa: BLE001  Anthropic failed -> try the OpenAI backup
                if seg or not self.cfg.openai_api_key:
                    raise
                await emit({"type": "note", "text": "main model unavailable, using the backup model"})
                try:
                    await self._stream_openai(backup_system_prompt(system),
                                              await asyncio.to_thread(self.store.history, conv_id,
                                                                      self.cfg.history_char_budget),
                                              emit, seg)
                except Exception:  # noqa: BLE001
                    raise primary_err
                full_parts.append("".join(seg))
                break
            seg_text = "".join(seg)
            full_parts.append(seg_text)

            if looped:
                await emit({"type": "note", "text": "repetition detected and trimmed"})
                break
            if final is None:
                break

            if final.stop_reason == "refusal" and not seg_text.strip() and plan.model != self.cfg.model_smart:
                full_parts.pop()  # the strongest model declined; retry once on the smart model
                plan = replace(plan, model=self.cfg.model_smart)
                continue

            if final.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": final.content})
                results = []
                for block in final.content:
                    if getattr(block, "type", "") == "tool_use":
                        out, is_err = await asyncio.to_thread(run_tool, block.name, dict(block.input or {}), self.store, user_id)
                        await emit({"type": "tool", "name": block.name, "input": block.input,
                                    "output": out[:2000], "error": is_err})
                        results.append({"type": "tool_result", "tool_use_id": block.id,
                                        "content": out, "is_error": is_err})
                messages.append({"role": "user", "content": results})
                continue

            if final.stop_reason == "max_tokens" and continuations < self.cfg.max_continuations:
                continuations += 1
                messages.append({"role": "assistant", "content": final.content})
                messages.append({"role": "user", "content": "Continue exactly where you stopped. Do not "
                                 "repeat anything you already wrote."})
                continue
            break

        answer = await asyncio.to_thread(lambda: collapse_repeats(trim_loop("".join(full_parts))))
        if not answer.strip():
            answer = ("I got stuck producing that answer. Could you rephrase or give me a little more "
                      "detail? I'd like to get this right.")
            await emit({"type": "delta", "text": answer})

        # ---- self-verification for hard problems (separate model call, can append a correction)
        if plan.verify and len(answer) > 80:
            try:
                correction = await self._verify(plan, user_text, answer)
            except Exception:  # noqa: BLE001  verification is best-effort, never break the chat
                correction = None
            if correction:
                block = ("\n\n---\n**Correction after double-checking my work:** " + correction["problem"]
                         + "\n\n" + correction["corrected_answer"])
                await emit({"type": "correction", "text": block})
                answer += block

        await asyncio.to_thread(self.store.add_message, conv_id, "assistant", answer)
        await emit({"type": "done", "conversation_id": conv_id, "final": answer})
        return answer

    # ---- OpenAI backup ----------------------------------------------------------------
    async def _stream_openai(self, system: str, history: List[Dict[str, str]], emit: Emit,
                             collected: List[str]) -> None:
        """Streams a plain-text answer from OpenAI's strongest model (Responses API). No tools."""
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Missing dependency for the backup: pip install openai") from exc
        client = AsyncOpenAI(api_key=self.cfg.openai_api_key, max_retries=3, timeout=600.0)
        msgs = [{"role": m["role"], "content": m["content"]} for m in history]
        last_err: Optional[Exception] = None
        for model in [m for m in dict.fromkeys([self.cfg.openai_model, self.cfg.openai_model_alt]) if m]:
            base: Dict[str, Any] = dict(model=model, instructions=system, input=msgs, stream=True,
                                        max_output_tokens=min(self.cfg.max_tokens, 64000))
            attempts = []
            if self.cfg.openai_reasoning:
                attempts.append({**base, "reasoning": {"effort": self.cfg.openai_reasoning}})
            attempts.append(base)
            for kwargs in attempts:
                try:
                    stream = await client.responses.create(**kwargs)
                    async for event in stream:
                        if getattr(event, "type", "") == "response.output_text.delta":
                            collected.append(event.delta)
                            await emit({"type": "delta", "text": event.delta})
                    return
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    if collected:
                        raise
                    continue
        assert last_err is not None
        raise last_err

    # ---- verification -----------------------------------------------------------------
    async def _verify(self, plan: Plan, question: str, answer: str) -> Optional[Dict[str, str]]:
        resp = await self.client().messages.create(
            model=plan.model if plan.model == self.cfg.model_max else self.cfg.model_smart,
            max_tokens=32000, system=VERIFIER_PROMPT,
            messages=[{"role": "user", "content": f"QUESTION:\n{question}\n\nPROPOSED ANSWER:\n{answer}"}],
        )
        text = "".join(getattr(b, "text", "") for b in resp.content)
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
        if str(data.get("verdict", "")).lower() == "error" and str(data.get("corrected_answer", "")).strip():
            return {"problem": str(data.get("problem", "")).strip() or "I found a mistake.",
                    "corrected_answer": str(data["corrected_answer"]).strip()}
        return None


# --------------------------------------------------------------------------------------
# 9. WEB API (FastAPI)  -  streaming SSE + simple JSON endpoint + tiny built-in chat page
# --------------------------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Deep OQ</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{margin:0;background:#0b1030;color:#e8ebff;font:16px/1.6 Georgia,serif;display:flex;flex-direction:column;height:100vh}
 #log{flex:1;overflow:auto;padding:24px;max-width:860px;width:100%;margin:0 auto;box-sizing:border-box}
 .u{background:#27308f;border-radius:16px;padding:10px 16px;margin:14px 0 14px auto;width:fit-content;max-width:80%;white-space:pre-wrap}
 .a{margin:14px 0;white-space:pre-wrap}.t{color:#8c93c9;font-size:13px}
 form{display:flex;gap:8px;padding:16px;max-width:860px;width:100%;margin:0 auto;box-sizing:border-box}
 textarea{flex:1;border-radius:14px;border:1px solid #2c356e;background:#10163f;color:inherit;padding:12px;font:inherit;resize:none}
 button{border:0;border-radius:14px;background:#6c5ce7;color:#fff;padding:0 22px;font:inherit;cursor:pointer}
</style></head><body><div id="log"></div>
<form id="f"><textarea id="m" rows="2" placeholder="Ask Deep OQ... (Shift+Enter for new line)"></textarea><button>Send</button></form>
<script>
let cid=null;const log=document.getElementById('log'),m=document.getElementById('m');
function add(c,t){const d=document.createElement('div');d.className=c;d.textContent=t;log.appendChild(d);log.scrollTop=1e9;return d}
async function send(){const q=m.value.trim();if(!q)return;m.value='';add('u',q);const a=add('a','');const t=add('t','thinking...');
 const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:q,conversation_id:cid})});
 const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
 while(true){const {done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let i;
  while((i=buf.indexOf('\n\n'))>=0){const line=buf.slice(0,i);buf=buf.slice(i+2);if(!line.startsWith('data: '))continue;
   const ev=JSON.parse(line.slice(6));
   if(ev.type=='meta'){cid=ev.conversation_id;t.textContent=ev.model+(ev.thinking?' (thinking deeply)':'')}
   else if(ev.type=='delta'||ev.type=='correction'){a.textContent+=ev.text;log.scrollTop=1e9}
   else if(ev.type=='tool'){t.textContent='used tool: '+ev.name}
   else if(ev.type=='error'){a.textContent+='\n[error] '+ev.message}
   else if(ev.type=='done'){t.textContent=''}}}}
document.getElementById('f').onsubmit=e=>{e.preventDefault();send()};
m.onkeydown=e=>{if(e.key=='Enter'&&!e.shiftKey){e.preventDefault();send()}};
</script></body></html>"""


def create_app():
    from fastapi import Depends, FastAPI, HTTPException, Request  # type: ignore
    from fastapi.middleware.cors import CORSMiddleware  # type: ignore
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse  # type: ignore
    from pydantic import BaseModel  # type: ignore

    store = make_store(CFG)
    engine = Engine(CFG, store)
    app = FastAPI(title="Deep OQ v2")
    origins = [o.strip() for o in CFG.cors_origins.split(",") if o.strip()] or ["*"]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    hits: Dict[str, List[float]] = {}

    def client_ip(request: Request) -> str:
        fwd = request.headers.get("x-forwarded-for", "")  # behind Render / Railway the real IP is in here
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "?"

    async def guard(request: Request):
        if CFG.api_token:
            if request.headers.get("authorization", "") != f"Bearer {CFG.api_token}":
                raise HTTPException(status_code=401, detail="unauthorized")
        if CFG.rate_limit_per_min > 0:
            ip = client_ip(request)
            now = time.time()
            if len(hits) > 5000:  # forget old visitors so memory does not grow forever
                for k in [k for k, v in hits.items() if not v or now - v[-1] > 60]:
                    hits.pop(k, None)
            window = [t for t in hits.get(ip, []) if now - t < 60]
            if len(window) >= CFG.rate_limit_per_min:
                raise HTTPException(status_code=429, detail="slow down a little")
            window.append(now)
            hits[ip] = window

    class ChatIn(BaseModel):
        message: Optional[str] = None
        prompt: Optional[str] = None
        query: Optional[str] = None
        conversation_id: Optional[str] = None
        session_id: Optional[str] = None
        user_id: Optional[str] = "default"
        mode: Optional[str] = "auto"        # auto | fast | smart | max
        research: Optional[bool] = False    # the Research button -> uses the most powerful model

        def resolved_mode(self) -> str:
            return "max" if self.research else (self.mode or "auto")

        def text(self) -> str:
            return (self.message or self.prompt or self.query or "").strip()

    async def prepare(body: ChatIn) -> Tuple[str, str, str]:
        text = body.text()
        if not text:
            raise HTTPException(status_code=400, detail="empty message")
        user_id = (body.user_id or "default")[:100]
        conv_id = await asyncio.to_thread(store.ensure_conversation, body.conversation_id or body.session_id,
                                          user_id, text)
        return conv_id, user_id, text

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return INDEX_HTML

    @app.get("/health")
    async def health():
        return {"ok": True, "api_key_set": bool(CFG.api_key), "active": engine.active,
                "waiting": engine.waiting, "database": store.kind, "fast": CFG.model_fast, "smart": CFG.model_smart,
                "max": CFG.model_max,
                "openai_backup": CFG.openai_model if CFG.openai_api_key else None}

    background: set = set()

    async def sse(conv_id: str, user_id: str, text: str, mode: str = "auto"):
        q: asyncio.Queue = asyncio.Queue()

        async def emit(ev: Dict[str, Any]):
            await q.put(ev)

        async def worker():
            try:
                await engine.run_turn(conv_id, user_id, text, emit, mode)
            except Exception as e:  # noqa: BLE001
                msg = str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: the model request failed."
                await q.put({"type": "error", "message": msg})
            finally:
                await q.put(None)

        # The worker is NOT cancelled when the client disconnects (e.g. a phone that loses signal), so the
        # full answer is still finished and saved; it will be there when the chat is reloaded.
        task = asyncio.create_task(worker())
        background.add(task)
        task.add_done_callback(background.discard)
        while True:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=15)
            except asyncio.TimeoutError:
                yield ": ping\n\n"   # keep-alive so proxies / mobile networks do not cut the stream
                continue
            if ev is None:
                break
            yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"

    async def effective_mode(body: ChatIn, user_id: str, request: Request) -> str:
        """Applies the per-user daily limit on the expensive max model (falls back to smart)."""
        mode = body.resolved_mode()
        if mode == "max" and CFG.daily_max_limit > 0:
            who = user_id if user_id != "default" else client_ip(request)
            used = await asyncio.to_thread(store.bump_usage, who, "max")
            if used > CFG.daily_max_limit:
                return "smart"
        return mode

    @app.post("/api/chat", dependencies=[Depends(guard)])
    @app.post("/chat", dependencies=[Depends(guard)])
    async def chat_stream(body: ChatIn, request: Request):
        conv_id, user_id, text = await prepare(body)
        mode = await effective_mode(body, user_id, request)
        return StreamingResponse(sse(conv_id, user_id, text, mode), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/chat/sync", dependencies=[Depends(guard)])
    async def chat_sync(body: ChatIn, request: Request):
        conv_id, user_id, text = await prepare(body)
        mode = await effective_mode(body, user_id, request)
        events: List[Dict[str, Any]] = []

        async def emit(ev: Dict[str, Any]):
            events.append(ev)

        try:
            answer = await engine.run_turn(conv_id, user_id, text, emit, mode)
        except RuntimeError as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        meta = next((e for e in events if e["type"] == "meta"), {})
        return {"reply": answer, "conversation_id": conv_id, "model": meta.get("model"),
                "kind": meta.get("kind")}

    @app.get("/api/conversations", dependencies=[Depends(guard)])
    async def conversations(user_id: str = "default"):
        return await asyncio.to_thread(store.list_conversations, user_id)

    @app.get("/api/conversations/{conv_id}", dependencies=[Depends(guard)])
    async def conversation(conv_id: str):
        return await asyncio.to_thread(store.messages, conv_id)

    @app.delete("/api/conversations/{conv_id}", dependencies=[Depends(guard)])
    async def delete_conversation(conv_id: str):
        await asyncio.to_thread(store.delete_conversation, conv_id)
        return {"ok": True}

    @app.get("/api/memory", dependencies=[Depends(guard)])
    async def get_memory(user_id: str = "default"):
        return {"facts": await asyncio.to_thread(store.facts, user_id)}

    @app.delete("/api/memory", dependencies=[Depends(guard)])
    async def clear_memory(user_id: str = "default"):
        await asyncio.to_thread(store.clear_facts, user_id)
        return {"ok": True}

    return app


# --------------------------------------------------------------------------------------
# 10. CLI
# --------------------------------------------------------------------------------------


async def _cli_chat() -> None:
    store = make_store(CFG)
    engine = Engine(CFG, store)
    conv = store.ensure_conversation(None, "cli", "terminal chat")
    print("Deep OQ v2 terminal chat. Type 'exit' to quit.\n")
    loop = asyncio.get_running_loop()

    async def emit(ev: Dict[str, Any]) -> None:
        if ev["type"] in ("delta", "correction"):
            print(ev["text"], end="", flush=True)
        elif ev["type"] == "tool":
            print(f"\n[tool: {ev['name']}]\n", end="", flush=True)
        elif ev["type"] == "meta":
            print(f"[{ev['model']}{' + thinking' if ev['thinking'] else ''}]\n", flush=True)

    while True:
        try:
            q = await loop.run_in_executor(None, lambda: input("\nyou > "))
        except (EOFError, KeyboardInterrupt):
            break
        if q.strip().lower() in ("exit", "quit"):
            break
        if not q.strip():
            continue
        print("\nOQ  > ", end="", flush=True)
        try:
            await engine.run_turn(conv, "cli", q, emit)
        except Exception as e:  # noqa: BLE001
            print(f"\n[error] {e}")
        print()


def main(argv: List[str]) -> None:
    cmd = argv[1] if len(argv) > 1 else "serve"
    if cmd == "chat":
        asyncio.run(_cli_chat())
    elif cmd == "serve":
        import uvicorn  # type: ignore
        port = _env_int("PORT", 8000)  # Render sets PORT
        workers = max(1, _env_int("WEB_CONCURRENCY", 1))
        if workers > 1:  # several processes = several CPU cores handling users in parallel
            uvicorn.run("deep_oq_v2:create_app", factory=True, host="0.0.0.0", port=port, workers=workers)
        else:
            uvicorn.run(create_app(), host="0.0.0.0", port=port)
    elif cmd == "selftest":
        _selftest()
    else:
        print("usage: python deep_oq_v2.py [serve|chat|selftest]")


# --------------------------------------------------------------------------------------
# 11. SELF-TEST (no API key or network needed)
# --------------------------------------------------------------------------------------


def _selftest() -> None:
    looped = "Problem: Let $f: " * 12
    assert detect_loop(looped), "loop detector missed the screenshot failure"
    assert collapse_repeats("Answer Answer You can use heat. You can use heat.").count("heat") == 1
    assert "```\nx\nx\n```" in collapse_repeats("Hi.\n\n```\nx\nx\n```")
    assert safe_calc("2^10 + sqrt(16)") == 1028
    assert abs(safe_calc("sin(pi/2)") - 1) < 1e-12
    for bad in ("__import__('os').system('ls')", "(1).__class__", "9**9**9", "open('x')"):
        try:
            safe_calc(bad)
            raise AssertionError(f"should have rejected {bad}")
        except (ToolError, SyntaxError, TypeError):
            pass
    assert route("hi").model == CFG.model_fast and route("hi").thinking_budget == 0
    hard = route("For x in R, let f(x)=|log 2 - sin x| and g(x)=f(f(x)). Which statement is correct at x=0?")
    assert hard.model == CFG.model_smart and hard.verify
    assert route("hi", mode="max").model == CFG.model_max
    assert route("hi", mode="fast").model == CFG.model_fast
    assert "Anthropic" not in backup_system_prompt(build_system_prompt([]))
    try:
        make_store(Config(database_url="mysql://x"))
        raise AssertionError("non-postgres URL should be rejected")
    except RuntimeError:
        pass
    print("selftest OK")


if __name__ == "__main__":
    main(sys.argv)
