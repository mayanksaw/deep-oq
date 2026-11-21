"""
proof_checker.py - makes Deep OQ's maths proofs honest.

An AI can write a proof that reads well and still contains false steps. This module catches that:

  1. The question is turned into a checkable statement and tested numerically first. A false statement
     gets a counterexample instead of a fake proof, and a true one gets a "tightness map" (where it is
     almost an equality) that tells the AI which steps are dangerous.
  2. The AI writes a proof. A SEPARATE call lists every inequality / identity the proof asserts, so the
     author cannot hide a weak step.
  3. Every listed step is tested at tens of thousands of random points, including values near 0, near 1
     and very large. A counterexample is confirmed with 50-digit arithmetic so rounding errors cannot
     cause a false alarm. Steps that fail are sent back to the AI with the exact counterexample, and it
     rewrites the proof (up to 3 rounds).
  4. The reader is told honestly what was checked. Passing the check means "no counterexample found",
     which catches mistakes but is not a formal proof.

Use inside deep_oq.py (see the 6-line hook at the bottom of this file's docstring in the chat message):

    import proof_checker
    res = proof_checker.prove(question, lambda m, t, k: chat(m, t, k, smart=True, prefer=EXAM_MODEL), emit=emit)
    if res: return res.text

Needs: pip install numpy sympy
Test:  python proof_checker.py --selftest        Demo: python proof_checker.py --demo
"""
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application, parse_expr,
                                        standard_transformations)

# --------------------------------------------------------------------------
# Safe parsing of maths text
# --------------------------------------------------------------------------

_FUNCS = {
    "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "abs": sp.Abs, "Abs": sp.Abs, "min": sp.Min, "max": sp.Max,
    "floor": sp.floor, "ceiling": sp.ceiling,
}
_CONSTS = {"pi": sp.pi, "E": sp.E}
_TRANS = standard_transformations + (implicit_multiplication_application, convert_xor)
_OK_CHARS = re.compile(r"^[0-9A-Za-z_+\-*/^().,\s<>=!]*$")
_REL = re.compile(r"(<=|>=|==|<|>|=|≤|≥)")


class CheckError(ValueError):
    pass


def _clean(s: str) -> str:
    for a, b in (("−", "-"), ("×", "*"), ("÷", "/"), ("≤", "<="), ("≥", ">="), ("√", "sqrt"), ("π", "pi"),
                 ("²", "**2"), ("³", "**3"), ("·", "*")):
        s = s.replace(a, b)
    return s.strip()


def parse_expr_safe(text: str, names: List[str]):
    s = _clean(text)
    if not s or "__" in s or not _OK_CHARS.match(s):
        raise CheckError("unreadable or unsafe expression: " + text[:80])
    local: Dict[str, object] = {}
    local.update(_FUNCS)
    local.update(_CONSTS)
    for n in names:
        local[n] = sp.Symbol(n)
    for ident in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", s):
        if ident not in local:
            raise CheckError("unknown name '%s' (declare it as a variable)" % ident)
    return parse_expr(s, local_dict=local, transformations=_TRANS)


def split_chain(text: str):
    """'1 < f < 2' -> (['1','f','2'], ['<','<'])."""
    parts = _REL.split(text)
    if len(parts) < 3:
        raise CheckError("not an inequality or equation: " + text[:80])
    exprs = [p.strip() for p in parts[0::2]]
    ops = ["==" if o == "=" else ("<=" if o == "≤" else (">=" if o == "≥" else o)) for o in parts[1::2]]
    if any(not e for e in exprs):
        raise CheckError("an expression is missing in: " + text[:80])
    return exprs, ops


# --------------------------------------------------------------------------
# Sampling: values everywhere, but especially near the edges where proofs break
# --------------------------------------------------------------------------


def _pos(n, rng):
    k = rng.random(n)
    out = np.empty(n)
    m = k < 0.42
    out[m] = 10 ** rng.uniform(-6, 6, m.sum())
    m = (k >= 0.42) & (k < 0.72)
    out[m] = rng.uniform(0, 4, m.sum())
    m = (k >= 0.72) & (k < 0.86)
    out[m] = 10 ** rng.uniform(-9, -3, m.sum())
    m = k >= 0.86
    out[m] = 10 ** rng.uniform(3, 9, m.sum())
    out[out <= 0] = 1e-12
    return out


def _unit(n, rng):
    k = rng.random(n)
    out = np.empty(n)
    m = k < 0.4
    out[m] = rng.uniform(0, 1, m.sum())
    m = (k >= 0.4) & (k < 0.6)
    out[m] = 10 ** rng.uniform(-9, -1, m.sum())
    m = (k >= 0.6) & (k < 0.8)
    out[m] = 1 - 10 ** rng.uniform(-9, -1, m.sum())
    m = k >= 0.8
    out[m] = rng.uniform(0.3, 0.7, m.sum())
    return np.clip(out, 1e-12, 1 - 1e-12)


def _interval(lo, hi, n, rng):
    k = rng.random(n)
    w = hi - lo
    out = lo + w * rng.random(n)
    m = (k >= 0.6) & (k < 0.8)
    out[m] = lo + w * 10 ** rng.uniform(-8, -1, m.sum())
    m = k >= 0.8
    out[m] = hi - w * 10 ** rng.uniform(-8, -1, m.sum())
    return np.clip(out, lo + w * 1e-12, hi - w * 1e-12)


def sample_var(kind: str, n: int, rng) -> np.ndarray:
    kind = (kind or "real").strip().lower()
    if kind in ("pos", "positive", ">0"):
        return _pos(n, rng)
    if kind in ("nonneg", ">=0", "nonnegative"):
        v = _pos(n, rng)
        v[rng.random(n) < 0.03] = 0.0
        return v
    if kind in ("unit", "(0,1)"):
        return _unit(n, rng)
    if kind in ("int", "integer", "posint", "natural"):
        small = rng.integers(1, 60, n).astype(float)
        big = np.floor(10 ** rng.uniform(2, 6, n))
        return np.where(rng.random(n) < 0.8, small, big)
    if kind in ("real", "any"):
        return _pos(n, rng) * np.where(rng.random(n) < 0.5, -1.0, 1.0)
    m = re.fullmatch(r"[\(\[]\s*([^,]+?)\s*,\s*([^,]+?)\s*[\)\]]", kind)
    if m:
        lo, hi = float(sp.N(parse_expr_safe(m.group(1), []))), float(sp.N(parse_expr_safe(m.group(2), [])))
        if lo < hi:
            return _interval(lo, hi, n, rng)
    raise CheckError("unknown variable type '%s' (use pos, nonneg, unit, real, int or (lo,hi))" % kind)


# --------------------------------------------------------------------------
# Evaluating claims
# --------------------------------------------------------------------------


@dataclass
class ClaimResult:
    id: str
    text: str
    status: str            # ok | false | unparsed | untestable
    tested: int = 0
    example: Optional[Dict[str, float]] = None
    detail: str = ""


def _lambdify(symbols, expr):
    fn = sp.lambdify(symbols, expr, modules=["numpy"])

    def ev(env):
        with np.errstate(all="ignore"):
            r = fn(*[env[str(s)] for s in symbols])
        return np.asarray(r, dtype=float)
    return ev


def _free_names(*exprs) -> List[str]:
    out = set()
    for e in exprs:
        out |= {str(s) for s in e.free_symbols}
    return sorted(out)


def _truth(vals, ops, tol=1e-9):
    """(ok, violation) arrays for a chain. vals are arrays, one per expression."""
    n = len(vals[0])
    ok = np.ones(n, bool)
    viol = np.zeros(n)
    for i, op in enumerate(ops):
        l, r = vals[i], vals[i + 1]
        d = l - r
        scale = 1 + np.abs(l) + np.abs(r)
        sc = tol * scale
        if op in ("<", "<="):
            bad, amt = d > sc, d
        elif op in (">", ">="):
            bad, amt = -d > sc, -d
        else:
            bad, amt = np.abs(d) > sc, np.abs(d)
        ok &= ~bad
        viol = np.maximum(viol, np.where(bad, amt / scale, 0.0))
    return ok, viol


def _hp_truth(exprs, ops, point):
    """Same chain at 50 digits. Returns (holds, worst violation)."""
    sub = {sp.Symbol(k): sp.Float(v, 50) for k, v in point.items()}
    vals = [sp.N(e.subs(sub), 50) for e in exprs]
    worst, holds = 0, True
    for i, op in enumerate(ops):
        d = vals[i] - vals[i + 1]
        if op in ("<", "<="):
            bad, amt = d > 0, d
        elif op in (">", ">="):
            bad, amt = d < 0, -d
        else:
            bad, amt = abs(d) > 1e-30, abs(d)
        if bad:
            holds = False
            worst = max(worst, float(amt))
    return holds, worst


class Spec:
    """Variables, definitions and claims of one proof."""

    def __init__(self, variables: Dict[str, str], lets: Optional[Dict[str, str]] = None):
        self.vars = dict(variables or {})
        self.lets_text = dict(lets or {})
        for k in self.lets_text:
            self.vars.pop(k, None)
        self.names = list(self.vars) + list(self.lets_text)

    def lets(self):
        out = {}
        for k, v in self.lets_text.items():
            out[sp.Symbol(k)] = parse_expr_safe(v, self.names)
        return out

    def expand(self, expr):
        lets = self.lets()
        for _ in range(6):
            nxt = expr.subs(lets)
            if nxt == expr:
                break
            expr = nxt
        return expr


def _sample_env(spec: Spec, n: int, rng) -> Dict[str, np.ndarray]:
    return {k: sample_var(t, n, rng) for k, t in spec.vars.items()}


def check_claim(spec: Spec, claim: str, assume: Optional[List[str]] = None, cid: str = "?",
                n: int = 30000, seed: int = 1, min_points: int = 300) -> ClaimResult:
    rng = np.random.default_rng(seed)
    text = claim.strip()
    try:
        if "<=>" in text:
            left, right = text.split("<=>", 1)
            sides = [split_chain(left), split_chain(right)]
        else:
            sides = [split_chain(text)]
        parsed = [([spec.expand(parse_expr_safe(e, spec.names)) for e in ex], ops) for ex, ops in sides]
        asm = [(lambda ex, ops: ([spec.expand(parse_expr_safe(e, spec.names)) for e in ex], ops))(*split_chain(a))
               for a in (assume or [])]
        used = _free_names(*[e for ex, _ in parsed + asm for e in ex])
        missing = [u for u in used if u not in spec.vars]
        if missing:
            raise CheckError("the step uses %s, which is not declared as a variable" % ", ".join(missing))
    except CheckError as e:
        return ClaimResult(cid, text, "unparsed", detail=str(e))
    except Exception as e:  # noqa: BLE001
        return ClaimResult(cid, text, "unparsed", detail="could not read this step: %s" % str(e)[:100])

    syms = [sp.Symbol(u) for u in sorted(spec.vars)]
    evs = [([_lambdify(syms, e) for e in ex], ops) for ex, ops in parsed]
    aevs = [([_lambdify(syms, e) for e in ex], ops) for ex, ops in asm]
    pts: List[Dict[str, float]] = []
    kept = 0
    bad_env = None
    for _ in range(20):
        env = _sample_env(spec, n, rng)
        valid = np.ones(n, bool)
        mask = np.ones(n, bool)
        for fs, ops in aevs:
            vals = [f(env) * np.ones(n) for f in fs]
            ok, _v = _truth(vals, ops, 1e-12)
            valid &= np.all([np.isfinite(v) for v in vals], axis=0)
            mask &= ok
        mask &= valid
        if not mask.any():
            continue
        truths, viols = [], []
        for fs, ops in evs:
            vals = [f(env) * np.ones(n) for f in fs]
            ok, v = _truth(vals, ops)
            fin = np.all([np.isfinite(x) for x in vals], axis=0)
            mask &= fin
            truths.append(ok)
            viols.append(v)
        kept += int(mask.sum())
        if len(truths) == 2:
            wrong = mask & (truths[0] != truths[1])
            amt = np.where(wrong, 1.0, 0.0)
        else:
            wrong = mask & ~truths[0]
            amt = np.where(wrong, viols[0], 0.0)
        if wrong.any():
            order = np.argsort(-amt)[:6]
            for i in order:
                if not wrong[i]:
                    continue
                pt = {k: float(env[k][i]) for k in spec.vars}
                confirmed = False
                if len(parsed) == 2:
                    h0, _ = _hp_truth(*parsed[0], pt)
                    h1, _ = _hp_truth(*parsed[1], pt)
                    confirmed = h0 != h1
                else:
                    h, _w = _hp_truth(*parsed[0], pt)
                    confirmed = not h
                if confirmed:
                    return ClaimResult(cid, text, "false", kept, pt, _explain(parsed[0], pt, len(parsed) == 2))
        if kept >= max(min_points, 20000):
            break
    if kept < 20:
        return ClaimResult(cid, text, "untestable", kept,
                           detail="almost no random points satisfy the conditions of this step, so it could not be tested")
    return ClaimResult(cid, text, "ok", kept)


def _explain(parsed, pt, iff):
    sub = {sp.Symbol(k): sp.Float(v, 30) for k, v in pt.items()}
    exprs, ops = parsed
    vals = [float(sp.N(e.subs(sub), 30)) for e in exprs]
    if iff:
        return "the two sides of the equivalence disagree here"
    shown = []
    for i, op in enumerate(ops):
        shown.append("%.10g %s %.10g" % (vals[i], op, vals[i + 1]))
    return "evaluates to: " + "; ".join(shown) + "  (this is false)"


def _fmt_point(pt):
    return ", ".join("%s=%.6g" % (k, v) for k, v in pt.items())


# --------------------------------------------------------------------------
# Whole-statement scan: is it even true, and where is it tight?
# --------------------------------------------------------------------------


def scan_statement(variables: Dict[str, str], statement: str, lets: Optional[Dict[str, str]] = None,
                   n: int = 120000, seed: int = 3) -> dict:
    spec = Spec(variables, lets)
    res = check_claim(spec, statement, None, "statement", n=min(n, 40000), seed=seed)
    out = {"holds": res.status != "false", "status": res.status, "counterexample": res.example,
           "detail": res.detail, "hint": ""}
    if res.status in ("unparsed", "untestable", "false"):
        return out
    exprs, ops = split_chain(statement)
    parsed = [spec.expand(parse_expr_safe(e, spec.names)) for e in exprs]
    syms = [sp.Symbol(u) for u in sorted(spec.vars)]
    rng = np.random.default_rng(seed + 1)
    env = _sample_env(spec, n, rng)
    notes = []
    for idx in range(1, len(parsed) - 1):
        v = _lambdify(syms, parsed[idx])(env) * np.ones(n)
        fin = np.isfinite(v)
        if not fin.any():
            continue
        vv = np.where(fin, v, np.nan)
        lo, hi = np.nanmin(vv), np.nanmax(vv)
        ilo, ihi = int(np.nanargmin(vv)), int(np.nanargmax(vv))
        lo_pt = {k: float(env[k][ilo]) for k in spec.vars}
        hi_pt = {k: float(env[k][ihi]) for k in spec.vars}
        notes.append("The middle expression ranges over about [%.6g, %.6g] on the tested points. Its largest value is at "
                     "%s and its smallest at %s." % (lo, hi, _fmt_point(hi_pt), _fmt_point(lo_pt)))
        for side, bound_text in ((exprs[0], "lower"), (exprs[-1], "upper")):
            try:
                b = float(sp.N(parse_expr_safe(side, [])))
            except Exception:  # noqa: BLE001
                continue
            gap = (lo - b) if bound_text == "lower" else (b - hi)
            if abs(gap) < 0.01 * max(1.0, abs(b)):
                where = lo_pt if bound_text == "lower" else hi_pt
                notes.append("TIGHT: it gets within %.2g of the %s bound %g near %s, so every step of a proof must stay "
                             "valid there; a bound that is loose near that point will fail." % (abs(gap), bound_text, b,
                                                                                               _fmt_point(where)))
    out["hint"] = " ".join(notes)
    return out


# --------------------------------------------------------------------------
# The AI side: formalise, prove, extract the steps, check, repair
# --------------------------------------------------------------------------

SYNTAX = ("Write every formula in plain computer syntax: * for multiplication, ** for powers, sqrt(), abs(), exp(), log(), "
          "sin(), cos(), pi. Relations are <, <=, >, >=, = (a chain like 1 < f < 2 is allowed). No LaTeX, no words.")

FORMALISE_PROMPT = """[TASK:FORMALISE]
Turn this maths question into a checkable statement. Reply with ONLY a JSON object:
{{"variables": {{"x": "pos"}}, "statement": "1 < f(x) < 2"}}
variables: every free variable with its type: pos (>0), nonneg, unit ((0,1)), real, int, or an interval like "(0,pi)".
statement: the thing to be proved, as ONE relation or chain, using only those variables. {syntax}
If the question is not a statement about variables that can be tested numerically (for example it asks for a value, or
needs an existence proof), reply with exactly {{"variables": {{}}, "statement": ""}}.

QUESTION:
{question}"""

PROOF_PROMPT = """[TASK:PROOF]
Prove the following completely and rigorously. Number the key steps (1), (2), ... Give each step a reason. State every
case. Never skip a step with "clearly" or "obviously".

THE STATEMENT AS CHECKED BY A COMPUTER:
{statement}
{landscape}
Rules learned from numerical testing: every inequality you assert as a step is going to be tested by a computer at tens of
thousands of random points, including values near 0, near 1 and very large. A step that is false at even one point will
be caught. A bound that is too weak near the tight point of the statement is the most common failure, so do not replace
an expression by a looser one unless you are sure the looser bound still works everywhere it is used.

QUESTION:
{question}
{style}"""

EXTRACT_PROMPT = """[TASK:EXTRACT]
Below is a maths proof. List EVERY mathematical claim the proof asserts as a step, so a computer can test each one at
random numbers. Include every inequality, identity and equivalence, including the small ones and the final conclusion.
Do not judge them; just list them faithfully. Reply with ONLY a JSON object:
{{"variables": {{"a": "pos", "x": "pos", "u": "pos"}},
  "let": {{"p": "1/sqrt(1+x)"}},
  "claims": [{{"id": "(1)", "assume": ["u+v<1"], "claim": "t <= (1-r)**2/4"}}]}}
variables: every free variable the claims use, with its type: pos, nonneg, unit ((0,1)), real, int or an interval like "(0,1)".
let: names that are defined from other variables (so they are computed, not sampled). Leave {{}} if none.
claim: one relation or chain, or an equivalence written A <=> B where A and B are relations.
assume: conditions under which the step is claimed (a case, a sub-case, a definition like r = 1 - u - v written as the
relation r = 1-u-v). A claim that holds only inside a case must list the case here.
{syntax}

THE PROOF:
{proof}"""

REPAIR_PROMPT = """[TASK:PROOF]
A computer tested every step of your proof at random numbers. These steps are FALSE or could not be checked:

{problems}

Fix the proof. Do not patch the failing step with a hand-wave. Find an argument whose every step is true for all allowed
values, and re-check each step yourself with a concrete number before you write it. Remember the statement is tight near:
{tight}
Write the complete corrected proof again, with numbered steps, in the same style."""


def _json_in(text: str):
    t = re.sub(r"```(?:json)?", "", text or "")
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except ValueError:
                    return None
    return None


@dataclass
class ProofResult:
    text: str
    verified: bool
    rounds: int = 0
    checked: int = 0
    failed: List[ClaimResult] = field(default_factory=list)
    statement_false: bool = False


def _problems(results: List[ClaimResult]) -> str:
    lines = []
    for r in results:
        if r.status == "false":
            lines.append("- Step %s is FALSE: %s\n    counterexample: %s; %s" % (r.id, r.text, _fmt_point(r.example or {}), r.detail))
        elif r.status in ("unparsed", "untestable"):
            lines.append("- Step %s could not be checked: %s\n    %s" % (r.id, r.text, r.detail))
    return "\n".join(lines)


def _footer(results: List[ClaimResult], rounds: int) -> str:
    ok = [r for r in results if r.status == "ok"]
    skipped = [r for r in results if r.status in ("unparsed", "untestable")]
    pts = min((r.tested for r in ok), default=0)
    s = ("\n\n*Checked by computer: %d of %d steps were tested at random numbers (at least %s points each, including values "
         "near 0, near 1 and very large) and no counterexample was found" % (len(ok), len(results), format(pts, ",")))
    if rounds > 1:
        s += "; the proof was rewritten %d time%s after steps failed the test" % (rounds - 1, "" if rounds == 2 else "s")
    if skipped:
        s += "; %d step%s could not be machine-checked" % (len(skipped), "" if len(skipped) == 1 else "s")
    return s + ". This catches mistakes but is not a formal proof.*"


def _strip_block(text: str) -> str:
    return re.sub(r"```(?:json|claims)?\s*\{.*?\}\s*```", "", text, flags=re.S).strip()


def prove(question: str, ask: Callable[[list, float, int], str], rounds: int = 3,
          emit: Optional[Callable[[str, str], None]] = None, style: str = "", n: int = 30000) -> Optional[ProofResult]:
    """Returns a ProofResult, or None when the question is not a checkable statement (use the normal path then)."""
    say = emit or (lambda k, v: None)
    say("status", "Reading the statement")
    f = _json_in(ask([{"role": "user", "content": FORMALISE_PROMPT.format(question=question[:6000], syntax=SYNTAX)}], 0.0, 600))
    if not f or not str(f.get("statement", "")).strip() or not isinstance(f.get("variables"), dict) or not f["variables"]:
        return None
    variables, statement = {str(k): str(v) for k, v in f["variables"].items()}, str(f["statement"]).strip()
    say("status", "Testing the statement with numbers")
    scan = scan_statement(variables, statement)
    if scan["status"] == "unparsed":
        return None
    if not scan["holds"]:
        return ProofResult(
            "I tested this statement with numbers before trying to prove it, and it is **false**. For example at %s:\n\n%s\n\n"
            "So it cannot be proved as written. If I misread the problem, send it again (or a photo) and I will check it again."
            % (_fmt_point(scan["counterexample"] or {}), scan["detail"]), True, 0, 0, [], True)
    landscape = ("\nWHAT NUMERICAL TESTING SHOWED ABOUT THE STATEMENT (it is true at every tested point): " + scan["hint"] + "\n"
                 if scan["hint"] else "")
    messages = [{"role": "user", "content": PROOF_PROMPT.format(statement=statement, landscape=landscape,
                                                               question=question[:6000], style=style)}]
    best: Optional[Tuple[str, List[ClaimResult]]] = None
    for rnd in range(1, rounds + 1):
        say("status", "Writing the proof" if rnd == 1 else "Rewriting the proof (round %d)" % rnd)
        proof = ask(messages, 0.2, 12000)
        say("status", "Checking every step with numbers")
        spec_json = _json_in(ask([{"role": "user", "content": EXTRACT_PROMPT.format(proof=proof[:30000], syntax=SYNTAX)}], 0.0, 4000))
        results: List[ClaimResult] = []
        if spec_json and isinstance(spec_json.get("claims"), list):
            try:
                spec = Spec({str(k): str(v) for k, v in (spec_json.get("variables") or {}).items()} or variables,
                            {str(k): str(v) for k, v in (spec_json.get("let") or {}).items()})
            except Exception:  # noqa: BLE001
                spec = None
            if spec is not None:
                for i, c in enumerate(spec_json["claims"][:60]):
                    if not isinstance(c, dict):
                        continue
                    results.append(check_claim(spec, str(c.get("claim", "")),
                                               [str(a) for a in (c.get("assume") or [])][:8],
                                               str(c.get("id") or i + 1), n=n))
        bad = [r for r in results if r.status == "false"]
        unchecked = [r for r in results if r.status in ("unparsed", "untestable")]
        score = (len(bad), len(unchecked))
        if best is None or score < (sum(1 for r in best[1] if r.status == "false"), sum(1 for r in best[1] if r.status in ("unparsed", "untestable"))):
            best = (proof, results)
        if results and not bad and len(unchecked) <= max(1, len(results) // 4):
            return ProofResult(_strip_block(proof) + _footer(results, rnd), True, rnd, len(results))
        if rnd == rounds:
            break
        problems = _problems(bad + unchecked) if results else "- The list of steps could not be read. Write numbered steps and a single clear conclusion."
        messages = messages + [{"role": "assistant", "content": proof},
                               {"role": "user", "content": REPAIR_PROMPT.format(problems=problems, tight=scan["hint"] or "(no tight point found)")}]
    proof, results = best  # type: ignore[misc]
    bad = [r for r in results if r.status == "false"]
    warn = ("**I could not produce a fully verified proof.** I tested every step with numbers and %s still failed. "
            "Treat the argument below as a draft, not a proof.\n\n" % (
                ("%d step%s" % (len(bad), "" if len(bad) == 1 else "s")) if bad else "some steps could not be checked"))
    if bad:
        warn += "Failing steps:\n" + "\n".join("- %s: %s (false at %s)" % (r.id, r.text, _fmt_point(r.example or {})) for r in bad[:6]) + "\n\n"
    return ProofResult(warn + _strip_block(proof), False, rounds, len(results), bad)


# --------------------------------------------------------------------------
# Self-test and demo (no AI model needed)
# --------------------------------------------------------------------------

PROBLEM = "1 < 1/sqrt(1+x) + 1/sqrt(1+a) + sqrt(a*x/(a*x+8)) < 2"


def _bad_proof_spec() -> dict:
    """The steps of the flawed proof you showed me."""
    return {"variables": {"a": "pos", "x": "pos", "u": "pos", "p": "pos", "q": "pos"}, "let": {}, "claims": [
        {"id": "2.1", "claim": "sqrt(u/(u+8)) <= (u+4)/(u+8)"},
        {"id": "(4)", "claim": "1/sqrt(1+x)+1/sqrt(1+a)+(a*x+4)/(a*x+8) < 2"},
        {"id": "(5)", "claim": "2-(a*x+4)/(a*x+8) = 4/(a*x+8)"},
        {"id": "(6)", "claim": "1/sqrt(1+p)+1/sqrt(1+q) <= 2/sqrt((1+p)*(1+q))"},
        {"id": "(8)", "claim": "sqrt((1+x)*(1+a)) > (a*x+8)/2"},
        {"id": "3", "claim": "2/sqrt(1+x) > 2"},
    ]}


def _good_proof_spec() -> dict:
    """A correct proof (p = 1/sqrt(1+x), q = 1/sqrt(1+a)), written as checkable steps."""
    return {"variables": {"a": "pos", "x": "pos", "u": "pos", "v": "pos"},
            "let": {"p": "1/sqrt(1+x)", "q": "1/sqrt(1+a)"}, "claims": [
                {"id": "T<1", "claim": "sqrt(a*x/(a*x+8)) < 1"},
                {"id": "id", "claim": "a*x/(a*x+8) = (1-p**2)*(1-q**2)/((1-p**2)*(1-q**2)+8*p**2*q**2)"},
                {"id": "U-equiv", "assume": ["p+q>1"],
                 "claim": "sqrt(a*x/(a*x+8)) < 2-p-q <=> (p+q-1)*(3-p-q)*(1-p**2)*(1-q**2) < 8*p**2*q**2*(2-p-q)**2"},
                {"id": "L-equiv", "assume": ["p+q<1"],
                 "claim": "sqrt(a*x/(a*x+8)) > 1-p-q <=> (1-p**2)*(1-q**2)*(p+q)*(2-p-q) > 8*p**2*q**2*(1-p-q)**2"},
                {"id": "L1", "assume": ["p+q<1"], "claim": "(1-p**2)*(1-q**2) > p*q"},
                {"id": "L2", "assume": ["p+q<1"], "claim": "(p+q)*(2-p-q) > 2*sqrt(p*q)"},
                {"id": "L3", "assume": ["p+q<1"], "claim": "4*sqrt(p*q)*(1-p-q)**2 < 1"},
                {"id": "L4", "assume": ["p+q<1"], "claim": "(1-p**2)*(1-q**2)*(p+q)*(2-p-q) > 8*p**2*q**2*(1-p-q)**2"},
            ]}


def _good_uv_spec() -> dict:
    """Upper bound, case p+q>1, written with u = 1-p, v = 1-q (u+v<1), t = uv, r = 1-u-v."""
    return {"variables": {"u": "unit", "v": "unit"}, "let": {"t": "u*v", "r": "1-u-v"}, "claims": [
        {"id": "amgm", "assume": ["u+v<1"], "claim": "t <= (1-r)**2/4"},
        {"id": "key", "assume": ["u+v<1"], "claim": "8*(1-r)**2*(r+t)**2 > r*(2-r)*t*(2+2*r+t)"},
        {"id": "A1", "assume": ["u+v<1", "r>=1/6"], "claim": "r*(2-r)*t*(2+2*r+t) <= 5/4*r*(1-r)**2"},
        {"id": "A2", "assume": ["u+v<1", "r>=1/6"], "claim": "8*(1-r)**2*(r+t)**2 >= 8*(1-r)**2*r**2"},
        {"id": "A3", "assume": ["u+v<1", "r>=1/6"], "claim": "8*(1-r)**2*r**2 > 5/4*r*(1-r)**2"},
        {"id": "B1", "assume": ["u+v<1", "r<1/6"], "claim": "8*(1-r)**2*(r+t)**2 >= 32*(1-r)**2*r*t"},
        {"id": "B2", "assume": ["u+v<1", "r<1/6"], "claim": "r*(2-r)*t*(2+2*r+t) <= 5.2*r*t"},
        {"id": "B3", "assume": ["u+v<1", "r<1/6"], "claim": "32*(1-r)**2 > 5.2"},
    ]}


def run_spec(spec_dict: dict) -> List[ClaimResult]:
    spec = Spec(spec_dict["variables"], spec_dict.get("let"))
    return [check_claim(spec, c["claim"], c.get("assume"), c["id"]) for c in spec_dict["claims"]]


def _selftest() -> None:
    # 1. every false step of the flawed proof is caught, and the one true step is not flagged
    res = {r.id: r for r in run_spec(_bad_proof_spec())}
    assert res["2.1"].status == "ok", res["2.1"]
    for k in ("(4)", "(5)", "(6)", "(8)", "3"):
        assert res[k].status == "false" and res[k].example, (k, res[k])
    # 2. a correct proof passes every step (including the equivalences)
    for r in run_spec(_good_proof_spec()) + run_spec(_good_uv_spec()):
        assert r.status == "ok", (r.id, r.status, r.example, r.detail)
        assert r.tested >= 300, (r.id, r.tested)
    # 3. the statement scan: true, and tight at both ends
    s = scan_statement({"a": "pos", "x": "pos"}, PROBLEM)
    assert s["holds"] and "TIGHT" in s["hint"], s
    f = scan_statement({"x": "pos"}, "x**2 > x")
    assert not f["holds"] and f["counterexample"]["x"] < 1
    # 4. unsafe or unreadable input is refused, not executed
    for bad in ("__import__('os').system('x') < 1", "open('f') < 2", "x.__class__ < 1"):
        r = check_claim(Spec({"x": "pos"}), bad)
        assert r.status == "unparsed", (bad, r)
    assert check_claim(Spec({"x": "pos"}), "y > 0").status == "unparsed"
    assert check_claim(Spec({"x": "pos"}), "x > 1", ["x < 0"]).status in ("untestable",)

    # 5. the full loop with a scripted fake model: bad proof first, then a good one
    flawed = {"variables": {"a": "pos", "x": "pos"}, "let": {}, "claims": [
        {"id": "(4)", "claim": "1/sqrt(1+x)+1/sqrt(1+a)+(a*x+4)/(a*x+8) < 2"}]}
    good = {"variables": {"a": "pos", "x": "pos", "u": "unit", "v": "unit"}, "let": {"p": "1/sqrt(1+x)", "q": "1/sqrt(1+a)"},
            "claims": [{"id": "L1", "assume": ["p+q<1"], "claim": "(1-p**2)*(1-q**2) > p*q"}]}
    state = {"proofs": 0, "extracts": 0, "prompts": []}

    def fake(messages, temperature, max_tokens):
        last = messages[-1]["content"]
        state["prompts"].append(last)
        if "[TASK:FORMALISE]" in last:
            return json.dumps({"variables": {"a": "pos", "x": "pos"}, "statement": PROBLEM})
        if "[TASK:PROOF]" in last and "FALSE" not in last:
            state["proofs"] += 1
            return "FIRST ATTEMPT proof, step (4): weaker bound.\n```json\n{}\n```"
        if "[TASK:PROOF]" in last:
            state["proofs"] += 1
            return "CORRECTED proof with p = 1/sqrt(1+x)."
        if "[TASK:EXTRACT]" in last:
            state["extracts"] += 1
            return json.dumps(flawed if "FIRST ATTEMPT" in last else good)
        raise AssertionError("unexpected prompt")

    status = []
    out = prove("Prove that for all a, x > 0: " + PROBLEM, fake, emit=lambda k, v: status.append(v))
    assert out and out.verified and out.rounds == 2 and state["proofs"] == 2, (out, state["proofs"])
    assert "CORRECTED" in out.text and "FIRST ATTEMPT" not in out.text and "Checked by computer" in out.text
    assert any("counterexample" in p and "(4)" in p for p in state["prompts"]), "the failing step must be sent back"
    assert "Rewriting the proof (round 2)" in status

    # 6. a false statement gets a counterexample, not a proof; a non-checkable question falls through
    def fake_false(messages, temperature, max_tokens):
        return json.dumps({"variables": {"x": "pos"}, "statement": "x**2 > x"})
    r = prove("Prove that x^2 > x for all x > 0", fake_false)
    assert r and r.statement_false and "false" in r.text
    assert prove("What is 2+2?", lambda m, t, k: json.dumps({"variables": {}, "statement": ""})) is None

    # 7. a model that never fixes its proof is reported honestly
    def fake_stubborn(messages, temperature, max_tokens):
        last = messages[-1]["content"]
        if "[TASK:FORMALISE]" in last:
            return json.dumps({"variables": {"a": "pos", "x": "pos"}, "statement": PROBLEM})
        if "[TASK:EXTRACT]" in last:
            return json.dumps(flawed)
        return "STUBBORN proof"
    st = prove("Prove it", fake_stubborn, rounds=2)
    assert st and not st.verified and "could not produce a fully verified proof" in st.text and st.failed
    print("proof_checker selftest passed")


def _demo() -> None:
    print("Steps of the flawed proof, tested with numbers:\n")
    for r in run_spec(_bad_proof_spec()):
        print("  %-5s %-6s %s" % (r.id, r.status.upper(), r.text))
        if r.status == "false":
            print("         at %s -> %s" % (_fmt_point(r.example), r.detail))
    print("\nSteps of a correct proof:\n")
    for r in run_spec(_good_proof_spec()) + run_spec(_good_uv_spec()):
        print("  %-8s %-4s (%s points) %s" % (r.id, r.status.upper(), format(r.tested, ","), r.text[:70]))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    elif "--demo" in sys.argv:
        _demo()
    else:
        print(__doc__)
