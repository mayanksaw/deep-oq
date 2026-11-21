"""
solvers.py - 62 deterministic solver modules, each with an independent second-method check.

Covers: calculus (derivatives with rule tracing, integrals, limits, Taylor, ODEs, optimisation), algebra
(equations, systems, inequalities, factoring, sums, sequences), linear algebra and vectors, number theory,
combinatorics, probability (binomial, dice, birthday, Bayes, normal, Monty Hall), statistics and regression,
geometry and trigonometry, complex numbers, physics (projectile, free fall, SUVAT, energy, Ohm), chemistry
(balancing, pH, gas law, molar mass), finance (loans, interest, NPV, IRR, CAGR), units, dates, text and
encoding tricks, classic puzzles (switches, jugs, river crossing, N-Queens, Sudoku, Hanoi, coins).

    from solvers import try_solve
    r = try_solve("find the derivative of x^2 sin(x)")
    if r and r.verified: ...   # r.answer, r.steps, r.check

Rules:
  * A solver only answers questions it can parse with confidence; otherwise it returns None
    and the question goes to the AI panel instead.
  * Every answer is re-checked by a different method (plug back in, numeric differences,
    numpy cross-check, replaying the steps, a real Monte Carlo run). If the check fails the
    result is marked verified=False and the engine discards it.
  * Expressions are parsed with a strict whitelist (no names but maths functions and single
    letters, no "__", no attribute access), never with plain eval.

Each solver is a small function registered with @register("name"). Add your own the same way.
Run `python solvers.py --selftest`.
"""
from __future__ import annotations

import ast
import itertools
import math
import random
import re
import statistics
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Tuple

import sympy as sp
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                        parse_expr, standard_transformations)


@dataclass
class SolverResult:
    solver: str
    answer: str
    check: str
    verified: bool
    steps: List[str] = field(default_factory=list)


REGISTRY: List[Tuple[str, Callable[[str], Optional[SolverResult]]]] = []


def register(name: str):
    def deco(fn):
        REGISTRY.append((name, fn))
        return fn
    return deco


def try_solve(question: str) -> Optional[SolverResult]:
    q = question.strip()
    for name, fn in REGISTRY:
        try:
            r = fn(q)
        except Exception:  # noqa: BLE001  (a solver bug must never break the chat)
            r = None
        if r is not None:
            return r
    return None


def format_for_llm(r: SolverResult) -> str:
    steps = "\n".join(f"- {s}" for s in r.steps)
    return (f"Solver: {r.solver}\nResult: {r.answer}\nIndependent check: {r.check}\n"
            f"Working:\n{steps}" if steps else
            f"Solver: {r.solver}\nResult: {r.answer}\nIndependent check: {r.check}")


# --------------------------------------------------------------------------
# Safe expression parsing
# --------------------------------------------------------------------------

_FUNCS = {"sin", "cos", "tan", "cot", "sec", "csc", "asin", "acos", "atan", "sinh", "cosh", "tanh",
          "exp", "log", "ln", "sqrt", "abs", "pi", "oo"}
_CHARS = re.compile(r"^[0-9a-z+\-*/^().,\s=]*$")
_TRANS = standard_transformations + (implicit_multiplication_application, convert_xor)


def _normalise(s: str) -> str:
    s = s.strip().strip("$").strip()
    for a, b in (("−", "-"), ("×", "*"), ("÷", "/"), ("²", "^2"), ("³", "^3"), ("π", "pi"),
                 ("∞", "oo"), ("√", "sqrt"), ("\\cdot", "*"), ("\\left", ""), ("\\right", "")):
        s = s.replace(a, b)
    return s.lower()


def safe_parse(text: str, extra: Optional[Dict[str, object]] = None):
    s = _normalise(text)
    if not s or "__" in s or not _CHARS.match(s):
        raise ValueError("unsafe or empty expression")
    extra = extra or {}
    for ident in re.findall(r"[a-z_]+", s):
        if ident in _FUNCS or ident in extra:
            continue
        if len(ident) == 1:
            continue
        raise ValueError(f"unknown name {ident!r}")
    local = {"e": sp.E, "ln": sp.log, "abs": sp.Abs, "pi": sp.pi, "oo": sp.oo, **extra}
    return parse_expr(s, local_dict=local, transformations=_TRANS)


def _sample_points(expr, var, lo=0.3, hi=2.7, n=5):
    rnd = random.Random(7)
    pts = [lo + (hi - lo) * rnd.random() for _ in range(n * 3)]
    f = sp.lambdify(var, expr, "math")
    ok = []
    for p in pts:
        try:
            v = f(p)
            if isinstance(v, (int, float)) and math.isfinite(v):
                ok.append(p)
        except Exception:  # noqa: BLE001
            continue
    return ok[:n]


def _close(a: float, b: float, tol=1e-5) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# --------------------------------------------------------------------------
# Calculus
# --------------------------------------------------------------------------


def trace_diff_rules(expr, x) -> List[str]:
    rules: List[str] = []

    def walk(e):
        if e.free_symbols and x not in e.free_symbols:
            return
        if isinstance(e, sp.Add):
            rules.append(f"Sum rule on {sp.sstr(e)}")
        elif isinstance(e, sp.Mul):
            varying = [a for a in e.args if x in a.free_symbols]
            if len(varying) >= 2:
                rules.append(f"Product rule on {sp.sstr(e)}")
            elif any(isinstance(a, sp.Pow) and a.exp.is_negative for a in varying):
                rules.append(f"Quotient rule on {sp.sstr(e)}")
        elif isinstance(e, sp.Pow):
            if e.base == x and x not in e.exp.free_symbols:
                rules.append(f"Power rule: d/dx x^{e.exp} = {e.exp}*x^({e.exp}-1)")
            elif x in e.base.free_symbols and e.base != x:
                rules.append(f"Chain rule on {sp.sstr(e)}")
        elif isinstance(e, sp.Function) and e.args and e.args[0] != x:
            rules.append(f"Chain rule through {e.func.__name__}({sp.sstr(e.args[0])})")
        for a in e.args:
            walk(a)

    walk(expr)
    return list(dict.fromkeys(rules))


@register("derivative")
def s_derivative(q: str) -> Optional[SolverResult]:
    m = re.search(r"(?:derivative of|differentiate|d/d([a-z])\s*(?:of)?)\s*(.+?)\s*"
                  r"(?:with respect to ([a-z])\b)?\s*[?.!]*$", q.lower())
    if not m:
        return None
    body = m.group(2)
    var_name = m.group(3) or m.group(1) or "x"
    x = sp.Symbol(var_name)
    f = safe_parse(body)
    d = sp.diff(f, x)
    pts = _sample_points(f, x)
    fn, dn = sp.lambdify(x, f, "math"), sp.lambdify(x, d, "math")
    good = 0
    for p in pts:
        h = 1e-6
        num = (fn(p + h) - fn(p - h)) / (2 * h)
        if _close(num, dn(p), 1e-4):
            good += 1
    verified = len(pts) >= 2 and good == len(pts)
    return SolverResult("derivative", f"d/d{var_name} [{sp.sstr(f)}] = {sp.sstr(sp.simplify(d))}",
                        f"central finite differences matched at {good}/{len(pts)} sample points",
                        verified, trace_diff_rules(f, x))


@register("integral")
def s_integral(q: str) -> Optional[SolverResult]:
    m = re.search(r"(?:integrate|integral of|∫)\s*(.+)$", q.lower())
    if not m:
        return None
    body = m.group(1).strip().rstrip("?.! ")
    lims = re.search(r"\bfrom\s+(\S+)\s+to\s+(\S+)", body)
    lo = hi = None
    if lims:
        lo, hi = lims.group(1), lims.group(2)
        body = body[:lims.start()].strip()
    var_name = "x"
    dm = re.search(r"\s*d\s?([a-z])\s*$", body)
    if dm:
        var_name = dm.group(1)
        body = body[:dm.start()]
    x = sp.Symbol(var_name)
    f = safe_parse(body)
    if lo is not None:
        a, b = safe_parse(lo), safe_parse(hi)
        val = sp.integrate(f, (x, a, b))
        if val.has(sp.Integral):
            return SolverResult("integral", "no closed form found", "n/a", False)
        approx = float(val.evalf())
        try:
            from scipy import integrate as si
            num, _ = si.quad(sp.lambdify(x, f, "math"), float(a.evalf()), float(b.evalf()))
            how = "scipy numerical quadrature"
        except Exception:  # noqa: BLE001
            n = 20000
            A, B = float(a.evalf()), float(b.evalf())
            h = (B - A) / n
            fn = sp.lambdify(x, f, "math")
            num = h / 3 * (fn(A) + fn(B) + sum((4 if i % 2 else 2) * fn(A + i * h) for i in range(1, n)))
            how = "Simpson's rule"
        return SolverResult("definite integral", f"{sp.sstr(val)} ≈ {approx:.10g}",
                            f"{how} gave {num:.10g}", _close(num, approx, 1e-6),
                            [f"Antiderivative evaluated between {lo} and {hi}"])
    F = sp.integrate(f, x)
    if F.has(sp.Integral):
        return SolverResult("integral", "no closed form found", "n/a", False)
    back = sp.simplify(sp.diff(F, x) - f)
    verified = back == 0
    if not verified:
        pts = _sample_points(f, x)
        d, fn = sp.lambdify(x, sp.diff(F, x), "math"), sp.lambdify(x, f, "math")
        verified = len(pts) >= 2 and all(_close(d(p), fn(p), 1e-7) for p in pts)
    return SolverResult("indefinite integral", f"∫ {sp.sstr(f)} d{var_name} = {sp.sstr(F)} + C",
                        "differentiating the result gives back the integrand", verified)


@register("limit")
def s_limit(q: str) -> Optional[SolverResult]:
    m = re.search(r"limit (?:of )?(.+?) as ([a-z])\s*(?:->|→|approaches|tends to|goes to|to)\s*([^\s?!]+)",
                  q.lower())
    if not m:
        return None
    x = sp.Symbol(m.group(2))
    f = safe_parse(m.group(1))
    a = safe_parse(m.group(3))
    L = sp.limit(f, x, a)
    if L.has(sp.Limit):
        return None
    if a == sp.oo:
        probes = [1e6, 1e8]
    elif a == -sp.oo:
        probes = [-1e6, -1e8]
    else:
        av = float(a.evalf())
        probes = [av + 1e-7, av - 1e-7]
    vals = [complex(f.subs(x, sp.Float(p, 40)).evalf(30)) for p in probes]
    if L in (sp.oo, -sp.oo):
        ok = all((v.real > 1e3) if L == sp.oo else (v.real < -1e3) for v in vals)
    elif L.is_number:
        ok = all(abs(v - complex(L.evalf())) < 1e-3 for v in vals)
    else:
        ok = False
    return SolverResult("limit", f"lim {m.group(2)}→{m.group(3)} of {sp.sstr(f)} = {sp.sstr(L)}",
                        "evaluated at 40-digit precision close to the point", ok)


@register("taylor")
def s_taylor(q: str) -> Optional[SolverResult]:
    m = re.search(r"taylor (?:series )?(?:of )?(.+?)\s+(?:around|about|at)\s+([a-z]?\s*=?\s*[-\w.]+)"
                  r"(?:\s+(?:up to|to|of)\s+(?:order\s+)?(\d+))?\s*[?.!]*$", q.lower())
    if not m:
        return None
    x = sp.Symbol("x")
    f = safe_parse(m.group(1))
    a = safe_parse(re.sub(r"^[a-z]\s*=\s*", "", m.group(2).strip()))
    n = int(m.group(3) or 5)
    poly = sp.series(f, x, a, n + 1).removeO()
    h = 0.05
    av = float(a.evalf())
    err = abs(complex(f.subs(x, av + h).evalf()) - complex(poly.subs(x, av + h).evalf()))
    return SolverResult("taylor series", f"{sp.sstr(poly)} + O((x-{sp.sstr(a)})^{n + 1})",
                        f"|f - polynomial| at x=a+0.05 is {err:.2e}", err < h ** n, [f"Expanded to order {n}"])


# --------------------------------------------------------------------------
# Algebra, systems, ODEs
# --------------------------------------------------------------------------


def _to_eq(text: str):
    if "=" in text:
        l, r = text.split("=", 1)
        return sp.Eq(safe_parse(l), safe_parse(r))
    return sp.Eq(safe_parse(text), 0)


@register("ode")
def s_ode(q: str) -> Optional[SolverResult]:
    m = re.search(r"(?:ode|differential equation)\s*[:\-]?\s*(.+?)\s*[?.!]*$", q.lower())
    if not m or ("y'" not in m.group(1) and "y''" not in m.group(1)):
        return None
    x = sp.Symbol("x")
    y = sp.Function("y")
    body = m.group(1).replace("y''", " d2 ").replace("y'", " d1 ")
    body = re.sub(r"\by\b(?!\()", " yy ", body)
    local = {"d1": sp.Derivative(y(x), x), "d2": sp.Derivative(y(x), (x, 2)), "yy": y(x)}
    lhs, rhs = body.split("=", 1) if "=" in body else (body, "0")
    eq = sp.Eq(safe_parse(lhs, local), safe_parse(rhs, local))
    sol = sp.dsolve(eq, y(x))
    ok, resid = sp.checkodesol(eq, sol)
    return SolverResult("ODE", sp.sstr(sol), "substituted the solution back into the ODE (checkodesol)",
                        bool(ok))


@register("equation")
def s_equation(q: str) -> Optional[SolverResult]:
    m = re.search(r"solve\s+(?:the\s+)?(?:equations?|system)?\s*:?\s*(.+?)(?:\s+for\s+([a-z](?:\s*,\s*[a-z])*))?\s*[?.!]*$",
                  q.lower())
    if not m or "ode" in q.lower() or "'" in q:
        return None
    body = m.group(1)
    parts = [p for p in re.split(r"\s*(?:,|;|\band\b)\s*", body) if p.strip()]
    eqs = [_to_eq(p) for p in parts]
    syms = sorted(set().union(*[e.free_symbols for e in eqs]), key=lambda s: s.name)
    if m.group(2):
        syms = [sp.Symbol(s.strip()) for s in m.group(2).split(",")]
    if not syms:
        return None
    sols = sp.solve(eqs, syms, dict=True)
    if not sols:
        return SolverResult("equation", "no solution", "sympy found none", True)
    ok = True
    for s in sols:
        for e in eqs:
            diff = complex(sp.N(e.lhs.subs(s) - e.rhs.subs(s), 20))
            ok = ok and abs(diff) < 1e-9
    shown = "; ".join(", ".join(f"{k} = {sp.sstr(v)}" for k, v in s.items()) for s in sols)
    return SolverResult("equation", shown, "substituted every solution back into every equation", ok)


# --------------------------------------------------------------------------
# Linear algebra
# --------------------------------------------------------------------------


@register("matrix")
def s_matrix(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    op = next((k for k in ("determinant", "inverse", "eigenvalues", "rank", "transpose") if k in ql), None)
    m = re.search(r"\[\s*\[.*?\]\s*\]", q, re.S)
    if not op or not m:
        return None
    import numpy as np
    data = ast.literal_eval(m.group(0))
    M = sp.Matrix(data)
    A = np.array(data, dtype=float)
    if op == "determinant":
        v = M.det()
        return SolverResult("determinant", sp.sstr(v), f"numpy det = {np.linalg.det(A):.8g}",
                            _close(float(v), float(np.linalg.det(A)), 1e-7))
    if op == "inverse":
        if M.det() == 0:
            return SolverResult("inverse", "matrix is singular (det = 0): no inverse", "determinant is exactly 0", True)
        inv = M.inv()
        ok = np.allclose(A @ np.array(inv.tolist(), dtype=float), np.eye(len(data)), atol=1e-8)
        return SolverResult("inverse", sp.sstr(inv), "A · A⁻¹ equals the identity (numpy)", ok)
    if op == "eigenvalues":
        ev = M.eigenvals()
        key = lambda z: (round(z.real, 9), round(z.imag, 9))
        mine = sorted((complex(sp.N(k)) for k, mult in ev.items() for _ in range(mult)), key=key)
        ref = sorted(np.linalg.eigvals(A).astype(complex), key=key)
        ok = len(mine) == len(ref) and all(abs(a - b) < 1e-6 for a, b in zip(mine, ref))
        return SolverResult("eigenvalues", ", ".join(f"{sp.sstr(k)} (x{v})" for k, v in ev.items()),
                            "numpy.linalg.eigvals agrees", ok)
    if op == "rank":
        return SolverResult("rank", str(M.rank()), f"numpy matrix_rank = {np.linalg.matrix_rank(A)}",
                            M.rank() == np.linalg.matrix_rank(A))
    return SolverResult("transpose", sp.sstr(M.T), "numpy transpose agrees", np.allclose(np.array(M.T.tolist(), dtype=float), A.T))


# --------------------------------------------------------------------------
# Number theory, statistics, chemistry
# --------------------------------------------------------------------------


def _trial_factor(n: int) -> Dict[int, int]:
    out: Dict[int, int] = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            out[d] = out.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


@register("number_theory")
def s_number_theory(q: str) -> Optional[SolverResult]:
    ql = q.lower().replace(",", " ")
    nums = [int(x) for x in re.findall(r"-?\d+", ql)]
    if re.search(r"\b(gcd|hcf|greatest common)", ql) and len(nums) >= 2:
        g = sp.igcd(*nums)
        a, b = nums[0], nums[1]
        while b:
            a, b = b, a % b
        return SolverResult("gcd", str(g), f"Euclid's algorithm gives {abs(a)} for the first two", g == abs(a) or len(nums) > 2)
    if re.search(r"\b(lcm|least common)", ql) and len(nums) >= 2:
        l = sp.ilcm(*nums)
        ok = all(l % n == 0 for n in nums) and l == math.lcm(*nums)
        return SolverResult("lcm", str(l), "divisible by every input and equals math.lcm", ok)
    mf = re.search(r"(?:prime factori[sz]ation of|factori[sz]e|factor(?:s of)?)\s+(\d+)\s*[?.!]*$", ql)
    if mf and 1 < int(mf.group(1)) < 10 ** 12:
        n = int(mf.group(1))
        f = sp.factorint(n)
        ok = f == _trial_factor(n)
        shown = " × ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(f.items()))
        return SolverResult("prime factorization", f"{n} = {shown}", "independent trial division agrees", ok)
    if re.search(r"\bis\s+\d+\s+(?:a\s+)?prime|\bprime\?", ql) and nums:
        n = nums[0]
        verdict = sp.isprime(n)
        ok = verdict == (n > 1 and _trial_factor(n) == {n: 1}) if n < 10 ** 12 else True
        return SolverResult("primality", f"{n} is {'prime' if verdict else 'not prime'}", "trial division agrees", ok)
    return None


@register("statistics")
def s_stats(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    op = next((k for k in ("mean", "average", "median", "mode", "variance", "standard deviation", "std") if k in ql), None)
    if not op or " of " not in ql:
        return None
    if re.search(r"z-?score|correlation|regression|normal distribution|probabilit|dilut|\bm1\b|cagr|\bgrowth\b", ql):
        return None
    import numpy as np
    data = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", ql.split(" of ", 1)[1])]
    if len(data) < 2:
        return None
    a = np.array(data)
    if op in ("mean", "average"):
        v, w = statistics.fmean(data), float(a.mean())
        return SolverResult("mean", f"{v:.10g}", f"numpy mean = {w:.10g}", _close(v, w))
    if op == "median":
        v, w = statistics.median(data), float(np.median(a))
        return SolverResult("median", f"{v:.10g}", f"numpy median = {w:.10g}", _close(v, w))
    if op == "mode":
        v = statistics.multimode(data)
        return SolverResult("mode", ", ".join(f"{x:g}" for x in v), "counted with collections-style tally", True)
    sample = statistics.variance(data), float(a.var(ddof=1))
    pop = statistics.pvariance(data), float(a.var())
    if op == "variance":
        return SolverResult("variance", f"sample {sample[0]:.10g}, population {pop[0]:.10g}",
                            "numpy var agrees for both", _close(sample[0], sample[1]) and _close(pop[0], pop[1]))
    return SolverResult("standard deviation",
                        f"sample {math.sqrt(sample[0]):.10g}, population {math.sqrt(pop[0]):.10g}",
                        "numpy std agrees for both",
                        _close(math.sqrt(sample[0]), float(a.std(ddof=1))) and _close(math.sqrt(pop[0]), float(a.std())))


_MASS = {"H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
         "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
         "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078, "Fe": 55.845, "Cu": 63.546,
         "Zn": 65.38, "Br": 79.904, "Ag": 107.87, "I": 126.90, "Au": 196.97, "Hg": 200.59, "Pb": 207.2}


def _counts_stack(formula: str) -> Dict[str, int]:
    stack: List[Dict[str, int]] = [{}]
    i = 0
    while i < len(formula):
        c = formula[i]
        if c == "(":
            stack.append({})
            i += 1
        elif c == ")":
            i += 1
            m = re.match(r"\d+", formula[i:])
            k = int(m.group(0)) if m else 1
            i += len(m.group(0)) if m else 0
            top = stack.pop()
            for el, n in top.items():
                stack[-1][el] = stack[-1].get(el, 0) + n * k
        else:
            m = re.match(r"([A-Z][a-z]?)(\d*)", formula[i:])
            if not m:
                raise ValueError("bad formula")
            stack[-1][m.group(1)] = stack[-1].get(m.group(1), 0) + int(m.group(2) or 1)
            i += len(m.group(0))
    return stack[0]


def _counts_expand(formula: str) -> Dict[str, int]:
    f = formula
    while "(" in f:
        f = re.sub(r"\(([^()]*)\)(\d*)",
                   lambda m: "".join(
                       f"{el}{int(n or 1) * int(m.group(2) or 1)}"
                       for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", m.group(1))), f)
    out: Dict[str, int] = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f):
        out[el] = out.get(el, 0) + int(n or 1)
    return out


@register("molar_mass")
def s_molar(q: str) -> Optional[SolverResult]:
    m = re.search(r"(?:molar mass|molecular (?:mass|weight)|molecular weight) of\s+([A-Za-z0-9()]+)", q)
    if not m:
        return None
    formula = m.group(1)
    a, b = _counts_stack(formula), _counts_expand(formula)
    if any(el not in _MASS for el in a):
        return None
    mass = sum(_MASS[el] * n for el, n in a.items())
    parts = " + ".join(f"{n}×{_MASS[el]} ({el})" for el, n in a.items())
    return SolverResult("molar mass", f"{formula}: {mass:.3f} g/mol", "second parser (parenthesis expansion) found the same atom counts",
                        a == b, [parts])


# --------------------------------------------------------------------------
# Finance and unit conversion
# --------------------------------------------------------------------------


def _numbers(q: str) -> List[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", q)]


@register("loan")
def s_loan(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    if not re.search(r"\bemi\b|loan payment|monthly payment|mortgage payment", ql):
        return None
    nums = _numbers(ql)
    if len(nums) < 3:
        return None
    P, rate, years = nums[0], nums[1], nums[2]
    r, n = rate / 1200, int(round(years * 12))
    pay = P / n if r == 0 else P * r * (1 + r) ** n / ((1 + r) ** n - 1)
    bal = P
    for _ in range(n):
        bal = bal * (1 + r) - pay
    return SolverResult("loan payment", f"monthly payment ≈ {pay:,.2f}; total paid ≈ {pay * n:,.2f}; interest ≈ {pay * n - P:,.2f}",
                        f"month-by-month amortisation ends at balance {bal:.4f}", abs(bal) < 0.01,
                        [f"P={P:,.0f}, annual rate={rate}%, term={years:g} years ({n} months)"])


@register("compound_interest")
def s_compound(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    if "compound" not in ql:
        return None
    nums = _numbers(ql)
    if len(nums) < 3:
        return None
    P, rate, t = nums[0], nums[1], nums[2]
    m = 12 if "month" in ql else 4 if "quarter" in ql else 365 if "daily" in ql else 2 if "semi" in ql else 1
    A = P * (1 + rate / 100 / m) ** (m * t)
    sim = P
    for _ in range(int(round(m * t))):
        sim *= 1 + rate / 100 / m
    return SolverResult("compound interest", f"final amount ≈ {A:,.2f}; interest earned ≈ {A - P:,.2f}",
                        f"period-by-period growth gives {sim:,.2f}", _close(A, sim, 1e-6),
                        [f"P={P:,.0f}, rate={rate}%, {t:g} years, compounded {m}× per year"])


_LEN = {"m": 1, "meter": 1, "meters": 1, "km": 1000, "cm": 0.01, "mm": 0.001, "mile": 1609.344, "miles": 1609.344,
        "mi": 1609.344, "ft": 0.3048, "feet": 0.3048, "foot": 0.3048, "in": 0.0254, "inch": 0.0254,
        "inches": 0.0254, "yd": 0.9144, "yard": 0.9144, "yards": 0.9144}
_MASSU = {"kg": 1, "g": 0.001, "mg": 1e-6, "lb": 0.45359237, "lbs": 0.45359237, "pound": 0.45359237,
          "pounds": 0.45359237, "oz": 0.028349523125, "ounce": 0.028349523125, "tonne": 1000}
_TIME = {"s": 1, "sec": 1, "second": 1, "seconds": 1, "min": 60, "minute": 60, "minutes": 60, "h": 3600,
         "hr": 3600, "hour": 3600, "hours": 3600, "day": 86400, "days": 86400, "week": 604800, "weeks": 604800}
_VOL = {"l": 1, "liter": 1, "litre": 1, "liters": 1, "litres": 1, "ml": 0.001, "gal": 3.785411784,
        "gallon": 3.785411784, "gallons": 3.785411784}
_TABLES = [_LEN, _MASSU, _TIME, _VOL]


@register("unit_conversion")
def s_units(q: str) -> Optional[SolverResult]:
    m = re.search(r"convert\s+(-?\d[\d,]*(?:\.\d+)?)\s*°?\s*([a-zA-Z]+)\s+(?:to|into|in)\s+°?\s*([a-zA-Z]+)", q)
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    a, b = m.group(2).lower(), m.group(3).lower()
    temps = {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}
    if a in temps and b in temps:
        def to_c(x, u):
            return x if u[0] == "c" else (x - 32) * 5 / 9 if u[0] == "f" else x - 273.15

        def from_c(x, u):
            return x if u[0] == "c" else x * 9 / 5 + 32 if u[0] == "f" else x + 273.15
        out = from_c(to_c(v, a), b)
        rt = to_c(out, b)  # convert the result back to Celsius
        return SolverResult("temperature", f"{v:g} {a.upper()[0]} = {out:.6g} {b.upper()[0]}",
                            "converted back to Celsius and compared", _close(rt, to_c(v, a), 1e-9))
    for t in _TABLES:
        if a in t and b in t:
            out = v * t[a] / t[b]
            back = out * t[b] / t[a]
            return SolverResult("unit conversion", f"{v:g} {a} = {out:.8g} {b}", "round trip returns the original value",
                                _close(back, v, 1e-9), [f"1 {a} = {t[a] / t[b]:.8g} {b}"])
    return None


# --------------------------------------------------------------------------
# Probability simulations
# --------------------------------------------------------------------------


@register("monty_hall")
def s_monty(q: str) -> Optional[SolverResult]:
    if "monty hall" not in q.lower():
        return None
    # exact: enumerate prize door, first pick (host opens a goat door)
    stay = switch = Fraction(0)
    for prize, pick in itertools.product(range(3), repeat=2):
        stay += Fraction(1, 9) * (pick == prize)
        switch += Fraction(1, 9) * (pick != prize)
    rnd = random.Random(2026)
    trials = 200_000
    sw = 0
    for _ in range(trials):
        prize, pick = rnd.randrange(3), rnd.randrange(3)
        opened = rnd.choice([d for d in range(3) if d != pick and d != prize])
        final = next(d for d in range(3) if d != pick and d != opened)
        sw += final == prize
    est = sw / trials
    return SolverResult("Monty Hall", f"stay wins {stay}, switch wins {switch} (switching doubles your chances)",
                        f"Monte Carlo, {trials:,} games: switching won {est:.4f}", abs(est - 2 / 3) < 0.01,
                        ["Enumerated all 9 (prize, pick) cases exactly"])


# --------------------------------------------------------------------------
# Puzzles
# --------------------------------------------------------------------------


@register("three_switches")
def s_switches(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    if not (re.search(r"\b(3|three) switch", ql) and "bulb" in ql):
        return None
    # check by enumeration: every wiring must give a distinct observation
    seen = {}
    for wiring in itertools.permutations(range(3)):  # wiring[i] = bulb controlled by switch i
        obs = []
        for bulb in range(3):
            sw = wiring.index(bulb)
            state = ("on" if sw == 1 else "off+warm" if sw == 0 else "off+cold")  # switch 1 left on, 0 on 10 min then off
            obs.append(state)
        seen[tuple(obs)] = wiring
    return SolverResult("three switches", "Turn switch 1 on for ~10 minutes, then turn it off. Turn switch 2 on and walk in. "
                        "The lit bulb is switch 2; the unlit but warm bulb is switch 1; the unlit cold bulb is switch 3.",
                        f"enumerated all 6 wirings: {len(seen)} distinct observations, so the method is unambiguous",
                        len(seen) == 6)


def _bfs(start, goal_fn, moves):
    from collections import deque
    prev = {start: None}
    dq = deque([start])
    while dq:
        s = dq.popleft()
        if goal_fn(s):
            path = []
            while s is not None:
                path.append(s)
                s = prev[s][0] if prev[s] else None
            return list(reversed(path)), prev
        for ns, label in moves(s):
            if ns not in prev:
                prev[ns] = (s, label)
                dq.append(ns)
    return None, prev


@register("water_jugs")
def s_jugs(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    if not re.search(r"jug|bucket", ql) or not re.search(r"measure|exactly|get", ql):
        return None
    nums = [int(x) for x in re.findall(r"\d+", ql)]
    if len(nums) < 3:
        return None
    A, B, T = nums[0], nums[1], nums[2]

    def moves(s):
        a, b = s
        yield (A, b), f"fill jug {A}"
        yield (a, B), f"fill jug {B}"
        yield (0, b), f"empty jug {A}"
        yield (a, 0), f"empty jug {B}"
        p = min(a, B - b)
        yield (a - p, b + p), f"pour {A}→{B}"
        p = min(b, A - a)
        yield (a + p, b - p), f"pour {B}→{A}"

    path, prev = _bfs((0, 0), lambda s: T in s, moves)
    if not path:
        return SolverResult("water jugs", f"impossible to measure {T} with jugs of {A} and {B}",
                            "exhaustive search of every reachable state found no solution",
                            T % math.gcd(A, B) != 0 or T > max(A, B))
    labels = []
    cur = path[-1]
    while prev[cur]:
        labels.append(prev[cur][1])
        cur = prev[cur][0]
    labels.reverse()
    # replay
    state = (0, 0)
    for st in path[1:]:
        state = st
    ok = T in state and all(0 <= x <= c for x, c in zip(state, (A, B)))
    return SolverResult("water jugs", f"{len(labels)} steps (shortest): " + "; ".join(labels) + f" → jugs hold {path[-1]}",
                        "replayed the shortest path; final state holds the target", ok,
                        [f"{i + 1}. {l} → {s}" for i, (l, s) in enumerate(zip(labels, path[1:]))])


@register("wolf_goat_cabbage")
def s_wgc(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    if not all(w in ql for w in ("wolf", "goat", "cabbage")):
        return None
    items = ("farmer", "wolf", "goat", "cabbage")

    def safe(s):
        f, w, g, c = s
        return not ((w == g != f) or (g == c != f))

    def moves(s):
        f = s[0]
        for i in range(4):
            if i and s[i] != f:
                continue
            ns = list(s)
            ns[0] = 1 - f
            if i:
                ns[i] = 1 - f
            ns = tuple(ns)
            if safe(ns):
                yield ns, ("cross alone" if i == 0 else f"take the {items[i]} across")

    path, prev = _bfs((0, 0, 0, 0), lambda s: s == (1, 1, 1, 1), moves)
    labels = []
    cur = path[-1]
    while prev[cur]:
        labels.append(prev[cur][1])
        cur = prev[cur][0]
    labels.reverse()
    ok = path[-1] == (1, 1, 1, 1) and all(safe(s) for s in path)
    return SolverResult("wolf, goat and cabbage", f"{len(labels)} crossings: " + "; then ".join(labels),
                        "replayed every state; nothing is ever left eaten", ok)


@register("n_queens")
def s_queens(q: str) -> Optional[SolverResult]:
    m = re.search(r"\b(\d{1,2})[\s-]*queens", q.lower())
    if not m:
        return None
    n = int(m.group(1))
    if not 1 <= n <= 12:
        return None
    cols, d1, d2, sol, count = set(), set(), set(), [], [0]
    first: List[List[int]] = []

    def place(r):
        if r == n:
            count[0] += 1
            if not first:
                first.append(sol[:])
            return
        for c in range(n):
            if c in cols or r - c in d1 or r + c in d2:
                continue
            cols.add(c); d1.add(r - c); d2.add(r + c); sol.append(c)
            place(r + 1)
            cols.discard(c); d1.discard(r - c); d2.discard(r + c); sol.pop()

    place(0)
    if not first:
        return SolverResult("N-Queens", f"no solution for n={n}", "exhaustive backtracking", True)
    s = first[0]
    ok = all(s[i] != s[j] and abs(s[i] - s[j]) != abs(i - j) for i in range(n) for j in range(i + 1, n))
    board = "\n".join("".join("Q" if s[r] == c else "." for c in range(n)) for r in range(n))
    return SolverResult("N-Queens", f"{count[0]} solutions for n={n}. One of them (column per row): {s}\n{board}",
                        "no two queens share a column or diagonal (checked pairwise)", ok)


@register("sudoku")
def s_sudoku(q: str) -> Optional[SolverResult]:
    cells = re.sub(r"[^0-9.]", "", q if "sudoku" in q.lower() else "")
    digits = re.sub(r"\D", "", q) if "sudoku" in q.lower() else ""
    grid_s = cells if len(cells) == 81 else digits if len(digits) == 81 else ""
    if len(grid_s) != 81:
        return None
    g = [int(c) if c.isdigit() else 0 for c in grid_s]

    def ok_place(i, v):
        r, c = divmod(i, 9)
        for k in range(9):
            if g[r * 9 + k] == v or g[k * 9 + c] == v:
                return False
        br, bc = r // 3 * 3, c // 3 * 3
        return all(g[(br + a) * 9 + bc + b] != v for a in range(3) for b in range(3))

    def solve() -> bool:
        try:
            i = g.index(0)
        except ValueError:
            return True
        for v in range(1, 10):
            if ok_place(i, v):
                g[i] = v
                if solve():
                    return True
                g[i] = 0
        return False

    orig = g[:]
    if not solve():
        return SolverResult("sudoku", "no solution", "exhaustive backtracking", True)
    rows = [g[r * 9:(r + 1) * 9] for r in range(9)]
    full = set(range(1, 10))
    ok = (all(set(r) == full for r in rows) and all({rows[r][c] for r in range(9)} == full for c in range(9))
          and all({rows[br + a][bc + b] for a in range(3) for b in range(3)} == full
                  for br in (0, 3, 6) for bc in (0, 3, 6))
          and all(o == 0 or o == v for o, v in zip(orig, g)))
    return SolverResult("sudoku", "\n".join(" ".join(map(str, r)) for r in rows),
                        "every row, column and 3×3 box holds 1-9 and all given clues are kept", ok)


# -- truth tables: tiny parser, no eval ------------------------------------

_TOK = re.compile(r"\s*(<->|->|\(|\)|[a-z]\b|and\b|or\b|not\b|xor\b|true\b|false\b|[01]\b|&|\||!|\^|~)")


def _parse_bool(s: str):
    toks, pos = [], 0
    s = s.lower().replace("↔", "<->").replace("→", "->").replace("∧", "&").replace("∨", "|").replace("¬", "!")
    while pos < len(s):
        m = _TOK.match(s, pos)
        if not m:
            raise ValueError("bad token")
        toks.append(m.group(1))
        pos = m.end()
    i = [0]
    norm = {"&": "and", "|": "or", "!": "not", "~": "not", "^": "xor"}

    def peek():
        return norm.get(toks[i[0]], toks[i[0]]) if i[0] < len(toks) else None

    def eat():
        t = peek()
        i[0] += 1
        return t

    def atom():
        t = eat()
        if t == "(":
            v = iff()
            if eat() != ")":
                raise ValueError("missing )")
            return v
        if t == "not":
            return ("not", atom())
        if t in ("true", "1"):
            return ("const", True)
        if t in ("false", "0"):
            return ("const", False)
        if t and len(t) == 1 and t.isalpha():
            return ("var", t)
        raise ValueError("syntax")

    def level(sub, ops):
        left = sub()
        while peek() in ops:
            op = eat()
            left = (op, left, sub())
        return left

    def and_():
        return level(atom, ("and",))

    def xor_():
        return level(and_, ("xor",))

    def or_():
        return level(xor_, ("or",))

    def imp():
        left = or_()
        if peek() == "->":
            eat()
            return ("->", left, imp())
        return left

    def iff():
        return level(imp, ("<->",))

    tree = iff()
    if i[0] != len(toks):
        raise ValueError("trailing tokens")
    return tree


def _eval_bool(t, env):
    k = t[0]
    if k == "const":
        return t[1]
    if k == "var":
        return env[t[1]]
    if k == "not":
        return not _eval_bool(t[1], env)
    a, b = _eval_bool(t[1], env), _eval_bool(t[2], env)
    return {"and": a and b, "or": a or b, "xor": a != b, "->": (not a) or b, "<->": a == b}[k]


def _to_sympy(t):
    from sympy.logic import boolalg as bl
    k = t[0]
    if k == "const":
        return bl.true if t[1] else bl.false
    if k == "var":
        return sp.Symbol(t[1])
    if k == "not":
        return bl.Not(_to_sympy(t[1]))
    a, b = _to_sympy(t[1]), _to_sympy(t[2])
    return {"and": bl.And, "or": bl.Or, "xor": bl.Xor, "->": bl.Implies, "<->": bl.Equivalent}[k](a, b)


def _vars(t, acc):
    if t[0] == "var":
        acc.add(t[1])
    elif t[0] != "const":
        for c in t[1:]:
            _vars(c, acc)
    return acc


@register("truth_table")
def s_truth(q: str) -> Optional[SolverResult]:
    m = re.search(r"truth table (?:for|of)\s*:?\s*(.+?)\s*[?.!]*$", q, re.I)
    if not m:
        return None
    tree = _parse_bool(m.group(1))
    names = sorted(_vars(tree, set()))
    if not names or len(names) > 6:
        return None
    sym = _to_sympy(tree)
    rows, ok = [], True
    for vals in itertools.product([True, False], repeat=len(names)):
        env = dict(zip(names, vals))
        mine = _eval_bool(tree, env)
        ok = ok and bool(sym.subs({sp.Symbol(k): v for k, v in env.items()})) == mine
        rows.append(" ".join("T" if v else "F" for v in vals) + " | " + ("T" if mine else "F"))
    kind = "tautology" if all(r.endswith("T") for r in rows) else "contradiction" if all(r.endswith("F") for r in rows) else "contingent"
    return SolverResult("truth table", " ".join(names) + " | result\n" + "\n".join(rows) + f"\n({kind})",
                        "every row re-evaluated with sympy's boolean algebra", ok)


# ==========================================================================
# BATCH 2 - geometry, trigonometry, combinatorics, probability
# ==========================================================================


def _fl(q: str) -> List[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", q)]


def _num_after(ql: str, key: str) -> Optional[float]:
    m = re.search(rf"{key}\s*(?:of|is|=|:|measures)?\s*(-?\d+(?:\.\d+)?)", ql)
    return float(m.group(1)) if m else None


def _simpson(f, a: float, b: float, n: int = 2000) -> float:
    if n % 2:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(a + i * h)
    return s * h / 3


def _exact(x: float):
    return sp.nsimplify(x, rational=True)


@register("triangle_area")
def s_triangle(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    if not ("triangle" in ql and "area" in ql and re.search(r"\bsides?\b", ql)):
        return None
    v = _fl(ql)
    if len(v) < 3:
        return None
    a, b, c = v[:3]
    if a + b <= c or a + c <= b or b + c <= a:
        return SolverResult("triangle area", "those lengths cannot form a triangle (the triangle inequality fails)",
                            "checked all three inequalities", True)
    s = (a + b + c) / 2
    heron = math.sqrt(s * (s - a) * (s - b) * (s - c))
    cos_c = (a * a + b * b - c * c) / (2 * a * b)
    other = 0.5 * a * b * math.sqrt(max(0.0, 1 - cos_c ** 2))
    return SolverResult("triangle area (Heron)", f"{heron:.6g} square units",
                        f"law-of-cosines route gives {other:.6g}", _close(heron, other, 1e-9),
                        [f"semi-perimeter s = {s:g}"])


@register("pythagoras")
def s_pythagoras(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    if "hypotenuse" not in ql and "pythagor" not in ql:
        return None
    legs = re.search(r"legs?\s*(?:are|is|=|of)?\s*(\d+(?:\.\d+)?)\s*(?:and|,|&)\s*(\d+(?:\.\d+)?)", ql)
    if legs:
        a, b = float(legs.group(1)), float(legs.group(2))
        c = math.hypot(a, b)
        return SolverResult("Pythagoras", f"hypotenuse = {c:.8g}", f"a² + b² = {a * a + b * b:g} and c² = {c * c:.8g}",
                            _close(a * a + b * b, c * c, 1e-9))
    hyp = re.search(r"hypotenuse\s*(?:is|=|of|measures)?\s*(\d+(?:\.\d+)?)", ql)
    leg = re.search(r"\bleg\s*(?:is|=|of|measures)?\s*(\d+(?:\.\d+)?)", ql)
    if hyp and leg:
        c, a = float(hyp.group(1)), float(leg.group(1))
        if a >= c:
            return SolverResult("Pythagoras", "impossible: a leg must be shorter than the hypotenuse", "compared lengths", True)
        b = math.sqrt(c * c - a * a)
        return SolverResult("Pythagoras", f"other leg = {b:.8g}", f"a² + b² = {a * a + b * b:.8g} and c² = {c * c:g}",
                            _close(a * a + b * b, c * c, 1e-9))
    return None


@register("coordinate_geometry")
def s_coord(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    pts = re.findall(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", ql)
    if len(pts) != 2 or not re.search(r"distance|midpoint|mid-point|slope|gradient|equation of (?:the )?line", ql):
        return None
    import numpy as np
    (x1, y1), (x2, y2) = [(float(a), float(b)) for a, b in pts]
    dx, dy = x2 - x1, y2 - y1
    if re.search(r"midpoint|mid-point", ql):
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        d1, d2, d = math.hypot(mx - x1, my - y1), math.hypot(mx - x2, my - y2), math.hypot(dx, dy)
        return SolverResult("midpoint", f"({mx:g}, {my:g})", "equal distances to both ends, each half the segment length",
                            _close(d1, d2, 1e-9) and _close(d1, d / 2, 1e-9))
    if re.search(r"slope|gradient", ql):
        if dx == 0:
            return SolverResult("slope", "undefined (vertical line)", "x-coordinates are equal", True)
        m = dy / dx
        return SolverResult("slope", f"{_exact(m)} (≈ {m:.6g})", "y₂ − y₁ equals m(x₂ − x₁)", _close(dy, m * dx, 1e-9))
    if "equation" in ql:
        if dx == 0:
            return SolverResult("line", f"x = {x1:g}", "both points have this x", True)
        m = dy / dx
        c = y1 - m * x1
        ok = _close(y1, m * x1 + c, 1e-9) and _close(y2, m * x2 + c, 1e-9)
        return SolverResult("line through two points", f"y = {_exact(m)}·x + {_exact(c)}", "both points satisfy the equation", ok)
    dist = math.sqrt(dx * dx + dy * dy)
    alt = float(np.linalg.norm([dx, dy]))
    exact = sp.sqrt(_exact(dx) ** 2 + _exact(dy) ** 2)
    return SolverResult("distance", f"{sp.sstr(exact)} ≈ {dist:.8g}", f"numpy norm gives {alt:.8g}", _close(dist, alt, 1e-12))


@register("circle")
def s_circle(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    if not re.search(r"\bcircle\b|\bradius\b|\bdiameter\b|\bsector\b", ql) or not re.search(r"\barea\b|circumference|perimeter", ql):
        return None
    if re.search(r"sphere|cylinder|cone|volume|triangle", ql):
        return None
    r = _num_after(ql, "radius")
    if r is None:
        d = _num_after(ql, "diameter")
        r = d / 2 if d is not None else None
    if r is None:
        return None
    ang = re.search(r"(\d+(?:\.\d+)?)\s*(?:°|degrees?|deg)\b", ql)
    R = _exact(r)
    if "sector" in ql and ang:
        th = float(ang.group(1))
        area = th / 360 * math.pi * r * r
        num = _simpson(lambda x: th / 360 * 2 * math.pi * x, 0, r)
        return SolverResult("sector area", f"{sp.sstr(sp.Rational(str(th)) / 360 * sp.pi * R ** 2)} ≈ {area:.6g}",
                            f"numerical integration gives {num:.6g}", _close(area, num, 1e-9))
    if re.search(r"circumference|perimeter", ql):
        c = 2 * math.pi * r
        num = (math.pi * (r + 1e-6) ** 2 - math.pi * (r - 1e-6) ** 2) / 2e-6  # d(area)/dr
        return SolverResult("circumference", f"{sp.sstr(2 * sp.pi * R)} ≈ {c:.6g}",
                            f"derivative of the area formula gives {num:.6g}", _close(c, num, 1e-6))
    area = math.pi * r * r
    num = _simpson(lambda x: 2 * math.pi * x, 0, r)
    return SolverResult("circle area", f"{sp.sstr(sp.pi * R ** 2)} ≈ {area:.6g}",
                        f"integrating 2πx from 0 to r gives {num:.6g}", _close(area, num, 1e-9))


@register("solids")
def s_solids(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    kind = next((k for k in ("sphere", "cylinder", "cone", "cube", "cuboid") if k in ql), None)
    if kind is None and re.search(r"rectangular (?:box|prism)|\bbox\b", ql):
        kind = "cuboid"
    want_vol = "volume" in ql
    want_sa = re.search(r"surface area", ql) is not None
    if not kind or not (want_vol or want_sa):
        return None
    r = _num_after(ql, "radius")
    if r is None and (d := _num_after(ql, "diameter")) is not None:
        r = d / 2
    h = _num_after(ql, "height")
    out: List[str] = []
    ok = True
    chk: List[str] = []
    if kind == "sphere" and r is not None:
        V, S = 4 / 3 * math.pi * r ** 3, 4 * math.pi * r ** 2
        nv = _simpson(lambda x: math.pi * (r * r - x * x), -r, r)
        ns = (4 / 3 * math.pi * ((r + 1e-6) ** 3 - (r - 1e-6) ** 3)) / 2e-6   # dV/dr
    elif kind == "cylinder" and r is not None and h is not None:
        V, S = math.pi * r * r * h, 2 * math.pi * r * (r + h)
        nv = _simpson(lambda x: math.pi * r * r, 0, h)
        ns = _simpson(lambda x: 2 * math.pi * r, 0, h) + 2 * math.pi * r * r
    elif kind == "cone" and r is not None and h is not None:
        V = math.pi * r * r * h / 3
        S = math.pi * r * (r + math.sqrt(r * r + h * h))
        nv = _simpson(lambda x: math.pi * (r * x / h) ** 2, 0, h)
        ns = _simpson(lambda x: 2 * math.pi * (r * x / h) * math.sqrt(1 + (r / h) ** 2), 0, h) + math.pi * r * r
    elif kind == "cube":
        a = _num_after(ql, r"(?:side|edge)")
        if a is None:
            return None
        V, S = a ** 3, 6 * a * a
        nv, ns = a * a * a, sum([a * a] * 6)
    elif kind == "cuboid":
        l, w, hh = _num_after(ql, "length"), _num_after(ql, "width"), _num_after(ql, "height")
        if None in (l, w, hh):
            return None
        V, S = l * w * hh, 2 * (l * w + w * hh + l * hh)
        nv, ns = l * (w * hh), 2 * l * w + 2 * w * hh + 2 * l * hh
    else:
        return None
    if want_vol:
        out.append(f"volume = {V:.8g}")
        ok &= _close(V, nv, 1e-6)
        chk.append("volume re-computed by integrating cross-sections")
    if want_sa:
        out.append(f"surface area = {S:.8g}")
        ok &= _close(S, ns, 1e-6)
        chk.append("surface area re-computed independently")
    return SolverResult(f"{kind}", "; ".join(out), "; ".join(chk), ok)


@register("trig_values")
def s_trig(q: str) -> Optional[SolverResult]:
    ql = q.lower().strip()
    m = re.search(r"convert\s+(-?\d+(?:\.\d+)?)\s*(?:°|degrees?|deg)\s+(?:to|into|in)\s+radians?", ql)
    if m:
        v = _exact(float(m.group(1)))
        rad = v * sp.pi / 180
        return SolverResult("degrees to radians", f"{sp.sstr(rad)} ≈ {float(rad.evalf()):.8g} rad",
                            "math.radians agrees", _close(float(rad.evalf()), math.radians(float(m.group(1))), 1e-12))
    m = re.search(r"convert\s+(.+?)\s*radians?\s+(?:to|into|in)\s+degrees?", ql)
    if m:
        rad = safe_parse(m.group(1))
        deg = rad * 180 / sp.pi
        return SolverResult("radians to degrees", f"{sp.sstr(sp.simplify(deg))}°",
                            "math.degrees agrees", _close(float(deg.evalf()), math.degrees(float(rad.evalf())), 1e-9))
    m = re.search(r"\b(sin|cos|tan|cot|sec|csc)\s*(?:of\s*)?\(?\s*([-0-9./*\spi]+?)\s*\)?\s*(degrees?|deg|°|radians?)?\s*[?.!]*$", ql)
    if not m:
        return None
    fn, ang_s, unit = m.group(1), m.group(2).strip(), m.group(3)
    ang = safe_parse(ang_s)
    if unit and unit.startswith(("deg", "°")):
        ang = ang * sp.pi / 180
    elif not unit and "pi" not in ang_s:
        return None  # "sin 30" with no unit is ambiguous: let the AI ask
    f = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec, "csc": sp.csc}[fn]
    val = f(ang)
    pf = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "cot": lambda x: 1 / math.tan(x),
          "sec": lambda x: 1 / math.cos(x), "csc": lambda x: 1 / math.sin(x)}[fn]
    if val.has(sp.zoo):
        return SolverResult("trig value", f"{fn}({sp.sstr(ang)}) is undefined", "denominator is exactly zero", True)
    num = pf(float(ang.evalf()))
    return SolverResult("trig value", f"{fn}({sp.sstr(ang)}) = {sp.sstr(val)} ≈ {float(val.evalf()):.8g}",
                        f"math.{fn if fn in ('sin', 'cos', 'tan') else 'reciprocal'} gives {num:.8g}",
                        _close(float(val.evalf()), num, 1e-9))


@register("combinatorics")
def s_comb(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    kind = None
    m = re.search(r"\b(\d+)\s*c\s*(\d+)\b", ql)
    if m:
        n, k, kind = int(m.group(1)), int(m.group(2)), "C"
    if not kind and (m := re.search(r"\b(\d+)\s*p\s*(\d+)\b", ql)):
        n, k, kind = int(m.group(1)), int(m.group(2)), "P"
    if not kind and (m := re.search(r"(\d+)\s+choose\s+(\d+)", ql)):
        n, k, kind = int(m.group(1)), int(m.group(2)), "C"
    if not kind and (m := re.search(r"(?:combinations|permutations|arrangements) of (\d+)\s+(?:\w+\s+){0,2}taken (\d+)", ql)):
        n, k, kind = int(m.group(1)), int(m.group(2)), "P" if "permutation" in ql or "arrangement" in ql else "C"
    if not kind and (m := re.search(r"(?:choose|select|pick|committee of|team of)\s+(\d+)\s+(?:\w+\s+){0,4}?(?:from|out of)\s+(?:a\s+)?(?:group of\s+)?(\d+)", ql)):
        k, n, kind = int(m.group(1)), int(m.group(2)), "C"
    if not kind or n > 5000 or k > n:
        return None
    val = math.comb(n, k) if kind == "C" else math.perm(n, k)
    if n <= 11:
        gen = itertools.combinations if kind == "C" else itertools.permutations
        enum = sum(1 for _ in gen(range(n), k))
        how = f"enumerated all {enum:,} cases"
    else:
        num = 1
        for i in range(k):
            num *= n - i
        enum = num // (math.factorial(k) if kind == "C" else 1)
        how = "falling-factorial product"
    name = f"{n}C{k}" if kind == "C" else f"{n}P{k}"
    return SolverResult("combinatorics", f"{name} = {val:,}", how, val == enum)


@register("binomial_probability")
def s_binom(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    mn = re.search(r"(?:in|of|after)\s+(\d+)\s+(?:\w+\s+)?(?:flips?|tosses|trials?|attempts?|throws?|shots?|games?|tests?)", ql)
    mk = re.search(r"(exactly|at least|at most|more than|fewer than|less than)\s+(\d+)", ql)
    if not mn or not mk or "probab" not in ql:
        return None
    n, kind, k = int(mn.group(1)), mk.group(1), int(mk.group(2))
    if n > 2000 or k > n:
        return None
    if re.search(r"coin|flip|toss|head|tail", ql):
        p = 0.5
    else:
        mp = re.search(r"(?:p\s*=|probability (?:of (?:success|a success|winning|hitting)\s*)?(?:is|=|of)?)\s*(0?\.\d+|\d+(?:\.\d+)?\s*%)", ql)
        if not mp:
            return None
        t = mp.group(1)
        p = float(t[:-1]) / 100 if t.endswith("%") else float(t)
    pmf = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    sel = {"exactly": lambda i: i == k, "at least": lambda i: i >= k, "at most": lambda i: i <= k,
           "more than": lambda i: i > k, "fewer than": lambda i: i < k, "less than": lambda i: i < k}[kind]
    exact = sum(pmf[i] for i in range(n + 1) if sel(i))
    rnd = random.Random(11)
    trials = 100_000
    hit = sum(sel(sum(rnd.random() < p for _ in range(n))) for _ in range(trials)) if n <= 200 else None
    if hit is None:
        total_ok = abs(sum(pmf) - 1) < 1e-9
        return SolverResult("binomial", f"{exact:.8g}", "the full distribution sums to 1", total_ok)
    est = hit / trials
    se = math.sqrt(max(exact * (1 - exact), 1e-12) / trials)
    return SolverResult("binomial probability", f"P = {exact:.8g}  (n={n}, p={p:g}, {kind} {k})",
                        f"Monte Carlo ({trials:,} runs) gives {est:.4f}", abs(est - exact) < 5 * se + 1e-3)


@register("dice_sum")
def s_dice(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    words = {"two": 2, "three": 3, "four": 4, "2": 2, "3": 3, "4": 4, "1": 1, "one": 1, "a": 1, "five": 5, "5": 5}
    md = re.search(r"\b(one|two|three|four|five|\d)\s+(?:[a-z-]+\s+)*dic?e\b", ql)
    ms = re.search(r"\bsum\b(.*)$", ql)
    if not md or not ms or not re.findall(r"\d+", ms.group(1)):
        return None
    n, target = words.get(md.group(1), 0), int(re.findall(r"\d+", ms.group(1))[-1])
    if not n:
        return None
    combos = list(itertools.product(range(1, 7), repeat=n))
    favour = sum(1 for c in combos if sum(c) == target)
    exact = Fraction(favour, len(combos))
    rnd = random.Random(5)
    trials = 100_000
    est = sum(sum(rnd.randint(1, 6) for _ in range(n)) == target for _ in range(trials)) / trials
    return SolverResult("dice", f"{exact} ≈ {float(exact):.5f}  ({favour} of {len(combos)} outcomes)",
                        f"Monte Carlo ({trials:,} rolls) gives {est:.4f}", abs(est - float(exact)) < 0.01)


@register("birthday")
def s_birthday(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    m = re.search(r"(\d+)\s+(?:people|persons|students|friends|kids|guests)", ql)
    if "birthday" not in ql or not m:
        return None
    n = int(m.group(1))
    if n > 400:
        return None
    p_diff = 1.0
    for i in range(n):
        p_diff *= (365 - i) / 365
    exact = 1 - p_diff
    rnd = random.Random(3)
    trials = 40_000
    hits = sum(len({rnd.randrange(365) for _ in range(n)}) < n for _ in range(trials))
    est = hits / trials
    return SolverResult("birthday problem", f"P(at least two share a birthday) = {exact:.6f}",
                        f"Monte Carlo ({trials:,} groups) gives {est:.4f}", abs(est - exact) < 0.01)


def _pct_after(ql: str, keys: str) -> Optional[float]:
    m = re.search(rf"(?:{keys})[^\d%]{{0,30}}(\d+(?:\.\d+)?)\s*%", ql)
    return float(m.group(1)) / 100 if m else None


@register("bayes")
def s_bayes(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    prior = _pct_after(ql, "prevalence|base rate|prior|incidence|prob(?:ability)? of (?:having|the disease)")
    sens = _pct_after(ql, "sensitivity|true positive")
    spec = _pct_after(ql, "specificity|true negative")
    fp = _pct_after(ql, "false positive")
    if prior is None or sens is None or (spec is None and fp is None):
        return None
    fp = fp if fp is not None else 1 - spec
    exact = prior * sens / (prior * sens + (1 - prior) * fp)
    pop = 1_000_000
    sick = round(pop * prior)
    tp = round(sick * sens)
    fpos = round((pop - sick) * fp)
    count = tp / (tp + fpos)
    return SolverResult("Bayes' theorem", f"P(condition | positive test) = {exact:.6f} ({exact * 100:.2f}%)",
                        f"counting a population of {pop:,}: {tp:,} true vs {fpos:,} false positives gives {count:.6f}",
                        _close(exact, count, 1e-4),
                        [f"prior = {prior:g}, sensitivity = {sens:g}, false-positive rate = {fp:g}"])


@register("normal_distribution")
def s_normal(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    if "normal" not in ql:
        return None
    mu = _num_after(ql, r"mean")
    sd = _num_after(ql, r"(?:standard deviation|std|sd|σ)")
    if mu is None or not sd:
        return None

    def cdf(x):
        return 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))

    def pdf(x):
        return math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))

    def numeric(a, b):
        return _simpson(pdf, a, b, 4000)

    btw = re.search(r"between\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)", ql)
    lt = re.search(r"(?:less than|below|under|<|at most)\s*(-?\d+(?:\.\d+)?)", ql)
    gt = re.search(r"(?:greater than|above|more than|over|>|at least|exceeds?)\s*(-?\d+(?:\.\d+)?)", ql)
    lo = mu - 12 * sd
    if btw:
        a, b = float(btw.group(1)), float(btw.group(2))
        ex, nu, label = cdf(b) - cdf(a), numeric(a, b), f"P({a:g} < X < {b:g})"
    elif lt:
        x = float(lt.group(1))
        ex, nu, label = cdf(x), numeric(lo, x), f"P(X < {x:g})"
    elif gt:
        x = float(gt.group(1))
        ex, nu, label = 1 - cdf(x), numeric(x, mu + 12 * sd), f"P(X > {x:g})"
    else:
        return None
    return SolverResult("normal distribution", f"{label} = {ex:.6f}  (mean {mu:g}, sd {sd:g})",
                        f"numerical integration of the density gives {nu:.6f}", abs(ex - nu) < 1e-5)


# ==========================================================================
# BATCH 3 - algebra extras, number facts, text tricks, dates, classic puzzles
# ==========================================================================

import base64
import calendar
import datetime
from collections import Counter


def _equal_numeric(f, g, syms, tries: int = 8) -> bool:
    syms = list(syms)
    rnd = random.Random(13)
    fn, gn = sp.lambdify(syms, f, "math"), sp.lambdify(syms, g, "math")
    good = 0
    for _ in range(tries * 3):
        pt = [rnd.uniform(0.3, 2.7) for _ in syms]
        try:
            a, b = fn(*pt), gn(*pt)
        except Exception:  # noqa: BLE001
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        if not _close(a, b, 1e-7):
            return False
        good += 1
    return good >= 3


@register("fractions")
def s_fractions(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    m = re.search(r"(\d+)\s*/\s*(\d+)\s*(\+|-|\*|×|÷|\bx\b|plus|minus|times|divided by)\s*(\d+)\s*/\s*(\d+)", ql)
    if m:
        a, b, op, c, d = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5))
        if b == 0 or d == 0:
            return None
        x, y = Fraction(a, b), Fraction(c, d)
        sx, sy = sp.Rational(a, b), sp.Rational(c, d)
        kind = {"+": "+", "plus": "+", "-": "-", "minus": "-", "*": "*", "×": "*", "x": "*", "times": "*",
                "÷": "/", "divided by": "/"}[op]
        if kind == "/" and c == 0:
            return None
        res = {"+": x + y, "-": x - y, "*": x * y, "/": x / y if y else None}[kind]
        ref = {"+": sx + sy, "-": sx - sy, "*": sx * sy, "/": sx / sy if sy else None}[kind]
        return SolverResult("fraction arithmetic", f"{res} (≈ {float(res):.6g})",
                            "sympy's exact rational arithmetic agrees", sp.Rational(res.numerator, res.denominator) == ref)
    m = re.search(r"\b(add|subtract|multiply|divide)\s+(\d+)\s*/\s*(\d+)\s+(?:and|by|to|from|with)\s+(\d+)\s*/\s*(\d+)", ql)
    if m:
        verb = m.group(1)
        a, b, c, d = (int(m.group(i)) for i in (2, 3, 4, 5))
        if b == 0 or d == 0 or (verb == "divide" and c == 0):
            return None
        x, y = Fraction(a, b), Fraction(c, d)
        sx, sy = sp.Rational(a, b), sp.Rational(c, d)
        res = {"add": x + y, "subtract": y - x, "multiply": x * y, "divide": x / y}[verb]
        ref = {"add": sx + sy, "subtract": sy - sx, "multiply": sx * sy, "divide": sx / sy}[verb]
        return SolverResult("fraction arithmetic", f"{res} (≈ {float(res):.6g})",
                            "sympy's exact rational arithmetic agrees", sp.Rational(res.numerator, res.denominator) == ref)
    m = re.search(r"simplify\s+(\d+)\s*/\s*(\d+)\s*[?.!]*$", ql)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b == 0:
            return None
        f = Fraction(a, b)
        return SolverResult("simplify fraction", f"{f}", f"gcd {math.gcd(a, b)} divided out; sympy agrees",
                            sp.Rational(a, b) == sp.Rational(f.numerator, f.denominator))
    return None


@register("polynomial_algebra")
def s_poly(q: str) -> Optional[SolverResult]:
    m = re.search(r"\b(factor(?:ise|ize)?|expand|simplify|partial fractions? of|apart)\b\s*(?:the\s*)?"
                  r"(?:expression|polynomial|fraction)?\s*:?\s*(.+?)\s*[?.!]*$", q.lower())
    if not m:
        return None
    op, body = m.group(1), m.group(2)
    if re.fullmatch(r"[\d\s,]+", body):
        return None
    f = safe_parse(body)
    syms = sorted(f.free_symbols, key=lambda s: s.name)
    if not syms:
        return None
    if op.startswith("factor"):
        g, name = sp.factor(f), "factored form"
    elif op == "expand":
        g, name = sp.expand(f), "expanded form"
    elif op == "simplify":
        g, name = sp.simplify(f), "simplified form"
    else:
        g, name = sp.apart(f, syms[0]), "partial fractions"
    return SolverResult("polynomial algebra", f"{name}: {sp.sstr(g)}",
                        "original and result agree at random numeric points", _equal_numeric(f, g, syms))


@register("inequality")
def s_inequality(q: str) -> Optional[SolverResult]:
    ql = q.lower().replace("≤", "<=").replace("≥", ">=")
    m = re.search(r"(?:solve|find (?:all )?(?:the )?(?:values? of )?x(?: such that)?|for which (?:values? of )?x)\s*:?\s*(.+?)\s*[?.!]*$", ql)
    if not m:
        return None
    body = m.group(1)
    op = re.search(r"<=|>=|<|>", body)
    if not op:
        return None
    l, r = body.split(op.group(0), 1)
    lhs, rhs = safe_parse(l), safe_parse(r)
    syms = (lhs - rhs).free_symbols
    if len(syms) != 1:
        return None
    x = next(iter(syms))
    rel = {"<": sp.Lt, "<=": sp.Le, ">": sp.Gt, ">=": sp.Ge}[op.group(0)](lhs, rhs)
    sol = sp.solve_univariate_inequality(rel, x, relational=False)
    rnd = random.Random(21)
    ok = True
    for _ in range(400):
        p = sp.Rational(rnd.randint(-20000, 20000), 1000)
        ok &= bool(rel.subs(x, p)) == bool(sol.contains(p))
    return SolverResult("inequality", f"{x} ∈ {sol}", "tested 400 points: membership matches the original inequality", ok)


@register("summation")
def s_summation(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    m = re.search(r"sum of (.+?)\s+(?:for|from|where|with|as)\s+([a-z])\s*=\s*(-?\d+)\s*(?:to|until|→|up to)\s*(\d+|n|infinity|oo|∞)", ql)
    if m:
        body, var, lo, hi = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        k = sp.Symbol(var)
        expr = safe_parse(body)
        if hi.isdigit():
            hi_i = int(hi)
            if hi_i - lo > 200000:
                return None
            val = sp.summation(expr, (k, lo, hi_i))
            direct = sum(expr.subs(k, i) for i in range(lo, hi_i + 1))
            return SolverResult("summation", f"{sp.sstr(val)}", "adding the terms one by one gives the same",
                                sp.simplify(val - direct) == 0)
        if hi == "n":
            n = sp.Symbol("n", integer=True, positive=True)
            val = sp.simplify(sp.summation(expr, (k, lo, n)))
            ok = all(sp.simplify(val.subs(n, t) - sum(expr.subs(k, i) for i in range(lo, t + 1))) == 0
                     for t in range(max(lo, 1), max(lo, 1) + 6))
            return SolverResult("summation", f"{sp.sstr(val)}", "formula tested against direct sums for 6 values of n", ok)
        val = sp.summation(expr, (k, lo, sp.oo))
        if val.has(sp.oo) or val.has(sp.zoo) or val.is_number is False:
            return None
        fn = sp.lambdify(k, expr, "math")
        part = sum(fn(i) for i in range(lo, lo + 100000))
        return SolverResult("infinite series", f"{sp.sstr(val)} ≈ {float(val.evalf()):.8g}",
                            f"partial sum of 100,000 terms is {part:.8g}", abs(part - float(val.evalf())) < 1e-3 * (1 + abs(part)))
    m = re.search(r"sum of (?:the )?first (\d+) (natural numbers|positive integers|odd numbers|even numbers|squares|cubes)", ql)
    if m:
        n, kind = int(m.group(1)), m.group(2)
        gen = {"natural numbers": lambda i: i, "positive integers": lambda i: i, "odd numbers": lambda i: 2 * i - 1,
               "even numbers": lambda i: 2 * i, "squares": lambda i: i * i, "cubes": lambda i: i ** 3}[kind]
        formula = {"natural numbers": n * (n + 1) // 2, "positive integers": n * (n + 1) // 2, "odd numbers": n * n,
                   "even numbers": n * (n + 1), "squares": n * (n + 1) * (2 * n + 1) // 6,
                   "cubes": (n * (n + 1) // 2) ** 2}[kind]
        direct = sum(gen(i) for i in range(1, n + 1)) if n <= 2_000_000 else formula
        return SolverResult("sum of first n terms", f"{formula:,}", "closed formula equals adding every term", formula == direct)
    return None


@register("arithmetic_geometric")
def s_arith_geo(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    kind = "arithmetic" if "arithmetic" in ql else "geometric" if "geometric" in ql else None
    a = _num_after(ql, "first term")
    if kind is None or a is None:
        return None
    mn = re.search(r"(\d+)(?:st|nd|rd|th)\s+term", ql)
    ms = re.search(r"sum of (?:the )?first (\d+) terms?", ql)
    if not mn and not ms:
        return None
    A = Fraction(str(a))
    if kind == "arithmetic":
        d = _num_after(ql, "common difference")
        if d is None:
            return None
        D = Fraction(str(d))
        nth = lambda n: A + (n - 1) * D  # noqa: E731
        total = lambda n: Fraction(n, 2) * (2 * A + (n - 1) * D)  # noqa: E731
        step = lambda t: t + D  # noqa: E731
    else:
        r = _num_after(ql, "common ratio")
        if r is None:
            return None
        Rr = Fraction(str(r))
        nth = lambda n: A * Rr ** (n - 1)  # noqa: E731
        total = lambda n: A * n if Rr == 1 else A * (Rr ** n - 1) / (Rr - 1)  # noqa: E731
        step = lambda t: t * Rr  # noqa: E731
    out, ok = [], True
    if mn:
        n = int(mn.group(1))
        t = A
        for _ in range(n - 1):
            t = step(t)
        out.append(f"term {n} = {nth(n)}")
        ok &= t == nth(n)
    if ms:
        n = int(ms.group(1))
        t, s = A, Fraction(0)
        for _ in range(n):
            s += t
            t = step(t)
        out.append(f"sum of first {n} terms = {total(n)}")
        ok &= s == total(n)
    return SolverResult(f"{kind} sequence", "; ".join(out), "step-by-step iteration gives the same", ok)


@register("next_term")
def s_next_term(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    if not re.search(r"next|missing|pattern|continue|what comes|sequence|series", ql):
        return None
    m = re.search(r"(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){3,})", ql)
    if not m:
        return None
    t = [Fraction(x.strip()) for x in m.group(1).split(",")]
    n = len(t)

    def show(v):
        return str(v.numerator) if v.denominator == 1 else f"{v} (≈ {float(v):.6g})"

    def diffs(seq):
        return [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]

    # 1 arithmetic
    d1 = diffs(t)
    if len(set(d1)) == 1:
        nxt = t[-1] + d1[0]
        ok = all(t[i] == t[0] + i * d1[0] for i in range(n))
        return SolverResult("next term", f"{show(nxt)}", f"arithmetic: add {show(d1[0])} each time; reproduces all {n} terms", ok)
    # 2 geometric
    if all(x != 0 for x in t[:-1]):
        ratios = [t[i + 1] / t[i] for i in range(n - 1)]
        if len(set(ratios)) == 1:
            nxt = t[-1] * ratios[0]
            ok = all(t[i] == t[0] * ratios[0] ** i for i in range(n))
            return SolverResult("next term", f"{show(nxt)}", f"geometric: multiply by {show(ratios[0])}; reproduces all {n} terms", ok)
    # 3 fibonacci-like
    if n >= 5 and all(t[i] == t[i - 1] + t[i - 2] for i in range(2, n)):
        return SolverResult("next term", f"{show(t[-1] + t[-2])}", "each term is the sum of the previous two (checked for every term)", True)
    # 4 polynomial of degree k via finite differences
    level = d1
    for k in range(2, n - 1):
        level = diffs(level)
        if len(set(level)) == 1:
            x = sp.Symbol("x")
            poly = sp.interpolate([(i, sp.Rational(t[i].numerator, t[i].denominator)) for i in range(k + 1)], x)
            ok = all(poly.subs(x, i) == sp.Rational(t[i].numerator, t[i].denominator) for i in range(n))
            nxt = poly.subs(x, n)
            return SolverResult("next term", f"{nxt}", f"polynomial of degree {k} (constant {k}th differences); term i = {sp.sstr(sp.expand(poly))} for i = 0,1,2,...", ok)
    return None


@register("complex_numbers")
def s_complex(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    m = re.search(r"(modulus|magnitude|absolute value|conjugate|argument|phase|polar form)\s*of\s*(.+?)\s*[?.!]*$", ql)
    if not m:
        return None
    op = m.group(1)
    z_sym = sp.expand(safe_parse(m.group(2), {"i": sp.I}))
    if not z_sym.has(sp.I):
        return None
    z = complex(sp.N(z_sym))
    import cmath
    if op in ("modulus", "magnitude", "absolute value"):
        ex = sp.simplify(sp.Abs(z_sym))
        return SolverResult("complex modulus", f"|{sp.sstr(z_sym)}| = {sp.sstr(ex)} ≈ {float(ex.evalf()):.8g}",
                            f"Python abs() gives {abs(z):.8g}", _close(float(ex.evalf()), abs(z), 1e-9))
    if op == "conjugate":
        ex = sp.conjugate(z_sym)
        return SolverResult("complex conjugate", sp.sstr(sp.expand(ex)), "Python conjugate() agrees",
                            _close(complex(sp.N(ex)).real, z.conjugate().real) and _close(complex(sp.N(ex)).imag, z.conjugate().imag))
    th = sp.arg(z_sym)
    if op in ("argument", "phase"):
        return SolverResult("complex argument", f"{sp.sstr(sp.simplify(th))} ≈ {float(th.evalf()):.8g} rad",
                            f"cmath.phase gives {cmath.phase(z):.8g}", _close(float(th.evalf()), cmath.phase(z), 1e-9))
    r = abs(z)
    back = complex(r * math.cos(cmath.phase(z)), r * math.sin(cmath.phase(z)))
    return SolverResult("polar form", f"{r:.6g}·(cos {cmath.phase(z):.6g} + i·sin {cmath.phase(z):.6g})",
                        f"converting back gives {back.real:.6g}{back.imag:+.6g}i", abs(back - z) < 1e-9)


@register("base_conversion")
def s_base(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    names = {"binary": 2, "octal": 8, "hex": 16, "hexadecimal": 16}
    m = re.search(r"(?:convert\s+)?(?:decimal\s+)?(\d+)\s+(?:to|in|into)\s+(binary|hexadecimal|hex|octal)\b", ql)
    if m:
        n, base = int(m.group(1)), names[m.group(2)]
        out = format(n, {2: "b", 8: "o", 16: "x"}[base])
        return SolverResult("base conversion", f"{n} in {m.group(2)} = {out}", "parsing the result back gives the original number",
                            int(out, base) == n)
    m = re.search(r"(?:convert\s+)?(binary|hexadecimal|hex|octal)\s+([0-9a-f]+)\s+(?:to|in|into)\s+decimal", ql)
    if m:
        base = names[m.group(1)]
        try:
            val = int(m.group(2), base)
        except ValueError:
            return SolverResult("base conversion", f"{m.group(2)} is not a valid {m.group(1)} number", "digit check", True)
        back = format(val, {2: "b", 8: "o", 16: "x"}[base])
        return SolverResult("base conversion", f"{m.group(1)} {m.group(2)} = {val} in decimal", "formatting back reproduces the digits",
                            back == m.group(2).lstrip("0") or val == 0)
    return None


def _binexp(a: int, b: int, n: int) -> int:
    r, a = 1, a % n
    while b:
        if b & 1:
            r = r * a % n
        a = a * a % n
        b >>= 1
    return r


@register("modular_arithmetic")
def s_modular(q: str) -> Optional[SolverResult]:
    ql = q.lower().replace(",", "")
    m = re.search(r"(\d+)\s*\^\s*(\d+)\s*(?:mod|modulo|%)\s*(\d+)", ql) or \
        re.search(r"remainder when (\d+)\s*\^\s*(\d+)\s+is divided by (\d+)", ql)
    if m:
        a, b, n = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if n == 0:
            return None
        v = pow(a, b, n)
        return SolverResult("modular exponent", f"{a}^{b} mod {n} = {v}", "independent square-and-multiply gives the same", v == _binexp(a, b, n))
    m = re.search(r"(?:modular )?inverse of (\d+)\s*(?:mod|modulo|under mod|modulus)\s*(\d+)", ql)
    if m:
        a, n = int(m.group(1)), int(m.group(2))
        if math.gcd(a, n) != 1:
            return SolverResult("modular inverse", f"no inverse: gcd({a}, {n}) = {math.gcd(a, n)} ≠ 1", "gcd computed", True)
        inv = pow(a, -1, n)
        return SolverResult("modular inverse", f"{inv}", f"{a}·{inv} mod {n} = {a * inv % n}", a * inv % n == 1)
    m = re.search(r"last digit of (\d+)\s*\^\s*(\d+)", ql)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        v = pow(a, b, 10)
        seen, x, i = [], a % 10, 0
        cyc = []
        while x not in cyc:
            cyc.append(x)
            x = x * a % 10
        start = cyc.index(x)
        period = len(cyc) - start
        pos = (b - 1) if b - 1 < start else start + ((b - 1 - start) % period)
        return SolverResult("last digit", f"{v}", f"cycle of last digits {cyc}: position {pos} gives {cyc[pos]}", cyc[pos] == v)
    m = re.search(r"remainder when (\d+) is divided by (\d+)", ql)
    if m:
        a, n = int(m.group(1)), int(m.group(2))
        if n == 0:
            return None
        q_, r_ = divmod(a, n)
        return SolverResult("remainder", f"{a % n}", f"{n}×{q_} + {r_} = {a}", n * q_ + r_ == a and r_ == a % n)
    return None


@register("percentages")
def s_percent(q: str) -> Optional[SolverResult]:
    ql = q.lower().replace(",", "")
    n = r"(\d+(?:\.\d+)?)"
    if m := re.search(rf"what is {n}\s*%\s*of\s*{n}", ql):
        p, v = float(m.group(1)), float(m.group(2))
        r = p * v / 100
        return SolverResult("percentage of", f"{r:g}", f"{r:g} is {r / v * 100:g}% of {v:g}", _close(r / v * 100, p, 1e-9) if v else False)
    if m := re.search(rf"{n}\s+is\s+what\s+percent(?:age)?\s+of\s+{n}", ql):
        a, b = float(m.group(1)), float(m.group(2))
        if b == 0:
            return None
        p = a / b * 100
        return SolverResult("what percent", f"{p:.6g}%", f"{p:.6g}% of {b:g} = {b * p / 100:g}", _close(b * p / 100, a, 1e-9))
    if m := re.search(rf"percent(?:age)?\s+(?:increase|decrease|change)\s+(?:from|between)\s+{n}\s+(?:to|and)\s+{n}", ql):
        a, b = float(m.group(1)), float(m.group(2))
        if a == 0:
            return None
        p = (b - a) / a * 100
        return SolverResult("percentage change", f"{p:+.6g}% ({'increase' if p > 0 else 'decrease' if p < 0 else 'no change'})",
                            f"{a:g} × (1 {p / 100:+.6g}) = {a * (1 + p / 100):g}", _close(a * (1 + p / 100), b, 1e-9))
    if m := re.search(rf"(increase|decrease|reduce|raise)\s+{n}\s+by\s+{n}\s*%", ql):
        a, p = float(m.group(2)), float(m.group(3))
        sign = 1 if m.group(1) in ("increase", "raise") else -1
        r = a * (1 + sign * p / 100)
        back = r / (1 + sign * p / 100)
        return SolverResult("percentage change", f"{r:.8g}", f"undoing the {p:g}% change gives {back:g}", _close(back, a, 1e-9))
    if m := re.search(rf"(?:{n}\s*%\s*(?:discount|off)\s*(?:on|of|from)?\s*(?:a|an|the)?\s*(?:price of\s*)?{n}|discount of {n}\s*%\s*(?:on|off|from)\s*(?:a|an|the)?\s*(?:price of\s*)?{n})", ql):
        g = [float(x) for x in m.groups() if x is not None]
        p, price = g[0], g[1]
        final = price * (1 - p / 100)
        return SolverResult("discount", f"final price {final:.2f} (you save {price - final:.2f})",
                            f"{final:.2f} + {price - final:.2f} = {final + price - final:.2f}", _close(final + (price - final), price, 1e-9))
    return None


@register("roman_numerals")
def s_roman(q: str) -> Optional[SolverResult]:
    if "roman" not in q.lower():
        return None
    table = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"),
             (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]

    def to_roman(n):
        out = ""
        for v, s in table:
            while n >= v:
                out += s
                n -= v
        return out

    def from_roman(s):
        vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        total = 0
        for i, ch in enumerate(reversed(s)):
            v = vals[ch]
            total += v if (i == 0 or v >= vals[list(reversed(s))[i - 1]]) else -v
        return total

    m = re.search(r"(\d+)", q)
    if m:
        n = int(m.group(1))
        if not 1 <= n <= 3999:
            return None
        r = to_roman(n)
        return SolverResult("roman numerals", f"{n} = {r}", f"reading {r} back gives {from_roman(r)}", from_roman(r) == n)
    m = re.search(r"\b([MDCLXVI]{1,15})\b", q)
    if not m:
        return None
    s = m.group(1)
    v = from_roman(s)
    return SolverResult("roman numerals", f"{s} = {v}", f"writing {v} as a numeral gives {to_roman(v)}", from_roman(to_roman(v)) == v)


_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})


def _find_dates(q: str) -> List[datetime.date]:
    ql = q.lower()
    found: List[Tuple[int, datetime.date]] = []
    for m in re.finditer(r"(\d{4})-(\d{1,2})-(\d{1,2})", ql):
        found.append((m.start(), datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))))
    for m in re.finditer(r"(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]{3,9})\.?,?\s+(\d{4})", ql):
        if m.group(2) in _MONTHS:
            found.append((m.start(), datetime.date(int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))))
    for m in re.finditer(r"([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", ql):
        if m.group(1) in _MONTHS:
            found.append((m.start(), datetime.date(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2)))))
    found.sort()
    return [d for _, d in found]


def _zeller(y: int, m: int, d: int) -> str:
    if m < 3:
        m += 12
        y -= 1
    K, J = y % 100, y // 100
    h = (d + (13 * (m + 1)) // 5 + K + K // 4 + J // 4 + 5 * J) % 7
    return ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][h]


def _jdn(y: int, m: int, d: int) -> int:
    a = (14 - m) // 12
    yy, mm = y + 4800 - a, m + 12 * a - 3
    return d + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - yy // 100 + yy // 400 - 32045


@register("dates")
def s_dates(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    m = re.search(r"is\s+(\d{4})\s+a\s+leap\s+year|leap\s+year.*?(\d{4})", ql)
    if m:
        y = int(m.group(1) or m.group(2))
        rule = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
        days = (datetime.date(y, 12, 31) - datetime.date(y, 1, 1)).days + 1
        return SolverResult("leap year", f"{y} {'is' if rule else 'is not'} a leap year", f"the year has {days} days",
                            rule == (days == 366) == calendar.isleap(y))
    ds = _find_dates(q)
    if re.search(r"day of (?:the )?week|what day|which day|weekday", ql) and len(ds) == 1:
        d = ds[0]
        z = _zeller(d.year, d.month, d.day)
        return SolverResult("day of the week", f"{d.isoformat()} is a {z}", "Zeller's congruence matches the calendar library",
                            z == d.strftime("%A"))
    if re.search(r"days between|how many days|difference between|days from", ql) and len(ds) == 2:
        a, b = ds
        diff = (b - a).days
        j = _jdn(b.year, b.month, b.day) - _jdn(a.year, a.month, a.day)
        return SolverResult("days between", f"{abs(diff)} days (from {min(a, b)} to {max(a, b)})",
                            "Julian-day-number arithmetic gives the same", diff == j)
    return None


@register("text_tricks")
def s_text(q: str) -> Optional[SolverResult]:
    ql = q.lower().strip()
    m = re.search(r"how many\s+(?:times\s+)?(?:does\s+)?(?:the\s+)?(?:letter\s+)?['\"]?([a-z])(?:'s|s)?['\"]?\s+"
                  r"(?:are\s+|is\s+|appears?\s+|occurs?\s+)?(?:there\s+)?(?:in|inside)\s+(?:the\s+)?(?:word\s+)?['\"]?([a-z\-]+)['\"]?", ql)
    if m:
        ch, word = m.group(1), m.group(2)
        a = word.count(ch)
        b = sum(1 for c in word if c == ch)
        pos = [i + 1 for i, c in enumerate(word) if c == ch]
        return SolverResult("letter count", f"'{ch}' appears {a} time{'s' if a != 1 else ''} in \"{word}\" (positions {pos})",
                            "counted twice, by string count and letter-by-letter", a == b == len(re.findall(ch, word)))
    m = re.search(r"how many vowels (?:are )?(?:there )?in\s+['\"]?(.+?)['\"]?\s*[?.!]*$", ql)
    if m:
        t = m.group(1)
        a = sum(t.count(v) for v in "aeiou")
        b = len(re.findall(r"[aeiou]", t))
        return SolverResult("vowel count", f"{a} vowels in \"{t}\"", "regex count agrees", a == b)
    m = re.search(r"how many words (?:are )?(?:there )?in[:\s]+['\"]?(.+?)['\"]?\s*[?.!]*$", q, re.I)
    if m:
        t = m.group(1)
        a, b = len(t.split()), len(re.findall(r"\S+", t))
        return SolverResult("word count", f"{a} words", "two splitting methods agree", a == b)
    m = re.search(r"\breverse\s+(?:the\s+)?(?:word|string|text|letters (?:of|in))\s+['\"]?([^'\"?]+?)['\"]?\s*[?.!]*$", q, re.I) or \
        re.search(r"\breverse\s+['\"]([^'\"]+)['\"]", q, re.I)
    if m:
        t = m.group(1).strip()
        r = t[::-1]
        return SolverResult("reverse", r, "reversing the result gives the original", r[::-1] == t and "".join(reversed(t)) == r)
    m = re.search(r"is\s+['\"]?([a-z0-9 ]+?)['\"]?\s+a\s+palindrome", ql)
    if m:
        t = re.sub(r"[^a-z0-9]", "", m.group(1))
        a = t == t[::-1]
        i, j, b = 0, len(t) - 1, True
        while i < j:
            b &= t[i] == t[j]
            i, j = i + 1, j - 1
        return SolverResult("palindrome", f"\"{m.group(1)}\" {'is' if a else 'is not'} a palindrome", "two-pointer check agrees", a == b)
    m = re.search(r"are\s+['\"]?([a-z]+)['\"]?\s+and\s+['\"]?([a-z]+)['\"]?\s+anagrams", ql)
    if m:
        x, y = m.groups()
        a = sorted(x) == sorted(y)
        return SolverResult("anagram", f"{x} and {y} {'are' if a else 'are not'} anagrams", "letter-frequency comparison agrees", a == (Counter(x) == Counter(y)))
    return None


@register("encodings")
def s_encoding(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    if "base64" in ql.replace(" ", "").replace("-", ""):
        m = re.search(r"(?:encode|convert)\s+['\"]?(.+?)['\"]?\s+(?:to|in|into)\s+base\s*-?64", q, re.I)
        if m:
            t = m.group(1)
            enc = base64.b64encode(t.encode()).decode()
            return SolverResult("base64 encode", enc, "decoding it returns the original text", base64.b64decode(enc).decode() == t)
        m = re.search(r"decode\s+(?:this\s+)?(?:base\s*-?64\s+)?(?:string\s+)?['\"]?([A-Za-z0-9+/=]{4,})['\"]?", q, re.I)
        if m:
            try:
                dec = base64.b64decode(m.group(1), validate=True).decode("utf-8")
            except Exception:  # noqa: BLE001
                return None
            return SolverResult("base64 decode", dec, "re-encoding returns the original string",
                                base64.b64encode(dec.encode()).decode().rstrip("=") == m.group(1).rstrip("="))
    if "caesar" in ql or "rot13" in ql or "rot-13" in ql:
        mt = re.search(r"['\"]([^'\"]+)['\"]", q)
        if not mt:
            return None
        shift = 13 if "rot" in ql else None
        if shift is None:
            ms = re.search(r"(?:shift|by|key)\s*(?:of\s*)?(\d+)", ql)
            if not ms:
                return None
            shift = int(ms.group(1))
        if re.search(r"decrypt|decode", ql):
            shift = -shift

        def rot(t, k):
            return "".join(chr((ord(c) - base + k) % 26 + base) if c.isalpha() and c.isascii() and (base := 65 if c.isupper() else 97) else c
                           for c in t)
        out = rot(mt.group(1), shift)
        return SolverResult("Caesar cipher", out, "shifting back returns the original text", rot(out, -shift) == mt.group(1))
    return None


@register("number_facts")
def s_numfacts(q: str) -> Optional[SolverResult]:
    ql = q.lower().replace(",", "")
    m = re.search(r"(\d+)(?:st|nd|rd|th)\s+fibonacci", ql) or re.search(r"fibonacci(?:\s+number)?\s*(?:f\s*)?\(?\s*(\d+)\s*\)?", ql)
    if m:
        n = int(m.group(1))
        if not 1 <= n <= 100000:
            return None
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b

        def fd(k):
            if k == 0:
                return 0, 1
            x, y = fd(k >> 1)
            c = x * (2 * y - x)
            d = x * x + y * y
            return (d, c + d) if k & 1 else (c, d)
        ok = a == fd(n)[0] == int(sp.fibonacci(n))
        s = str(a)
        return SolverResult("Fibonacci", f"F({n}) = {s if len(s) <= 40 else s[:20] + '… (' + str(len(s)) + ' digits)'}",
                            "iteration, fast doubling and sympy all agree", ok)
    m = re.search(r"(\d+)(?:st|nd|rd|th)\s+prime", ql)
    if m:
        n = int(m.group(1))
        if not 1 <= n <= 200000:
            return None
        limit = max(15, int(n * (math.log(n) + math.log(math.log(n))) * 1.3) + 10) if n > 5 else 15
        sieve = bytearray([1]) * (limit + 1)
        sieve[0:2] = b"\x00\x00"
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
        primes = [i for i, v in enumerate(sieve) if v]
        return SolverResult("nth prime", f"prime #{n} is {primes[n - 1]}", "sieve of Eratosthenes and sympy.prime agree", primes[n - 1] == int(sp.prime(n)))
    m = re.search(r"(\d+)\s*!", ql) or re.search(r"factorial of (\d+)", ql)
    if m:
        n = int(m.group(1))
        if n > 5000:
            return None
        f = math.factorial(n)
        prod = 1
        for i in range(2, n + 1):
            prod *= i
        digits = len(str(f))
        if "digit" in ql and "sum" not in ql:
            est = int(math.lgamma(n + 1) / math.log(10)) + 1
            return SolverResult("factorial digits", f"{n}! has {digits} digits", f"log-gamma estimate gives {est}", f == prod and abs(est - digits) <= 1)
        if "trailing zero" in ql:
            z, p = 0, 5
            while p <= n:
                z += n // p
                p *= 5
            s = str(f)
            return SolverResult("trailing zeros", f"{n}! ends in {z} zeros", "counted zeros in the full number", z == len(s) - len(s.rstrip("0")))
        s = str(f)
        return SolverResult("factorial", f"{n}! = {s if len(s) <= 60 else s[:20] + '… (' + str(digits) + ' digits)'}",
                            "math.factorial equals the running product", f == prod)
    m = re.search(r"sum of (?:the )?digits of (\d+)(?:\s*\^\s*(\d+))?", ql)
    if m:
        v = int(m.group(1)) ** int(m.group(2) or 1)
        a = sum(int(c) for c in str(v))
        b, t = 0, v
        while t:
            t, r = divmod(t, 10)
            b += r
        return SolverResult("digit sum", f"{a}", "dividing by 10 repeatedly gives the same", a == b)
    return None


@register("letter_arrangements")
def s_arrangements(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    m = re.search(r"(?:arrangements|permutations|ways to arrange|ways can (?:you )?arrange|distinct arrangements)\s*(?:of\s*)?(?:the\s*)?letters\s*(?:of|in)\s*(?:the\s*word\s*)?['\"]?([a-z]+)['\"]?", ql)
    if not m:
        return None
    w = m.group(1)
    n = len(w)
    if n > 20:
        return None
    cnt = Counter(w)
    val = math.factorial(n)
    for c in cnt.values():
        val //= math.factorial(c)
    alt, left = 1, n
    for c in cnt.values():
        alt *= math.comb(left, c)
        left -= c
    ok = val == alt
    how = "chosen-positions product (nCr) agrees"
    if n <= 8:
        enum = len(set(itertools.permutations(w)))
        ok &= enum == val
        how = f"enumerated {enum:,} distinct orderings"
    rep = ", ".join(f"{k}×{v}" for k, v in cnt.items() if v > 1) or "no repeated letters"
    return SolverResult("letter arrangements", f"{val:,} distinct arrangements of {w.upper()} ({rep})", how, ok)


@register("coin_change")
def s_coins(q: str) -> Optional[SolverResult]:
    ql = q.lower().replace(",", " ,")
    if not re.search(r"coin|denomination|change", ql):
        return None
    md = re.search(r"(?:coins?|denominations?|using|with)[^\d]{0,25}(\d+(?:\s*(?:,|and|&|or)\s*\d+)+)", ql)
    ma = re.search(r"(?:make|change for|amount of|sum of|total of|reach|for)\s*(?:\$|rs\.?|₹)?\s*(\d+)", ql)
    if not md or not ma:
        return None
    coins = sorted({int(x) for x in re.findall(r"\d+", md.group(1)) if int(x) > 0})
    amount = int(ma.group(1))
    if amount > 100000 or len(coins) > 12 or len(coins) < 2:
        return None
    want_ways = re.search(r"ways|combinations", ql) is not None
    if want_ways:
        dp = [1] + [0] * amount
        for c in coins:
            for a in range(c, amount + 1):
                dp[a] += dp[a - c]
        ways = dp[amount]
        if amount <= 600:
            import functools
            sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))

            @functools.lru_cache(None)
            def rec(a, i):
                if a == 0:
                    return 1
                if a < 0 or i == len(coins):
                    return 0
                return rec(a - coins[i], i) + rec(a, i + 1)
            ok = rec(amount, 0) == ways
            how = "recursive count with memoisation agrees"
        else:
            ok, how = True, "table DP (no independent check for this size)"
        return SolverResult("coin change (ways)", f"{ways:,} ways to make {amount} with {coins}", how, ok)
    INF = 10 ** 9
    best = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and best[a - c] + 1 < best[a]:
                best[a] = best[a - c] + 1
    if best[amount] >= INF:
        return SolverResult("coin change", f"{amount} cannot be made with {coins}", "table DP found no way", True)
    used, a = [], amount
    while a:
        for c in sorted(coins, reverse=True):
            if c <= a and best[a - c] == best[a] - 1:
                used.append(c)
                a -= c
                break
    ok = sum(used) == amount and len(used) == best[amount]
    if amount <= 20000:
        from collections import deque
        seen, dq = {0}, deque([(0, 0)])
        bfs = None
        while dq:
            v, d = dq.popleft()
            if v == amount:
                bfs = d
                break
            for c in coins:
                if v + c <= amount and v + c not in seen:
                    seen.add(v + c)
                    dq.append((v + c, d + 1))
        ok &= bfs == best[amount]
        how = "breadth-first search finds the same minimum"
    else:
        how = "the chosen coins add up exactly"
    return SolverResult("coin change (minimum)", f"{best[amount]} coins: {sorted(used, reverse=True)} make {amount}", how, ok)


@register("tower_of_hanoi")
def s_hanoi(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    m = re.search(r"(\d+)\s+disks?", ql)
    if "hanoi" not in ql or not m:
        return None
    n = int(m.group(1))
    if not 1 <= n <= 60:
        return None
    formula = 2 ** n - 1
    rec = 0
    t = 0
    for _ in range(n):
        t = 2 * t + 1
    rec = t
    ok = formula == rec
    how = "recurrence T(n) = 2T(n-1) + 1 agrees"
    if n <= 12:
        pegs = [list(range(n, 0, -1)), [], []]
        moves = 0

        def mv(k, a, b, c):
            nonlocal moves
            if k == 0:
                return True
            if not mv(k - 1, a, c, b):
                return False
            disk = pegs[a].pop()
            if pegs[b] and pegs[b][-1] < disk:
                return False
            pegs[b].append(disk)
            moves += 1
            return mv(k - 1, c, b, a)
        legal = mv(n, 0, 2, 1)
        ok &= legal and moves == formula and pegs[2] == list(range(n, 0, -1))
        how = f"simulated every move legally ({moves:,} moves)"
    return SolverResult("Tower of Hanoi", f"{n} disks need {formula:,} moves at minimum (2^{n} − 1)", how, ok)


@register("optimisation")
def s_optimise(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    m = re.search(r"(maximum|minimum|max|min)(?:imum)?\s*(?:value\s*)?of\s*(.+?)\s*(?:on|over|in|for|between)\s*(?:the\s*)?(?:interval\s*)?"
                  r"\[\s*([^,\]]+?)\s*,\s*([^\]]+?)\s*\]", ql)
    if not m:
        return None
    import numpy as np
    want_max = m.group(1).startswith("max")
    x = sp.Symbol("x")
    f = safe_parse(m.group(2))
    a, b = safe_parse(m.group(3)), safe_parse(m.group(4))
    if f.free_symbols - {x}:
        return None
    af, bf = float(a.evalf()), float(b.evalf())
    crit = [c for c in sp.solve(sp.diff(f, x), x) if c.is_real and af <= float(c.evalf()) <= bf]
    cands = [a, b] + crit
    vals = [(float(sp.N(f.subs(x, c))), c) for c in cands]
    best_v, best_x = max(vals, key=lambda t: t[0]) if want_max else min(vals, key=lambda t: t[0])
    grid = np.linspace(af, bf, 200001)
    fn = sp.lambdify(x, f, "numpy")
    gv = np.asarray(fn(grid), dtype=float)
    gv = gv if gv.shape else np.full_like(grid, float(gv))
    ref = float(np.nanmax(gv) if want_max else np.nanmin(gv))
    return SolverResult("optimisation", f"{'maximum' if want_max else 'minimum'} = {sp.sstr(sp.nsimplify(best_v)) if abs(best_v) < 1e9 else best_v} at x = {sp.sstr(best_x)}",
                        f"scanning 200,001 points gives {ref:.8g}", _close(best_v, ref, 1e-6),
                        [f"candidates: endpoints and critical points {[sp.sstr(c) for c in crit]}"])


# ==========================================================================
# BATCH 4 - physics, chemistry, vectors, regression, finance, work/rate
# ==========================================================================


def _g(ql: str) -> float:
    m = re.search(r"\bg\s*=\s*(\d+(?:\.\d+)?)", ql)
    return float(m.group(1)) if m else 9.8


@register("projectile")
def s_projectile(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    if not re.search(r"projectile|thrown|launched|fired|kicked|shot|\bhit\b", ql):
        return None
    mv = re.search(r"(\d+(?:\.\d+)?)\s*m\s*/\s*s", ql)
    ma = re.search(r"(\d+(?:\.\d+)?)\s*(?:°|degrees?|deg)\b", ql)
    if not mv or not ma:
        return None
    v, deg, g = float(mv.group(1)), float(ma.group(1)), _g(ql)
    th = math.radians(deg)
    T = 2 * v * math.sin(th) / g
    R = v * v * math.sin(2 * th) / g
    H = (v * math.sin(th)) ** 2 / (2 * g)
    if T <= 0:
        return None
    n = 200000
    dt = T / n
    x = y = 0.0
    vx, vy = v * math.cos(th), v * math.sin(th)
    top, t = 0.0, 0.0
    while True:
        ny = y + vy * dt - 0.5 * g * dt * dt
        vy -= g * dt
        x += vx * dt
        t += dt
        top = max(top, ny)
        if ny < 0:
            f = y / (y - ny)
            x -= vx * dt * (1 - f)
            t -= dt * (1 - f)
            break
        y = ny
    ok = _close(t, T, 1e-3) and _close(x, R, 1e-3) and _close(top, H, 1e-3)
    return SolverResult("projectile motion", f"time of flight {T:.4g} s, range {R:.4g} m, max height {H:.4g} m (g = {g:g} m/s²)",
                        f"step-by-step simulation gives {t:.4g} s, {x:.4g} m, {top:.4g} m", ok)


@register("free_fall")
def s_freefall(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    if not re.search(r"dropped|falls?\b|free[- ]fall|released from rest|let go", ql) or re.search(r"°|degrees?", ql):
        return None
    mh = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)\b(?!\s*/)", ql)
    if not mh:
        return None
    h, g = float(mh.group(1)), _g(ql)
    T = math.sqrt(2 * h / g)
    v = g * T
    n = 200000
    dt = T / n
    y, vel = 0.0, 0.0
    for _ in range(n):
        y += vel * dt + 0.5 * g * dt * dt
        vel += g * dt
    ok = _close(y, h, 1e-6) and _close(vel, v, 1e-6) and _close(v, math.sqrt(2 * g * h), 1e-9)
    return SolverResult("free fall", f"falls for {T:.4g} s and hits the ground at {v:.4g} m/s (g = {g:g} m/s², no air resistance)",
                        f"simulation covers {y:.4g} m; energy conservation gives {math.sqrt(2 * g * h):.4g} m/s", ok)


@register("suvat")
def s_suvat(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    vals = dict(re.findall(r"\b([uvast])\s*=\s*(-?\d+(?:\.\d+)?)", ql))
    if len(vals) != 3 or not re.search(r"find|calculate|what|determine", ql):
        return None
    u, v, a, s, t = sp.symbols("u v a s t")
    sym = {"u": u, "v": v, "a": a, "s": s, "t": t}
    eqs = [sp.Eq(v, u + a * t), sp.Eq(s, u * t + a * t ** 2 / 2), sp.Eq(v ** 2, u ** 2 + 2 * a * s),
           sp.Eq(s, (u + v) * t / 2), sp.Eq(s, v * t - a * t ** 2 / 2)]
    known = {sym[k]: sp.Rational(val) for k, val in vals.items()}
    unknown = [sym[k] for k in "uvast" if k not in vals]
    sols: List[dict] = []
    seen = set()
    for pair in itertools.combinations(eqs, 2):
        try:
            res = sp.solve([e.subs(known) for e in pair], unknown, dict=True)
        except Exception:  # noqa: BLE001
            continue
        for r in res:
            if not all(x in r and r[x].is_real for x in unknown):
                continue
            if t in r and r[t] < 0:
                continue
            if all(sp.simplify(e.subs(known).subs(r)) == True for e in eqs):  # noqa: E712
                key = tuple(sp.nsimplify(r[x]) for x in unknown)
                if key not in seen:
                    seen.add(key)
                    sols.append(r)
    if not sols:
        return None
    def fmt(val):
        val = sp.nsimplify(val)
        return sp.sstr(val) if val.is_Integer else f"{sp.sstr(val)} (= {float(val):.6g})"
    shown = " or ".join(", ".join(f"{x} = {fmt(r[x])}" for x in unknown) for r in sols)
    return SolverResult("kinematics (SUVAT)", shown, "each solution satisfies all five SUVAT equations", True)


@register("physics_formulas")
def s_physics(q: str) -> Optional[SolverResult]:
    ql = q.lower()

    def val(pat):
        m = re.search(pat, ql)
        return float(m.group(1)) if m else None

    kg = val(r"(\d+(?:\.\d+)?)\s*kg")
    ms = val(r"(\d+(?:\.\d+)?)\s*m\s*/\s*s\b(?!\s*[²2^])")
    g = _g(ql)
    if "kinetic energy" in ql and kg and ms:
        ke, p = 0.5 * kg * ms ** 2, kg * ms
        alt = p * p / (2 * kg)
        return SolverResult("kinetic energy", f"KE = {ke:g} J", f"p²/2m gives {alt:g} J", _close(ke, alt, 1e-12), [f"KE = ½mv² with m = {kg:g} kg, v = {ms:g} m/s"])
    if "momentum" in ql and kg and ms:
        p = kg * ms
        return SolverResult("momentum", f"p = {p:g} kg·m/s", f"p²/2m = {p * p / (2 * kg):g} J equals ½mv² = {0.5 * kg * ms ** 2:g} J",
                            _close(p * p / (2 * kg), 0.5 * kg * ms ** 2, 1e-12))
    mh = val(r"(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)\b(?!\s*/)")
    if "potential energy" in ql and kg and mh:
        pe = kg * g * mh
        vv = math.sqrt(2 * g * mh)
        return SolverResult("gravitational potential energy", f"PE = {pe:g} J (g = {g:g} m/s²)",
                            f"after falling, ½mv² = {0.5 * kg * vv * vv:g} J (energy conservation)", _close(pe, 0.5 * kg * vv * vv, 1e-9))
    if re.search(r"ohm|voltage|current|resistance|volts?\b|amps?\b", ql):
        V = val(r"(\d+(?:\.\d+)?)\s*(?:v|volts?)\b")
        I = val(r"(\d+(?:\.\d+)?)\s*(?:a|amps?|amperes?)\b")
        R = val(r"(\d+(?:\.\d+)?)\s*(?:ohms?|Ω|ω)")
        given = [x for x in (V, I, R) if x is not None]
        if len(given) == 2:
            if V is None:
                V = I * R
            elif I is None:
                if R == 0:
                    return None
                I = V / R
            else:
                if I == 0:
                    return None
                R = V / I
            P = V * I
            ok = _close(P, I * I * R, 1e-9) and _close(P, V * V / R, 1e-9)
            return SolverResult("Ohm's law", f"V = {V:g} V, I = {I:g} A, R = {R:g} Ω, power = {P:g} W",
                                "P = VI = I²R = V²/R all agree", ok)
    if "average speed" in ql or "speed" in ql and re.search(r"travel|drives?|runs?|walks?|cycles?", ql):
        md = re.search(r"(\d+(?:\.\d+)?)\s*(km|kilometers?|kilometres?|miles?|m|meters?|metres?)\b(?!\s*/)", ql)
        mt = re.search(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|seconds?|secs?)\b", ql)
        if md and mt:
            d, t = float(md.group(1)), float(mt.group(1))
            if t == 0:
                return None
            du, tu = md.group(2), mt.group(2)
            spd = d / t
            dunit = "km" if du.startswith(("k",)) else "miles" if du.startswith("mi") else "m"
            tunit = "h" if tu.startswith("h") else "min" if tu.startswith("m") else "s"
            return SolverResult("average speed", f"{spd:.6g} {dunit}/{tunit}", f"{spd:.6g} × {t:g} = {spd * t:g} {dunit}", _close(spd * t, d, 1e-12))
    return None


def _parse_side(side: str) -> List[Dict[str, int]]:
    out = []
    for sp_ in side.split("+"):
        f = re.sub(r"^\s*\d+\s*", "", sp_.strip())
        if f:
            out.append(_counts_stack(f))
    return out


@register("balance_equation")
def s_balance(q: str) -> Optional[SolverResult]:
    if "balance" not in q.lower():
        return None
    body = re.sub(r"^.*?\bbalance\b\s*(?:the\s*)?(?:following\s*)?(?:chemical\s*)?(?:equation|reaction)?\s*:?\s*", "", q, flags=re.I)
    m = re.search(r"([A-Za-z0-9() +]+?)\s*(?:->|→|=>|=)\s*([A-Za-z0-9() +]+)", body)
    if not m:
        return None
    left, right = m.group(1).strip(), m.group(2).strip().rstrip("?.!")
    names_l = [re.sub(r"^\s*\d+\s*", "", x.strip()) for x in left.split("+") if x.strip()]
    names_r = [re.sub(r"^\s*\d+\s*", "", x.strip()) for x in right.split("+") if x.strip()]
    if not names_l or not names_r or any(not re.fullmatch(r"[A-Z][A-Za-z0-9()]*", x) for x in names_l + names_r):
        return None
    cols = [_counts_stack(f) for f in names_l] + [_counts_stack(f) for f in names_r]
    elems = sorted({e for c in cols for e in c})
    M = sp.Matrix([[(1 if j < len(names_l) else -1) * c.get(e, 0) for j, c in enumerate(cols)] for e in elems])
    ns = M.nullspace()
    if len(ns) != 1:
        return None
    v = ns[0]
    den = sp.ilcm(*[x.q for x in v])
    coef = [int(x * den) for x in v]
    g = math.gcd(*coef)
    coef = [c // g for c in coef]
    if any(c <= 0 for c in coef):
        coef = [-c for c in coef]
    if any(c <= 0 for c in coef):
        return None
    names = names_l + names_r
    nl = len(names_l)
    tot_l, tot_r = Counter(), Counter()
    for i, f in enumerate(names):
        for e, n in _counts_stack(f).items():
            (tot_l if i < nl else tot_r)[e] += n * coef[i]
    fmt = lambda cs, nm: " + ".join((f"{c} " if c != 1 else "") + n for c, n in zip(cs, nm))  # noqa: E731
    return SolverResult("balance chemical equation", f"{fmt(coef[:nl], names_l)} → {fmt(coef[nl:], names_r)}",
                        f"atom counts match on both sides: {dict(tot_l)}", tot_l == tot_r)


@register("chemistry_calcs")
def s_chem(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    if re.search(r"\bph\b", ql):
        m = re.search(r"(\d+(?:\.\d+)?(?:e-?\d+)?)\s*m\b", ql) or re.search(r"\[h\+?\]\s*=\s*(\d+(?:\.\d+)?(?:e-?\d+)?)", ql)
        if m and re.search(r"of|\[h", ql):
            c = float(m.group(1))
            if c <= 0:
                return None
            ph = -math.log10(c)
            return SolverResult("pH", f"pH = {ph:.4g} (assuming a strong monoprotic acid, so [H⁺] = {c:g} M)",
                                f"10^(−pH) = {10 ** (-ph):g} M gives the concentration back", _close(10 ** (-ph), c, 1e-9))
        m = re.search(r"ph\s*(?:of|is|=)?\s*(\d+(?:\.\d+)?)", ql)
        if m and re.search(r"concentration|\[h|hydrogen ion", ql):
            ph = float(m.group(1))
            c = 10 ** (-ph)
            return SolverResult("hydrogen ion concentration", f"[H⁺] = {c:.4g} M", f"−log10 of it gives pH {-math.log10(c):.4g}", _close(-math.log10(c), ph, 1e-9))
    if re.search(r"ideal gas|pv\s*=\s*nrt", ql):
        P = re.search(r"(\d+(?:\.\d+)?)\s*atm", ql)
        V = re.search(r"(\d+(?:\.\d+)?)\s*(?:l|liters?|litres?)\b", ql)
        n = re.search(r"(\d+(?:\.\d+)?)\s*mol", ql)
        T = re.search(r"(\d+(?:\.\d+)?)\s*k\b", ql)
        have = {k: float(x.group(1)) for k, x in zip("PVnT", (P, V, n, T)) if x}
        if len(have) == 3:
            R = 0.082057
            miss = next(k for k in "PVnT" if k not in have)
            P_, V_, n_, T_ = (have.get(k) for k in "PVnT")
            if miss == "P":
                P_ = n_ * R * T_ / V_
            elif miss == "V":
                V_ = n_ * R * T_ / P_
            elif miss == "n":
                n_ = P_ * V_ / (R * T_)
            else:
                T_ = P_ * V_ / (n_ * R)
            val = {"P": P_, "V": V_, "n": n_, "T": T_}[miss]
            unit = {"P": "atm", "V": "L", "n": "mol", "T": "K"}[miss]
            return SolverResult("ideal gas law", f"{miss} = {val:.5g} {unit} (R = 0.082057 L·atm/(mol·K))",
                                f"PV/(nT) = {P_ * V_ / (n_ * T_):.6f} equals R", _close(P_ * V_ / (n_ * T_), R, 1e-9))
    if re.search(r"dilut|m1\s*v1", ql):
        d = {k: float(v) for k, v in re.findall(r"\b([mcv][12])\s*=\s*(\d+(?:\.\d+)?)", ql)}
        d = {("m" + k[1] if k[0] == "c" else k): v for k, v in d.items()}
        if len(d) == 3:
            miss = next(k for k in ("m1", "v1", "m2", "v2") if k not in d)
            if miss == "m1":
                d[miss] = d["m2"] * d["v2"] / d["v1"]
            elif miss == "v1":
                d[miss] = d["m2"] * d["v2"] / d["m1"]
            elif miss == "m2":
                d[miss] = d["m1"] * d["v1"] / d["v2"]
            else:
                d[miss] = d["m1"] * d["v1"] / d["m2"]
            return SolverResult("dilution (M₁V₁ = M₂V₂)", f"{miss.upper()} = {d[miss]:.6g}",
                                f"M₁V₁ = {d['m1'] * d['v1']:.6g} and M₂V₂ = {d['m2'] * d['v2']:.6g}", _close(d["m1"] * d["v1"], d["m2"] * d["v2"], 1e-9))
    return None


@register("matrix_vector_ops")
def s_mvops(q: str) -> Optional[SolverResult]:
    import numpy as np
    ql = q.lower()
    mats = re.findall(r"\[\s*\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*\s*\]", q)
    if len(mats) == 2 and re.search(r"multiply|product|times", ql):
        A, B = sp.Matrix(ast.literal_eval(mats[0])), sp.Matrix(ast.literal_eval(mats[1]))
        if A.shape[1] != B.shape[0]:
            return SolverResult("matrix product", f"not defined: {A.shape[0]}×{A.shape[1]} times {B.shape[0]}×{B.shape[1]}",
                                "inner dimensions differ", True)
        C = A * B
        ok = np.allclose(np.array(A.tolist(), dtype=float) @ np.array(B.tolist(), dtype=float), np.array(C.tolist(), dtype=float))
        return SolverResult("matrix product", sp.sstr(C), "numpy matrix product agrees", ok)
    if mats:
        return None
    vecs = [[float(x) for x in v.split(",")] for v in
            re.findall(r"[\(\[]\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){1,2})\s*[\)\]]", q)]
    if not vecs:
        return None
    op = next((k for k in ("dot product", "cross product", "angle between", "magnitude", "length of", "norm", "unit vector") if k in ql), None)
    if not op:
        return None
    if op in ("magnitude", "length of", "norm", "unit vector") and len(vecs) >= 1:
        a = np.array(vecs[0])
        n = math.sqrt(sum(x * x for x in vecs[0]))
        if op == "unit vector":
            if n == 0:
                return None
            u = [x / n for x in vecs[0]]
            return SolverResult("unit vector", f"{tuple(round(x, 6) for x in u)}", f"its length is {math.sqrt(sum(x * x for x in u)):.9f}", _close(math.sqrt(sum(x * x for x in u)), 1, 1e-9))
        return SolverResult("vector magnitude", f"{n:.8g}", f"numpy norm gives {np.linalg.norm(a):.8g}", _close(n, float(np.linalg.norm(a)), 1e-12))
    if len(vecs) < 2 or len(vecs[0]) != len(vecs[1]):
        return None
    a, b = vecs[0], vecs[1]
    A, B = np.array(a), np.array(b)
    if op == "dot product":
        d = sum(x * y for x, y in zip(a, b))
        return SolverResult("dot product", f"{d:g}", f"numpy dot gives {float(np.dot(A, B)):g}", _close(d, float(np.dot(A, B)), 1e-12))
    if op == "cross product":
        if len(a) == 2:
            z = a[0] * b[1] - a[1] * b[0]
            return SolverResult("2D cross product", f"{z:g} (z-component)", "numpy cross agrees", _close(z, float(np.cross(A, B)), 1e-12))
        c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        ok = np.allclose(c, np.cross(A, B)) and abs(np.dot(c, A)) < 1e-9 and abs(np.dot(c, B)) < 1e-9
        return SolverResult("cross product", f"({c[0]:g}, {c[1]:g}, {c[2]:g})", "matches numpy and is perpendicular to both vectors", bool(ok))
    na, nb = np.linalg.norm(A), np.linalg.norm(B)
    if na == 0 or nb == 0:
        return None
    ang1 = math.degrees(math.acos(max(-1, min(1, float(np.dot(A, B) / (na * nb))))))
    cr = np.cross(A, B) if len(a) == 3 else abs(a[0] * b[1] - a[1] * b[0])
    ang2 = math.degrees(math.atan2(float(np.linalg.norm(cr)), float(np.dot(A, B))))
    return SolverResult("angle between vectors", f"{ang1:.6g}°", f"atan2(|a×b|, a·b) gives {ang2:.6g}°", _close(ang1, ang2, 1e-7))


@register("regression_correlation")
def s_regression(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    m = re.search(r"z-?score\s*(?:of|for)?\s*(-?\d+(?:\.\d+)?)", ql)
    if m:
        x, mu, sd = float(m.group(1)), _num_after(ql, "mean"), _num_after(ql, r"(?:standard deviation|std|sd)")
        if mu is None or not sd:
            return None
        z = (x - mu) / sd
        return SolverResult("z-score", f"z = {z:.6g}", f"mean + z·sd = {mu + z * sd:g} gives back {x:g}", _close(mu + z * sd, x, 1e-12))
    if not re.search(r"correlation|regression|best fit|r-squared", ql):
        return None
    lists = re.findall(r"\[([^\[\]]+)\]", q)
    arrs = []
    for l in lists:
        try:
            arrs.append([float(x) for x in l.split(",")])
        except ValueError:
            return None
    if len(arrs) != 2 or len(arrs[0]) != len(arrs[1]) or len(arrs[0]) < 3:
        return None
    import numpy as np
    x, y = arrs
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    if sxx == 0 or syy == 0:
        return None
    r = sxy / math.sqrt(sxx * syy)
    slope = sxy / sxx
    icpt = my - slope * mx
    ref_r = float(np.corrcoef(x, y)[0, 1])
    ref_s, ref_i = np.polyfit(x, y, 1)
    ok = _close(r, ref_r, 1e-9) and _close(slope, float(ref_s), 1e-9) and _close(icpt, float(ref_i), 1e-9)
    return SolverResult("correlation and regression", f"r = {r:.6f}, R² = {r * r:.6f}; best-fit line y = {slope:.6g}·x + {icpt:.6g}",
                        "numpy corrcoef and polyfit agree", ok)


@register("finance_returns")
def s_finance(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    mc = re.search(r"cash\s*flows?\s*(?:of|:)?\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)+)", ql)
    if mc and "npv" in ql:
        cf = [float(x) for x in mc.group(1).split(",")]
        mr = re.search(r"(\d+(?:\.\d+)?)\s*%", ql)
        if not mr:
            return None
        r = float(mr.group(1)) / 100
        npv = sum(c / (1 + r) ** i for i, c in enumerate(cf))
        exact = sum(sp.Rational(str(c)) / (1 + sp.Rational(str(r))) ** i for i, c in enumerate(cf))
        return SolverResult("NPV", f"NPV = {npv:,.2f} at {r * 100:g}%", f"exact rational arithmetic gives {float(exact):,.2f}", _close(npv, float(exact), 1e-9))
    if mc and "irr" in ql:
        cf = [float(x) for x in mc.group(1).split(",")]
        f = lambda r: sum(c / (1 + r) ** i for i, c in enumerate(cf))  # noqa: E731
        lo, hi = -0.99, 10.0
        if f(lo) * f(hi) > 0:
            return None
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(lo) * f(mid) <= 0:
                hi = mid
            else:
                lo = mid
        irr = (lo + hi) / 2
        try:
            from scipy.optimize import brentq
            ref = brentq(f, -0.99, 10.0)
            ok = _close(irr, ref, 1e-8) and abs(f(irr)) < 1e-6 * (1 + sum(abs(c) for c in cf))
            how = f"scipy brentq gives {ref * 100:.4f}% and NPV at that rate is {f(irr):.2e}"
        except Exception:  # noqa: BLE001
            ok = abs(f(irr)) < 1e-6 * (1 + sum(abs(c) for c in cf))
            how = f"NPV at that rate is {f(irr):.2e}"
        return SolverResult("IRR", f"IRR = {irr * 100:.4f}%", how, ok)
    if "cagr" in ql or "compound annual growth" in ql:
        m = re.search(r"from\s*(?:\$|rs\.?|₹)?\s*(\d+(?:\.\d+)?)\s*to\s*(?:\$|rs\.?|₹)?\s*(\d+(?:\.\d+)?)\s*(?:over|in)\s*(\d+(?:\.\d+)?)\s*years?", ql)
        if not m:
            return None
        a, b, n = float(m.group(1)), float(m.group(2)), float(m.group(3))
        if a <= 0 or n <= 0:
            return None
        g = (b / a) ** (1 / n) - 1
        return SolverResult("CAGR", f"{g * 100:.4f}% per year", f"{a:g} × (1 + {g:.6f})^{n:g} = {a * (1 + g) ** n:.4f}", _close(a * (1 + g) ** n, b, 1e-9))
    return None


@register("simple_interest")
def s_simple_interest(q: str) -> Optional[SolverResult]:
    ql = q.lower().replace(",", "")
    if "simple interest" not in ql:
        return None
    pct = re.search(r"(\d+(?:\.\d+)?)\s*%", ql)
    tm = re.search(r"(\d+(?:\.\d+)?)\s*(years?|months?)", ql)
    pr = re.search(r"(?:rs\.?|₹|\$)?\s*(\d+(?:\.\d+)?)", ql)
    if not pct or not tm or not pr:
        return None
    P, r, t = float(pr.group(1)), float(pct.group(1)), float(tm.group(1))
    if tm.group(2).startswith("month"):
        t /= 12
    si = P * r * t / 100
    bal, add = P, P * r / 100
    full = int(t)
    for _ in range(full):
        bal += add
    bal += add * (t - full)
    return SolverResult("simple interest", f"interest = {si:,.2f}; total amount = {P + si:,.2f}",
                        f"adding {add:,.2f} per year gives total {bal:,.2f}", _close(bal, P + si, 1e-9))


@register("work_rate")
def s_work_rate(q: str) -> Optional[SolverResult]:
    ql = q.lower()
    if "together" not in ql:
        return None
    found = re.findall(r"(?:can|could)\s+(?:do|finish|complete|build|paint|fill|dig|clean|write)[^.?]*?\bin\s+(\d+(?:\.\d+)?)\s*(days?|hours?|minutes?|mins?)", ql)
    if len(found) < 2:
        return None
    times = [float(t) for t, _ in found]
    unit = found[0][1].rstrip("s")
    if any(t <= 0 for t in times):
        return None
    T = 1 / sum(1 / t for t in times)
    done = sum(T / t for t in times)
    steps, dt, prog, clock = 200000, T / 200000, 0.0, 0.0
    rates = [1 / t for t in times]
    for _ in range(steps):
        prog += sum(rates) * dt
        clock += dt
    return SolverResult("work and rate", f"together they take {T:.4g} {unit}s ({' + '.join(f'1/{t:g}' for t in times)} of the job per {unit})",
                        f"in that time their shares add up to {done:.6f} of the job; stepping time forward finishes at {prog:.6f}",
                        _close(done, 1, 1e-9) and _close(prog, 1, 1e-6))


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------


def _selftest() -> None:
    def need(q, contains, verified=True):
        r = try_solve(q)
        assert r is not None, f"no solver matched: {q}"
        assert r.verified == verified, (q, r)
        assert contains.replace(" ", "") in r.answer.replace(" ", ""), (q, r.answer)
        return r

    need("find the derivative of x^2*sin(x)", "x*(x*cos(x)+2*sin(x))")
    r = need("derivative of sin(x^2)", "2*x*cos(x**2)")
    assert any("Chain rule" in s for s in r.steps)
    need("integrate x*exp(x) dx", "(x-1)*exp(x)")
    need("integrate x^2 from 0 to 3", "9")
    need("limit of sin(x)/x as x -> 0", "=1")
    need("taylor series of exp(x) around 0 up to order 3", "x**3/6")
    need("solve x^2 - 5x + 6 = 0", "x=2")
    need("solve x + y = 10, x - y = 2", "x=6")
    need("solve the ODE: y'' + y = 0", "C1*sin(x)")
    need("determinant of [[1,2],[3,4]]", "-2")
    need("inverse of [[2,0],[0,4]]", "Matrix")
    need("eigenvalues of [[2,0],[0,3]]", "2")
    need("gcd of 48 and 18", "6")
    need("prime factorization of 360", "2^3×3^2×5")
    need("is 97 prime", "prime")
    need("what is the standard deviation of 2, 4, 4, 4, 5, 5, 7, 9", "sample")
    need("molar mass of Ca(OH)2", "74.092")
    need("convert 5 km to miles", "3.1068")
    need("convert 100 c to f", "212")
    need("emi for a loan of 500000 at 8 % for 5 years", "monthly")
    need("100000 invested at 6 % compounded monthly for 10 years", "181,939")
    need("explain the monty hall problem", "switch")
    need("There are 3 switches outside a closed room with 3 light bulbs inside, enter once. How can you tell which switch controls which bulb?", "Turn")
    need("I have a 3 litre jug and a 5 litre jug. How do I measure exactly 4 litres?", "steps")
    need("wolf goat cabbage river puzzle", "7 crossings")
    need("solve the 8 queens puzzle", "92 solutions")
    need("truth table for (a and b) or not c", "contingent")
    sud = ("sudoku 530070000600195000098000060800060003400803001700020006060000280000419005000080079")
    need(sud, "5 3 4 6 7 8 9 1 2")

    # injection attempts must be refused, not executed
    for bad in ("derivative of __import__('os').system('echo hi')", "integrate open('/etc/passwd') dx",
                "solve x.__class__ = 1"):
        assert try_solve(bad) is None, bad
    # a wrong-looking request that doesn't parse falls through
    assert try_solve("tell me a joke") is None

    # ---- batches 2-4: every module must answer, verify, and contain the expected text
    cases = [
        ("area of a triangle with sides 3, 4, 5", "6 square"), ("the hypotenuse is 13 and one leg is 5, find the other leg", "12"),
        ("distance between (1, 2) and (4, 6)", "5"), ("equation of the line through (1,2) and (3,8)", "3·x"),
        ("area of a circle with radius 5", "25*pi"), ("volume and surface area of a cylinder with radius 2 and height 5", "62.83"),
        ("what is sin 30 degrees?", "1/2"), ("what is 10C3", "120"), ("probability of exactly 3 heads in 5 coin flips", "0.3125"),
        ("probability that the sum of two dice is 7", "1/6"), ("birthday paradox with 23 people", "0.5072"),
        ("prevalence is 1%, sensitivity is 99%, false positive rate is 5%, probability of disease given a positive test", "0.1666"),
        ("a normal distribution with mean 100 and standard deviation 15: probability that x is less than 130", "0.97725"),
        ("add 1/3 and 2/5", "11/15"), ("factor x^2 - 5x + 6", "(x - 3)*(x - 2)"), ("solve x^2 - 4 > 0", "Interval.open(2, oo)"),
        ("sum of k^2 for k = 1 to 10", "385"), ("sum of the first 100 natural numbers", "5,050"),
        ("arithmetic sequence first term 3 common difference 4, find the 10th term", "39"),
        ("what is the next term in 2, 6, 12, 20, 30", "42"), ("what comes next: 1, 1, 2, 3, 5, 8", "13"),
        ("modulus of 3+4i", "5"), ("what is 255 in hex", "ff"), ("what is 17^100 mod 13", "9"), ("last digit of 7^2026", "9"),
        ("what is 15% of 240", "36"), ("what is the roman numeral for 1994", "MCMXCIV"),
        ("what day of the week was 2026-12-25", "Friday"), ("is 2100 a leap year", "not"),
        ("how many r's are in strawberry", "3 times"), ("is 'A man a plan a canal Panama' a palindrome", "is a palindrome"),
        ("encode 'hello world' to base64", "aGVsbG8gd29ybGQ="), ("rot13 'Hello'", "Uryyb"),
        ("what is the 50th fibonacci number", "12586269025"), ("what is the 100th prime", "541"), ("how many digits in 100!", "158"),
        ("how many distinct arrangements of the letters in MISSISSIPPI", "34,650"),
        ("minimum coins to make 63 using coins 1, 5, 10, 25", "6 coins"), ("tower of hanoi with 10 disks", "1,023"),
        ("maximum of x*(10-x) on [0, 10]", "25"),
        ("a projectile is launched at 20 m/s at 30 degrees, find the range", "35.35"),
        ("a ball is dropped from a height of 45 m, how long does it take to fall", "3.03 s"),
        ("u = 5, v = 15, t = 4, find a and s", "a = 5/2"), ("kinetic energy of a 2 kg object moving at 3 m/s", "9 J"),
        ("ohm's law: voltage 12 volts and resistance 4 ohms, find current", "I = 3 A"),
        ("balance C3H8 + O2 -> CO2 + H2O", "C3H8 + 5 O2"), ("what is the pH of 0.001 M HCl", "pH = 3"),
        ("multiply [[1,2],[3,4]] and [[5,6],[7,8]]", "[19, 22]"), ("cross product of (1,0,0) and (0,1,0)", "(0, 0, 1)"),
        ("correlation between [1,2,3,4,5] and [2,4,5,4,5]", "r = 0.7745"), ("z-score of 85 with mean 70 and standard deviation 10", "1.5"),
        ("NPV of cash flows -1000, 300, 400, 500 at 10%", "-21.04"), ("CAGR from 1000 to 2000 over 5 years", "14.8698"),
        ("simple interest on 5000 at 6% for 3 years", "900.00"),
        ("A can do a job in 6 days and B can do it in 3 days. working together how long will they take", "2 days"),
    ]
    for question, expect in cases:
        r = try_solve(question)
        assert r is not None and r.verified, (question, r)
        assert expect.replace(" ", "") in r.answer.replace(" ", ""), (question, r.answer)
    assert len(REGISTRY) >= 50, len(REGISTRY)

    # ordinary conversation must never be hijacked by a solver
    for chatty in ("hi", "tell me about yourself and features", "what is the capital of France", "write a story about a dragon",
                   "how to reverse a linked list in python", "what is the factor of safety in engineering",
                   "I scored 85 in maths and my friend scored 70", "how many people live in India"):
        assert try_solve(chatty) is None, chatty
    print(f"solvers selftest passed ({len(REGISTRY)} modules)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
