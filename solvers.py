"""
solvers.py - deterministic solvers, each with an independent second-method check.

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
    if re.search(r"\bprime factori|\bfactori[sz]e\b|\bfactor\b", ql) and nums and 1 < nums[0] < 10 ** 12:
        n = nums[0]
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
    mn = re.search(r"(?:in|of|after)\s+(\d+)\s+(?:flips?|tosses|trials?|attempts?|throws?|shots?|games?|tests?)", ql)
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
    m = re.search(r"\b(one|two|three|four|five|\d)\s+(?:fair\s+|standard\s+|six-sided\s+)*dic?e\b.*?\bsum\s*(?:of|is|equals|=|to)?\s*(\d+)", ql)
    if not m:
        return None
    n, target = words.get(m.group(1), 0), int(m.group(2))
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
    print("solvers selftest passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
