#!/usr/bin/env python3
"""
agent_tools
===========
The tools an AI agent can call to solve exam problems reliably:

  lookup_concept  find the core concept, formulas, method and trap for a chapter
  math_tool       exact maths with sympy (solve, limit, diff, integrate, areas, series, matrices, ODEs, ...)
  calc            numeric calculator with physics/chemistry constants and variables
  molar_mass      molar mass from a chemical formula, e.g. CuSO4.5H2O
  ph_calc         pH of strong/weak acids and bases, and buffers

Every tool takes plain arguments and returns a plain string, so it works with any agent framework.
TOOL_SCHEMAS describes them in the format the Anthropic API expects, and run_tool() runs one by name.

Command line (handy for testing)
--------------------------------
    python agent_tools.py --selftest
    python agent_tools.py math_tool '{"operation":"integrate","expression":"x*exp(x)","lower":"0","upper":"1"}'

Safety: expressions are checked before parsing (no attribute access, no double underscores, limited characters).
sympy parsing is still not a full sandbox, so run this on your own machine with trusted settings.
"""

import json
import math
import re
import sys

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from concepts_db import format_chapter, search

X = sp.symbols("x", real=True)
Y = sp.Function("y")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_ALLOWED = re.compile(r"^[A-Za-z0-9_\s\.\,\+\-\*/\^\(\)\[\]=<>!]*$")
_ATTRIBUTE = re.compile(r"[A-Za-z_\)\]]\s*\.\s*[A-Za-z_]")

CONSTANTS = {
    "NA": 6.022e23, "R": 8.314, "F": 96485, "h": 6.626e-34, "kB": 1.381e-23,
    "c": 2.998e8, "eps0": 8.854e-12, "mu0": 4 * math.pi * 1e-7, "qe": 1.602e-19,
    "me": 9.109e-31, "G": 6.674e-11, "hc_eVnm": 1240.0,
}
MAX_OUTPUT = 4000


# ----------------------------------------------------------------- parsing

def _check(text):
    text = str(text)
    if "__" in text or not _ALLOWED.match(text) or _ATTRIBUTE.search(text):
        raise ValueError(f"Expression not allowed or contains unsupported characters: {text!r}")
    return text


def parse(text, local=None):
    text = _check(text).strip()
    names = {"x": X, "y": Y}
    if local:
        names.update(local)
    return parse_expr(text, local_dict=names, transformations=_TRANSFORMS)


def fmt(value):
    """Exact form, plus a decimal when helpful."""
    out = str(value)
    try:
        if getattr(value, "is_number", False) and not getattr(value, "is_Integer", False):
            dec = sp.N(value, 8)
            if str(dec) != out:
                out += f"  (decimal: {dec})"
    except Exception:
        pass
    return out


# ----------------------------------------------------------------- math helpers

def _cuts(f, g, lo, hi):
    try:
        roots = sp.solve(sp.Eq(f, g), X)
    except NotImplementedError:
        roots = []
    keep = []
    for r in roots:
        v = complex(sp.N(r))
        if abs(v.imag) < 1e-12 and float(lo) < v.real < float(hi):
            keep.append(sp.simplify(r))
    return sorted(set(keep), key=lambda c: float(sp.N(c)))


def _piecewise(f, g, lo, hi, pick):
    pts = [lo] + _cuts(f, g, lo, hi) + [hi]
    total, notes = sp.Integer(0), []
    for a, b in zip(pts[:-1], pts[1:]):
        m = (a + b) / 2
        fm, gm = sp.N(f.subs(X, m)), sp.N(g.subs(X, m))
        use, name = pick(f, g, fm, gm)
        piece = sp.simplify(sp.integrate(use, (X, a, b)))
        notes.append(f"on [{a}, {b}] use {name}: {piece}")
        total += piece
    return sp.simplify(total), pts, notes


# ----------------------------------------------------------------- tools

def lookup_concept(query, subject=None, max_results=2):
    results = search(str(query), k=int(max_results), subject=subject)
    if not results:
        return "No matching chapter found. Try different keywords."
    return "\n\n".join(format_chapter(c) for c in results)


def math_tool(operation, expression=None, f=None, g=None, lower=None, upper=None, point=None,
              direction="+-", order=1, even=False, multiplier=1, variables=None, equations=None,
              matrix=None, a=None, b=None, c=None, d=None, n=None, r=None, p=None, k=None,
              initial=None):
    op = operation
    if op == "evaluate":
        return fmt(sp.N(parse(expression), 10))

    if op == "simplify":
        return fmt(sp.simplify(parse(expression)))

    if op == "solve":
        if equations:
            names = variables or ["x", "y"]
            syms = sp.symbols(" ".join(names))
            if not isinstance(syms, tuple):
                syms = (syms,)
            local = {str(s): s for s in syms}
            eqs = []
            for e in equations:
                e = _check(e)
                if "=" in e:
                    left, right = e.split("=", 1)
                    eqs.append(parse(left, local) - parse(right, local))
                else:
                    eqs.append(parse(e, local))
            return fmt(sp.solve(eqs, syms, dict=True))
        text = _check(expression)
        if "=" in text:
            left, right = text.split("=", 1)
            expr = parse(left) - parse(right)
        else:
            expr = parse(text)
        return "Solutions: " + fmt(sp.solve(expr, X))

    if op == "limit":
        pt = sp.oo if str(point).strip() in ("oo", "inf", "infinity") else parse(point)
        return "Limit: " + fmt(sp.limit(parse(expression), X, pt, dir=direction))

    if op == "diff":
        return fmt(sp.simplify(sp.diff(parse(expression), X, int(order))))

    if op == "integrate":
        e = parse(expression)
        if lower is None or upper is None:
            return "Indefinite integral (add +C): " + fmt(sp.integrate(e, X))
        return "Definite integral: " + fmt(sp.simplify(sp.integrate(e, (X, parse(lower), parse(upper)))))

    if op in ("area_between", "area_under_min"):
        F, G = parse(f), parse(g)
        lo, hi = sp.nsimplify(parse(lower)), sp.nsimplify(parse(upper))
        if op == "area_between":
            pick = lambda F_, G_, fm, gm: ((F_ - G_), "(f-g)") if fm >= gm else ((G_ - F_), "(g-f)")
        else:
            pick = lambda F_, G_, fm, gm: (F_, "f") if fm <= gm else (G_, "g")
        area, pts, notes = _piecewise(F, G, lo, hi, pick)
        if op == "area_under_min" and even:
            area = sp.simplify(2 * area)
        lines = ["Split points: " + str(pts)] + notes
        label = "Area (doubled for even symmetry)" if (op == "area_under_min" and even) else "Area"
        lines.append(f"{label}: {fmt(area)}")
        if multiplier not in (1, "1", None):
            m = parse(multiplier)
            lines.append(f"{multiplier} x area: {fmt(sp.simplify(m * area))}")
        return "\n".join(lines)

    if op == "max_min":
        e = parse(expression)
        lo, hi = sp.nsimplify(parse(lower)), sp.nsimplify(parse(upper))
        crit = [t for t in sp.solve(sp.diff(e, X), X) if t.is_real and float(lo) <= float(t) <= float(hi)]
        vals = [(t, sp.simplify(e.subs(X, t))) for t in [lo, hi] + crit]
        mx = max(vals, key=lambda t: float(sp.N(t[1])))
        mn = min(vals, key=lambda t: float(sp.N(t[1])))
        return (f"Candidates (x, f): {vals}\nMaximum {fmt(mx[1])} at x={mx[0]}\n"
                f"Minimum {fmt(mn[1])} at x={mn[0]}")

    if op == "ap_sum":
        A, D, N = parse(a), parse(d), parse(n)
        return f"Last term: {fmt(sp.simplify(A + (N - 1) * D))}\nSum: {fmt(sp.simplify(N / 2 * (2 * A + (N - 1) * D)))}"

    if op == "gp_sum":
        A, R_ = parse(a), parse(r)
        if n is None or str(n) in ("oo", "inf"):
            if abs(float(sp.N(R_))) >= 1:
                return "Infinite GP does not converge (|r| >= 1)."
            return "Sum to infinity: " + fmt(sp.simplify(A / (1 - R_)))
        N = parse(n)
        return "Sum: " + fmt(sp.simplify(A * (R_**N - 1) / (R_ - 1)))

    if op == "quadratic":
        A, B, C_ = parse(a), parse(b), parse(c)
        D_ = sp.simplify(B**2 - 4 * A * C_)
        return (f"Discriminant: {fmt(D_)}\nSum of roots: {fmt(sp.simplify(-B / A))}\n"
                f"Product of roots: {fmt(sp.simplify(C_ / A))}\nRoots: {fmt(sp.solve(A * X**2 + B * X + C_, X))}")

    if op == "matrix":
        M = sp.Matrix(sp.sympify(_check(matrix)))
        out = [f"Determinant: {fmt(M.det())}", f"Rank: {M.rank()}"]
        if M.is_square and M.det() != 0:
            out += [f"Inverse: {M.inv().tolist()}", f"Adjoint: {M.adjugate().tolist()}"]
        else:
            out.append("No inverse (singular or non-square).")
        return "\n".join(out)

    if op == "ode":
        eq = parse(expression)
        ics = None
        if initial:
            left, right = _check(initial).split("=", 1)
            ics = {parse(left): parse(right)}
        sol = sp.dsolve(eq, Y(X), ics=ics) if ics else sp.dsolve(eq, Y(X))
        return "Solution: " + str(sol)

    if op == "binomial":
        N, K, Pp = int(n), int(k), sp.nsimplify(p)
        prob = sp.binomial(N, K) * Pp**K * (1 - Pp) ** (N - K)
        return f"P(X={K}) = {fmt(prob)}\nMean = {fmt(N * Pp)}\nVariance = {fmt(N * Pp * (1 - Pp))}"

    if op == "combinatorics":
        N, K = int(n), int(k)
        return f"nPk = {sp.factorial(N) / sp.factorial(N - K)}\nnCk = {sp.binomial(N, K)}"

    raise ValueError(f"Unknown operation: {op}")


def calc(expression, variables=None):
    local = {k: sp.Float(v) for k, v in CONSTANTS.items()}
    for key, val in (variables or {}).items():
        if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", str(key)):
            raise ValueError(f"Bad variable name: {key!r}")
        local[str(key)] = sp.nsimplify(val) if isinstance(val, str) else sp.Float(val)
    value = sp.N(parse(expression, local), 10)
    return f"Result: {value}"


_MASS = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Br": 79.904, "Ag": 107.87,
    "Sn": 118.71, "I": 126.90, "Ba": 137.33, "Au": 196.97, "Hg": 200.59, "Pb": 207.2, "Ti": 47.867,
    "V": 50.942, "As": 74.922, "Rb": 85.468, "Sr": 87.62, "Cd": 112.41, "Pt": 195.08, "U": 238.03,
}


def _formula_mass(formula):
    tokens = re.findall(r"([A-Z][a-z]?|\(|\)|\d+)", formula)
    if "".join(tokens) != formula:
        raise ValueError(f"Cannot read formula: {formula!r}")
    stack = [[]]
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "(":
            stack.append([])
        elif t == ")":
            group = stack.pop()
            mult = 1
            if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                mult = int(tokens[i + 1])
                i += 1
            stack[-1].extend((el, cnt * mult) for el, cnt in group)
        elif t.isdigit():
            raise ValueError(f"Unexpected number in {formula!r}")
        else:
            mult = 1
            if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                mult = int(tokens[i + 1])
                i += 1
            stack[-1].append((t, mult))
        i += 1
    total = 0.0
    for el, cnt in stack[0]:
        if el not in _MASS:
            raise ValueError(f"Element not in table: {el}")
        total += _MASS[el] * cnt
    return total


def molar_mass(formula):
    formula = formula.strip().replace("·", ".").replace(" ", "")
    total = 0.0
    for part in formula.split("."):
        m = re.match(r"^(\d+)(.*)$", part)
        mult, body = (int(m.group(1)), m.group(2)) if m else (1, part)
        total += mult * _formula_mass(body)
    return f"Molar mass of {formula} = {total:.3f} g/mol (standard atomic masses; round as your exam expects)"


def ph_calc(kind, concentration, ka=None, kb=None, salt=None, pka=None):
    conc = float(concentration)
    kw = 1e-14
    if kind == "strong_acid":
        h = (conc + math.sqrt(conc**2 + 4 * kw)) / 2
        return f"[H+] = {h:.4g} M, pH = {-math.log10(h):.4f}"
    if kind == "strong_base":
        oh = (conc + math.sqrt(conc**2 + 4 * kw)) / 2
        return f"[OH-] = {oh:.4g} M, pOH = {-math.log10(oh):.4f}, pH = {14 + math.log10(oh):.4f}"
    if kind == "weak_acid":
        K = float(ka)
        h = (-K + math.sqrt(K * K + 4 * K * conc)) / 2
        return f"[H+] = {h:.4g} M, pH = {-math.log10(h):.4f}, degree of dissociation = {h / conc:.4g}"
    if kind == "weak_base":
        K = float(kb)
        oh = (-K + math.sqrt(K * K + 4 * K * conc)) / 2
        return f"[OH-] = {oh:.4g} M, pOH = {-math.log10(oh):.4f}, pH = {14 + math.log10(oh):.4f}"
    if kind == "buffer":
        ph = float(pka) + math.log10(float(salt) / conc)
        return f"Henderson-Hasselbalch: pH = {ph:.4f} (concentration = acid, salt = conjugate base)"
    raise ValueError("kind must be strong_acid, strong_base, weak_acid, weak_base or buffer")


# ----------------------------------------------------------------- schemas

_S = {"type": "string"}
TOOL_SCHEMAS = [
    {
        "name": "lookup_concept",
        "description": "Look up the core concept, key formulas, solving method and common trap for a chapter. "
                       "Use this first to choose the right method. Query with topic words, e.g. 'area under curve' "
                       "or 'projectile motion'.",
        "input_schema": {"type": "object", "properties": {
            "query": _S,
            "subject": {"type": "string", "enum": ["maths", "physics", "chemistry", "biology"]},
            "max_results": {"type": "integer"}}, "required": ["query"]},
    },
    {
        "name": "math_tool",
        "description": (
            "Exact maths with sympy. Variable is x (use y(x) only for ODEs). Use ** or ^ for powers, exp(x), sqrt(x), "
            "sin(x), oo for infinity. Operations: evaluate, simplify, solve (expression with optional '=', or equations "
            "+ variables for a system), limit (expression, point, direction), diff (expression, order), integrate "
            "(expression, optional lower/upper), area_between (f, g, lower, upper), area_under_min (f, g, lower, upper, "
            "even, multiplier; for even symmetry give the x>=0 form on [0, upper]), max_min (expression, lower, upper), "
            "ap_sum (a, d, n), gp_sum (a, r, optional n), quadratic (a, b, c), matrix (matrix like [[1,2],[3,4]]), "
            "ode (expression like Derivative(y(x), x) - y(x); optional initial like y(0)=1), binomial (n, p, k), "
            "combinatorics (n, k)."),
        "input_schema": {"type": "object", "properties": {
            "operation": {"type": "string", "enum": [
                "evaluate", "simplify", "solve", "limit", "diff", "integrate", "area_between", "area_under_min",
                "max_min", "ap_sum", "gp_sum", "quadratic", "matrix", "ode", "binomial", "combinatorics"]},
            "expression": _S, "f": _S, "g": _S, "lower": _S, "upper": _S, "point": _S,
            "direction": {"type": "string", "enum": ["+", "-", "+-"]},
            "order": {"type": "integer"}, "even": {"type": "boolean"}, "multiplier": _S,
            "variables": {"type": "array", "items": _S}, "equations": {"type": "array", "items": _S},
            "matrix": _S, "a": _S, "b": _S, "c": _S, "d": _S, "n": _S, "r": _S, "p": _S, "k": _S,
            "initial": _S}, "required": ["operation"]},
    },
    {
        "name": "calc",
        "description": "Numeric calculator for physics/chemistry arithmetic. Pass an expression and a variables object "
                       "(name to number). Built-in constants: NA, R (8.314), F (96485), h, kB, c, eps0, mu0, qe, me, G, "
                       "hc_eVnm (1240). Override any constant (for example F=96500 or c=3e8) via variables. "
                       "Example: expression 'u**2*sin(2*t)/g', variables {u:20, t:0.5235988, g:10}.",
        "input_schema": {"type": "object", "properties": {
            "expression": _S, "variables": {"type": "object"}}, "required": ["expression"]},
    },
    {
        "name": "molar_mass",
        "description": "Molar mass of a chemical formula, e.g. 'H2SO4', 'Ca(OH)2' or 'CuSO4.5H2O'.",
        "input_schema": {"type": "object", "properties": {"formula": _S}, "required": ["formula"]},
    },
    {
        "name": "ph_calc",
        "description": "pH of a solution. kind: strong_acid, strong_base, weak_acid (needs ka), weak_base (needs kb), "
                       "buffer (concentration = acid, salt = conjugate base concentration, pka).",
        "input_schema": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["strong_acid", "strong_base", "weak_acid", "weak_base", "buffer"]},
            "concentration": {"type": "number"}, "ka": {"type": "number"}, "kb": {"type": "number"},
            "salt": {"type": "number"}, "pka": {"type": "number"}}, "required": ["kind", "concentration"]},
    },
]

_TOOLS = {"lookup_concept": lookup_concept, "math_tool": math_tool, "calc": calc,
          "molar_mass": molar_mass, "ph_calc": ph_calc}


def run_tool(name, arguments):
    """Run a tool by name. Always returns a string (errors are returned as text for the model to read)."""
    try:
        if name not in _TOOLS:
            return f"Error: unknown tool {name!r}"
        result = _TOOLS[name](**(arguments or {}))
        result = str(result)
        return result if len(result) <= MAX_OUTPUT else result[:MAX_OUTPUT] + "\n...(output cut)"
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"


# ----------------------------------------------------------------- self-test

def selftest():
    bad = 0

    def expect(label, got, *needles):
        nonlocal bad
        ok = all(n in got for n in needles)
        print(("PASS " if ok else "FAIL ") + label + ("" if ok else f"\n   got: {got}"))
        bad += 0 if ok else 1

    expect("JEE 2022 area x9", run_tool("math_tool", {
        "operation": "area_under_min", "f": "x**2 + 5/12", "g": "2 - 8*x/3", "lower": "0",
        "upper": "3/4", "even": True, "multiplier": "9"}), "9 x area: 6")
    expect("definite integral", run_tool("math_tool", {
        "operation": "integrate", "expression": "x*exp(x)", "lower": "0", "upper": "1"}), "Definite integral: 1")
    expect("area between", run_tool("math_tool", {
        "operation": "area_between", "f": "x", "g": "x**2", "lower": "0", "upper": "1"}), "Area: 1/6")
    expect("limit", run_tool("math_tool", {"operation": "limit", "expression": "sin(3*x)/(2*x)", "point": "0"}), "3/2")
    expect("solve equation", run_tool("math_tool", {"operation": "solve", "expression": "x**2 - 5*x + 6 = 0"}), "[2, 3]")
    expect("solve system", run_tool("math_tool", {
        "operation": "solve", "equations": ["x + y = 5", "x - y = 1"], "variables": ["x", "y"]}), "3", "2")
    expect("max/min", run_tool("math_tool", {
        "operation": "max_min", "expression": "x**3 - 3*x", "lower": "-2", "upper": "3"}), "Maximum 18")
    expect("AP sum", run_tool("math_tool", {"operation": "ap_sum", "a": "3", "d": "4", "n": "15"}), "Sum: 465")
    expect("matrix", run_tool("math_tool", {"operation": "matrix", "matrix": "[[2,1],[5,3]]"}), "Determinant: 1")
    expect("ODE with condition", run_tool("math_tool", {
        "operation": "ode", "expression": "Derivative(y(x), x) - y(x)", "initial": "y(0)=2"}), "2*exp(x)")
    expect("binomial", run_tool("math_tool", {"operation": "binomial", "n": "10", "p": "1/2", "k": "3"}), "15/128")
    expect("calc with variables", run_tool("calc", {
        "expression": "u**2*sin(2*t)/g", "variables": {"u": 20, "t": math.pi / 6, "g": 10}}), "34.64")
    expect("calc constant", run_tool("calc", {"expression": "hc_eVnm/500"}), "2.48")
    expect("molar mass water", run_tool("molar_mass", {"formula": "H2O"}), "18.015")
    expect("molar mass hydrate", run_tool("molar_mass", {"formula": "CuSO4.5H2O"}), "249.677")
    expect("molar mass brackets", run_tool("molar_mass", {"formula": "Ca(OH)2"}), "74.09")
    expect("pH strong acid", run_tool("ph_calc", {"kind": "strong_acid", "concentration": 0.001}), "pH = 3.0000")
    expect("pH very dilute acid", run_tool("ph_calc", {"kind": "strong_acid", "concentration": 1e-8}), "pH = 6.9783")
    expect("pH weak acid", run_tool("ph_calc", {"kind": "weak_acid", "concentration": 0.1, "ka": 1.8e-5}), "pH = 2.8")
    expect("buffer", run_tool("ph_calc", {"kind": "buffer", "concentration": 0.1, "salt": 1.0, "pka": 4.74}), "5.74")
    expect("concept lookup", run_tool("lookup_concept", {"query": "area under curve", "max_results": 1}), "Area Under Curves")
    expect("blocks unsafe code", run_tool("math_tool", {
        "operation": "evaluate", "expression": "__import__('os').system('echo hi')"}), "Error")
    expect("blocks attribute access", run_tool("math_tool", {
        "operation": "evaluate", "expression": "x.free_symbols"}), "Error")
    expect("unknown tool", run_tool("nope", {}), "unknown tool")
    print("\nAll tool tests passed." if bad == 0 else f"\n{bad} test(s) failed.")
    return 1 if bad else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--selftest":
        return selftest()
    name = argv[0]
    args = json.loads(argv[1]) if len(argv) > 1 else {}
    print(run_tool(name, args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
