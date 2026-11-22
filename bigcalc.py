#!/usr/bin/env python3
"""
bigcalc.py - exact and arbitrary-precision engine for very large / very precise problems.

Pure standard library (decimal, fractions, ast): no numpy, scipy, sympy or mpmath needed.

  * Integers and fractions are EXACT (2**4423-1, 5000!, comb(10**6, 5)...).
  * Transcendental functions and constants run at any number of digits (--digits 500).
  * High-precision integrals, series and closed-form recovery (integer relations by LLL
    on 40-100+ digit numbers, so far longer formulas than double precision allows).

Commands
  calc       "factorial(2000)"  /  "sqrt(2)" --digits 200  /  "powmod(3, 10**20, 10**9+7)"
  hpintegral "ln(1+x)/(1+x^2)" 0 1 --digits 40
  hpsum      "1/n^2" --lo 1                       (positive terms, Van Wijngaarden + CVZ)
  hpsum      "1/n" --lo 1 --alt                   (alternating, CVZ acceleration)
  hpsum      "n^2" --lo 1 --hi 100000             (finite: EXACT rational result)
  hpidentify "pi*ln(2)/8" --digits 40 [--basis "1,pi,ln(2)"]
  selftest
Nothing is hardcoded as an answer; selftest keeps its own oracles separate.
"""
import argparse
import ast
import decimal
import math
import re
import sys
import time
from decimal import Decimal, localcontext, getcontext
from fractions import Fraction

try:
    sys.set_int_max_str_digits(0)
except AttributeError:
    pass

GUARD = 25
D = Decimal


# ======================================================================================
# arbitrary-precision constants and functions (Decimal)
# ======================================================================================
_cache = {}


def _ctx_setup(c, prec):
    c.prec = prec
    c.Emax = decimal.MAX_EMAX
    c.Emin = decimal.MIN_EMIN


def _cached(name, fn):
    key = (name, getcontext().prec)
    if key not in _cache:
        with localcontext() as c:
            c.prec += 12
            v = fn()
        _cache[key] = +v
    return +_cache[key]


def _atan_inv(n):
    x = D(1) / n
    x2 = x * x
    p, s, k = x, x, 1
    eps = D(10) ** (-(getcontext().prec + 3))
    while abs(p) > eps:
        p = -p * x2
        k += 2
        s += p / k
    return s


def d_pi():
    return _cached("pi", lambda: 16 * _atan_inv(5) - 4 * _atan_inv(239))


def d_e():
    return _cached("e", lambda: D(1).exp())


def d_atan(x):
    x = D(x)
    with localcontext() as c:
        c.prec += 10
        neg = x < 0
        if neg:
            x = -x
        inv = x > 1
        if inv:
            x = 1 / x
        k = 0
        while x > D("0.01"):
            x = x / (1 + (1 + x * x).sqrt())
            k += 1
        x2 = x * x
        p, s, j = x, x, 1
        eps = D(10) ** (-(c.prec + 3))
        while abs(p) > eps:
            p = -p * x2
            j += 2
            s += p / j
        s *= 2 ** k
        if inv:
            s = d_pi() / 2 - s
        if neg:
            s = -s
    return +s


def d_sincos(x):
    x = D(x)
    with localcontext() as c:
        c.prec += max(0, x.adjusted()) + 12
        half = d_pi() / 2
        q = int((x / half).to_integral_value())
        r = x - q * half
        r2 = r * r
        eps = D(10) ** (-(c.prec + 3))
        s = t = r
        k = 1
        while abs(t) > eps:
            t = -t * r2 / ((2 * k) * (2 * k + 1))
            s += t
            k += 1
        co = t = D(1)
        k = 1
        while abs(t) > eps:
            t = -t * r2 / ((2 * k - 1) * (2 * k))
            co += t
            k += 1
        sn, cs = [(s, co), (co, -s), (-s, -co), (-co, s)][q % 4]
    return +sn, +cs


def d_sin(x):
    return d_sincos(x)[0]


def d_cos(x):
    return d_sincos(x)[1]


def d_tan(x):
    s, c = d_sincos(x)
    return s / c


def d_asin(x):
    x = D(x)
    if abs(x) > 1:
        raise ValueError("asin domain")
    if abs(x) == 1:
        return d_pi() / 2 * (1 if x > 0 else -1)
    return d_atan(x / (1 - x * x).sqrt())


def d_acos(x):
    return d_pi() / 2 - d_asin(x)


def d_sinh(x):
    x = D(x)
    return (x.exp() - (-x).exp()) / 2


def d_cosh(x):
    x = D(x)
    return (x.exp() + (-x).exp()) / 2


def d_tanh(x):
    x = D(x)
    a, b = x.exp(), (-x).exp()
    return (a - b) / (a + b)


def d_catalan():
    def calc():
        s, t, n = D(0), D(1), 0           # t = 1/C(2n,n)
        eps = D(10) ** (-(getcontext().prec + 3))
        while True:
            term = t / ((2 * n + 1) ** 2)
            s += term
            if term < eps:
                break
            n += 1
            t = t * n / (2 * (2 * n - 1))  # 1/C(2n,n) recurrence
        return d_pi() / 8 * (2 + D(3).sqrt()).ln() + D(3) / 8 * s
    return _cached("catalan", calc)


def d_zeta(s):
    """Borwein's algorithm; s real > 0, s != 1."""
    s = D(s)
    with localcontext() as c:
        c.prec += 8
        n = int(c.prec * 1.31) + 4
        t = Fraction(1)
        ds, acc = [], Fraction(0)
        for i in range(n + 1):
            acc += t
            ds.append(acc)
            t = t * 4 * (n + i) * (n - i) / ((2 * i + 1) * (2 * i + 2)) if i < n else t
        dn = D(ds[n].numerator) / D(ds[n].denominator)
        tot = D(0)
        for k in range(n):
            dk = D(ds[k].numerator) / D(ds[k].denominator)
            term = (dk - dn) / ((D(k + 1).ln() * s).exp())
            tot += -term if k % 2 else term
        z = -tot / (dn * (1 - (D(2) ** (1 - s) if s == s.to_integral_value() and s >= 1
                             else ((1 - s) * D(2).ln()).exp())))
    return +z


# ======================================================================================
# safe expression evaluator: exact int/Fraction, Decimal when irrational
# ======================================================================================
def norm(v):
    if isinstance(v, Fraction) and v.denominator == 1:
        return int(v.numerator)
    return v


def to_dec(v):
    if isinstance(v, D):
        return v
    if isinstance(v, int):
        return D(v)
    if isinstance(v, Fraction):
        return D(v.numerator) / D(v.denominator)
    raise TypeError(f"cannot convert {type(v)}")


def exact_root(a, q):
    """Exact q-th root of a nonnegative int/Fraction, or None."""
    def iroot(n):
        if n < 0:
            return None
        r = round(n ** (1.0 / q)) if n < 10 ** 300 else None
        if r is None:
            lo, hi = 0, 1 << (n.bit_length() // q + 1)
            while lo < hi:
                mid = (lo + hi) // 2
                if mid ** q < n:
                    lo = mid + 1
                else:
                    hi = mid
            r = lo
        for cand in (r - 1, r, r + 1):
            if cand >= 0 and cand ** q == n:
                return cand
        return None
    if isinstance(a, int):
        r = iroot(a)
        return r
    n, d = iroot(a.numerator), iroot(a.denominator)
    return None if n is None or d is None else Fraction(n, d)


def v_pow(a, b):
    if isinstance(a, D) or isinstance(b, D):
        a, b = to_dec(a), to_dec(b)
        if b == b.to_integral_value() and abs(b) < 10 ** 9:
            return a ** int(b)
        if a <= 0:
            raise ValueError("power of non-positive base with non-integer exponent")
        return (b * a.ln()).exp()
    b = Fraction(b)
    if b.denominator == 1:
        e = int(b)
        base_bits = (a if isinstance(a, int) else max(a.numerator, a.denominator)).bit_length()
        if abs(e) * base_bits > 4 * 10 ** 8:
            raise SystemExit("result too large to compute (limit about 4e8 bits)")
        if isinstance(a, int):
            return a ** e if e >= 0 else Fraction(1, a ** -e)
        return norm(Fraction(a) ** e)
    r = exact_root(Fraction(a) if not isinstance(a, int) else a, b.denominator)
    if r is not None:
        return v_pow(r, b.numerator)
    return v_pow(to_dec(a), to_dec(b))


def v_floor(v):
    if isinstance(v, int):
        return v
    if isinstance(v, Fraction):
        return v.numerator // v.denominator
    return int(v.to_integral_value(rounding=decimal.ROUND_FLOOR))


def binop(op, a, b):
    if isinstance(a, D) or isinstance(b, D):
        a, b = to_dec(a), to_dec(b)
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b
        if op == "//":
            return D(v_floor(a / b))
        if op == "%":
            return a - b * D(v_floor(a / b))
        if op == "**":
            return v_pow(a, b)
    if op == "**":
        return v_pow(a, b)
    if isinstance(a, int) and isinstance(b, int):
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return norm(Fraction(a, b))
        if op == "//":
            return a // b
        if op == "%":
            return a % b
    fa, fb = Fraction(a), Fraction(b)
    if op == "+":
        return norm(fa + fb)
    if op == "-":
        return norm(fa - fb)
    if op == "*":
        return norm(fa * fb)
    if op == "/":
        return norm(fa / fb)
    if op == "//":
        return v_floor(fa / fb)
    if op == "%":
        return norm(fa - fb * v_floor(fa / fb))
    raise ValueError(op)


def f_sqrt(v):
    if isinstance(v, (int, Fraction)) and v >= 0:
        r = exact_root(v, 2)
        if r is not None:
            return r
    v = to_dec(v)
    if v < 0:
        raise ValueError("sqrt of negative number")
    return v.sqrt()


def _fib(n):
    def fd(k):
        if k == 0:
            return 0, 1
        a, b = fd(k >> 1)
        c = a * (2 * b - a)
        d = a * a + b * b
        return (d, c + d) if k & 1 else (c, d)
    return fd(int(n))[0]


def _is_prime(n):
    n = int(n)
    if n < 2:
        return 0
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in small:
        if n % p == 0:
            return int(n == p)
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return 0
    return 1


def _nextprime(n):
    n = int(n) + 1
    while not _is_prime(n):
        n += 1
    return n


def _need_int(*a):
    for v in a:
        if not isinstance(v, int):
            raise ValueError("integer argument required")


def _cap(n, limit, name):
    if n > limit:
        raise SystemExit(f"{name}: argument too large (limit {limit})")
    return n


FUNCS = {
    "sqrt": f_sqrt,
    "exp": lambda v: to_dec(v).exp(),
    "ln": lambda v: to_dec(v).ln(),
    "log": lambda v: to_dec(v).ln(),
    "log10": lambda v: to_dec(v).log10(),
    "log2": lambda v: to_dec(v).ln() / D(2).ln(),
    "sin": d_sin, "cos": d_cos, "tan": d_tan,
    "asin": d_asin, "acos": d_acos, "atan": d_atan,
    "sinh": d_sinh, "cosh": d_cosh, "tanh": d_tanh,
    "zeta": lambda s: d_zeta(to_dec(s)),
    "abs": lambda v: abs(v),
    "floor": v_floor,
    "ceil": lambda v: -v_floor(-v if not isinstance(v, D) else -v),
    "factorial": lambda n: (_need_int(n), math.factorial(_cap(n, 5 * 10 ** 6, "factorial")))[1],
    "comb": lambda n, k: (_need_int(n, k), math.comb(n, k))[1],
    "perm": lambda n, k: (_need_int(n, k), math.perm(n, k))[1],
    "gcd": lambda *a: (_need_int(*a), math.gcd(*a))[1],
    "lcm": lambda *a: (_need_int(*a), math.lcm(*a))[1],
    "isqrt": lambda n: (_need_int(n), math.isqrt(n))[1],
    "fib": lambda n: (_need_int(n), _fib(_cap(n, 10 ** 8, "fib")))[1],
    "powmod": lambda a, e, m: (_need_int(a, e, m), pow(a, e, m))[1],
    "modinv": lambda a, m: (_need_int(a, m), pow(a, -1, m))[1],
    "is_prime": lambda n: (_need_int(n), _is_prime(n))[1],
    "nextprime": lambda n: (_need_int(n), _nextprime(n))[1],
}
CONSTS = {"pi": d_pi, "e": d_e, "catalan": d_catalan, "G": d_catalan}


def Q(text):
    return norm(Fraction(text))


_LIT = re.compile(r"(?<![\w.])((?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)")


def _wrap_literals(src):
    def rep(m):
        t = m.group(1)
        return f"Q('{t}')" if re.search(r"[.eE]", t) else t
    return _LIT.sub(rep, src)


class Expr:
    """Parsed once, evaluated many times (inside a Decimal context set by the caller)."""

    def __init__(self, text, variables=()):
        src = _wrap_literals(text.replace("^", "**"))
        self.tree = ast.parse(src, mode="eval").body
        self.vars = tuple(variables)
        self._check(self.tree)

    def _check(self, n):
        ok = (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant, ast.Load,
              ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
              ast.USub, ast.UAdd)
        if not isinstance(n, ok):
            raise SystemExit(f"unsupported syntax in expression: {type(n).__name__}")
        if isinstance(n, ast.Name) and n.id not in CONSTS and n.id not in self.vars \
                and n.id not in FUNCS and n.id != "Q":
            raise SystemExit(f"unknown name: {n.id}")
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.keywords:
                raise SystemExit("only plain function calls are allowed")
        for ch in ast.iter_child_nodes(n):
            self._check(ch)

    def __call__(self, *vals):
        env = dict(zip(self.vars, vals))
        return self._ev(self.tree, env)

    def _ev(self, n, env):
        if isinstance(n, ast.Constant):
            if isinstance(n.value, bool) or not isinstance(n.value, (int, float)):
                raise SystemExit("unsupported constant")
            return n.value if isinstance(n.value, int) else Q(repr(n.value))
        if isinstance(n, ast.Name):
            if n.id in env:
                return env[n.id]
            if n.id in CONSTS:
                return CONSTS[n.id]()
            raise SystemExit(f"{n.id} used without arguments")
        if isinstance(n, ast.UnaryOp):
            v = self._ev(n.operand, env)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp):
            ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
                   ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**"}
            return binop(ops[type(n.op)], self._ev(n.left, env), self._ev(n.right, env))
        if isinstance(n, ast.Call):
            name = n.func.id
            args = [self._ev(a, env) for a in n.args]
            if name == "Q":
                return Q(args[0]) if isinstance(args[0], str) else args[0]
            return FUNCS[name](*args)
        raise SystemExit("unsupported expression")


def _q_literal_fix():
    """Q('...') arrives as a Constant str; let the evaluator return the string itself."""
    orig = Expr._ev

    def ev(self, n, env):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return n.value
        return orig(self, n, env)
    Expr._ev = ev


_q_literal_fix()


def digits_ctx(digits):
    c = localcontext()
    ctx = c.__enter__()
    _ctx_setup(ctx, digits + GUARD)
    return c


class Prec:
    def __init__(self, digits):
        self.digits = digits

    def __enter__(self):
        self.c = localcontext()
        ctx = self.c.__enter__()
        _ctx_setup(ctx, self.digits + GUARD)
        return ctx

    def __exit__(self, *a):
        return self.c.__exit__(*a)


# ======================================================================================
# rendering
# ======================================================================================
def render(v, digits, full=False):
    lines = []
    if isinstance(v, int):
        s = str(v)
        neg = s.startswith("-")
        body = s.lstrip("-")
        if len(body) <= 400 or full:
            lines.append(s)
        else:
            lines.append(f"{'-' if neg else ''}{body[:40]}...{body[-40:]}")
        lines.append(f"  ({len(body)} decimal digits, {v.bit_length()} bits)")
        if len(body) > 400 and not full:
            lines.append("  (use --full or --save FILE for every digit)")
    elif isinstance(v, Fraction):
        p, q = str(v.numerator), str(v.denominator)
        if len(p) + len(q) > 300 and not full:
            lines.append(f"{p[:30]}...{p[-20:]} / {q[:30]}...{q[-20:]}")
        else:
            lines.append(f"{v.numerator}/{v.denominator}")
        with Prec(digits) as ctx:
            ctx.prec = digits
            lines.append(f"  = {+(D(v.numerator) / D(v.denominator))}")
    else:
        with Prec(digits) as ctx:
            ctx.prec = digits
            lines.append(str(+v))
    return "\n".join(lines)


# ======================================================================================
# integer relations (fast LLL) for high-precision closed forms
# ======================================================================================
def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def lll_fast(B, delta=Fraction(99, 100)):
    n = len(B)
    B = [list(map(int, r)) for r in B]
    Bs = [None] * n
    bn = [None] * n
    mu = [[Fraction(0)] * n for _ in range(n)]
    valid = [0]

    def ensure(k):
        for i in range(valid[0], k + 1):
            v = [Fraction(x) for x in B[i]]
            for j in range(i):
                mu[i][j] = Fraction(_dot(B[i], Bs[j])) / bn[j]
                v = [a - mu[i][j] * b for a, b in zip(v, Bs[j])]
            Bs[i] = v
            bn[i] = _dot(v, v)
        valid[0] = max(valid[0], k + 1)

    k = 1
    while k < n:
        ensure(k)
        for j in range(k - 1, -1, -1):
            q = round(mu[k][j])
            if q:
                B[k] = [a - q * b for a, b in zip(B[k], B[j])]
                valid[0] = min(valid[0], k)
                ensure(k)
        if bn[k] >= (delta - mu[k][k - 1] ** 2) * bn[k - 1]:
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            valid[0] = min(valid[0], k - 1)
            k = max(k - 1, 1)
    return B


def fmt_combo(coefs, names):
    parts = []
    for c, nm in zip(coefs, names):
        if c == 0:
            continue
        a = abs(c)
        if nm == "1":
            term = str(a)
        elif a == 1:
            term = nm
        elif a.denominator == 1:
            term = f"{a.numerator}*{nm}"
        else:
            term = f"({a})*{nm}"
        parts.append(("-" if c < 0 else "+", term))
    if not parts:
        return "0"
    s = ("-" if parts[0][0] == "-" else "") + parts[0][1]
    for sg, t in parts[1:]:
        s += f" {sg} {t}"
    return s


def hp_relation(v, basis, digits_ok, digits, maxcoef):
    """Find rational combination v = sum r_i * basis_i. v is a Decimal accurate to about
    digits_ok digits. Returns dict or None."""
    with Prec(digits):
        vals = [to_dec(Expr(b)()) for b in basis]
        xs = [v] + vals
        n = len(xs)
        S = max(digits_ok - 3, 8)
        scale = 10 ** S
        B = [[1 if i == j else 0 for j in range(n)] + [int((xs[i] * scale).to_integral_value())]
             for i in range(n)]
        for r in lll_fast(B):
            c = list(r[:n])
            if c[0] == 0 or max(abs(t) for t in c) > maxcoef:
                continue
            g = math.gcd(*[abs(t) for t in c])
            c = [t // g for t in c]
            resid = abs(sum(D(ci) * xi for ci, xi in zip(c, xs)))
            spread = sum(abs(D(ci) * xi) for ci, xi in zip(c, xs))
            err = D(10) ** (-digits_ok)
            tol = 8 * (abs(c[0]) * err + spread * D(10) ** (-(digits + GUARD - 3))) + D(10) ** (-(digits_ok + 2))
            if resid > tol:
                continue
            nvec = 1.0
            for ci in c:
                nvec *= 2 * abs(ci) + 1
            expected = nvec * float(tol / max(spread, D("1e-300")))
            if expected > 1e-6:
                continue
            coefs = [Fraction(-t, c[0]) for t in c[1:]]
            return {"expr": fmt_combo(coefs, basis), "residual": float(resid),
                    "expected": expected, "basis": basis}
    return None


DEFAULT_LADDER = [
    ["1"], ["1", "pi"], ["pi"], ["pi^2"], ["pi^3"], ["ln(2)"], ["catalan"], ["zeta(3)"],
    ["pi*ln(2)"], ["sqrt(2)"], ["sqrt(3)"], ["sqrt(5)"], ["pi*sqrt(2)"], ["pi*sqrt(3)"],
    ["1", "ln(2)"], ["1", "pi^2"], ["1", "sqrt(2)"], ["1", "sqrt(3)"], ["1", "sqrt(5)"],
    ["1", "pi", "ln(2)"], ["pi", "pi*ln(2)"], ["pi", "ln(2)"], ["pi^2", "ln(2)^2"],
    ["pi^2", "pi*ln(2)"], ["catalan", "pi"], ["pi*ln(2)", "catalan"], ["pi^2", "1"],
    ["pi*catalan", "zeta(3)"], ["pi*catalan", "pi^2*ln(2)", "zeta(3)"],
    ["pi^3", "pi*ln(2)^2", "zeta(3)"], ["pi^2", "ln(2)^2", "1", "pi*ln(2)"],
    ["catalan", "pi", "pi*ln(2)", "1"], ["pi^3", "pi^2*ln(2)", "zeta(3)", "pi*catalan"],
    ["1", "pi", "ln(2)", "pi^2", "ln(2)^2", "pi*ln(2)"],
    ["pi^4", "zeta(3)*ln(2)", "pi^2*ln(2)^2", "ln(2)^4", "zeta(3)", "pi^2", "ln(2)^2"],
]


def hp_identify(v, digits_ok, digits, basis=None, maxcoef=10 ** 5):
    ladder = [basis] if basis else DEFAULT_LADDER
    for b in ladder:
        if len(b) + 1 > 9 and not basis:
            continue
        r = hp_relation(v, b, digits_ok, digits, maxcoef)
        if r:
            return r
    return None


def report_identify(v, digits_ok, digits, basis, maxcoef=10 ** 5):
    t0 = time.time()
    r = hp_identify(v, digits_ok, digits, basis, maxcoef)
    if r is None:
        print(f"  closed form: none found in the constant sets tried "
              f"(accuracy ~{digits_ok} digits, coefficient bound {maxcoef})")
        print("  try --basis with the constants you expect, or more --digits")
        return None
    print(f"  closed form: {r['expr']}")
    print(f"    fit residual {r['residual']:.1e}; expected chance coincidences "
          f"{r['expected']:.1e}  ({time.time() - t0:.1f}s)")
    return r


# ======================================================================================
# high-precision integration (tanh-sinh in Decimal)
# ======================================================================================
INF = "inf"


def _tanh_sinh_core(f, a, b, digits, maxlevel):
    pi2 = d_pi() / 2
    tmax = D(math.log(2 * digits) + 0.7)
    eps = D(10) ** (-digits)
    T = D(0)
    prev = None
    bad = tot = 0
    diff = None
    for m in range(maxlevel + 1):
        h = D(1) / D(2 ** m)
        K = int((tmax / h).to_integral_value(rounding=decimal.ROUND_CEILING))
        ks = range(0, K + 1) if m == 0 else range(1, K + 1, 2)
        for k in ks:
            t = k * h
            et = t.exp()
            em = 1 / et
            u = pi2 * (et - em) / 2
            eu = u.exp()
            chu = (eu + 1 / eu) / 2
            w = pi2 * (et + em) / 2 / (chu * chu)
            dist = (b - a) / (1 + eu * eu)
            xs = [a + dist] if k == 0 else [b - dist, a + dist]
            for x in xs:
                if not (a < x < b):
                    continue
                tot += 1
                try:
                    val = f(x)
                    if not val.is_finite():
                        raise ArithmeticError
                    T += w * val
                except (ArithmeticError, ValueError, ZeroDivisionError):
                    bad += 1
        I = T * h * (b - a) / 2
        if prev is not None:
            diff = abs(I - prev)
            if diff <= eps * max(D(1), abs(I)):
                prev = I
                break
        prev = I
    if diff is None or diff == 0:
        ach = digits
    else:
        ach = max(0, min(digits, int(-(diff / max(D(1), abs(prev))).log10())))
    return prev, ach, bad, tot


def hp_integrate(f, a, b, digits, maxlevel=9):
    """a, b: Decimal or INF / '-inf'. f: Decimal -> Decimal."""
    if a == "-inf" and b == INF:
        I1, a1, b1, t1 = hp_integrate(f, "-inf", D(0), digits, maxlevel)
        I2, a2, b2, t2 = hp_integrate(f, D(0), INF, digits, maxlevel)
        return I1 + I2, min(a1, a2), b1 + b2, t1 + t2
    if b == INF:
        g = lambda u: f(a + u / (1 - u)) / (1 - u) ** 2
        return _tanh_sinh_core(g, D(0), D(1), digits, maxlevel)
    if a == "-inf":
        g = lambda u: f(b - u / (1 - u)) / (1 - u) ** 2
        return _tanh_sinh_core(g, D(0), D(1), digits, maxlevel)
    return _tanh_sinh_core(f, a, b, digits, maxlevel)


def parse_bound(text):
    t = text.strip().lower()
    if t in ("inf", "+inf", "infinity"):
        return INF
    if t in ("-inf", "-infinity"):
        return "-inf"
    return to_dec(Expr(text)())


def parse_basis(text):
    return [t.strip() for t in re.split(r",(?![^()]*\))", text)] if text else None


# ======================================================================================
# series
# ======================================================================================
def cvz_alternating(a, prec):
    """sum_{k>=0} (-1)^k a(k), a(k) 'nice' (e.g. completely monotone). Cohen-Villegas-Zagier."""
    n = int(prec * 1.31) + 6
    d = (3 + D(8).sqrt()) ** n
    d = (d + 1 / d) / 2
    b = D(-1)
    c = -d
    s = D(0)
    for k in range(n):
        c = b - c
        s += c * a(k)
        b = (k + n) * (k - n) * b / ((D(k) + D("0.5")) * (k + 1))
    return s / d


def van_wijngaarden(a, prec, jmax=1500):
    """sum_{m>=1} a(m) = sum_{k>=1} (-1)^(k-1) b_k,  b_k = sum_j 2^j a(2^j k); then CVZ."""
    eps = D(10) ** (-(prec + 4))

    def b(k):
        s = D(0)
        for j in range(jmax):
            term = (2 ** j) * a((2 ** j) * k)
            s += term
            if j > 2 and abs(term) < eps * max(abs(s), eps):
                return s
        raise ArithmeticError("terms do not decay fast enough (series diverges or converges"
                              " too slowly for this method)")
    return cvz_alternating(lambda k: b(k + 1), prec)


# ======================================================================================
# commands
# ======================================================================================
def cmd_calc(a):
    t0 = time.time()
    with Prec(a.digits):
        v = Expr(a.expr)()
        out = render(v, a.digits, a.full)
        if a.save:
            with open(a.save, "w") as fh:
                if isinstance(v, D):
                    with localcontext() as c:
                        c.prec = a.digits
                        fh.write(str(+v) + "\n")
                elif isinstance(v, Fraction):
                    fh.write(f"{v.numerator}/{v.denominator}\n")
                else:
                    fh.write(str(v) + "\n")
    print(out)
    print(f"  [{time.time() - t0:.2f}s]")
    return v


def cmd_hpintegral(a):
    t0 = time.time()
    with Prec(a.digits):
        f_expr = Expr(a.expr, (a.var,))
        lo, hi = parse_bound(a.lo), parse_bound(a.hi)
        f = lambda x: to_dec(f_expr(x))
        I, ach, bad, tot = hp_integrate(f, lo, hi, a.digits, a.maxlevel)
        with localcontext() as c:
            c.prec = a.digits
            shown = +I
        print(f"integral of {a.expr} d{a.var} from {a.lo} to {a.hi}")
        print(f"  value = {shown}")
        print(f"  converged to ~{ach} digits ({tot} evaluations, {bad} skipped); "
              f"[{time.time() - t0:.1f}s]")
        if bad > 0.05 * max(tot, 1):
            print("  warning: many evaluation points failed (singularity?); accuracy suspect")
        if ach >= 12:
            report_identify(I, ach, a.digits, parse_basis(a.basis), a.maxcoef)
        else:
            print("  too few converged digits to attempt a closed form")
    return I


def cmd_hpsum(a):
    t0 = time.time()
    with Prec(a.digits):
        prec = getcontext().prec
        f_expr = Expr(a.expr, (a.var,))
        lo = int(a.lo)
        if a.hi not in ("inf", "infinity"):
            hi = int(a.hi)
            if hi - lo > 5 * 10 ** 6:
                raise SystemExit("finite range too long (limit 5e6 terms)")
            tot = 0
            for n in range(lo, hi + 1):
                tot = binop("+", tot, f_expr(n))
            print(f"finite sum of {a.expr} for {a.var} = {lo}..{hi}  (exact arithmetic)")
            print("  " + render(tot, a.digits, a.full).replace("\n", "\n  "))
            print(f"  [{time.time() - t0:.1f}s]")
            return tot
        try:
            if a.alt:
                v = cvz_alternating(lambda k: to_dec(f_expr(lo + k)), prec)
                method = "Cohen-Villegas-Zagier"
            else:
                v = van_wijngaarden(lambda m: to_dec(f_expr(lo + m - 1)), prec)
                method = "Van Wijngaarden + CVZ"
        except ArithmeticError as e:
            print(f"  failed: {e}")
            return None
        with localcontext() as c:
            c.prec = a.digits
            shown = +v
        print(f"infinite sum of {a.expr} for {a.var} >= {lo}  ({method})")
        print(f"  value = {shown}   [{time.time() - t0:.1f}s]")
        report_identify(v, a.digits - 2, a.digits, parse_basis(a.basis), a.maxcoef)
    return v


def cmd_hpidentify(a):
    with Prec(a.digits):
        v = to_dec(Expr(a.value)())
        print(f"value = {v:.{min(a.digits, 60)}}")
        report_identify(v, a.digits - 2, a.digits, parse_basis(a.basis), a.maxcoef)


# ======================================================================================
# self test (oracles live here only)
# ======================================================================================
def selftest(_a=None):
    res = []

    def t(name, ok, detail=""):
        res.append(bool(ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")

    def val(expr, digits=60):
        with Prec(digits):
            return Expr(expr)()

    def head(expr, n, digits=70):
        with Prec(digits) as c:
            c.prec = digits
            return str(+to_dec(Expr(expr)()))[:n]

    # --- exact big arithmetic
    t("factorial(100) has 158 digits and ends in 24 zeros",
      len(str(val("factorial(100)"))) == 158 and str(val("factorial(100)")).endswith("0" * 24))
    t("Mersenne 2^127-1 is prime, 2^128-1 is not", val("is_prime(2**127-1)") == 1 and val("is_prime(2**128-1)") == 0)
    t("fib(1000) satisfies fib(n)=fib(n-1)+fib(n-2) at 1000",
      val("fib(1000)") == val("fib(999)+fib(998)"))
    t("exact fractions: 1/3+1/6 = 1/2", val("1/3+1/6") == Fraction(1, 2))
    t("exact sqrt of a perfect square fraction", val("sqrt(49/16)") == Fraction(7, 4))
    t("powmod matches pow", val("powmod(3, 10**20, 10**9+7)") == pow(3, 10 ** 20, 10 ** 9 + 7))
    t("Fermat: 2^(p-1) mod p = 1 for p = 2^61-1", val("powmod(2, 2**61-2, 2**61-1)") == 1)
    # --- high precision, cross-checked without recalled digits
    with Prec(100) as c:
        s, co = d_sincos(D(7) / 3)
        t("sin^2+cos^2 = 1 at 100 digits", abs(s * s + co * co - 1) < D(10) ** -99)
        t("exp(ln 3) = 3 at 100 digits", abs(D(3).ln().exp() - 3) < D(10) ** -99)
        t("tan(atan(0.7)) = 0.7", abs(d_tan(d_atan(D("0.7"))) - D("0.7")) < D(10) ** -98)
        t("sin(1000.5) vs sin(1000.5 - 318*pi) (large-argument reduction)",
          abs(d_sin(D("1000.5")) - d_sin(D("1000.5") - 318 * d_pi())) < D(10) ** -95)
        t("Machin pi: sin(pi) ~ 0 and cos(pi) ~ -1", abs(d_sin(d_pi())) < D(10) ** -98
          and abs(d_cos(d_pi()) + 1) < D(10) ** -98)
        t("zeta(2) = pi^2/6", abs(d_zeta(D(2)) - d_pi() ** 2 / 6) < D(10) ** -95)
        t("zeta(4) = pi^4/90", abs(d_zeta(D(4)) - d_pi() ** 4 / 90) < D(10) ** -95)
    # --- well-known digits (recalled; second line of defence)
    t("pi digits", head("pi", 52) == "3.14159265358979323846264338327950288419716939937510"[:52])
    t("e digits", head("e", 52) == "2.71828182845904523536028747135266249775724709369995"[:52])
    t("sqrt(2) digits", head("sqrt(2)", 42) == "1.41421356237309504880168872420969807856967"[:42])
    t("ln(2) digits", head("ln(2)", 42) == "0.69314718055994530941723212145817656807550"[:42])
    t("Catalan digits", head("catalan", 37) == "0.915965594177219015054603514932384110774"[:37])
    t("zeta(3) digits", head("zeta(3)", 40) == "1.202056903159594285399738161511449990764"[:40])
    # --- hard integrals and series with closed-form recovery
    import io
    import contextlib

    def run(fn, **k):
        class A:
            pass
        a = A()
        a.__dict__.update(k)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = fn(a)
        return r, buf.getvalue()

    r, out = run(cmd_hpintegral, expr="ln(1+x)/(1+x^2)", lo="0", hi="1", var="x", digits=40,
                 maxlevel=9, basis=None, maxcoef=10 ** 5)
    t("hp integral ln(1+x)/(1+x^2) -> (1/8)*pi*ln(2)", "(1/8)*pi*ln(2)" in out)
    r, out = run(cmd_hpintegral, expr="x^2/sin(x)", lo="0", hi="pi/2", var="x", digits=40,
                 maxlevel=9, basis="pi*catalan,zeta(3)", maxcoef=10 ** 5)
    t("hp integral x^2/sin x on (0,pi/2) -> 2*pi*G - (7/2)*zeta(3)",
      "2*pi*catalan" in out and "(7/2)*zeta(3)" in out, out.strip().splitlines()[-2] if out else "")
    r, out = run(cmd_hpintegral, expr="ln(x)^2/(1+x^2)", lo="0", hi="inf", var="x", digits=40,
                 maxlevel=9, basis=None, maxcoef=10 ** 5)
    t("hp integral ln(x)^2/(1+x^2) on (0,inf) -> (1/8)*pi^3", "(1/8)*pi^3" in out)
    r, out = run(cmd_hpsum, expr="1/n^2", var="n", lo="1", hi="inf", alt=False, digits=40,
                 maxcoef=10 ** 5, basis=None, full=False)
    t("hp sum 1/n^2 -> (1/6)*pi^2", "(1/6)*pi^2" in out)
    r, out = run(cmd_hpsum, expr="1/n", var="n", lo="1", hi="inf", alt=True, digits=40,
                 maxcoef=10 ** 5, basis=None, full=False)
    t("hp alternating harmonic -> ln(2)", "closed form: ln(2)" in out)
    r, out = run(cmd_hpsum, expr="1/(2*n+1)", var="n", lo="0", hi="inf", alt=True, digits=40,
                 maxcoef=10 ** 5, basis=None, full=False)
    t("Leibniz series -> (1/4)*pi", "(1/4)*pi" in out)
    r, out = run(cmd_hpsum, expr="n^3", var="n", lo="1", hi="100000", alt=False, digits=40,
                 maxcoef=10 ** 5, basis=None, full=False)
    t("exact finite sum of cubes = (n(n+1)/2)^2", r == (100000 * 100001 // 2) ** 2)
    print(f"\n{sum(res)}/{len(res)} bigcalc self-tests passed")
    return all(res)


# ======================================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, *args):
        s = sp.add_parser(name)
        for a in args:
            s.add_argument(*a[0], **a[1])
        s.set_defaults(fn=fn)

    dg = (("--digits",), {"type": int, "default": 50})
    bs = (("--basis",), {"default": None, "help": "constants, e.g. 'pi,ln(2),1'"})
    mc = (("--maxcoef",), {"type": int, "default": 10 ** 5})
    add("calc", cmd_calc, (("expr",), {}), dg, (("--full",), {"action": "store_true"}),
        (("--save",), {"default": None}))
    add("hpintegral", cmd_hpintegral, (("expr",), {}), (("lo",), {}), (("hi",), {}),
        (("--var",), {"default": "x"}), (("--digits",), {"type": int, "default": 40}),
        (("--maxlevel",), {"type": int, "default": 9}), bs, mc)
    add("hpsum", cmd_hpsum, (("expr",), {}), (("--var",), {"default": "n"}),
        (("--lo",), {"default": "1"}), (("--hi",), {"default": "inf"}),
        (("--alt",), {"action": "store_true"}), (("--digits",), {"type": int, "default": 40}),
        (("--full",), {"action": "store_true"}), bs, mc)
    add("hpidentify", cmd_hpidentify, (("value",), {}), (("--digits",), {"type": int, "default": 40}), bs, mc)
    add("selftest", selftest)
    args = p.parse_args()
    r = args.fn(args)
    if args.cmd == "selftest":
        sys.exit(0 if r else 1)


if __name__ == "__main__":
    main()
