#!/usr/bin/env python3
"""
math_symbols
============
Turns maths written in words or LaTeX into plain symbols.

    "(b squared - k) divided by a"   ->  "(b² - k) ÷ a"
    "square root of D"               ->  "√D"
    "30 degrees", "theta"            ->  "30°", "θ"
    "$a' < 0$"                       ->  "a' < 0"
    "\\frac{a}{b}", "x^{2}"          ->  "(a ÷ b)", "x²"

Use it on every AI reply before showing it:

    from math_symbols import to_symbols
    clean = to_symbols(ai_reply)

It is safe to run more than once on the same text (so it works on text that is still streaming in).

Options
-------
to_symbols(text, words=True, slash=True)
    words=False  leave English phrases like "greater than" and "divided by" alone
    slash=False  do not turn a/b into a ÷ b

Command line:
    python math_symbols.py "k squaredb squared divided by 2"
    echo "text" | python math_symbols.py
"""

import re
import sys

_SUP = str.maketrans("0123456789+-=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ")
_SUB = str.maketrans("0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")
_SUP_OK = set("0123456789+-=()ni")
_SUB_OK = set("0123456789+-=()")

_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ",
    "eta": "η", "theta": "θ", "vartheta": "θ", "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
    "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ",
    "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
_COMMANDS = {
    "times": "×", "cdot": "·", "div": "÷", "pm": "±", "mp": "∓", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥",
    "neq": "≠", "ne": "≠", "approx": "≈", "equiv": "≡", "sim": "∼", "propto": "∝", "infty": "∞", "sum": "∑",
    "prod": "∏", "int": "∫", "oint": "∮", "partial": "∂", "nabla": "∇", "rightarrow": "→", "to": "→",
    "leftarrow": "←", "leftrightarrow": "↔", "Rightarrow": "⇒", "implies": "⇒", "Leftrightarrow": "⇔",
    "iff": "⇔", "therefore": "∴", "because": "∵", "in": "∈", "notin": "∉", "subset": "⊂", "cup": "∪",
    "cap": "∩", "forall": "∀", "exists": "∃", "angle": "∠", "perp": "⊥", "parallel": "∥", "degree": "°",
    "circ": "°", "ldots": "...", "dots": "...", "cdots": "...", "prime": "′", "hbar": "ħ", "ell": "ℓ",
}
_FUNCS = ["sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
          "log", "ln", "lim", "exp", "max", "min", "det", "gcd"]
_VULGAR = {"1/2": "½", "1/3": "⅓", "2/3": "⅔", "1/4": "¼", "3/4": "¾", "1/5": "⅕", "2/5": "⅖",
           "3/5": "⅗", "4/5": "⅘", "1/6": "⅙", "5/6": "⅚", "1/8": "⅛", "3/8": "⅜", "5/8": "⅝", "7/8": "⅞"}


def _sup(s):
    s = s.strip()
    if s and all(ch in _SUP_OK for ch in s):
        return s.translate(_SUP)
    return "^" + s if len(s) == 1 else "^(" + s + ")"


def _sub(s):
    s = s.strip()
    if s and all(ch in _SUB_OK for ch in s):
        return s.translate(_SUB)
    return "_" + s if len(s) == 1 else "_(" + s + ")"


def _group(s, i):
    """s[i] must be '{'. Return (content, index after the matching '}') or None."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    return None


def _simple(s):
    return re.fullmatch(r"[A-Za-z0-9²³′']+", s) is not None


# ---------------------------------------------------------------- LaTeX

_FRAC = re.compile(r"(?<![A-Za-z])\\?frac\s*\{")
_SQRT = re.compile(r"(?<![A-Za-z])\\?sqrt\s*(?:\[(\d+)\])?\s*\{")


def _latex(text):
    # \frac{a}{b}  (also "frac{a}{b}" when the backslash was lost)
    while True:
        m = _FRAC.search(text)
        if not m:
            break
        g1 = _group(text, m.end() - 1)
        if not g1:
            text = text[:m.start()] + text[m.end():]
            continue
        rest = g1[1]
        while rest < len(text) and text[rest] == " ":
            rest += 1
        g2 = _group(text, rest) if rest < len(text) and text[rest] == "{" else None
        if not g2:
            text = text[:m.start()] + g1[0] + text[g1[1]:]
            continue
        num, den = _latex(g1[0]).strip(), _latex(g2[0]).strip()
        key = f"{num}/{den}"
        out = _VULGAR[key] if key in _VULGAR else f"({num} ÷ {den})"
        text = text[:m.start()] + out + text[g2[1]:]

    # \sqrt{x}, \sqrt[3]{x}
    while True:
        m = _SQRT.search(text)
        if not m:
            break
        g = _group(text, m.end() - 1)
        if not g:
            text = text[:m.start()] + "√" + text[m.end():]
            continue
        inner = _latex(g[0]).strip()
        root = m.group(1)
        sign = {"3": "∛", "4": "∜"}.get(root, "√" if not root or root == "2" else _sup(root) + "√")
        out = sign + (inner if _simple(inner) else f"({inner})")
        text = text[:m.start()] + out + text[g[1]:]

    # \text{...}, \mathrm{...} and similar: keep the content
    text = re.sub(r"\\(?:text|textbf|textit|mathrm|mathbf|mathit|mathcal|operatorname|boxed|mbox)\s*\{([^{}]*)\}",
                  r"\1", text)
    # degrees written as ^\circ or ^{\circ}
    text = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "°", text)
    # powers and subscripts
    text = re.sub(r"\^\s*\{([^{}]*)\}", lambda m: _sup(m.group(1)), text)
    text = re.sub(r"\^\s*(-?[0-9A-Za-z])", lambda m: _sup(m.group(1)), text)
    text = re.sub(r"_\s*\{([^{}]*)\}", lambda m: _sub(m.group(1)), text)
    text = re.sub(r"_(\d)", lambda m: _sub(m.group(1)), text)
    # \left \right and spacing commands
    text = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)\s*", "", text)
    text = re.sub(r"\\(?:quad|qquad|,|;|:|!|\s)", " ", text)
    # symbol commands
    names = sorted(list(_GREEK) + list(_COMMANDS), key=len, reverse=True)
    pattern = r"\\(" + "|".join(names) + r")(?![A-Za-z])"
    text = re.sub(pattern, lambda m: _GREEK.get(m.group(1)) or _COMMANDS[m.group(1)], text)
    text = re.sub(r"\\(" + "|".join(_FUNCS) + r")(?![A-Za-z])", r"\1", text)
    text = re.sub(r"\\([{}])", r"\1", text)
    # remove maths delimiters
    text = re.sub(r"\\[()\[\]]", "", text)
    text = text.replace("$", "")
    return text


# ---------------------------------------------------------------- words

_UNITS = re.compile(
    r"\b(?:m|km|cm|mm|kg|g|mg|mol|J|kJ|N|V|W|C|A|rad|L|K|eV|Pa|Hz)\s*/\s*"
    r"(?:s|h|min|m|kg|g|mol|L|K|C|rad)(?:[²³]|\^\d)?(?![A-Za-z])")
_URL = re.compile(r"https?://\S+")


def _words(text):
    t = text
    # powers and roots
    t = re.sub(r"\s*to the power(?: of)?\s+(-?\w+)", lambda m: _sup(m.group(1)), t, flags=re.I)
    t = re.sub(r"\s*squared", "²", t, flags=re.I)
    t = re.sub(r"\s*cubed", "³", t, flags=re.I)
    t = re.sub(r"(?:the\s+)?cube root of\s*", "∛", t, flags=re.I)
    t = re.sub(r"(?:the\s+)?square root of\s*", "√", t, flags=re.I)
    # comparisons and operations
    t = re.sub(r"\s+(?:is\s+)?greater than or equal to\s+", " ≥ ", t, flags=re.I)
    t = re.sub(r"\s+(?:is\s+)?less than or equal to\s+", " ≤ ", t, flags=re.I)
    t = re.sub(r"\s+(?:is\s+)?not equal to\s+", " ≠ ", t, flags=re.I)
    t = re.sub(r"\s+(?:is\s+)?approximately equal to\s+", " ≈ ", t, flags=re.I)
    t = re.sub(r"\s+(?:is\s+)?greater than\s+", " > ", t, flags=re.I)
    t = re.sub(r"\s+(?:is\s+)?less than\s+", " < ", t, flags=re.I)
    t = re.sub(r"\s+(?:is\s+)?equal to\s+", " = ", t, flags=re.I)
    t = re.sub(r"\s+equals\s+", " = ", t, flags=re.I)
    t = re.sub(r"\s+plus or minus\s+", " ± ", t, flags=re.I)
    t = re.sub(r"\s+divided by\s+", " ÷ ", t, flags=re.I)
    t = re.sub(r"\s+multiplied by\s+", " × ", t, flags=re.I)
    t = re.sub(r"(\d|\)|\b[A-Za-z])\s+times\s+(?=\d|\(|[A-Za-z]\b)", r"\1 × ", t)
    t = re.sub(r"\s+plus\s+", " + ", t, flags=re.I)
    t = re.sub(r"\s+minus\s+", " - ", t, flags=re.I)
    t = re.sub(r"\s+implies\s+", " ⇒ ", t, flags=re.I)
    t = re.sub(r"\btherefore\b", "∴", t, flags=re.I)
    t = re.sub(r"\binfinity\b", "∞", t, flags=re.I)
    # degrees and percent
    t = re.sub(r"(?<=\d)\s*degrees?\s+(?:Celsius|centigrade)\b", "°C", t, flags=re.I)
    t = re.sub(r"(?<=[\d\)θ])\s*degrees?\b(?!\s+of\b)", "°", t, flags=re.I)
    t = re.sub(r"(?<=\d)\s*(?:per\s?cent|percent)\b", "%", t, flags=re.I)
    # greek letter names (lowercase only, so ordinary capitalised words are not touched)
    lower = {k: v for k, v in _GREEK.items() if k[0].islower() and k not in ("varepsilon", "vartheta", "varphi")}
    t = re.sub(r"\b(" + "|".join(sorted(lower, key=len, reverse=True)) + r")\b",
               lambda m: lower[m.group(1)], t)
    return t


def _slashes(text):
    """a/b -> a ÷ b for maths-looking cases; units, URLs and words like and/or are left alone."""
    store = []

    def keep(m):
        store.append(m.group(0))
        return f"\u0000{len(store) - 1}\u0000"

    t = _URL.sub(keep, text)
    t = _UNITS.sub(keep, t)
    t = re.sub(r"(?<![\d/])(\d)/(\d)(?![\d/])", lambda m: _VULGAR.get(m.group(0), m.group(0)), t)
    pat = re.compile(r"((?<![A-Za-z])[A-Za-z]'?|[\d\)²³])\s*/\s*([\d\(]|[A-Za-z](?![A-Za-z]))")
    for _ in range(3):
        t = pat.sub(r"\1 ÷ \2", t)
    return re.sub(r"\u0000(\d+)\u0000", lambda m: store[int(m.group(1))], t)


def to_symbols(text, words=True, slash=True):
    """Convert maths words and LaTeX in `text` to symbols. Returns a new string."""
    t = _latex(text)
    if words:
        t = _words(t)
    if slash:
        t = _slashes(t)
    return re.sub(r"[ \t]{2,}", " ", t)


if __name__ == "__main__":
    source = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else sys.stdin.read()
    print(to_symbols(source))
