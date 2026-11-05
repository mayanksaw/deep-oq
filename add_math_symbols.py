"""
Makes Deep OQ write maths with SYMBOLS instead of words.

  Before:  sin squared theta plus cos squared theta = 1,  theta = 90 degrees
  After:   sin²θ + cos²θ = 1,  θ = 90°

  Before:  integral from 0 to 1 of x to the power 6 dx
  After:   ∫₀¹ x⁶ dx

Two things are added to deep_oq.py:
  1. A rule in the AI instructions: always use symbols (x² √ ∫ π θ ° ± ÷ ≤ ≥ ...) and never the
     words squared, power, degree, theta, plus, minus, divided by, integral, frac ...
  2. A safety filter (fix_math) that turns any leftover words or LaTeX (frac, theta, ^2, begin aligned ...)
     into symbols before the answer is shown. Code blocks are never touched.

HOW TO USE
  1. Put this file in the same folder as deep_oq.py
  2. Run:  python add_math_symbols.py
  3. Upload the updated deep_oq.py to GitHub (a backup is saved as deep_oq.py.bak4, do not upload it)

If any step cannot find its place in your file, nothing is changed.
"""
import os
import shutil
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else "deep_oq.py"

EDITS = []


def edit(old, new):
    EDITS.append((old, new))


MATH_RULE_CODE = r'''
MATH_RULE = (
    " MATHS FORMAT: write every formula with symbols, never with words. Use + − × ÷ = ≠ ≤ ≥ ± ∓ ≈ √ ∛ ∫ ∑ π θ α β ° "
    "and superscripts and subscripts: x² x³ xⁿ sin²θ cos²A x₁ ∫₀¹. Write fractions as a/b or (a+b)/(c+d). "
    "Write an integral as ∫₀¹ (x²+1) dx and an angle as 90°. "
    "NEVER write the words squared, cubed, power, degree, degrees, theta, plus, minus, divided by, times, equals, "
    "integral, fraction or frac inside maths, and never write 'less than or equal to' (use ≤). "
    "Do NOT use LaTeX: no backslash commands, no dollar signs, no begin/end aligned blocks, no & signs. "
    "Example: write sin²θ + cos²θ = 1 and θ = 90°, never 'sin squared theta plus cos squared theta equals 1'. "
    "Put each step of a calculation on its own line."
)

'''

FIX_MATH_CODE = r'''
# ---------- maths filter: turn leftover words and LaTeX into symbols (code blocks are never touched) ----------
_SUP = {**dict(zip("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")), "+": "⁺", "-": "⁻", "−": "⁻", "=": "⁼", "(": "⁽", ")": "⁾", "n": "ⁿ", "i": "ⁱ"}
_SUB = {**dict(zip("0123456789", "₀₁₂₃₄₅₆₇₈₉")), "+": "₊", "-": "₋", "−": "₋", "=": "₌", "(": "₍", ")": "₎"}
_GREEK = {"theta": "θ", "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "phi": "φ",
          "pi": "π", "lambda": "λ", "mu": "μ", "omega": "ω", "sigma": "σ",
          "Delta": "Δ", "Sigma": "Σ", "Omega": "Ω", "Phi": "Φ", "Theta": "Θ"}
_CMD = {"times": "×", "cdot": "·", "div": "÷", "pm": "±", "mp": "∓", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥",
        "neq": "≠", "ne": "≠", "approx": "≈", "infty": "∞", "circ": "°", "to": "→", "rightarrow": "→",
        "Rightarrow": "⇒", "implies": "⇒", "int": "∫", "sum": "∑", "prod": "∏", "left": "", "right": "",
        "quad": "  ", "qquad": "  ", "sin": "sin", "cos": "cos", "tan": "tan", "cot": "cot", "sec": "sec",
        "csc": "csc", "ln": "ln", "log": "log", "exp": "exp", "lim": "lim", "max": "max", "min": "min"}
_FN = r"(?:sin|cos|tan|cot|sec|csc|cosec|ln|log|exp)"
_L = r"(?:[0-9)\]²³⁴⁵⁶⁷⁸⁹ⁿθπ°]|\b[A-Za-z])"
_R = r"(?=[0-9(\[√∛θπ−-]|[A-Za-z]\b|" + _FN + r"\b|sqrt\b)"
_STOP = {"is", "be", "are", "was", "were", "and", "or", "then", "it", "the", "so", "also", "been", "being", "get", "gets"}
_EXP = r"(-?\d+(?:\.\d+)?|[A-Za-z]\b|\([^)]*\))"


def _scr(s, table):
    s = s.strip()
    if s and all(c in table for c in s):
        return "".join(table[c] for c in s)
    return None


def _sup(s):
    s = s.strip()
    inner = s[1:-1] if (s.startswith("(") and s.endswith(")") and len(s) > 2) else s
    r = _scr(inner, _SUP)
    if r is not None:
        return r
    return "^" + (s if len(s) == 1 or (s.startswith("(") and s.endswith(")")) else "(" + s + ")")


def _frac(a, b):
    simple = re.compile(r"^[\w.²³θπ]+$")
    a, b = a.strip(), b.strip()
    return (a if simple.match(a) else "(" + a + ")") + "/" + (b if simple.match(b) else "(" + b + ")")


def _integral(a, b):
    a, b = {"pi": "π", "infinity": "∞"}.get(a, a), {"pi": "π", "infinity": "∞"}.get(b, b)
    sa, sb = _scr(a, _SUB), _scr(b, _SUP)
    if sa and sb:
        return "∫" + sa + sb
    return "∫[" + a + "→" + b + "]"


def _latex(s):
    if not re.search(r"\\|\$|\^|&=", s):
        return s
    s = re.sub(r"\\\[(.+?)\\\]|\\\((.+?)\\\)", lambda m: m.group(1) or m.group(2), s, flags=re.S)
    s = re.sub(r"\$\$(.+?)\$\$", r"\1", s, flags=re.S)
    s = re.sub(r"\$([^$\n]*[\\^{][^$\n]*)\$", r"\1", s)
    env = bool(re.search(r"\\begin\{", s))
    s = re.sub(r"\\(?:begin|end)\{[A-Za-z*]+\}", "", s)
    s = re.sub(r"\\(?:text|mathrm|operatorname|mathbf|textbf)\{([^{}]*)\}", r"\1", s)
    for _ in range(4):
        s = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", lambda m: _frac(m.group(1), m.group(2)), s)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}",
               lambda m: "√" + (m.group(1) if re.fullmatch(r"[\w.]+", m.group(1)) else "(" + m.group(1) + ")"), s)
    s = re.sub(r"\\int\s*_\{?([^\s^{}]+)\}?\s*\^\{?([^\s{}]+)\}?", lambda m: _integral(m.group(1), m.group(2)), s)
    s = re.sub(r"\^\s*\{?\\circ\}?", "°", s)
    s = re.sub(r"\\\\", "\n", s)
    s = re.sub(r"\\[,;:! ]", " ", s)
    s = re.sub(r"\\([A-Za-z]+)", lambda m: _GREEK.get(m.group(1), _CMD.get(m.group(1), m.group(0))), s)
    s = re.sub(r"\^\{([^{}]*)\}", lambda m: _sup(m.group(1)), s)
    s = re.sub(r"\^\(([^()]*)\)", lambda m: _sup(m.group(1)) if _scr(m.group(1), _SUP) else m.group(0), s)
    s = re.sub(r"\^(-?\d+|[A-Za-z]\b)", lambda m: _sup(m.group(1)), s)
    s = re.sub(r"_\{([^{}]*)\}", lambda m: _scr(m.group(1), _SUB) or "_" + m.group(1), s)
    if env:
        s = s.replace("&=", "=").replace("&", "")
    return s


def _pow_word(m):
    if m.group(1).lower() in _STOP:
        return m.group(0)
    return m.group(1) + ("²" if m.group(2) == "squared" else "³")


def _math_prose(s):
    s = _latex(s)
    s = re.sub(r"\bplus(?: or |/)minus\b", "±", s)
    s = re.sub(r"\bminus(?: or |/)plus\b", "∓", s)
    s = re.sub(r"\b[Tt]heta\b", "θ", s)
    s = re.sub(r"\b(sin|cos|tan|cot|sec|csc)\s+(alpha|beta|gamma|delta|phi)\b",
               lambda m: m.group(1) + " " + _GREEK[m.group(2)], s)
    s = re.sub(r"\b(sin|cos|tan|cot|sec|csc|ln|log)\s+(squared|cubed)\s*(?=(\S))",
               lambda m: m.group(1) + ("²" if m.group(2) == "squared" else "³")
               + ("" if re.match(r"[A-Za-zθπ]", m.group(3)) else " "), s)
    s = re.sub(r"(\b[A-Za-z0-9.θπ]{1,4}|\))\s+(squared|cubed)\b", _pow_word, s)
    s = re.sub(r"(\S)\s+(?:raised\s+)?to\s+the\s+power\s+(?:of\s+)?" + _EXP,
               lambda m: m.group(1) + _sup(m.group(2)), s)
    s = re.sub(r"(\d)\s*degrees?\s+(Celsius|Fahrenheit)\b", lambda m: m.group(1) + "°" + m.group(2)[0], s)
    s = re.sub(r"(\d(?:\.\d+)?)\s*degrees?\b(?!\s+(?:of|polynomial|equation|in|from|freedom))", r"\1°", s)
    s = re.sub(r"\b(?:is\s+)?less than or equal to\b", "≤", s)
    s = re.sub(r"\b(?:is\s+)?greater than or equal to\b", "≥", s)
    s = re.sub(r"\b(?:is\s+)?not equal to\b", "≠", s)
    s = re.sub(r"\bsquare root of\s+(\([^)]*\)|\d[\d.]*|[A-Za-z]\b)", lambda m: "√" + m.group(1), s)
    s = re.sub(r"\bcube root of\s+(\([^)]*\)|\d[\d.]*|[A-Za-z]\b)", lambda m: "∛" + m.group(1), s)
    s = re.sub(r"\b[Ii]ntegral from\s+(\S+)\s+to\s+(\S+)\s+of\b", lambda m: _integral(m.group(1), m.group(2)), s)
    for word, sym in ((r"plus", " + "), (r"minus", " − "), (r"(?:times|multiplied by)", " × "),
                      (r"(?:equals|is equal to)", " = ")):
        s = re.sub(r"(" + _L + r")\s+" + word + r"\s+" + _R, lambda m, sym=sym: m.group(1) + sym, s)
    s = re.sub(r"(" + _L + r")\s+divided by\s+" + _R, r"\1/", s)
    s = re.sub(r"(?<![\w)\]])minus\s+(?=\d)", "−", s)
    s = re.sub(r"(\d)\s*pi\b", r"\1π", s)
    s = re.sub(r"\bpi(?=\s*(?:/|radians))", "π", s)
    return s


def fix_math(text):
    """Use maths symbols instead of words in an answer. Fenced code and `inline code` stay exactly as they are."""
    if not isinstance(text, str) or not text:
        return text
    parts = text.split("```")
    for i in range(0, len(parts), 2):
        segs = parts[i].split("`")
        for j in range(0, len(segs), 2):
            segs[j] = _math_prose(segs[j])
        parts[i] = "`".join(segs)
    return "```".join(parts)


'''

# 1. the rule in the AI instructions
edit('''
SYSTEM = (''', MATH_RULE_CODE + '''SYSTEM = (''')
edit('''"behind Deep OQ."
)''', '''"behind Deep OQ."
    + MATH_RULE
)''')

# 2. the safety filter
edit('''def is_hard(text):''', FIX_MATH_CODE.lstrip("\n") + '''def is_hard(text):''')

# 3. use the filter on every answer the person sees
edit('''        answer = deep_oq(session, data["q"])''', '''        answer = fix_math(deep_oq(session, data["q"]))''')
edit('''        answer = chat(msgs, 0.3, 2048)''', '''        answer = fix_math(chat(msgs, 0.3, 2048))''')
edit('''        return jsonify(answer=chat(msgs, 0.2, 4096, smart=True))''',
     '''        return jsonify(answer=fix_math(chat(msgs, 0.2, 4096, smart=True)))''')
edit('''        return jsonify(_lab(request.get_json(silent=True) or {}))''',
     '''        res = _lab(request.get_json(silent=True) or {})
        if isinstance(res.get("answer"), str):
            res["answer"] = fix_math(res["answer"])
        return jsonify(res)''')


def main():
    if not os.path.exists(TARGET):
        sys.exit("Could not find %s. Put this script in the same folder as deep_oq.py." % TARGET)
    with open(TARGET, "r", encoding="utf-8") as fh:
        src = fh.read()
    if "def fix_math(" in src:
        sys.exit("The maths symbols fix is already in this file. Nothing to do.")
    for i, (old, new) in enumerate(EDITS, 1):
        n = src.count(old)
        if n != 1:
            sys.exit("Step %d: found %d matching places (expected 1). Nothing was changed. "
                     "Your file may be different from the one I saw." % (i, n))
        src = src.replace(old, new, 1)
    shutil.copyfile(TARGET, TARGET + ".bak4")
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(src)
    print("Done. Maths symbols fix added to %s (backup: %s.bak4)." % (TARGET, TARGET))
    print("Next: upload deep_oq.py to GitHub and wait for Render to say Live.")


if __name__ == "__main__":
    main()
