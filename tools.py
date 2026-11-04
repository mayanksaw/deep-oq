"""Deep OQ tools: exact math, text counting and live web search.

These run as normal code (the "Reasoning Gate"): numbers and counts are computed
by Python, not guessed by the language model. Web search gives current affairs.

Also holds the tool registry used by app.py and ai_advanced.py:
  tool(...)      decorator that registers a tool
  TOOLS          dict of all registered tools
  list_tools()   sorted tool names
  run_tool()     run one tool by name
  set_llm()      give the tools access to the language model
"""
import ast
import json
import math
import operator
import re

# ---------- tool registry ----------
TOOLS = {}
_llm_fn = None


def set_llm(fn):
    """fn(prompt, system='', image_url=None) -> str"""
    global _llm_fn
    _llm_fn = fn


def llm(prompt, system="", image_url=None):
    if _llm_fn is None:
        return "[Deep OQ] Model is not set."
    return _llm_fn(prompt, system, image_url)


ask = llm  # alias


def tool(name, category="general", description="", params=None):
    """Decorator: @tool("name", "category", "what it does", {"arg": "string"})"""
    def deco(fn):
        TOOLS[name] = {"fn": fn, "category": category,
                       "description": description, "params": params or {}}
        return fn
    return deco


def list_tools():
    return sorted(TOOLS)


def run_tool(name, **kwargs):
    if name not in TOOLS:
        return f"Unknown tool: {name}"
    try:
        return TOOLS[name]["fn"](**kwargs)
    except Exception as e:
        return f"[Deep OQ] Tool '{name}' failed: {e}"


# ---------- safe calculator (no eval of arbitrary code) ----------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {n: getattr(math, n) for n in (
    "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "log", "log10", "log2",
    "exp", "factorial", "floor", "ceil", "radians", "degrees", "gcd", "fabs")}
_FUNCS.update(abs=abs, round=round, min=min, max=max)
_NAMES = {"pi": math.pi, "e": math.e}


def _ev(n):
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
        return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
        a, b = _ev(n.left), _ev(n.right)
        if isinstance(n.op, ast.Pow) and abs(b) > 1000:
            raise ValueError("exponent too large")
        return _OPS[type(n.op)](a, b)
    if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
        return _OPS[type(n.op)](_ev(n.operand))
    if isinstance(n, ast.Name) and n.id in _NAMES:
        return _NAMES[n.id]
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
        return _FUNCS[n.func.id](*[_ev(a) for a in n.args])
    raise ValueError("unsupported expression")


def safe_calc(expr):
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
    if len(expr) > 300:
        raise ValueError("too long")
    val = _ev(ast.parse(expr, mode="eval").body)
    return int(val) if isinstance(val, float) and val.is_integer() and abs(val) < 1e15 else val


# ---------- exact text counting ----------
def count_text(text):
    words = re.findall(r"\S+", text)
    letters = re.findall(r"[^\W\d_]", text, re.UNICODE)
    return {"characters": len(text), "words": len(words),
            "letters": len(letters), "sentences": len(re.findall(r"[.!?]+", text))}


# ---------- live web search ----------
def web_search(query, n=5):
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        res = DDGS().text(query, max_results=n)
        return [{"title": r.get("title", ""), "body": r.get("body", ""),
                 "url": r.get("href", "")} for r in res]
    except Exception as e:
        print("web search failed:", e)
        return []


# ---------- router: decide which tools a question needs ----------
_NUM = re.compile(r"\d|calculate|compute|sum of|product of|how many|percent|%|sqrt|root|average")
_LIVE = re.compile(r"today|latest|news|current|currently|right now|price|rate|score|"
                   r"who is the|weather|this week|this month|this year|2025|2026|"
                   r"recent|update|trending|live", re.I)
_COUNT = re.compile(r"count|how many (words|letters|characters)|word count", re.I)


def run_tools(question, fast_chat):
    """Returns (tool_text, used_list). fast_chat(messages, temp, max_tokens) -> str"""
    if not (_NUM.search(question) or _LIVE.search(question) or _COUNT.search(question)):
        return "", []
    prompt = (
        "Decide which tools are needed to answer the question. Reply with ONLY JSON:\n"
        '{"calc": ["python-style arithmetic expressions to evaluate exactly"], '
        '"count": "exact text whose words/letters must be counted, or empty", '
        '"search": "short web search query if the answer depends on current/recent '
        'facts, or empty"}\n'
        "Use empty lists/strings when a tool is not needed. calc may only use numbers, "
        "+ - * / ** %, parentheses, sqrt, sin, cos, tan, log, log10, exp, factorial, "
        "pi, e.\n\nQUESTION:\n" + question)
    try:
        raw = fast_chat([{"role": "user", "content": prompt}], 0.0, 300)
        m = re.search(r"\{.*\}", raw, re.S)
        plan = json.loads(m.group(0)) if m else {}
    except Exception as e:
        print("router failed:", e)
        return "", []

    out, used = [], []
    for ex in (plan.get("calc") or [])[:6]:
        try:
            out.append(f"CALC {ex} = {safe_calc(str(ex))}")
            used.append("calc")
        except Exception:
            pass
    if plan.get("count"):
        out.append(f"COUNT of given text = {count_text(str(plan['count']))}")
        used.append("count")
    if plan.get("search"):
        hits = web_search(str(plan["search"]))
        if hits:
            used.append("web")
            out.append("WEB RESULTS for '%s':\n%s" % (plan["search"], "\n".join(
                f"- {h['title']}: {h['body']}" for h in hits)))
    return ("\n".join(out), used) if out else ("", [])


# ---------- basic tools shown in the website tool menu ----------
@tool("calculator", "math", "Exact arithmetic and math functions.", {"expression": "string"})
def _tool_calc(expression):
    return str(safe_calc(expression))


@tool("text_counter", "text", "Exact word, letter and sentence count.", {"text": "string"})
def _tool_count(text):
    return json.dumps(count_text(text))


@tool("web_search", "web", "Live web search for current information.", {"query": "string"})
def _tool_search(query):
    hits = web_search(query)
    if not hits:
        return "No web results found."
    return "\n".join(f"- {h['title']}: {h['body']} ({h['url']})" for h in hits)
