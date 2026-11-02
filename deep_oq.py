"""
Deep OQ - one-file AI app: Chat, Code, Photo and Video.
Made for Mayank Kumar Saw. Everything is in this single file.

WHAT IT DOES
  Chat   knowledge database + exact calculator/counter + live web search + plan/solve/verify
  Code   write, fix, explain, convert, review code in 16 languages (HTML and JavaScript can run)
  Photo  photo editor that runs in the browser (looks, sliders, rotate, crop, caption, download)
  Lab    image maker, photo vision, data analyst, business, robot, storyboard, music, farming, energy
  Video  trim, speed, resize, rotate, looks, MP4 / GIF / MP3 output (needs ffmpeg, see below)

RUN ON YOUR COMPUTER
  pip install flask groq gunicorn pypdf ddgs imageio-ffmpeg
  set GROQ_API_KEY=your_key            (Windows)   export GROQ_API_KEY=your_key   (Mac/Linux)
  python deep_oq.py                    -> http://localhost:5000
  python deep_oq.py ingest my_folder   (optional: add your notes and code docs to the database)

RUN ON RENDER (Web Service, language Python 3)
  Build command : pip install flask groq gunicorn pypdf ddgs imageio-ffmpeg
  Start command : gunicorn deep_oq:app --workers 1 --threads 4 --timeout 300
  Environment   : GROQ_API_KEY (required), ACCESS_KEY (optional password), PYTHON_VERSION 3.12.3
  Optional      : DEEP_OQ_MODEL, DEEP_OQ_SMART_MODEL, DEEP_OQ_DB, MAX_VIDEO_MB, MAX_VIDEO_SECONDS
"""
import ast
import datetime
import io
import json
import math
import operator
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import uuid

from flask import Flask, jsonify, request, send_file
from groq import Groq


# ==========================================================================
# KNOWLEDGE DATABASE (SQLite full-text search)
# ==========================================================================

DB_PATH = os.environ.get("DEEP_OQ_DB", "deep_oq.db")
CHUNK = 900      # characters per chunk
OVERLAP = 150


def _kb_conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5("
              "content, title UNINDEXED, source UNINDEXED, tokenize='porter unicode61')")
    c.execute("CREATE TABLE IF NOT EXISTS sources(source TEXT PRIMARY KEY, chunks INT)")
    c.execute("CREATE TABLE IF NOT EXISTS messages("
              "id INTEGER PRIMARY KEY, session TEXT, role TEXT, content TEXT, "
              "ts DATETIME DEFAULT CURRENT_TIMESTAMP)")
    return c


DB = _kb_conn()


def _split(text):
    text = re.sub(r"\s+", " ", text).strip()
    i = 0
    while i < len(text):
        yield text[i:i + CHUNK]
        i += CHUNK - OVERLAP


def kb_add_text(text, title="note", source="manual"):
    n = 0
    with DB:
        for ch in _split(text):
            DB.execute("INSERT INTO docs(content,title,source) VALUES(?,?,?)",
                       (ch, title, source))
            n += 1
        DB.execute("INSERT INTO sources(source,chunks) VALUES(?,?) "
                   "ON CONFLICT(source) DO UPDATE SET chunks=chunks+excluded.chunks",
                   (source, n))
    return n


def kb_search(query, k=6):
    words = re.findall(r"\w+", query.lower())
    if not words:
        return []
    q = " OR ".join(f'"{w}"' for w in words[:20])
    try:
        rows = DB.execute(
            "SELECT content,title,source FROM docs WHERE docs MATCH ? "
            "ORDER BY bm25(docs) LIMIT ?", (q, k)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [{"content": r[0], "title": r[1], "source": r[2]} for r in rows]


def kb_save_msg(session, role, content):
    with DB:
        DB.execute("INSERT INTO messages(session,role,content) VALUES(?,?,?)",
                   (session, role, content))


def kb_load_history(session, n=12):
    rows = DB.execute("SELECT role,content FROM messages WHERE session=? "
                      "ORDER BY id DESC LIMIT ?", (session, n)).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def kb_stats():
    d = DB.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    s = DB.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    return {"chunks": d, "sources": s}

# ==========================================================================
# TOOLS: exact calculator, text counter, web search
# ==========================================================================

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

# ==========================================================================
# CODE AND VIDEO STUDIOS
# ==========================================================================


MAX_UPLOAD_MB = int(os.environ.get("MAX_VIDEO_MB", "60"))
MAX_SECONDS = int(os.environ.get("MAX_VIDEO_SECONDS", "300"))   # ffmpeg time limit
VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".3gp"}
_video_slot = threading.Semaphore(1)  # one video job at a time keeps a small server alive


def ffmpeg_path():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# ---------------------------------------------------------------- code
CODE_RULES = {
    "write": ("Write {lang} code for the task below. Give complete, working code in ONE "
              "fenced code block (no placeholders, include imports and a small usage "
              "example), then at most 5 short lines explaining how it works."),
    "fix": ("Find and fix the bugs in this {lang} code. Give the full corrected code in ONE "
            "fenced code block, then a short bullet list of what was wrong and why."),
    "explain": ("Explain this {lang} code for a learner: what it does overall, then the key "
                "parts step by step, and the time complexity if it applies. Do not rewrite it."),
    "convert": ("Convert this code from {lang} to {target}. Keep the behaviour identical and "
                "use idiomatic {target}. Give the result in ONE fenced code block, then short "
                "notes on anything that works differently."),
    "review": ("Review this {lang} code. List problems ranked by importance (bugs, security, "
               "performance, readability), each with a concrete fix. If useful, end with an "
               "improved version in ONE fenced code block."),
}
LANGS = {"Python", "JavaScript", "TypeScript", "Java", "C++", "C", "C#", "Go", "Rust", "PHP",
         "Kotlin", "Swift", "SQL", "HTML/CSS", "Bash", "Dart"}


def _code():
    d = request.get_json(silent=True) or {}
    mode = d.get("mode", "write")
    lang = d.get("lang", "Python")
    target = d.get("target", "JavaScript")
    task = str(d.get("task", "")).strip()[:4000]
    code = str(d.get("code", "")).strip()[:20000]
    if mode not in CODE_RULES or lang not in LANGS or target not in LANGS:
        return jsonify(error="Pick a mode and language from the lists."), 400
    if mode == "write" and not task:
        return jsonify(error="Describe what the code should do."), 400
    if mode != "write" and not code:
        return jsonify(error="Paste the code first."), 400
    prompt = CODE_RULES[mode].format(lang=lang, target=target)
    if task:
        prompt += "\n\nTASK / NOTES:\n" + task
    if code:
        prompt += "\n\nCODE:\n```\n" + code + "\n```"
    msgs = [
        {"role": "system", "content": "You are Deep OQ, an expert software engineer created by "
         "Mayank Kumar Saw. Write correct, modern, readable code. Use fenced code blocks with "
         "the language name. Be concise outside the code. Never use tables, horizontal lines "
         "(---) or HTML tags such as <br>; use short bullet lists."},
        {"role": "user", "content": prompt},
    ]
    try:
        return jsonify(answer=chat(msgs, 0.2, 4096, smart=True))
    except Exception as e:
        print("code failed:", e)
        return jsonify(error=str(e)[:300]), 500


# ---------------------------------------------------------------- video
def _num(name, default, lo, hi):
    try:
        v = float(request.form.get(name, default))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def source_duration(exe, src):
    """Seconds, read from `ffmpeg -i` output (ffprobe is not always installed). None if unknown."""
    import re
    try:
        p = subprocess.run([exe, "-hide_banner", "-i", src], capture_output=True, text=True, timeout=30)
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        pass
    return None


def build_command(exe, src, dst, o):
    """o: dict of validated options. Returns the ffmpeg argument list (never a shell string)."""
    fmt = o["format"]
    cmd = [exe, "-y", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if o["start"] > 0:
        cmd += ["-ss", f"{o['start']:.2f}"]
    if o["end"] > o["start"]:
        cmd += ["-t", f"{o['end'] - o['start']:.2f}"]   # input option: length BEFORE speed change
    elif fmt == "gif":
        cmd += ["-t", "15"]
    cmd += ["-i", src]

    if fmt == "mp3":
        return cmd + ["-vn", "-c:a", "libmp3lame", "-q:a", "4", dst]

    vf = []
    if o["rotate"] == "cw":
        vf.append("transpose=1")
    elif o["rotate"] == "ccw":
        vf.append("transpose=2")
    elif o["rotate"] == "180":
        vf += ["hflip", "vflip"]
    if o["filter"] == "mono":
        vf.append("hue=s=0")
    elif o["filter"] == "vivid":
        vf.append("eq=saturation=1.4:contrast=1.08")
    elif o["filter"] == "bright":
        vf.append("eq=brightness=0.08:contrast=1.05")
    elif o["filter"] == "warm":
        vf.append("colorbalance=rs=.12:gs=.02:bs=-.12")
    elif o["filter"] == "cool":
        vf.append("colorbalance=rs=-.1:bs=.14")
    if o["speed"] != 1:
        vf.append(f"setpts=PTS/{o['speed']}")
    height = o["height"]
    if fmt == "gif":
        vf.append(f"fps=12,scale=-2:min(ih\\,{min(height or 480, 480)}):flags=lanczos")
        graph = ",".join(vf) + ",split[a][b];[a]palettegen[p];[b][p]paletteuse"
        return cmd + ["-an", "-vf", graph, "-loop", "0", dst]
    if height:
        vf.append(f"scale=-2:min(ih\\,{height})")
    else:  # h264 needs even dimensions
        vf.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
    cmd += ["-vf", ",".join(vf)]
    if o["mute"]:
        cmd += ["-an"]
    else:
        if o["speed"] != 1:
            cmd += ["-af", f"atempo={o['speed']}"]
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    return cmd + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                  "-movflags", "+faststart", dst]


def _video():
    exe = ffmpeg_path()
    if not exe:
        return jsonify(error="Video tools are not installed on this server."), 503
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Choose a video file first."), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in VIDEO_EXT:
        return jsonify(error="Use an MP4, MOV, WebM, MKV, AVI or 3GP video."), 400
    fmt = request.form.get("format", "mp4")
    if fmt not in ("mp4", "gif", "mp3"):
        fmt = "mp4"
    height = int(_num("height", 0, 0, 1080))
    o = {
        "format": fmt,
        "start": _num("start", 0, 0, 36000),
        "end": _num("end", 0, 0, 36000),
        "speed": round(_num("speed", 1, 0.5, 2), 2),
        "height": height if height in (0, 240, 360, 480, 720, 1080) else 0,
        "mute": request.form.get("mute") == "1",
        "rotate": request.form.get("rotate", "none"),
        "filter": request.form.get("filter", "none"),
    }
    if not _video_slot.acquire(blocking=False):
        return jsonify(error="Another video is being processed. Try again in a minute."), 503
    work = tempfile.mkdtemp(prefix="oq_")
    try:
        src = os.path.join(work, "in" + ext)
        dst = os.path.join(work, "out." + fmt)
        f.save(src)
        dur = source_duration(exe, src)
        if dur is not None:
            if o["start"] >= dur - 0.2:
                return jsonify(error=f"The start time is past the end of the video "
                                     f"({dur:.1f} seconds long)."), 422
            if o["end"] > dur:
                o["end"] = 0  # end after the video: just run to the end
        cmd = build_command(exe, src, dst, o)
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=MAX_SECONDS)
        except subprocess.TimeoutExpired:
            return jsonify(error="This took too long. Try a shorter clip or a lower resolution."), 504
        if p.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
            print("ffmpeg failed:", p.stderr[-600:])
            return jsonify(error="Could not edit this video. Check that the file plays, "
                                 "and that the trim times are inside the video."), 422
        with open(dst, "rb") as fh:  # read into memory so the temp folder can be removed
            data = fh.read()
    finally:
        shutil.rmtree(work, ignore_errors=True)
        _video_slot.release()
    import io
    mime = {"mp4": "video/mp4", "gif": "image/gif", "mp3": "audio/mpeg"}[fmt]
    name = f"deep-oq-{uuid.uuid4().hex[:6]}.{fmt}"
    return send_file(io.BytesIO(data), mimetype=mime, as_attachment=False, download_name=name)

# ==========================================================================
# AI CORE AND WEB APP
# ==========================================================================

# Two-tier models: FAST for simple chat, SMART (120B reasoning model) for hard questions.
FAST_MODEL = os.environ.get("DEEP_OQ_MODEL", "llama-3.3-70b-versatile")
SMART_MODEL = os.environ.get("DEEP_OQ_SMART_MODEL", "openai/gpt-oss-120b")
client = Groq(api_key=os.environ["GROQ_API_KEY"])
ACCESS_KEY = os.environ.get("ACCESS_KEY", "")  # set on Render to lock the site
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = (MAX_UPLOAD_MB + 2) * 1024 * 1024

SYSTEM = (
    "You are Deep OQ, an advanced AI assistant created by Mayank Kumar Saw. "
    "You are accurate, clear and honest. You write clean, working code, solve "
    "hard reasoning and math problems step by step, and say so when you are "
    "unsure. Reply in the language the user writes in. Do not add a Sources line. "
    "When KNOWLEDGE is provided, use it if relevant and prefer it over guessing; "
    "if it is not relevant, ignore it. Write in short paragraphs, simple headings and "
    "bullet lists. Never use tables, horizontal lines (---) or HTML tags such as <br>. "
    "If asked what model or engine powers you, say: you are Deep OQ, an AI assistant built by "
    "Mayank Kumar Saw on top of large language models from other providers, and you do not "
    "know which exact model is running at any moment. Never claim to have been trained from "
    "scratch by Mayank Kumar Saw, and never deny using other models."
)

HARD_HINTS = ("code", "function", "bug", "prove", "calculate", "solve", "why",
              "how many", "algorithm", "logic", "step", "compare", "design")


FAST_CANDIDATES = []
for _m in (FAST_MODEL, "llama-3.3-70b-versatile", "openai/gpt-oss-120b",
           "openai/gpt-oss-20b", "llama-3.1-8b-instant"):
    if _m not in FAST_CANDIDATES:
        FAST_CANDIDATES.append(_m)
_working = {"model": None}  # remembers the first model that works on this account


def _call(model, messages, temperature, max_tokens, effort=None):
    kw = dict(model=model, messages=messages, temperature=temperature)
    if model.startswith("openai/gpt-oss"):
        # reasoning models spend tokens on thinking, so allow a bigger budget
        kw["max_completion_tokens"] = max(max_tokens, 4096 if effort == "high" else 2048)
        kw["reasoning_effort"] = effort or "low"
    else:
        kw["max_tokens"] = max_tokens
    r = client.chat.completions.create(**kw)
    return (r.choices[0].message.content or "").strip()


def chat(messages, temperature=0.3, max_tokens=2048, smart=False):
    if smart:
        try:
            text = _call(SMART_MODEL, messages, temperature, max_tokens, "high")
            if text:
                return text
        except Exception as e:
            print("smart model failed, using fast models:", e)
    order = FAST_CANDIDATES[:]
    if _working["model"] in order:
        order.remove(_working["model"])
        order.insert(0, _working["model"])
    last = None
    for m in order:  # e.g. a 404 "model not found" moves on to the next model
        try:
            text = _call(m, messages, temperature, max_tokens)
            if text:
                _working["model"] = m
                return text
        except Exception as e:
            last = e
            print(f"model {m} failed:", e)
    raise last or RuntimeError("no model returned an answer")


def is_hard(text):
    t = text.lower()
    return len(t) > 200 or any(h in t for h in HARD_HINTS)


def knowledge_block(question):
    hits = kb_search(question, k=6)
    if not hits:
        return ""
    parts = [f"[{i+1}] ({h['title']}) {h['content']}" for i, h in enumerate(hits)]
    return "KNOWLEDGE:\n" + "\n\n".join(parts)


def deep_oq(session, question):
    history = kb_load_history(session, 12)
    know = knowledge_block(question)
    tool_text, used = run_tools(question, lambda m, t, k: chat(m, t, k))
    today = datetime.date.today().strftime("%A, %d %B %Y")
    sysmsg = (SYSTEM + f" Today's date is {today}. TOOL RESULTS (calculator, counter, "
              "web search) are exact or current: trust them over your own arithmetic "
              "and memory, and do not mention the tools.")
    base = [{"role": "system", "content": sysmsg}] + history
    user = ((know + "\n\n" if know else "")
            + ("TOOL RESULTS:\n" + tool_text + "\n\n" if tool_text else "")
            + f"QUESTION:\n{question}")

    if not (is_hard(question) or "calc" in used or "count" in used):
        return chat(base + [{"role": "user", "content": user}], 0.5)

    plan = chat(base + [{"role": "user", "content":
        f"{user}\n\nWrite a short numbered plan to answer this correctly. "
        "Do not answer yet."}], 0.2, 600, smart=True)

    draft = chat(base + [{"role": "user", "content":
        f"{user}\n\nPLAN:\n{plan}\n\nFollow the plan and give the full answer. "
        "Do arithmetic and logic one step at a time."}], 0.2, smart=True)

    return chat([{"role": "system", "content": sysmsg}, {"role": "user", "content":
        f"{user}\n\nDRAFT ANSWER:\n{draft}\n\nCheck the draft for mistakes (math, "
        "logic, code bugs, missed requirements). Output ONLY the corrected final "
        "answer for the user, with no mention of this review."}], 0.1, smart=True)


@app.get("/")
def index():
    return PAGE, 200, {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-cache"}


@app.get("/stats")
def stats():
    return jsonify(kb_stats())


def denied():
    return ACCESS_KEY and request.headers.get("X-Access-Key", "") != ACCESS_KEY



@app.post("/teach")
def teach():
    if denied():
        return jsonify(error="locked"), 401
    t = (request.get_json() or {}).get("text", "")
    n = kb_add_text(t, title="pasted note", source="pasted-" + uuid.uuid4().hex[:8])
    return jsonify(chunks_added=n)


@app.post("/ask")
def ask():
    if denied():
        return jsonify(error="locked"), 401
    data = request.get_json()
    session = data.get("session") or "default"
    try:
        answer = deep_oq(session, data["q"])
        kb_save_msg(session, "user", data["q"])
        kb_save_msg(session, "assistant", answer)
    except Exception as e:
        print("ask failed:", e)
        return jsonify(error=str(e)[:300]), 500
    return jsonify(answer=answer)

# ==========================================================================
# THE PAGE (HTML, CSS and JavaScript)
# ==========================================================================
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deep OQ</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 48 48'%3E%3Ccircle cx='24' cy='24' r='15' fill='none' stroke='%23161B4A' stroke-width='4.5'/%3E%3Ccircle cx='34.6' cy='34.6' r='4.6' fill='%23F2A02E'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400..800&family=Source+Serif+4:opsz,wght@8..60,400..700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#EAF0F3; --surface:#F8FAFB; --ink:#161B4A; --ink-2:#51577A; --line:#C9D4DB;
  --accent:#F2A02E; --accent-ink:#2B1A00; --river:#23797E;
  --bubble:#161B4A; --bubble-ink:#F3F5FF;
  --code-bg:#0F1438; --code-ink:#E6E9FF; --code-line:#262D66;
  --err:#B3321F; --shade:rgba(10,14,40,.45);
  --ui:"Bricolage Grotesque","Noto Sans Devanagari",system-ui,sans-serif;
  --read:"Source Serif 4","Noto Serif Devanagari",Georgia,serif;
  --mono:"JetBrains Mono",ui-monospace,Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0E1233; --surface:#151A45; --ink:#EEF0FF; --ink-2:#A3A9D1; --line:#2A3070;
    --bubble:#2A3192; --bubble-ink:#FFFFFF;
    --code-bg:#090C26; --code-line:#222962; --err:#FF8D78;
  }
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--read);
  display:grid;grid-template-rows:auto minmax(0,1fr);height:100dvh;-webkit-font-smoothing:antialiased}
button,textarea{font:inherit;color:inherit}
:focus-visible{outline:3px solid var(--river);outline-offset:2px}

/* top bar */
.top{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px clamp(16px,3vw,28px)}
.brand{display:flex;align-items:center;gap:10px;font-family:var(--ui);font-weight:700;font-size:21px;letter-spacing:-.025em}
.tools{display:flex;gap:8px}
.btn{font-family:var(--ui);font-weight:500;font-size:14px;padding:8px 14px;border-radius:999px;
  border:1px solid var(--line);background:transparent;cursor:pointer;display:inline-flex;align-items:center;gap:8px;
  transition:background .15s,border-color .15s}
.btn:hover{background:var(--surface);border-color:var(--ink-2)}
.badge{font-size:12px;min-width:22px;padding:1px 7px;border-radius:999px;background:var(--ink);color:var(--bg);text-align:center}

/* mark */
.mark{color:var(--ink);flex:none;display:block}
.orbit{transform-origin:24px 24px;transform-box:view-box}
.hero-mark .orbit{animation:settle 1.7s cubic-bezier(.2,.7,.2,1) both}
.thinking .mark .orbit{animation:spin 1.1s linear infinite}
@keyframes settle{from{transform:rotate(-400deg)}to{transform:rotate(0)}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes pulse{50%{opacity:.25}}
@media (prefers-reduced-motion:reduce){
  .hero-mark .orbit{animation:none}
  .thinking .mark .orbit{animation:pulse 1.2s ease-in-out infinite}
}

/* scroll area */
main{overflow-y:auto;scroll-behavior:smooth}
.col{max-width:720px;margin:0 auto;padding:0 clamp(16px,3vw,28px)}

/* empty state */
.hero{padding-top:clamp(28px,9vh,96px);padding-bottom:24px}
.chatting .hero{display:none}
.hero h1{font-family:var(--ui);font-weight:700;font-size:clamp(34px,6vw,54px);line-height:1.04;letter-spacing:-.035em;margin:22px 0 12px}
.hero p{margin:0 0 28px;font-size:18px;line-height:1.55;color:var(--ink-2);max-width:46ch}
.ideas{list-style:none;margin:0;padding:0;border-bottom:1px solid var(--line)}
.ideas li{border-top:1px solid var(--line)}
.ideas button{display:block;width:100%;text-align:left;background:none;border:0;cursor:pointer;
  padding:15px 4px;font-size:18px;line-height:1.4;transition:background .15s,color .15s,padding .15s}
.ideas button:hover{color:var(--river);padding-left:12px}

/* conversation */
#log{padding-bottom:28px}
.turn{margin:26px 0}
.user{display:flex;justify-content:flex-end}
.bubble{max-width:82%;padding:11px 17px;border-radius:22px 22px 6px 22px;background:var(--bubble);color:var(--bubble-ink);
  font-size:17.5px;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere}
.bot{display:grid;grid-template-columns:30px minmax(0,1fr);gap:14px}
.bot .mark{margin-top:2px}
.body{font-size:17.5px;line-height:1.68;overflow-wrap:anywhere}
.body>:first-child{margin-top:0}
.body p{margin:0 0 .9em}
.body ul,.body ol{margin:0 0 .9em;padding-left:1.35em}
.body li{margin:.25em 0}
.body h3{font-family:var(--ui);font-size:19px;letter-spacing:-.015em;margin:1.2em 0 .4em}
.body code{font-family:var(--mono);font-size:.86em;padding:1px 6px;border-radius:6px;background:rgba(127,130,170,.2)}
.code{margin:0 0 1em;border-radius:14px;overflow:hidden;background:var(--code-bg);color:var(--code-ink)}
.code-bar{display:flex;justify-content:space-between;align-items:center;padding:6px 8px 6px 14px;
  border-bottom:1px solid var(--code-line);font-family:var(--ui);font-size:13px;color:#9AA2DA}
.code pre{margin:0;padding:14px;overflow-x:auto}
.code pre code{background:none;padding:0;font-size:14px;line-height:1.6;color:inherit}
.mini{font-family:var(--ui);font-size:13px;font-weight:500;background:none;border:0;border-radius:8px;padding:4px 10px;cursor:pointer;color:inherit;opacity:.8}
.mini:hover{opacity:1;background:rgba(127,130,170,.2)}
.acts{margin-top:2px;color:var(--ink-2)}
.wait{color:var(--ink-2);font-family:var(--ui);font-size:15px;padding-top:3px}
.err .body{color:var(--err)}
.err .body p{margin-bottom:.5em}

/* composer */
.composer{padding:10px clamp(16px,3vw,28px) 16px}
.field{display:flex;align-items:flex-end;gap:8px;background:var(--surface);border:1px solid var(--line);
  border-radius:28px;padding:8px 8px 8px 20px;transition:border-color .15s,box-shadow .15s}
.field:focus-within{border-color:var(--river);box-shadow:0 0 0 4px rgba(35,121,126,.18)}
.field textarea{flex:1;border:0;background:transparent;resize:none;outline:none;
  font-family:var(--read);font-size:17.5px;line-height:1.5;padding:7px 0;max-height:200px;min-height:30px}
.field textarea::placeholder{color:var(--ink-2);opacity:.8}
.send{flex:none;width:42px;height:42px;border-radius:50%;border:0;background:var(--accent);color:var(--accent-ink);
  cursor:pointer;display:grid;place-items:center;transition:transform .15s,opacity .15s}
.send:hover:not(:disabled){transform:scale(1.06)}
.send:disabled{opacity:.35;cursor:default}
.hint{margin:7px 6px 0;font-family:var(--ui);font-size:12.5px;color:var(--ink-2)}

/* knowledge drawer */
.shade{position:fixed;inset:0;background:var(--shade);opacity:0;visibility:hidden;transition:opacity .25s,visibility .25s}
.drawer{position:fixed;top:0;right:0;bottom:0;width:min(420px,100%);background:var(--surface);border-left:1px solid var(--line);
  padding:24px;display:flex;flex-direction:column;gap:14px;transform:translateX(100%);visibility:hidden;
  transition:transform .28s cubic-bezier(.2,.7,.2,1),visibility .28s}
.open .shade{opacity:1;visibility:visible}
.open .drawer{transform:none;visibility:visible}
.drawer header{display:flex;justify-content:space-between;align-items:center}
.drawer h2{font-family:var(--ui);font-size:24px;letter-spacing:-.02em;margin:0}
.drawer p{margin:0;color:var(--ink-2);line-height:1.55}
.drawer textarea{flex:1;min-height:180px;resize:none;border:1px solid var(--line);border-radius:16px;padding:14px;
  background:var(--bg);font-family:var(--read);font-size:16.5px;line-height:1.55}
.primary{font-family:var(--ui);font-weight:600;font-size:15px;padding:12px 18px;border:0;border-radius:999px;
  background:var(--accent);color:var(--accent-ink);cursor:pointer}
.primary:disabled{opacity:.4;cursor:default}
.status{font-family:var(--ui);font-size:14px;color:var(--river);min-height:1.3em}
.btn,.brand{white-space:nowrap}
@media (max-width:520px){.btn span.long{display:none}.bubble{max-width:90%}.btn{padding:7px 12px;font-size:13px}.brand{font-size:19px}}

/* tabs */
.top{flex-wrap:wrap;justify-content:flex-start}
.tabs{display:flex;gap:4px;margin-left:10px;padding:4px;border:1px solid var(--line);border-radius:999px}
.tab{font-family:var(--ui);font-weight:500;font-size:14.5px;padding:7px 16px;border:0;border-radius:999px;background:none;cursor:pointer;transition:background .15s,color .15s}
.tab:hover{background:var(--surface)}
.tab[aria-current=page]{background:var(--ink);color:var(--bg)}
.tools{margin-left:auto}
body:not([data-view=chat]) #chatTools{display:none}
.views{position:relative;min-height:0}
.view{display:none;position:absolute;inset:0;min-height:0}
#v-chat.on{display:grid;grid-template-rows:minmax(0,1fr) auto}
.studio.on{display:block;overflow-y:auto}
@media (max-width:720px){.tabs{order:3;width:100%;margin:0;justify-content:space-between}.tab{flex:1;padding:7px 6px}}

/* studios */
.wrap{max-width:1180px;margin:0 auto;padding:4px clamp(16px,3vw,28px) 56px}
.st-title{font-family:var(--ui);font-weight:700;font-size:clamp(28px,4.5vw,40px);letter-spacing:-.03em;line-height:1.1;margin:16px 0 8px}
.st-sub{margin:0 0 26px;font-size:17px;line-height:1.55;color:var(--ink-2);max-width:62ch}
.grid2{display:grid;grid-template-columns:minmax(0,350px) minmax(0,1fr);gap:28px;align-items:start}
.grid2.even{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}
.grid2.rev>.stage,.grid2.rev>.stagecol{grid-column:2;grid-row:1}
.grid2.rev>.panel{grid-column:1;grid-row:1}
@media (min-width:901px){.stagecol,.grid2.rev>.stage{position:sticky;top:6px}}
@media (max-width:900px){.grid2,.grid2.even{grid-template-columns:minmax(0,1fr)}.grid2.rev>.stage,.grid2.rev>.stagecol,.grid2.rev>.panel{grid-column:auto;grid-row:auto}}
.panel{display:grid;gap:18px;min-width:0}
.panel fieldset{border:0;padding:0;margin:0;min-inline-size:0;display:grid;gap:20px;transition:opacity .2s}
.panel fieldset:disabled{opacity:.4}
.grp{display:grid;gap:10px}
.grp h3{font-family:var(--ui);font-size:16px;font-weight:600;letter-spacing:-.01em;margin:0}
.lab{display:grid;gap:6px;font-family:var(--ui);font-size:14px;font-weight:500;color:var(--ink-2)}
.in{font-family:var(--ui);font-size:15px;color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:10px 12px;width:100%;box-sizing:border-box}
.in:focus-visible{outline-offset:0}
textarea.in{resize:vertical;line-height:1.5}
.mono{font-family:var(--mono)!important;font-size:14px!important;line-height:1.55;tab-size:4;min-height:230px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.seg{display:flex;flex-wrap:wrap;gap:6px}
.seg button{font-family:var(--ui);font-weight:500;font-size:14px;padding:7px 13px;border-radius:999px;border:1px solid var(--line);background:transparent;cursor:pointer;transition:background .15s,color .15s}
.seg button:hover{background:var(--surface)}
.seg button[aria-pressed=true]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.sl{display:grid;grid-template-columns:86px minmax(0,1fr) 36px;align-items:center;gap:10px;font-family:var(--ui);font-size:14px;color:var(--ink-2)}
.sl output{text-align:right;color:var(--ink);font-variant-numeric:tabular-nums}
input[type=range]{width:100%;accent-color:var(--river);margin:0}
input[type=color]{width:100%;height:42px;padding:3px;border:1px solid var(--line);border-radius:12px;background:var(--surface);cursor:pointer}
.stage{border:1px solid var(--line);border-radius:18px;background:var(--surface);min-height:340px;display:grid;overflow:hidden;position:relative}
.stage>*{grid-area:1/1}
.stage canvas{max-width:100%;max-height:72vh;justify-self:center;align-self:center;display:block}
.stage.over{border-color:var(--river);box-shadow:0 0 0 4px rgba(35,121,126,.18)}
.drop{display:grid;gap:6px;place-content:center;text-align:center;padding:30px 20px;cursor:pointer;font-family:var(--ui);color:var(--ink-2);margin:12px;border:2px dashed var(--line);border-radius:12px;font-size:16px;line-height:1.5}
.drop strong{color:var(--ink);font-size:19px;font-weight:600}
.drop:hover{border-color:var(--river)}
.drop:focus-within{outline:3px solid var(--river);outline-offset:2px}
.stage video{width:100%;max-height:60vh;background:#000;display:block}
.stagecol{display:grid;gap:16px;min-width:0}
.result{display:grid;gap:12px;border:1px solid var(--line);border-radius:18px;background:var(--surface);padding:16px;min-width:0}
.result h3{font-family:var(--ui);font-size:17px;margin:0}
.result video,.result img{width:100%;max-height:60vh;border-radius:12px;background:#000;object-fit:contain}
.result audio{width:100%}
.actions{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
a.primary{text-decoration:none;display:inline-block}
.muted{color:var(--ink-2);font-family:var(--ui);font-size:15px;line-height:1.5;margin:0}
.outpane{min-height:300px;border:1px solid var(--line);border-radius:16px;padding:18px;background:var(--surface);min-width:0}
.outpane .body{font-size:16.5px}
.preview{width:100%;height:340px;border:1px solid var(--line);border-radius:12px;background:#fff}
.consoleout{font-family:var(--mono);font-size:13.5px;line-height:1.5;background:var(--code-bg);color:var(--code-ink);border-radius:12px;padding:12px;white-space:pre-wrap;min-height:44px;margin:0;overflow-x:auto}
.note{font-family:var(--ui);font-size:14px;color:var(--err);margin:0}
.tbl{overflow-x:auto;margin:0 0 1em;border-radius:12px}
.body table{border-collapse:collapse;width:100%;font-family:var(--ui);font-size:15px;line-height:1.5}
.body th{text-align:left;font-weight:600;padding:9px 14px;background:rgba(127,130,170,.2)}
.body td{padding:9px 14px;vertical-align:top}
.body tbody tr:nth-child(even) td{background:rgba(127,130,170,.09)}
[hidden]{display:none!important}
</style>
</head>
<body>
<header class="top">
  <div class="brand"><span id="brandMark"></span>Deep OQ</div>
  <nav class="tabs" aria-label="Sections">
    <button class="tab" type="button" data-view="chat" aria-current="page">Chat</button>
    <button class="tab" type="button" data-view="code">Code</button>
    <button class="tab" type="button" data-view="photo">Photo</button>
    <button class="tab" type="button" data-view="video">Video</button>
    <button class="tab" type="button" data-view="lab">Lab</button>
  </nav>
  <div class="tools" id="chatTools">
    <button class="btn" id="kbBtn" type="button" aria-haspopup="dialog">Knowledge <span class="badge" id="count">0</span></button>
    <button class="btn" id="newBtn" type="button">New chat</button>
  </div>
</header>

<div class="views">
<section class="view on" id="v-chat">
<main id="scroll">
  <div class="col">
    <section class="hero" id="hero">
      <span id="heroMark"></span>
      <h1>What are you working on?</h1>
      <p>Ask about code, maths or today's news. Anything you add under Knowledge, Deep OQ checks first.</p>
      <ul class="ideas">
        <li><button type="button">Write a Python function that merges two sorted lists</button></li>
        <li><button type="button">What is in the news in India today?</button></li>
        <li><button type="button">Calculate 18% GST on ₹2,499</button></li>
        <li><button type="button">Explain how a transformer model works</button></li>
      </ul>
    </section>
    <div id="log" role="log" aria-live="polite"></div>
  </div>
</main>

<form class="composer" id="form">
  <div class="col" style="padding:0">
    <div class="field">
      <textarea id="q" rows="1" placeholder="Ask Deep OQ" aria-label="Message" autocomplete="off"></textarea>
      <button class="send" id="send" type="submit" aria-label="Send" disabled>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
      </button>
    </div>
    <div class="hint">Shift + Enter for a new line</div>
  </div>
</form>
</section>

<section class="view studio" id="v-code">
  <div class="wrap">
    <h2 class="st-title">Code</h2>
    <p class="st-sub">Write, fix, explain, convert or review code in 16 languages. Choose what you need, then press the button.</p>
    <div class="grid2 even">
      <div class="panel">
        <div class="seg" id="cmodes" role="group" aria-label="What to do">
          <button type="button" data-m="write" aria-pressed="true">Write</button>
          <button type="button" data-m="fix" aria-pressed="false">Fix</button>
          <button type="button" data-m="explain" aria-pressed="false">Explain</button>
          <button type="button" data-m="convert" aria-pressed="false">Convert</button>
          <button type="button" data-m="review" aria-pressed="false">Review</button>
        </div>
        <div class="row2">
          <label class="lab">Language<select class="in" id="clang"><option>Python</option><option>JavaScript</option><option>TypeScript</option><option>Java</option><option>C++</option><option>C</option><option>C#</option><option>Go</option><option>Rust</option><option>PHP</option><option>Kotlin</option><option>Swift</option><option>SQL</option><option>HTML/CSS</option><option>Bash</option><option>Dart</option></select></label>
          <label class="lab" id="ctgtWrap" hidden>Convert to<select class="in" id="ctgt"><option>Python</option><option>JavaScript</option><option>TypeScript</option><option>Java</option><option>C++</option><option>C</option><option>C#</option><option>Go</option><option>Rust</option><option>PHP</option><option>Kotlin</option><option>Swift</option><option>SQL</option><option>HTML/CSS</option><option>Bash</option><option>Dart</option></select></label>
        </div>
        <label class="lab"><span id="ctaskL">What should the code do?</span>
          <textarea class="in" id="ctask" rows="3" placeholder="A function that checks if a number is prime"></textarea></label>
        <label class="lab"><span id="ccodeL">Starting code (optional)</span>
          <textarea class="in mono" id="ccode" rows="10" spellcheck="false" autocapitalize="off"></textarea></label>
        <div><button class="primary" id="crun" type="button">Write code</button></div>
      </div>
      <div class="panel">
        <div class="outpane" id="cout" aria-live="polite"><p class="muted">Your result appears here.</p></div>
        <div id="crunbar" hidden><button class="btn" id="cpreview" type="button">Preview</button></div>
        <iframe class="preview" id="cframe" sandbox="allow-scripts" title="Preview" hidden></iframe>
        <pre class="consoleout" id="cconsole" hidden></pre>
      </div>
    </div>
  </div>
</section>

<section class="view studio" id="v-photo">
  <div class="wrap">
    <h2 class="st-title">Photo</h2>
    <p class="st-sub">Edit a picture right in your browser. The photo stays on your device and is never uploaded.</p>
    <div class="grid2 rev">
      <div class="stage" id="pstage">
        <label class="drop" id="pdrop"><input type="file" id="pfile" accept="image/*" hidden>
          <strong>Choose a photo</strong><span>or drop one here</span></label>
        <canvas id="pcanvas" hidden></canvas>
      </div>
      <div class="panel">
        <p class="note" id="pnote" hidden>This browser cannot apply colour effects. Use Chrome, Edge or Firefox for the full editor.</p>
        <fieldset id="pset" disabled>
          <div class="grp"><h3>Look</h3>
            <div class="seg" id="ppresets">
              <button type="button" data-p="orig" aria-pressed="true">Original</button>
              <button type="button" data-p="vivid" aria-pressed="false">Vivid</button>
              <button type="button" data-p="mono" aria-pressed="false">Black and white</button>
              <button type="button" data-p="sepia" aria-pressed="false">Sepia</button>
              <button type="button" data-p="cool" aria-pressed="false">Cool</button>
              <button type="button" data-p="warm" aria-pressed="false">Warm</button>
              <button type="button" data-p="fade" aria-pressed="false">Faded</button>
            </div></div>
          <div class="grp"><h3>Adjust</h3>
            <label class="sl">Brightness<input type="range" id="p-bri" min="50" max="150" value="100"><output id="o-bri">100</output></label>
            <label class="sl">Contrast<input type="range" id="p-con" min="50" max="150" value="100"><output id="o-con">100</output></label>
            <label class="sl">Saturation<input type="range" id="p-sat" min="0" max="200" value="100"><output id="o-sat">100</output></label>
            <label class="sl">Blur<input type="range" id="p-blur" min="0" max="10" value="0"><output id="o-blur">0</output></label>
          </div>
          <div class="grp"><h3>Shape</h3>
            <div class="seg">
              <button type="button" id="p-rl">Rotate left</button>
              <button type="button" id="p-rr">Rotate right</button>
              <button type="button" id="p-fh">Flip sideways</button>
              <button type="button" id="p-fv">Flip upside down</button>
            </div>
            <div class="row2">
              <label class="lab">Crop<select class="in" id="p-crop"><option value="free">Free</option><option value="1:1">Square 1:1</option><option value="4:5">Portrait 4:5</option><option value="3:2">Photo 3:2</option><option value="16:9">Wide 16:9</option><option value="9:16">Story 9:16</option></select></label>
              <label class="lab">Width<select class="in" id="p-width"><option value="0">Original</option><option value="1920">1920 px</option><option value="1080">1080 px</option><option value="720">720 px</option><option value="480">480 px</option></select></label>
            </div></div>
          <div class="grp"><h3>Text</h3>
            <input class="in" id="p-text" type="text" placeholder="Add a caption" aria-label="Caption text" maxlength="120">
            <label class="sl">Size<input type="range" id="p-tsize" min="2" max="20" value="8"><output id="o-tsize">8</output></label>
            <div class="row2">
              <label class="lab">Colour<input type="color" id="p-tcolor" value="#ffffff"></label>
              <label class="lab">Position<select class="in" id="p-tpos"><option value="bottom">Bottom</option><option value="middle">Middle</option><option value="top">Top</option></select></label>
            </div></div>
          <div class="grp"><h3>Save</h3>
            <div class="row2">
              <label class="lab">Format<select class="in" id="p-fmt"><option value="image/png">PNG</option><option value="image/jpeg">JPG</option><option value="image/webp">WebP</option></select></label>
            </div>
            <div class="actions"><button class="primary" id="p-save" type="button">Download photo</button>
              <button class="btn" id="p-reset" type="button">Reset</button>
              <label class="btn" for="pfile" style="cursor:pointer">Change photo</label></div>
          </div>
        </fieldset>
      </div>
    </div>
  </div>
</section>

<section class="view studio" id="v-lab">
  <div class="wrap">
    <h2 class="st-title">Lab</h2>
    <p class="st-sub">More skills: pictures, photo vision, data, business, robot, storyboards, music, farming and energy.</p>
    <div class="grid2 even">
      <div class="panel">
        <div class="seg" id="lmods" role="group" aria-label="Skill">
          <button type="button" data-m="image" aria-pressed="true">Image</button>
          <button type="button" data-m="vision" aria-pressed="false">Vision</button>
          <button type="button" data-m="data" aria-pressed="false">Data</button>
          <button type="button" data-m="enterprise" aria-pressed="false">Business</button>
          <button type="button" data-m="robot" aria-pressed="false">Robot</button>
          <button type="button" data-m="media" aria-pressed="false">Storyboard</button>
          <button type="button" data-m="music" aria-pressed="false">Music</button>
          <button type="button" data-m="agri" aria-pressed="false">Farming</button>
          <button type="button" data-m="energy" aria-pressed="false">Energy</button>
        </div>
        <label class="lab"><span id="ltextL">Describe the picture</span>
          <textarea class="in" id="ltext" rows="5"></textarea></label>
        <div class="row2">
          <label class="lab" id="lplaceW" hidden>City or village<input class="in" id="lplace" placeholder="Patna"></label>
          <label class="lab" id="lareaW" hidden><span id="lareaL">Area (m2)</span><input class="in" id="larea" type="number" min="1" value="1000"></label>
        </div>
        <label class="lab" id="lfileW" hidden><span id="lfileL">Photo</span>
          <input class="in" id="lfile" type="file" accept="image/*"></label>
        <div><button class="primary" id="lgo" type="button">Make image</button></div>
        <p class="muted" id="lstatus" aria-live="polite"></p>
      </div>
      <div class="panel">
        <div class="outpane" id="lout" aria-live="polite"><p class="muted">Your result appears here.</p></div>
      </div>
    </div>
  </div>
</section>

<section class="view studio" id="v-video">
  <div class="wrap">
    <h2 class="st-title">Video</h2>
    <p class="st-sub">Trim, speed up, resize, rotate or add a look. You can also turn a clip into a GIF or an MP3. Files up to 60 MB. Your video is sent to the server to be edited, then deleted.</p>
    <div class="grid2 rev">
      <div class="stagecol">
        <div class="stage" id="vstage">
          <label class="drop" id="vdrop"><input type="file" id="vfile" accept="video/*,.mkv" hidden>
            <strong>Choose a video</strong><span>or drop one here</span></label>
          <video id="vsrc" controls playsinline hidden></video>
        </div>
        <div class="result" id="vresult" hidden>
          <h3>Result</h3>
          <div id="vmedia"></div>
          <div class="actions"><a class="primary" id="vdl" download="deep-oq-edit.mp4">Download</a><span class="muted" id="vsize"></span></div>
        </div>
      </div>
      <div class="panel">
        <fieldset id="vset" disabled>
          <div class="grp"><h3>Make</h3>
            <div class="seg" id="vfmt">
              <button type="button" data-f="mp4" aria-pressed="true">MP4 video</button>
              <button type="button" data-f="gif" aria-pressed="false">GIF</button>
              <button type="button" data-f="mp3" aria-pressed="false">MP3 audio</button>
            </div></div>
          <div class="grp"><h3>Trim</h3>
            <div class="row2">
              <label class="lab">Start (seconds)<input class="in" id="v-start" type="number" min="0" step="0.1" value="0"></label>
              <label class="lab">End (seconds)<input class="in" id="v-end" type="number" min="0" step="0.1" placeholder="End of video"></label>
            </div>
            <div class="seg"><button type="button" id="v-setstart">Start here</button><button type="button" id="v-setend">End here</button></div>
          </div>
          <div class="grp" data-fmt="mp4 gif"><h3>Picture</h3>
            <div class="row2">
              <label class="lab">Speed<select class="in" id="v-speed"><option value="0.5">0.5x slower</option><option value="0.75">0.75x</option><option value="1" selected>1x normal</option><option value="1.25">1.25x</option><option value="1.5">1.5x</option><option value="2">2x faster</option></select></label>
              <label class="lab">Size<select class="in" id="v-height"><option value="0">Original</option><option value="1080">1080p</option><option value="720">720p</option><option value="480">480p</option><option value="360">360p</option><option value="240">240p</option></select></label>
              <label class="lab">Rotate<select class="in" id="v-rotate"><option value="none">None</option><option value="cw">90 degrees right</option><option value="ccw">90 degrees left</option><option value="180">180 degrees</option></select></label>
              <label class="lab">Look<select class="in" id="v-filter"><option value="none">None</option><option value="vivid">Vivid</option><option value="mono">Black and white</option><option value="bright">Brighter</option><option value="warm">Warm</option><option value="cool">Cool</option></select></label>
            </div></div>
          <div class="grp" data-fmt="mp4"><label style="display:flex;gap:10px;align-items:center;font-family:var(--ui);font-size:15px"><input type="checkbox" id="v-mute"> Remove the sound</label></div>
          <div class="actions"><button class="primary" id="v-go" type="button">Edit video</button><button class="btn" id="v-other" type="button" onclick="document.getElementById('vfile').click()">Change video</button></div>
          <div class="status" id="vstatus" role="status"></div>
        </fieldset>
      </div>
    </div>
  </div>
</section>
</div>

<div class="shade" id="shade"></div>
<aside class="drawer" id="drawer" role="dialog" aria-label="Knowledge" aria-modal="true">
  <header><h2>Knowledge</h2><button class="btn" id="closeKb" type="button">Close</button></header>
  <p>Paste notes, documentation or news. Deep OQ searches this text before it answers.</p>
  <textarea id="kt" placeholder="Paste text here" aria-label="Text to add"></textarea>
  <button class="primary" id="teach" type="button" disabled>Add to knowledge</button>
  <div class="status" id="kstatus" role="status"></div>
</aside>

<script>
const $=s=>document.querySelector(s);
const log=$('#log'),scroller=$('#scroll'),box=$('#q'),send=$('#send');
let sid=uid(),busy=false;

function uid(){return (crypto.randomUUID&&crypto.randomUUID())||String(Date.now())+Math.random()}
const mark=(s,c='')=>`<svg class="mark ${c}" width="${s}" height="${s}" viewBox="0 0 48 48" aria-hidden="true"><circle cx="24" cy="24" r="15" fill="none" stroke="currentColor" stroke-width="4.5"/><g class="orbit"><circle cx="34.6" cy="34.6" r="4.6" fill="var(--accent)"/></g></svg>`;
$('#brandMark').innerHTML=mark(30);
$('#heroMark').innerHTML=mark(92,'hero-mark');

/* small markdown renderer: code blocks, headings, lists, bold, italic, inline code */
const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
function inline(s){
  s=esc(s);
  s=s.replace(/&lt;br\s*\/?&gt;/gi,'<br>');
  s=s.replace(/`([^`\n]+)`/g,'<code>$1</code>');
  s=s.replace(/\*\*([^*\n]+)\*\*/g,'<strong>$1</strong>');
  s=s.replace(/(^|[^*])\*([^*\n]+)\*/g,'$1<em>$2</em>');
  return s;
}
function block(t){
  let out='',list=null,para=[];
  const fp=()=>{if(para.length){out+='<p>'+inline(para.join(' '))+'</p>';para=[]}};
  const fl=()=>{if(list){out+=`<${list.t}>`+list.items.map(x=>'<li>'+inline(x)+'</li>').join('')+`</${list.t}>`;list=null}};
  const lines=t.split('\n');
  const isRow=x=>/^\s*\|.*\|\s*$/.test(x);
  const isSep=x=>/^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(x)&&x.includes('|');
  const cells=r=>r.trim().replace(/^\|/,'').replace(/\|$/,'').split('|').map(c=>c.trim());
  for(let i=0;i<lines.length;i++){
    const l=lines[i].trimEnd();let m;
    if(!l.trim()){fp();fl();continue}
    if(/^\s*([-*_])\1{2,}\s*$/.test(l)){fp();fl();continue}   /* drop horizontal lines */
    if(isRow(l)&&i+1<lines.length&&isSep(lines[i+1])){
      fp();fl();
      const head=cells(l);let j=i+2;const body=[];
      while(j<lines.length&&isRow(lines[j])){body.push(cells(lines[j]));j++}
      i=j-1;
      out+='<div class="tbl"><table><thead><tr>'+head.map(c=>'<th>'+inline(c)+'</th>').join('')+'</tr></thead><tbody>'
        +body.map(r=>'<tr>'+r.map(c=>'<td>'+inline(c)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';
      continue;
    }
    if(isRow(l)||isSep(l)){continue}   /* stray table pieces: no raw | characters on screen */
    if(m=l.match(/^#{1,4}\s+(.*)/)){fp();fl();out+='<h3>'+inline(m[1])+'</h3>';continue}
    if(m=l.match(/^\s*[-*•]\s+(.*)/)){fp();if(!list||list.t!=='ul'){fl();list={t:'ul',items:[]}}list.items.push(m[1]);continue}
    if(m=l.match(/^\s*\d+[.)]\s+(.*)/)){fp();if(!list||list.t!=='ol'){fl();list={t:'ol',items:[]}}list.items.push(m[1]);continue}
    fl();para.push(l.trim());
  }
  fp();fl();return out;
}
function md(src){
  return src.split('```').map((p,i)=>{
    if(i%2===0)return block(p);
    const nl=p.indexOf('\n'),lang=nl>-1?p.slice(0,nl).trim():'',code=nl>-1?p.slice(nl+1):p;
    return `<div class="code"><div class="code-bar"><span>${esc(lang||'code')}</span><button type="button" class="mini" data-copy="code">Copy</button></div><pre><code>${esc(code.replace(/\n$/,''))}</code></pre></div>`;
  }).join('');
}

/* conversation */
function addUser(text){
  const d=document.createElement('div');d.className='turn user';
  const b=document.createElement('div');b.className='bubble';b.textContent=text;d.append(b);
  log.append(d);return d;
}
function addBot(){
  const d=document.createElement('div');d.className='turn bot thinking';
  d.innerHTML=mark(30)+'<div><div class="body"><div class="wait">Thinking</div></div></div>';
  log.append(d);return d;
}
function toEnd(){scroller.scrollTo({top:scroller.scrollHeight})}

async function api(url,body){
  const go=()=>fetch(url,{method:'POST',headers:{'Content-Type':'application/json','X-Access-Key':localStorage.ak||''},body:JSON.stringify(body)});
  let r=await go();
  if(r.status===401){localStorage.ak=prompt('Enter the access key')||'';r=await go()}
  return r;
}

async function ask(q){
  q=q.trim();if(!q||busy)return;
  busy=true;send.disabled=true;document.body.classList.add('chatting');
  box.value='';box.style.height='auto';
  addUser(q);const bot=addBot();toEnd();
  try{
    const r=await api('/ask',{q,session:sid});
    const j=await r.json().catch(()=>({}));
    if(r.status===401){localStorage.ak='';throw new Error('That access key was not accepted. Reload the page and enter it again.')}
    if(!r.ok||j.error)throw new Error(j.error||'The server did not answer.');
    bot.classList.remove('thinking');
    bot.raw=j.answer;
    bot.lastChild.innerHTML='<div class="body">'+md(j.answer)+'</div><div class="acts"><button type="button" class="mini" data-copy="msg">Copy</button></div>';
  }catch(e){
    bot.classList.remove('thinking');bot.classList.add('err');
    bot.lastChild.innerHTML='<div class="body"><p>Could not get an answer.</p><p>'+esc(e.message)+'</p></div><div class="acts"><button type="button" class="mini" data-retry>Try again</button></div>';
    bot.q=q;
  }
  busy=false;send.disabled=!box.value.trim();toEnd();box.focus();
}

log.addEventListener('click',e=>{
  const c=e.target.closest('[data-copy]');
  if(c){
    const text=c.dataset.copy==='msg'?c.closest('.turn').raw:c.closest('.code').querySelector('code').textContent;
    navigator.clipboard.writeText(text).then(()=>{c.textContent='Copied';setTimeout(()=>c.textContent='Copy',1400)});
    return;
  }
  const r=e.target.closest('[data-retry]');
  if(r){const t=r.closest('.turn');const q=t.q;t.previousElementSibling?.remove();t.remove();ask(q)}
});

/* composer */
box.addEventListener('input',()=>{box.style.height='auto';box.style.height=Math.min(box.scrollHeight,200)+'px';send.disabled=busy||!box.value.trim()});
box.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();ask(box.value)}});
$('#form').addEventListener('submit',e=>{e.preventDefault();ask(box.value)});
document.querySelectorAll('.ideas button').forEach(b=>b.onclick=()=>ask(b.textContent));
$('#newBtn').onclick=()=>{sid=uid();log.innerHTML='';document.body.classList.remove('chatting');box.focus()};

/* knowledge drawer */
const kt=$('#kt'),teachBtn=$('#teach'),ks=$('#kstatus');
async function stats(){try{const j=await (await fetch('/stats')).json();$('#count').textContent=j.chunks}catch(e){}}
function drawer(open){
  document.body.classList.toggle('open',open);
  $('#drawer').setAttribute('aria-hidden',!open);
  if(open){ks.textContent='';setTimeout(()=>kt.focus(),50)}else $('#kbBtn').focus();
}
$('#kbBtn').onclick=()=>drawer(true);
$('#closeKb').onclick=()=>drawer(false);
$('#shade').onclick=()=>drawer(false);
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&document.body.classList.contains('open'))drawer(false)});
kt.addEventListener('input',()=>teachBtn.disabled=!kt.value.trim());
teachBtn.onclick=async()=>{
  const t=kt.value.trim();if(!t)return;
  teachBtn.disabled=true;ks.textContent='Adding';
  try{
    const r=await api('/teach',{text:t});const j=await r.json();
    if(!r.ok)throw new Error(j.error||'failed');
    ks.textContent='Added '+j.chunks_added+(j.chunks_added===1?' chunk.':' chunks.');kt.value='';stats();
  }catch(e){ks.textContent='Could not add this text. Try again.';teachBtn.disabled=false}
};
$('#drawer').setAttribute('aria-hidden','true');
stats();box.focus();

/* ================= tabs ================= */
const VIEWS=['chat','code','photo','video','lab'];
function showView(v){
  if(!VIEWS.includes(v))v='chat';
  document.body.dataset.view=v;
  VIEWS.forEach(n=>{
    $('#v-'+n).classList.toggle('on',n===v);
    const t=document.querySelector('.tab[data-view="'+n+'"]');
    if(n===v)t.setAttribute('aria-current','page');else t.removeAttribute('aria-current');
  });
  if(location.hash!=='#'+v)history.replaceState(null,'','#'+v);
  if(v==='chat')box.focus();
}
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>showView(t.dataset.view));

/* copy buttons inside the studios (the chat log has its own handler) */
document.addEventListener('click',e=>{
  const c=e.target.closest('[data-copy="code"]');
  if(!c||c.closest('#log'))return;
  navigator.clipboard.writeText(c.closest('.code').querySelector('code').textContent)
    .then(()=>{c.textContent='Copied';setTimeout(()=>c.textContent='Copy',1400)});
});

/* ================= code ================= */
const MODES={
  write:{btn:'Write code',task:'What should the code do?',ph:'A function that checks if a number is prime',code:'Starting code (optional)'},
  fix:{btn:'Fix code',task:'What goes wrong? (optional)',ph:'It crashes when the list is empty',code:'Your code'},
  explain:{btn:'Explain code',task:'What do you want to know? (optional)',ph:'Why is there a second loop?',code:'Your code'},
  convert:{btn:'Convert code',task:'Notes (optional)',ph:'Use only the standard library',code:'Your code'},
  review:{btn:'Review code',task:'What should the review focus on? (optional)',ph:'Security and speed',code:'Your code'}
};
let cmode='write',lastCode='',lastLang='Python';
function setMode(m){
  cmode=m;const c=MODES[m];
  document.querySelectorAll('#cmodes button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.m===m));
  $('#crun').textContent=c.btn;$('#ctaskL').textContent=c.task;$('#ctask').placeholder=c.ph;$('#ccodeL').textContent=c.code;
  $('#ctgtWrap').hidden=m!=='convert';
}
document.querySelectorAll('#cmodes button').forEach(b=>b.onclick=()=>setMode(b.dataset.m));

$('#crun').onclick=async()=>{
  const out=$('#cout'),btn=$('#crun');
  const body={mode:cmode,lang:$('#clang').value,target:$('#ctgt').value,task:$('#ctask').value,code:$('#ccode').value};
  if(cmode==='write'&&!body.task.trim()){out.innerHTML='<p class="note">Describe what the code should do.</p>';$('#ctask').focus();return}
  if(cmode!=='write'&&!body.code.trim()){out.innerHTML='<p class="note">Paste the code first.</p>';$('#ccode').focus();return}
  btn.disabled=true;$('#crunbar').hidden=true;$('#cframe').hidden=true;$('#cconsole').hidden=true;
  out.innerHTML='<p class="muted">Working on it. Harder jobs can take up to a minute.</p>';
  try{
    const r=await api('/code',body);
    const j=await r.json().catch(()=>({}));
    if(!r.ok||j.error)throw new Error(j.error||'The server did not answer.');
    out.innerHTML='<div class="body">'+md(j.answer)+'</div>';
    const m=j.answer.match(/```[^\n]*\n([\s\S]*?)```/);
    lastCode=m?m[1]:'';lastLang=cmode==='convert'?body.target:body.lang;
    if(lastCode&&(lastLang==='HTML/CSS'||lastLang==='JavaScript')){
      $('#cpreview').textContent=lastLang==='HTML/CSS'?'Preview':'Run';$('#crunbar').hidden=false;
    }
  }catch(e){
    out.innerHTML='<p class="note">Could not get an answer. '+esc(e.message)+'</p>';
  }
  btn.disabled=false;
};
$('#cpreview').onclick=()=>{
  const fr=$('#cframe'),con=$('#cconsole');
  fr.hidden=false;con.textContent='';con.hidden=lastLang!=='JavaScript';
  if(lastLang==='HTML/CSS'){fr.srcdoc=lastCode;return}
  fr.hidden=true; /* JavaScript runs without a visible page; output goes to the console box */
  fr.srcdoc='<script>var s=function(m){parent.postMessage({oq:m},"*")};'
    +'console.log=console.info=console.warn=console.error=function(){s([].map.call(arguments,function(x){try{return typeof x==="object"?JSON.stringify(x):String(x)}catch(e){return String(x)}}).join(" "))};'
    +'onerror=function(m){s("Error: "+m)};<\/script><script>'+lastCode.replace(/<\/script/gi,'<\\/script')+'<\/script>';
  con.textContent='Running...\n';
};
window.addEventListener('message',e=>{
  if(e.source===$('#cframe').contentWindow&&e.data&&typeof e.data.oq==='string'){
    const con=$('#cconsole');con.textContent=(con.textContent==='Running...\n'?'':con.textContent)+e.data.oq+'\n';
  }
});

/* ================= photo ================= */
const FILTER_OK='filter' in CanvasRenderingContext2D.prototype;
if(!FILTER_OK)$('#pnote').hidden=false;
const PH={src:null,rot:0,fx:false,fy:false,rc:null};
const ASPECT={free:0,'1:1':1,'4:5':.8,'3:2':1.5,'16:9':16/9,'9:16':9/16};
const PRESETS={
  orig:{bri:100,con:100,sat:100,blur:0,gray:0,sepia:0,hue:0},
  vivid:{bri:104,con:112,sat:150,blur:0,gray:0,sepia:0,hue:0},
  mono:{bri:100,con:115,sat:100,blur:0,gray:100,sepia:0,hue:0},
  sepia:{bri:100,con:105,sat:100,blur:0,gray:0,sepia:85,hue:0},
  cool:{bri:100,con:105,sat:105,blur:0,gray:0,sepia:0,hue:-18},
  warm:{bri:102,con:105,sat:115,blur:0,gray:0,sepia:28,hue:-6},
  fade:{bri:112,con:84,sat:78,blur:0,gray:0,sepia:8,hue:0}
};
let fx={...PRESETS.orig};
const SL=['bri','con','sat','blur'];
function syncSliders(){SL.forEach(k=>{$('#p-'+k).value=fx[k];$('#o-'+k).textContent=fx[k]})}
function markPreset(name){document.querySelectorAll('#ppresets button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.p===name))}
let raf=0;
function queue(){if(!raf)raf=requestAnimationFrame(()=>{raf=0;compose($('#pcanvas'),1200,0,false)})}

function rotated(){
  if(PH.rc)return PH.rc;
  const s=PH.src,swap=PH.rot%2===1,W=swap?s.height:s.width,H=swap?s.width:s.height;
  const r=document.createElement('canvas');r.width=W;r.height=H;const c=r.getContext('2d');
  c.translate(W/2,H/2);c.scale(PH.fx?-1:1,PH.fy?-1:1);c.rotate(PH.rot*Math.PI/2);
  c.drawImage(s,-s.width/2,-s.height/2);
  return PH.rc=r;
}
function wrapLines(ctx,text,max){
  const words=text.split(/\s+/),lines=[];let cur='';
  for(const w of words){const t=cur?cur+' '+w:w;if(cur&&ctx.measureText(t).width>max){lines.push(cur);cur=w}else cur=t}
  if(cur)lines.push(cur);return lines;
}
function compose(target,maxEdge,maxWidth,whiteBg){
  if(!PH.src)return;
  const r=rotated(),W=r.width,H=r.height,a=ASPECT[$('#p-crop').value];
  let cw=W,ch=H;
  if(a){if(W/H>a)cw=Math.round(H*a);else ch=Math.round(W/a)}
  const cx=(W-cw)/2,cy=(H-ch)/2;
  let k=1;if(maxWidth)k=Math.min(1,maxWidth/cw);if(maxEdge)k=Math.min(k,maxEdge/Math.max(cw,ch));
  const ow=Math.max(1,Math.round(cw*k)),oh=Math.max(1,Math.round(ch*k));
  target.width=ow;target.height=oh;
  const ctx=target.getContext('2d');
  if(whiteBg){ctx.fillStyle='#fff';ctx.fillRect(0,0,ow,oh)}
  if(FILTER_OK)ctx.filter='brightness('+fx.bri+'%) contrast('+fx.con+'%) saturate('+fx.sat+'%) sepia('+fx.sepia+'%) grayscale('+fx.gray+'%) hue-rotate('+fx.hue+'deg) blur('+(fx.blur*ow/800).toFixed(2)+'px)';
  ctx.drawImage(r,cx,cy,cw,ch,0,0,ow,oh);
  ctx.filter='none';
  const txt=$('#p-text').value.trim();
  if(txt){
    const fs=Math.max(8,Math.round(ow*(+$('#p-tsize').value)/100));
    ctx.font='700 '+fs+'px "Bricolage Grotesque",system-ui,sans-serif';
    ctx.fillStyle=$('#p-tcolor').value;ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.shadowColor='rgba(0,0,0,.55)';ctx.shadowBlur=fs*.2;
    const lines=wrapLines(ctx,txt,ow*.9),lh=fs*1.2,total=lines.length*lh,pad=ow*.04,pos=$('#p-tpos').value;
    const y0=pos==='top'?pad:pos==='bottom'?oh-pad-total:(oh-total)/2;
    lines.forEach((l,i)=>ctx.fillText(l,ow/2,y0+lh/2+i*lh));
    ctx.shadowBlur=0;
  }
}
function loadPhoto(file){
  if(!file||!file.type.startsWith('image/'))return;
  const url=URL.createObjectURL(file),img=new Image();
  img.onload=()=>{
    const k=Math.min(1,3200/Math.max(img.naturalWidth,img.naturalHeight));
    const s=document.createElement('canvas');s.width=Math.round(img.naturalWidth*k);s.height=Math.round(img.naturalHeight*k);
    s.getContext('2d').drawImage(img,0,0,s.width,s.height);
    URL.revokeObjectURL(url);
    PH.src=s;resetPhoto();
    $('#pset').disabled=false;$('#pdrop').hidden=true;$('#pcanvas').hidden=false;
    queue();
  };
  img.onerror=()=>{URL.revokeObjectURL(url);$('#pdrop').querySelector('span').textContent='That file could not be opened as a picture. Try a JPG or PNG.'};
  img.src=url;
}
function resetPhoto(){
  PH.rot=0;PH.fx=false;PH.fy=false;PH.rc=null;fx={...PRESETS.orig};syncSliders();markPreset('orig');
  $('#p-crop').value='free';$('#p-width').value='0';$('#p-text').value='';$('#p-tsize').value=8;$('#o-tsize').textContent=8;
  $('#p-tcolor').value='#ffffff';$('#p-tpos').value='bottom';
}
$('#pfile').onchange=e=>{loadPhoto(e.target.files[0]);e.target.value=''};
SL.forEach(k=>$('#p-'+k).addEventListener('input',e=>{fx[k]=+e.target.value;$('#o-'+k).textContent=fx[k];markPreset('');queue()}));
document.querySelectorAll('#ppresets button').forEach(b=>b.onclick=()=>{fx={...PRESETS[b.dataset.p]};syncSliders();markPreset(b.dataset.p);queue()});
$('#p-rl').onclick=()=>{PH.rot=(PH.rot+3)%4;PH.rc=null;queue()};
$('#p-rr').onclick=()=>{PH.rot=(PH.rot+1)%4;PH.rc=null;queue()};
$('#p-fh').onclick=()=>{PH.fx=!PH.fx;PH.rc=null;queue()};
$('#p-fv').onclick=()=>{PH.fy=!PH.fy;PH.rc=null;queue()};
['p-crop','p-tpos'].forEach(id=>$('#'+id).addEventListener('change',queue));
['p-text','p-tcolor'].forEach(id=>$('#'+id).addEventListener('input',queue));
$('#p-tsize').addEventListener('input',e=>{$('#o-tsize').textContent=e.target.value;queue()});
$('#p-reset').onclick=()=>{resetPhoto();queue()};
$('#p-save').onclick=()=>{
  const type=$('#p-fmt').value,ext={'image/png':'png','image/jpeg':'jpg','image/webp':'webp'}[type];
  const out=document.createElement('canvas');
  compose(out,4096,+$('#p-width').value||0,type==='image/jpeg');
  out.toBlob(b=>{
    const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='deep-oq-photo.'+ext;
    document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),4000);
  },type,.92);
};
function dropZone(stage,handler){
  ['dragenter','dragover'].forEach(n=>stage.addEventListener(n,e=>{e.preventDefault();stage.classList.add('over')}));
  ['dragleave','drop'].forEach(n=>stage.addEventListener(n,e=>{e.preventDefault();stage.classList.remove('over')}));
  stage.addEventListener('drop',e=>handler(e.dataTransfer.files[0]));
}
dropZone($('#pstage'),loadPhoto);

/* ================= video ================= */
const VMAX=60*1024*1024;
let vfmt='mp4',vfile=null,vurl='',vres='';
const vstatus=m=>$('#vstatus').textContent=m;
function applyFmt(){
  document.querySelectorAll('#vfmt button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.f===vfmt));
  document.querySelectorAll('#vset [data-fmt]').forEach(el=>el.hidden=!el.dataset.fmt.split(' ').includes(vfmt));
}
document.querySelectorAll('#vfmt button').forEach(b=>b.onclick=()=>{vfmt=b.dataset.f;applyFmt()});
applyFmt();
function loadVideo(f){
  if(!f)return;
  if(f.size>VMAX){vstatus('That file is '+(f.size/1048576).toFixed(0)+' MB. The limit is 60 MB.');return}
  vfile=f;if(vurl)URL.revokeObjectURL(vurl);vurl=URL.createObjectURL(f);
  const v=$('#vsrc');v.src=vurl;v.hidden=false;$('#vdrop').hidden=true;
  $('#vset').disabled=false;$('#vresult').hidden=true;vstatus('');
  $('#v-start').value=0;$('#v-end').value='';
}
$('#vfile').onchange=e=>{loadVideo(e.target.files[0]);e.target.value=''};
dropZone($('#vstage'),loadVideo);
$('#v-setstart').onclick=()=>$('#v-start').value=$('#vsrc').currentTime.toFixed(1);
$('#v-setend').onclick=()=>$('#v-end').value=$('#vsrc').currentTime.toFixed(1);

function sendVideo(retry){
  const fd=new FormData();
  fd.append('file',vfile);fd.append('format',vfmt);
  fd.append('start',$('#v-start').value||0);fd.append('end',$('#v-end').value||0);
  fd.append('speed',$('#v-speed').value);fd.append('height',$('#v-height').value);
  fd.append('rotate',$('#v-rotate').value);fd.append('filter',$('#v-filter').value);
  fd.append('mute',$('#v-mute').checked?'1':'0');
  const x=new XMLHttpRequest();
  x.open('POST','/video');x.responseType='blob';x.setRequestHeader('X-Access-Key',localStorage.ak||'');
  x.upload.onprogress=e=>{if(e.lengthComputable)vstatus('Uploading '+Math.round(e.loaded/e.total*100)+'%')};
  x.upload.onload=()=>vstatus('Editing. This can take a minute or two.');
  const done=()=>{$('#v-go').disabled=false};
  x.onerror=()=>{vstatus('The connection failed. Check your internet and try again.');done()};
  x.onload=async()=>{
    if(x.status===401&&!retry){localStorage.ak=prompt('Enter the access key')||'';return sendVideo(true)}
    if(x.status===200){
      if(vres)URL.revokeObjectURL(vres);vres=URL.createObjectURL(x.response);
      const ext={mp4:'mp4',gif:'gif',mp3:'mp3'}[vfmt];
      const tag=vfmt==='gif'?'<img alt="Edited GIF" src="'+vres+'">':vfmt==='mp3'?'<audio controls src="'+vres+'"></audio>':'<video controls playsinline src="'+vres+'"></video>';
      $('#vmedia').innerHTML=tag;$('#vdl').href=vres;$('#vdl').download='deep-oq-edit.'+ext;
      $('#vsize').textContent=x.response.size<1048576?Math.max(1,Math.round(x.response.size/1024))+' KB':(x.response.size/1048576).toFixed(1)+' MB';
      $('#vresult').hidden=false;vstatus('Done.');
      $('#vresult').scrollIntoView({behavior:'smooth',block:'nearest'});
    }else{
      let msg='Could not edit this video.';
      try{msg=JSON.parse(await x.response.text()).error||msg}catch(e){}
      vstatus(msg);
    }
    done();
  };
  x.send(fd);
}
$('#v-go').onclick=()=>{if(!vfile)return;$('#v-go').disabled=true;vstatus('Uploading 0%');sendVideo(false)};

/* ================= lab ================= */
let lmod='image',limg='';
const LM={
  image:['Describe the picture','a red fox in a snowy forest, watercolor','Make image'],
  vision:['What should I look for? (optional)','What is in this photo?','Analyze photo'],
  data:['Paste CSV data, or choose a .csv file','name,sales\nA,10\nB,20','Analyze data'],
  enterprise:['Ask using your Knowledge notes','Summarize our refund policy and list action items','Ask'],
  robot:['Tell the robot what to do on a 10 by 10 grid','Walk to the far right edge, then down to the bottom','Run robot'],
  media:['Video idea','A 30 second ad for a coffee shop','Write storyboard'],
  music:['Describe the music','A calm piano tune in C major','Compose'],
  agri:['Crop (rice, wheat, maize, cotton, tomato, potato...)','rice','Plan irrigation'],
  energy:['Question (optional)','How much solar power can my roof make?','Check energy and weather']
};
const lstatus=m=>$('#lstatus').textContent=m;
function applyLab(){
  document.querySelectorAll('#lmods button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.m===lmod));
  const m=LM[lmod];
  $('#ltextL').textContent=m[0];$('#ltext').placeholder=m[1];$('#lgo').textContent=m[2];
  const w=lmod==='agri'||lmod==='energy';
  $('#lplaceW').hidden=!w;$('#lareaW').hidden=!w;
  $('#lareaL').textContent=lmod==='agri'?'Field area (m2)':'Solar panel area (m2)';
  $('#lfileW').hidden=!(lmod==='vision'||lmod==='agri'||lmod==='data');
  $('#lfile').accept=lmod==='data'?'.csv,text/csv,text/plain':'image/*';
  $('#lfileL').textContent=lmod==='data'?'CSV file':lmod==='agri'?'Plant photo (optional, checks for disease)':'Photo';
  $('#lfile').value='';limg='';lstatus('');
}
document.querySelectorAll('#lmods button').forEach(b=>b.onclick=()=>{lmod=b.dataset.m;applyLab()});
applyLab();
$('#lfile').onchange=e=>{
  const f=e.target.files[0];if(!f)return;
  if(lmod==='data'){const r=new FileReader();r.onload=()=>{$('#ltext').value=String(r.result).slice(0,20000);lstatus('File loaded.')};r.readAsText(f);return}
  const url=URL.createObjectURL(f),img=new Image();
  img.onload=()=>{
    const k=Math.min(1,1024/Math.max(img.naturalWidth,img.naturalHeight)),c=document.createElement('canvas');
    c.width=Math.round(img.naturalWidth*k);c.height=Math.round(img.naturalHeight*k);
    c.getContext('2d').drawImage(img,0,0,c.width,c.height);URL.revokeObjectURL(url);
    limg=c.toDataURL('image/jpeg',.85);lstatus('Photo ready.');
  };
  img.onerror=()=>{URL.revokeObjectURL(url);lstatus('That file could not be opened as a picture.')};
  img.src=url;
};
function labRobot(cmds){
  const c=$('#lcan'),g=c.getContext('2d'),s=30;let x=0,y=0,dx=1,dy=0,i=0;
  g.strokeStyle='#bbb';
  for(let n=0;n<=10;n++){g.beginPath();g.moveTo(n*s,0);g.lineTo(n*s,300);g.moveTo(0,n*s);g.lineTo(300,n*s);g.stroke()}
  const t=setInterval(()=>{
    if(i>=cmds.length){clearInterval(t);return}
    const k=cmds[i++];
    if(k==='L')[dx,dy]=[dy,-dx];else if(k==='R')[dx,dy]=[-dy,dx];
    else{x=Math.max(0,Math.min(9,x+dx));y=Math.max(0,Math.min(9,y+dy))}
    g.fillStyle='rgba(60,120,255,.35)';g.fillRect(x*s+3,y*s+3,s-6,s-6);
  },150);
}
function labPlay(m){
  const a=new (window.AudioContext||window.webkitAudioContext)(),b=60/m.tempo;let t=a.currentTime+.1;
  m.notes.forEach(([n,d])=>{
    if(n>0){const o=a.createOscillator(),g=a.createGain();o.type='triangle';o.frequency.value=440*Math.pow(2,(n-69)/12);
      g.gain.setValueAtTime(.2,t);g.gain.exponentialRampToValueAtTime(.001,t+d*b);o.connect(g);g.connect(a.destination);o.start(t);o.stop(t+d*b)}
    t+=d*b;
  });
}
$('#lgo').onclick=async()=>{
  const go=$('#lgo');go.disabled=true;lstatus('Working. Pictures and photos can take a little while.');
  try{
    const r=await api('/lab',{mod:lmod,text:$('#ltext').value,place:$('#lplace').value,area:$('#larea').value,image:(lmod==='vision'||lmod==='agri')?limg:''});
    const j=await r.json();
    if(r.status===401)throw new Error('That access key was not accepted. Reload the page and enter it again.');
    if(!r.ok||j.error)throw new Error(j.error||'The server did not answer.');
    const o=$('#lout');
    o.innerHTML=(j.image?'<img id="limgout" alt="Generated picture" style="max-width:100%;border-radius:12px;margin-bottom:12px">':'')
      +(j.answer?'<div class="body">'+md(j.answer)+'</div>':'')
      +(j.music?'<p><button class="btn" id="lplay" type="button">Play</button></p>':'')
      +(j.robot?'<canvas id="lcan" width="300" height="300" aria-label="Robot path" style="max-width:100%"></canvas>':'');
    if(j.image){const im=$('#limgout');im.onerror=()=>{im.remove();lstatus('The picture service did not answer. Try again in a moment.')};im.src=j.image}
    if(j.music)$('#lplay').onclick=()=>labPlay(j.music);
    if(j.robot)labRobot(j.robot);
    if(!j.image)lstatus('');
  }catch(e){lstatus(e.message)}
  go.disabled=false;
};

showView(location.hash.slice(1));
</script>
</body>
</html>
"""


# ==========================================================================
# MORE ROUTES AND START
# ==========================================================================
# ==========================================================================
# LAB: extra skills (image, photo vision, data, business, robot, media, music, farm, energy)
# ==========================================================================
import collections
import csv
import random
import urllib.parse
import urllib.request

VISION_MODEL = os.environ.get("DEEP_OQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
LAB_SYS = ("You are Deep OQ, an AI created by Mayank Kumar Saw. Be accurate, practical and concise. "
           "Use short paragraphs and bullet lists. Never use tables, horizontal lines (---) or HTML tags.")
KC = {"rice": 1.2, "wheat": 1.15, "maize": 1.2, "corn": 1.2, "cotton": 1.15, "tomato": 1.15,
      "potato": 1.15, "sugarcane": 1.25, "soybean": 1.1, "banana": 1.2, "vegetables": 1.0}


def _get_json(url):
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.load(r)


def forecast(place):
    """7-day weather, evapotranspiration and sunshine for a place (free Open-Meteo, no key)."""
    g = _get_json("https://geocoding-api.open-meteo.com/v1/search?count=1&name=" + urllib.parse.quote(place))
    if not g.get("results"):
        raise ValueError("Could not find that place. Try a city name.")
    p = g["results"][0]
    w = _get_json("https://api.open-meteo.com/v1/forecast?latitude=%s&longitude=%s&timezone=auto&forecast_days=7"
                  "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,"
                  "et0_fao_evapotranspiration,shortwave_radiation_sum" % (p["latitude"], p["longitude"]))
    d = w["daily"]
    return p["name"], [{k: (d[k][i] or 0) for k in d if k != "time"} | {"date": day}
                       for i, day in enumerate(d["time"])]


def _json_from(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0))
    except (AttributeError, ValueError):
        raise ValueError("The AI did not return a usable result. Please try again.")


def _vision(prompt, image):
    if not image.startswith("data:image/"):
        raise ValueError("Choose a photo first.")
    r = client.chat.completions.create(
        model=VISION_MODEL, max_tokens=1200, temperature=0.3,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image}}]}])
    return (r.choices[0].message.content or "").strip()


def _lab(d):
    mod = d.get("mod")
    text = str(d.get("text", "")).strip()[:20000]
    place = str(d.get("place", "")).strip()[:80]
    image = str(d.get("image", ""))[:6_000_000]
    try:
        area = min(max(float(d.get("area") or 100), 1), 1e7)
    except (TypeError, ValueError):
        area = 100.0

    def ask(task, smart=False):
        return chat([{"role": "system", "content": LAB_SYS}, {"role": "user", "content": task}],
                    0.4, 2048, smart)

    # ---- Agriculture + Energy/Environment: real forecast numbers, then AI advice
    if mod in ("agri", "energy"):
        if not place:
            raise ValueError("Type a city or village name.")
        name, days = forecast(place)
        if mod == "agri":
            kc = KC.get(text.lower(), 1.0)
            lines, total = [], 0.0
            for x in days:
                mm = max(0.0, kc * x["et0_fao_evapotranspiration"] - x["precipitation_sum"])
                total += mm * area
                lines.append(f'{x["date"]}: max {x["temperature_2m_max"]}C, rain {x["precipitation_sum"]} mm, '
                             f'water needed {mm:.1f} mm = {mm * area:.0f} litres')
            facts = (f"Crop: {text or 'general crop'} (crop factor {kc}). Field {area:.0f} m2 near {name}.\n"
                     "7-day irrigation need (crop evapotranspiration minus rain):\n" + "\n".join(lines)
                     + f"\nTotal: {total:.0f} litres.")
            out = ask(facts + "\n\nExplain this plan simply for a farmer: when to irrigate, how much, "
                      "and any weather risks. Keep the numbers exactly as given.")
            if image.startswith("data:image/"):
                out += "\n\n## Plant health from your photo\n" + _vision(
                    "You are a plant pathologist. Identify the crop, any visible disease, pest or nutrient "
                    "problem, how sure you are, and practical treatment and prevention steps. "
                    "If the photo does not show a plant, say so.", image)
            return {"answer": out}
        lines, warn, kwh = [], [], 0.0
        for x in days:
            e = x["shortwave_radiation_sum"] / 3.6 * area * 0.18 * 0.75  # MJ/m2 -> kWh, 18% panels, 75% system
            kwh += e
            lines.append(f'{x["date"]}: solar {e:.1f} kWh, max {x["temperature_2m_max"]}C, '
                         f'wind {x["wind_speed_10m_max"]} km/h, rain {x["precipitation_sum"]} mm')
            if x["temperature_2m_max"] >= 40: warn.append(f'{x["date"]}: extreme heat')
            if x["temperature_2m_min"] <= 0: warn.append(f'{x["date"]}: frost')
            if x["precipitation_sum"] >= 50: warn.append(f'{x["date"]}: heavy rain, flood risk')
            if x["wind_speed_10m_max"] >= 60: warn.append(f'{x["date"]}: strong wind, storm risk')
        facts = (f"Place: {name}. Solar panels: {area:.0f} m2.\n" + "\n".join(lines)
                 + f"\nTotal solar this week: {kwh:.0f} kWh.\nEarly warnings: "
                 + ("; ".join(warn) or "none this week."))
        return {"answer": ask(facts + "\n\nSummarize the expected solar output, the early warnings and "
                              "sensible grid or home actions. Keep the numbers exactly as given. "
                              + (("Also answer: " + text) if text else ""))}

    if mod == "vision":
        return {"answer": _vision(text or "Describe this photo in detail and point out anything "
                                  "important or unusual.", image)}
    if not text:
        raise ValueError("Type something first.")

    if mod == "image":
        p = ask("Rewrite this as one vivid, detailed image-generation prompt (max 60 words). "
                "Output only the prompt:\n" + text)
        url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(p)
               + "?width=1024&height=1024&nologo=true&seed=" + str(random.randint(1, 10**6)))
        return {"image": url, "answer": "Prompt used: " + p}

    if mod == "data":
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows:
            raise ValueError("Paste CSV data with a header row.")
        stats = []
        for col in rows[0]:
            vals = []
            for r in rows:
                try:
                    vals.append(float(str(r.get(col, "")).replace(",", "")))
                except ValueError:
                    pass
            if vals and len(vals) >= len(rows) / 2:
                stats.append(f"{col}: numeric, n={len(vals)}, min={min(vals):g}, max={max(vals):g}, "
                             f"mean={sum(vals) / len(vals):.4g}, sum={sum(vals):.6g}")
            else:
                top = collections.Counter(str(r.get(col, "")) for r in rows).most_common(3)
                stats.append(f"{col}: text, top values {top}")
        return {"answer": ask(f"{len(rows)} rows. Exact column statistics:\n" + "\n".join(stats)
                              + f"\n\nFirst rows:\n{rows[:5]}\n\nGive key findings, anything unusual, and 3 "
                              "recommendations. Use the exact statistics.", smart=True)}

    if mod == "enterprise":
        know = knowledge_block(text)
        return {"answer": ask((know + "\n\n" if know else "") + "REQUEST:\n" + text
                              + "\n\nAnswer as a business assistant. Use KNOWLEDGE if relevant and say if "
                              "nothing relevant was found. End with a short list of action items.", smart=True)}

    if mod == "media":
        return {"answer": ask("Write a video storyboard for this idea: a title, then 5 to 8 scenes, each with "
                              "a shot description, narration line and duration in seconds. Finish with music "
                              "and editing tips.\n\nIDEA: " + text)}

    if mod == "robot":
        j = _json_from(ask(
            'A robot is on a 10x10 grid, starting at the top-left cell facing east. Commands: F moves one '
            'cell forward (walls stop it), L turns left, R turns right. Return ONLY JSON like '
            '{"commands":["F","R","F"],"explain":"one short sentence"} with at most 150 commands '
            'for this goal: ' + text))
        cmds = [c for c in j.get("commands", []) if c in ("F", "L", "R")][:150]
        return {"robot": cmds, "answer": str(j.get("explain", ""))[:500]}

    if mod == "music":
        j = _json_from(ask(
            'Compose a short piece. Return ONLY JSON {"title":"...","tempo":90,"notes":[[midi_note,beats],...]} '
            'with 32 to 64 notes; midi_note 60 is middle C and 0 is a rest; beats between 0.25 and 4. '
            'Request: ' + text))
        try:
            notes = [[int(n), float(b)] for n, b in j.get("notes", [])
                     if 0 <= int(n) <= 127 and 0.1 <= float(b) <= 8][:200]
            tempo = max(40, min(240, int(j.get("tempo", 100))))
        except (TypeError, ValueError):
            raise ValueError("The AI did not return usable music. Please try again.")
        return {"music": {"tempo": tempo, "notes": notes}, "answer": "Title: " + str(j.get("title", "Untitled"))[:80]}

    raise ValueError("Pick one of the Lab skills.")


@app.post("/lab")
def lab_route():
    if denied():
        return jsonify(error="locked"), 401
    try:
        return jsonify(_lab(request.get_json(silent=True) or {}))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        print("lab failed:", e)
        return jsonify(error=str(e)[:300]), 500


@app.post("/code")
def code_route():
    if denied():
        return jsonify(error="locked"), 401
    return _code()


@app.post("/video")
def video_route():
    if denied():
        return jsonify(error="locked"), 401
    return _video()


@app.errorhandler(413)
def too_big(_e):
    return jsonify(error=f"This file is too large. The limit is {MAX_UPLOAD_MB} MB."), 413


def ingest(paths):
    """python deep_oq.py ingest my_folder  -> add .txt .md code .csv .json .pdf files to the database"""
    text_ext = {".txt", ".md", ".py", ".js", ".ts", ".java", ".cpp", ".c", ".html", ".css",
                ".json", ".csv", ".kt", ".go", ".rs", ".sql"}
    total = 0
    for p in paths:
        files = []
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                files += [os.path.join(root, n) for n in names]
        else:
            files = [p]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            try:
                if ext == ".pdf":
                    from pypdf import PdfReader
                    text = "\n".join((pg.extract_text() or "") for pg in PdfReader(f).pages)
                elif ext in text_ext:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                        text = fh.read()
                else:
                    continue
            except Exception as e:
                print("skip", f, e)
                continue
            if text.strip():
                n = kb_add_text(text, title=os.path.basename(f), source=f)
                total += n
                print(f"{f}: {n} chunks")
    print("done.", kb_stats(), "added", total)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "ingest":
        ingest(sys.argv[2:])
    else:
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
