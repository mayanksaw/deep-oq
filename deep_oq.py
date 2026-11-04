"""
Deep OQ - one-file AI app: Chat, Code, Photo and Video.
Made for Mayank Kumar Saw. Everything is in this single file.

WHAT IT DOES
  Chat   knowledge database + exact calculator/counter + live web search + plan/solve/verify
  Code   write, fix, explain, convert, review code in 19 languages (HTML and JavaScript can run)
  Photo  photo editor that runs in the browser (looks, sliders, rotate, crop, caption, download)
  Video  trim, speed, resize, rotate, looks, MP4 / GIF / MP3 output (needs ffmpeg, see below)
  Image  make pictures from a description
  Data   open a CSV: summary, charts, AI analysis
  Music  compose a piece, play it, download MIDI
  Work   emails, summaries, meeting actions, reports, SQL, video scripts
  Farm   water-need calculator, crop plans, plant problems
  Planet 7-day weather warnings, home energy and solar, carbon footprint
  About  profile page (details live in mayank_profile.py)

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
import urllib.parse
import urllib.request
import uuid
from html import escape as _esc

from flask import Flask, jsonify, request, send_file
from groq import Groq


def _friendly(e):
    """What users see when something fails. The real error stays in the server log only,
    so no model, provider or setup details ever reach the screen."""
    print("request failed:", repr(e)[:400])
    return "Deep OQ could not answer right now. Please try again in a moment."


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
         "Kotlin", "Swift", "SQL", "HTML/CSS", "Bash", "Dart", "Arduino (C++)",
         "MicroPython", "ROS 2 (Python)"}


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
        return jsonify(error=_friendly(e)), 500


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

# ---- About Mayank Kumar Saw: the details live in the separate file mayank_profile.py
try:
    from mayank_profile import PROFILE
except ImportError:  # the app still runs without that file, with a bare profile
    PROFILE = {"name": "Mayank Kumar Saw", "role": "Founder and developer of Deep OQ"}


def _profile_text():
    p = PROFILE
    out = [f"{p.get('name', '')} is the {str(p.get('role', '')).lower()}. {p.get('about', '')}".strip()]
    for e in p.get("education", []):
        bits = [e.get("level", ""), e.get("field", ""), e.get("place", ""), e.get("years", "")]
        line = ", ".join(b for b in bits if b)
        if e.get("note"):
            line += (" - " if line else "") + e["note"]
        if line:
            out.append("Education: " + line)
    for k in p.get("skills", []):
        out.append(f"Skills ({k.get('group', '')}): {k.get('items', '')}")
    out += [f"Project {x['name']}: {x['text']}" for x in p.get("projects", [])]
    if p.get("interests"):
        out.append("Interests: " + p["interests"])
    out += ["Goal: " + g for g in p.get("goals", [])]
    return "\n".join(out)


def _profile_html():
    p = PROFILE
    name = p.get("name", "")
    initials = "".join(w[0] for w in name.split()[:3]).upper()

    def section(title, body):
        return f'<h3 class="ab-h">{title}</h3>{body}' if body else ""

    def rows(pairs):
        items = "".join(f"<li><b>{_esc(a)}</b>" + (f"<span>{_esc(b)}</span>" if b else "") + "</li>"
                        for a, b in pairs)
        return f'<ul class="projects">{items}</ul>' if items else ""

    edu = []
    for e in p.get("education", []):
        title = " - ".join(x for x in (e.get("level", ""), e.get("field", "")) if x)
        detail = " · ".join(x for x in (e.get("place", ""), e.get("years", "")) if x)
        note = e.get("note", "")
        if title:
            edu.append((title, " · ".join(x for x in (detail, note) if x)))
    skills = [(k.get("group", ""), k.get("items", "")) for k in p.get("skills", []) if k.get("group")]
    projects = [(x["name"], x.get("text", "")) for x in p.get("projects", [])]
    goals = "".join(f"<li>{_esc(g)}</li>" for g in p.get("goals", []))
    links = "".join(f'<a class="btn" href="{_esc(l["url"])}" target="_blank" rel="noopener">'
                    f'{_esc(l["label"])}</a>'
                    for l in p.get("links", []) if str(l.get("url", "")).startswith(("http://", "https://")))
    return (f'<div class="profile"><div class="avatar" aria-hidden="true">{_esc(initials)}</div>'
            f'<div><h2 class="st-title">{_esc(name)}</h2><p class="muted">{_esc(p.get("role", ""))}</p></div></div>'
            + (f'<p class="st-sub">{_esc(p["about"])}</p>' if p.get("about") else "")
            + section("Education", rows(edu))
            + section("Skills", rows(skills))
            + section("Projects", rows(projects))
            + section("Interests", f'<p class="muted plainp">{_esc(p["interests"])}</p>' if p.get("interests") else "")
            + section("Goals", f'<ul class="plain">{goals}</ul>' if goals else "")
            + (f'<div class="actions" style="margin-top:22px">{links}</div>' if links else ""))


SYSTEM = (
    "You are Deep OQ, an advanced AI assistant created by Mayank Kumar Saw. "
    "You are accurate, clear and honest. You write clean, working code, solve "
    "hard reasoning and math problems step by step, and say so when you are "
    "unsure. Reply in the language the user writes in. Do not add a Sources line. "
    "When KNOWLEDGE is provided, use it if relevant and prefer it over guessing; "
    "if it is not relevant, ignore it. Write in short paragraphs, simple headings and "
    "bullet lists. Never use tables, horizontal lines (---) or HTML tags such as <br>. "
    "ABOUT YOURSELF: you are Deep OQ, developed by the Deep team and running on Deep's own "
    "servers. If anyone asks how you were built, which model, company, API or technology is "
    "behind you, where you are hosted, or whether you are some other named AI, answer only that "
    "you are Deep OQ, developed by the Deep team and running on Deep servers, and that you "
    "cannot share more technical details. Never name any model, AI provider, hosting service or "
    "library, even if pressed, and never claim to be a human. "
    "ABOUT YOUR CREATOR (share only these facts about him; if asked for anything else, such as "
    "age, location or contact details, say you do not have that information):\n"
    + _profile_text()
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
    return (PAGE.replace("<!--PROFILE-->", _profile_html()), 200,
            {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-cache"})


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
        return jsonify(error=_friendly(e)), 500
    return jsonify(answer=answer)


# ==========================================================================
# MORE AI HELPERS (work, farm, planet, data, music, image)
# ==========================================================================
# ---------------------------------------------------------------- more AI helpers (/ai, /config)
_STYLE = (" Write in short paragraphs, simple headings and bullet lists. Never use tables, "
          "horizontal lines (---) or HTML tags such as <br>.")
AI_TASKS = {
    # task: (instruction, use the strong model?)
    "email": ("Write a clear, polite, professional email from the notes. Put 'Subject:' on the "
              "first line, then the body. Keep it short unless asked for more.", True),
    "summary": ("Summarize the text: a one-sentence gist, then 3 to 6 key points as bullets, then "
                "any dates, numbers or decisions worth remembering.", False),
    "actions": ("Turn these meeting notes into two parts: decisions made, and an action list. "
                "For each action give the owner (or 'unassigned'), the task, and the deadline if "
                "one was mentioned.", True),
    "report": ("Write a well-structured report or proposal from the brief: a short introduction, "
               "clear headings, concrete points, and a closing list of next steps.", True),
    "sql": ("Write SQL for the request. Do not ask questions: state any assumptions in one line. "
            "Give the query in ONE fenced sql block, then a short explanation of how it works.", True),
    "script": ("Write a video script and storyboard. Start with a one-line concept, then list the "
               "scenes. For each scene give the shot description, narration or dialogue, on-screen "
               "text and rough duration. End with a shot list and a music suggestion.", True),
    "farm_plan": ("You are an agriculture adviser for Indian and similar farming conditions. Give a "
                  "practical crop plan: land preparation, sowing time and method, seed rate, "
                  "fertilizer schedule, irrigation timing, pests and diseases to watch, harvest time "
                  "and a realistic yield range. End by saying to check the plan with the nearest "
                  "Krishi Vigyan Kendra or local agriculture officer.", True),
    "farm_problem": ("You are an agriculture adviser. From the description, list the likely causes "
                     "ranked from most to least likely, how to confirm each one, what to do now "
                     "(organic first, then chemical options; tell the farmer to follow the product "
                     "label for doses), and how to prevent it. Say clearly that this is a guess "
                     "from a description and a local agriculture officer or lab should confirm it.", True),
    "data": ("You are a careful data analyst. You get a summary of a dataset and optionally a "
             "question. Give the key findings, anything that looks unusual, data quality problems, "
             "and 3 useful analyses or charts to try next. Use only numbers present in the summary. "
             "Never invent values.", True),
    "planet_warning": ("You get a short list of weather warnings for a place. Explain in plain "
                       "words what they mean and give practical steps for people, farms and homes "
                       "for each warning. Be calm and specific. If there are no warnings, say so "
                       "and give one useful tip.", False),
    "planet_energy": ("You get a home's electricity use and solar numbers. Give a practical savings "
                      "plan: the biggest savings first, with rough effect on the bill, plus tips for "
                      "the solar setup. Use only the numbers given.", False),
    "planet_carbon": ("You get a person's yearly carbon footprint breakdown. Give a realistic plan "
                      "to cut it: the three biggest actions first, with the rough saving of each "
                      "based on the numbers given.", False),
    "image_prompt": ("Rewrite the idea as ONE detailed image-generation prompt covering subject, "
                     "setting, lighting, style and mood, in under 60 words. Output only the prompt.", False),
    "music_params": ("Turn the description into settings for a music composer. Reply with ONLY a "
                     "JSON object like {\"mood\":\"happy\",\"key\":\"C\",\"tempo\":120,\"bars\":16,"
                     "\"sound\":\"piano\",\"drums\":true}. mood is one of happy, calm, sad, epic, "
                     "mysterious, playful. key is one of C, C#, D, D#, E, F, F#, G, G#, A, A#, B. "
                     "tempo is 60 to 180. bars is 8, 16 or 32. sound is one of piano, pad, chip, "
                     "strings. drums is true or false.", False),
}
IMAGE_URL = os.environ.get(
    "DEEP_OQ_IMAGE_URL",
    "https://image.pollinations.ai/prompt/{prompt}?width={w}&height={h}&seed={seed}&nologo=true")


@app.post("/ai")
def ai_route():
    if denied():
        return jsonify(error="locked"), 401
    d = request.get_json(silent=True) or {}
    task = d.get("task")
    text = str(d.get("text", "")).strip()[:12000]
    extra = str(d.get("extra", "")).strip()[:2000]
    if task not in AI_TASKS:
        return jsonify(error="Unknown task."), 400
    if not text:
        return jsonify(error="Add some text first."), 400
    instruction, smart = AI_TASKS[task]
    user = text + (("\n\nExtra instructions:\n" + extra) if extra else "")
    msgs = [{"role": "system", "content": "You are Deep OQ, an expert assistant created by Mayank "
             "Kumar Saw. " + instruction + _STYLE},
            {"role": "user", "content": user}]
    try:
        return jsonify(answer=chat(msgs, 0.3, 3072, smart=smart))
    except Exception as e:
        print("ai failed:", e)
        return jsonify(error=_friendly(e)), 500


_image_slots = threading.Semaphore(2)
IMAGE_SIZES = {(1024, 1024), (1344, 768), (768, 1344)}


@app.post("/image")
def image_route():
    """Makes a picture from a description. The request goes through this server so the
    browser never sees which image service is used."""
    if denied():
        return jsonify(error="locked"), 401
    d = request.get_json(silent=True) or {}
    prompt = str(d.get("prompt", "")).strip()[:900]
    try:
        w, h, seed = int(d.get("w", 1024)), int(d.get("h", 1024)), int(d.get("seed", 1)) % 10_000_000
    except (TypeError, ValueError):
        return jsonify(error="Check the picture settings."), 400
    if not prompt:
        return jsonify(error="Describe the picture first."), 400
    if (w, h) not in IMAGE_SIZES:
        return jsonify(error="Pick one of the listed shapes."), 400
    if not _image_slots.acquire(blocking=False):
        return jsonify(error="Deep OQ is busy making other pictures. Try again in a moment."), 503
    try:
        url = (IMAGE_URL.replace("{prompt}", urllib.parse.quote(prompt)).replace("{w}", str(w))
               .replace("{h}", str(h)).replace("{seed}", str(seed)))
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; DeepOQ)"})
        with urllib.request.urlopen(req, timeout=100) as r:
            ctype = r.headers.get("Content-Type", "")
            data = r.read(8 * 1024 * 1024 + 1)
        if not ctype.startswith("image/") or not data or len(data) > 8 * 1024 * 1024:
            raise ValueError("not an image")
        return app.response_class(data, mimetype=ctype.split(";")[0])
    except Exception as e:
        print("image failed:", repr(e)[:300])
        return jsonify(error="Could not make the picture right now. Try again in a minute."), 502
    finally:
        _image_slots.release()


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
@media (max-width:1240px){.tabs{order:3;width:100%;margin:0;justify-content:flex-start}}

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
.profile{display:flex;align-items:center;gap:20px;margin:18px 0 16px}
.profile .st-title{margin:0}
.profile .muted{margin-top:4px;font-size:17px}
.avatar{flex:none;width:84px;height:84px;border-radius:50%;background:var(--ink);color:var(--bg);display:grid;place-items:center;
  font-family:var(--ui);font-weight:700;font-size:27px;letter-spacing:-.02em;position:relative}
.avatar::after{content:"";position:absolute;right:3px;bottom:5px;width:17px;height:17px;border-radius:50%;background:var(--accent);border:3px solid var(--bg)}
.ab-h{font-family:var(--ui);font-size:19px;letter-spacing:-.015em;margin:26px 0 12px}
.projects{list-style:none;margin:0;padding:0;display:grid;gap:14px;max-width:62ch}
.projects li{display:grid;gap:2px}
.projects b{font-family:var(--ui);font-size:17px;font-weight:600}
.projects span{color:var(--ink-2);font-size:17px;line-height:1.5}
.plain{margin:0;padding-left:1.2em;display:grid;gap:8px;max-width:62ch;color:var(--ink-2);font-size:17px;line-height:1.5}
.plainp{max-width:62ch;font-size:17px;line-height:1.5}
.tabs{overflow-x:auto;scrollbar-width:none;max-width:100%}
.tabs::-webkit-scrollbar{display:none}
body>*{min-width:0;max-width:100vw}
.top{min-width:0;max-width:100%}
.tabs{min-width:0;flex:0 1 auto}
.tab{flex:none;white-space:nowrap}
.row3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
@media (max-width:560px){.row3{grid-template-columns:minmax(0,1fr)}}
.cardbox{border:1px solid var(--line);border-radius:16px;background:var(--surface);padding:16px;display:grid;gap:12px;min-width:0}
.kv{display:grid;grid-template-columns:1fr auto;gap:8px 18px;font-family:var(--ui);font-size:15px;margin:0}
.kv dt{color:var(--ink-2)}
.kv dd{margin:0;text-align:right;font-weight:600;font-variant-numeric:tabular-nums}
.big{font-family:var(--ui);font-weight:700;font-size:36px;letter-spacing:-.03em;line-height:1.1}
.sub2{font-family:var(--ui);font-size:14px;color:var(--ink-2);margin:0;line-height:1.5}
.days{display:grid;grid-template-columns:repeat(auto-fill,minmax(132px,1fr));gap:10px}
.day{border:1px solid var(--line);border-radius:14px;padding:10px 12px;background:var(--surface);font-family:var(--ui);font-size:14px;line-height:1.55}
.day b{display:block;font-size:15px;font-weight:600}
.alerts{display:grid;gap:8px}
.alert{border-radius:12px;padding:10px 14px;font-family:var(--ui);font-size:15px;line-height:1.45}
.alert.warn{background:rgba(242,160,46,.22)}
.alert.severe{background:rgba(209,61,40,.24)}
.alert.ok{background:rgba(35,121,126,.18)}
.apps{display:grid;gap:8px}
.approw{display:grid;grid-template-columns:minmax(0,1.5fr) 70px 46px 52px 34px;gap:6px;align-items:center}
.approw .in{padding:8px 9px;font-size:14px}
.approw .x{border:0;background:none;cursor:pointer;font-size:20px;color:var(--ink-2);border-radius:8px}
.approw .x:hover{background:var(--surface)}
.apphead{display:grid;grid-template-columns:minmax(0,1.5fr) 70px 46px 52px 34px;gap:6px;font-family:var(--ui);font-size:12.5px;color:var(--ink-2)}
canvas.chart{width:100%;height:320px;display:block;border:1px solid var(--line);border-radius:14px;background:var(--surface)}
.tbls{overflow:auto;max-height:340px;border:1px solid var(--line);border-radius:12px;background:var(--surface)}
.tbls table{border-collapse:collapse;width:100%;font-family:var(--ui);font-size:14px}
.tbls th{position:sticky;top:0;text-align:left;font-weight:600;padding:8px 12px;background:var(--bg)}
.tbls td{padding:7px 12px;vertical-align:top}
.tbls tr:nth-child(even) td{background:rgba(127,130,170,.08)}
.bars{display:grid;gap:8px}
.bar{display:grid;grid-template-columns:110px minmax(0,1fr) 64px;gap:10px;align-items:center;font-family:var(--ui);font-size:14px}
.bar i{display:block;height:12px;border-radius:6px;background:var(--river)}
.bar span:last-child{text-align:right;font-variant-numeric:tabular-nums}
canvas.roll{width:100%;height:230px;display:block}
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
    <button class="tab" type="button" data-view="image">Image</button>
    <button class="tab" type="button" data-view="data">Data</button>
    <button class="tab" type="button" data-view="music">Music</button>
    <button class="tab" type="button" data-view="work">Work</button>
    <button class="tab" type="button" data-view="farm">Farm</button>
    <button class="tab" type="button" data-view="planet">Planet</button>
    <button class="tab" type="button" data-view="about">About</button>
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
    <p class="st-sub">Write, fix, explain, convert or review code in 19 languages. Choose what you need, then press the button.</p>
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
          <label class="lab">Language<select class="in" id="clang"><option>Python</option><option>JavaScript</option><option>TypeScript</option><option>Java</option><option>C++</option><option>C</option><option>C#</option><option>Go</option><option>Rust</option><option>PHP</option><option>Kotlin</option><option>Swift</option><option>SQL</option><option>HTML/CSS</option><option>Bash</option><option>Dart</option><option>Arduino (C++)</option><option>MicroPython</option><option>ROS 2 (Python)</option></select></label>
          <label class="lab" id="ctgtWrap" hidden>Convert to<select class="in" id="ctgt"><option>Python</option><option>JavaScript</option><option>TypeScript</option><option>Java</option><option>C++</option><option>C</option><option>C#</option><option>Go</option><option>Rust</option><option>PHP</option><option>Kotlin</option><option>Swift</option><option>SQL</option><option>HTML/CSS</option><option>Bash</option><option>Dart</option><option>Arduino (C++)</option><option>MicroPython</option><option>ROS 2 (Python)</option></select></label>
        </div>
        <div class="grp"><h3>Robotics starters</h3><div class="seg" id="crobo"></div></div>
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

<section class="view studio" id="v-image">
  <div class="wrap">
    <h2 class="st-title">Image</h2>
    <p class="st-sub">Describe a picture and Deep OQ makes it. A picture can take 10 to 40 seconds.</p>
    <div class="grid2 rev">
      <div class="stage" id="istage">
        <div class="drop" id="idrop" style="cursor:default"><strong>Your picture appears here</strong><span>Describe it, then press Create image.</span></div>
        <img id="i-img" alt="The picture Deep OQ made" hidden>
      </div>
      <div class="panel">
        <label class="lab">Describe the picture
          <textarea class="in" id="i-prompt" rows="4" placeholder="A tiger resting under a banyan tree at sunrise"></textarea></label>
        <div><button class="btn" id="i-better" type="button">Improve my description</button></div>
        <div class="grp"><h3>Style</h3>
          <div class="seg" id="istyles">
            <button type="button" data-s="photo" aria-pressed="true">Photo</button>
            <button type="button" data-s="art" aria-pressed="false">Illustration</button>
            <button type="button" data-s="three" aria-pressed="false">3D</button>
            <button type="button" data-s="anime" aria-pressed="false">Anime</button>
            <button type="button" data-s="water" aria-pressed="false">Watercolour</button>
            <button type="button" data-s="logo" aria-pressed="false">Logo</button>
            <button type="button" data-s="none" aria-pressed="false">None</button>
          </div></div>
        <label class="lab">Shape<select class="in" id="i-size"><option value="1024x1024">Square</option><option value="1344x768">Wide</option><option value="768x1344">Tall</option></select></label>
        <div class="actions"><button class="primary" id="i-go" type="button">Create image</button>
          <button class="btn" id="i-again" type="button" disabled>New variation</button>
          <button class="btn" id="i-dl" type="button" disabled>Download</button></div>
        <div class="status" id="istatus" role="status"></div>
      </div>
    </div>
  </div>
</section>

<section class="view studio" id="v-data">
  <div class="wrap">
    <h2 class="st-title">Data</h2>
    <p class="st-sub">Open a CSV file to see a summary, draw charts and ask questions about it. The file is read in your browser. Only a short summary is sent when you ask for analysis.</p>
    <div class="stage" id="dstage" style="min-height:170px">
      <label class="drop"><input type="file" id="dfile" accept=".csv,.tsv,.txt,text/csv" hidden><strong>Choose a CSV file</strong><span>or drop one here. Up to 8 MB.</span></label>
    </div>
    <p class="note" id="dnote" hidden></p>
    <div id="dwork" hidden>
      <p class="muted" id="dinfo" style="margin-top:14px"></p>
      <h3 class="ab-h">Columns</h3>
      <div class="tbls"><table id="dsum"></table></div>
      <h3 class="ab-h">Chart</h3>
      <div class="seg" id="dkind" style="margin-bottom:12px">
        <button type="button" data-k="bar" aria-pressed="true">Bar</button>
        <button type="button" data-k="hist" aria-pressed="false">Histogram</button>
        <button type="button" data-k="line" aria-pressed="false">Line</button>
      </div>
      <div class="row3" style="margin-bottom:12px">
        <label class="lab" id="dxw">Group by<select class="in" id="d-x"></select></label>
        <label class="lab">Number column<select class="in" id="d-y"></select></label>
        <label class="lab" id="daggw">Combine with<select class="in" id="d-agg"><option value="sum">Sum</option><option value="avg">Average</option><option value="count">Count of rows</option></select></label>
      </div>
      <canvas class="chart" id="dchart" role="img" aria-label="Chart of your data"></canvas>
      <h3 class="ab-h">Ask about this data</h3>
      <div class="panel" style="max-width:720px">
        <textarea class="in" id="d-q" rows="2" placeholder="Which region sells the most? Leave this empty for an overview."></textarea>
        <div><button class="primary" id="d-go" type="button">Analyse</button></div>
        <div class="outpane" id="d-out" style="min-height:90px" aria-live="polite"><p class="muted">The analysis appears here.</p></div>
      </div>
    </div>
  </div>
</section>

<section class="view studio" id="v-music">
  <div class="wrap">
    <h2 class="st-title">Music</h2>
    <p class="st-sub">Deep OQ composes a short piece with a melody, chords, bass and drums. Play it here, or download it as a MIDI file for any music app.</p>
    <div class="grid2 rev">
      <div class="stagecol">
        <div class="stage" style="min-height:0;padding:14px"><canvas class="roll" id="m-roll" role="img" aria-label="Notes of the piece"></canvas></div>
        <div class="actions"><button class="primary" id="m-play" type="button">Play</button>
          <button class="btn" id="m-new" type="button">Compose a new piece</button>
          <button class="btn" id="m-dl" type="button">Download MIDI</button></div>
        <div class="status" id="m-status" role="status"></div>
      </div>
      <div class="panel">
        <label class="lab">Describe the music
          <textarea class="in" id="m-desc" rows="2" placeholder="A calm, rainy evening in the mountains"></textarea></label>
        <div><button class="btn" id="m-fromdesc" type="button">Set from description</button></div>
        <div class="row2">
          <label class="lab">Mood<select class="in" id="m-mood"><option value="happy">Happy</option><option value="calm">Calm</option><option value="sad">Sad</option><option value="epic">Epic</option><option value="mysterious">Mysterious</option><option value="playful">Playful</option></select></label>
          <label class="lab">Key<select class="in" id="m-key"><option>C</option><option>C#</option><option>D</option><option>D#</option><option>E</option><option>F</option><option>F#</option><option>G</option><option>G#</option><option>A</option><option>A#</option><option>B</option></select></label>
          <label class="lab">Length<select class="in" id="m-bars"><option value="8">8 bars</option><option value="16" selected>16 bars</option><option value="32">32 bars</option></select></label>
          <label class="lab">Sound<select class="in" id="m-sound"><option value="piano">Piano</option><option value="pad">Soft pad</option><option value="chip">Chiptune</option><option value="strings">Strings</option></select></label>
        </div>
        <label class="sl" style="grid-template-columns:70px minmax(0,1fr) 60px">Tempo<input type="range" id="m-tempo" min="60" max="180" value="110"><output id="m-tempoo">110 bpm</output></label>
        <label style="display:flex;gap:10px;align-items:center;font-family:var(--ui);font-size:15px"><input type="checkbox" id="m-drums" checked> Add drums</label>
      </div>
    </div>
  </div>
</section>

<section class="view studio" id="v-work">
  <div class="wrap">
    <h2 class="st-title">Work</h2>
    <p class="st-sub">Everyday office jobs, done in seconds. Choose a job, add your notes and read the result.</p>
    <div class="grid2 even">
      <div class="panel">
        <div class="seg" id="wkinds" role="group" aria-label="Job">
          <button type="button" data-k="email" aria-pressed="true">Email</button>
          <button type="button" data-k="summary" aria-pressed="false">Summary</button>
          <button type="button" data-k="actions" aria-pressed="false">Meeting actions</button>
          <button type="button" data-k="report" aria-pressed="false">Report</button>
          <button type="button" data-k="sql" aria-pressed="false">SQL</button>
          <button type="button" data-k="script" aria-pressed="false">Video script</button>
        </div>
        <label class="lab"><span id="wtextL">What should the email say?</span><textarea class="in" id="w-text" rows="9"></textarea></label>
        <label class="lab"><span id="wextraL">Tone (optional)</span><textarea class="in" id="w-extra" rows="2"></textarea></label>
        <div><button class="primary" id="w-go" type="button">Write email</button></div>
      </div>
      <div class="panel"><div class="outpane" id="w-out" aria-live="polite"><p class="muted">Your result appears here.</p></div></div>
    </div>
  </div>
</section>

<section class="view studio" id="v-farm">
  <div class="wrap">
    <h2 class="st-title">Farm</h2>
    <p class="st-sub">Work out how much water a crop needs, get a crop plan, or describe a plant problem. These are guides. Check important decisions with your local agriculture officer.</p>
    <div class="seg" id="fmodes" role="group" aria-label="Farm tool" style="margin-bottom:22px">
      <button type="button" data-m="water" aria-pressed="true">Water needs</button>
      <button type="button" data-m="plan" aria-pressed="false">Crop plan</button>
      <button type="button" data-m="problem" aria-pressed="false">Plant problem</button>
    </div>
    <div id="f-water" class="grid2 even">
      <div class="panel">
        <div class="row2">
          <label class="lab">Crop<select class="in" id="f-crop"></select></label>
          <label class="lab">Growth stage<select class="in" id="f-stage"><option value="ini">Just planted</option><option value="dev">Growing fast</option><option value="mid" selected>Fully grown</option><option value="end">Ripening</option></select></label>
          <label class="lab">Weather<select class="in" id="f-clim"><option value="3">Cool (3 mm/day)</option><option value="4">Mild (4 mm/day)</option><option value="5" selected>Warm (5 mm/day)</option><option value="6.5">Hot (6.5 mm/day)</option><option value="8">Very hot and dry (8 mm/day)</option></select></label>
          <label class="lab">Water system<select class="in" id="f-eff"><option value="0.9">Drip (90%)</option><option value="0.75" selected>Sprinkler (75%)</option><option value="0.6">Flood (60%)</option></select></label>
          <label class="lab">Field size<input class="in" id="f-area" type="number" min="0" step="0.1" value="1"></label>
          <label class="lab">Unit<select class="in" id="f-unit"><option value="4046.86">Acre</option><option value="10000">Hectare</option><option value="1">Square metre</option></select></label>
          <label class="lab">Useful rain (mm/day)<input class="in" id="f-rain" type="number" min="0" step="0.5" value="0"></label>
        </div>
      </div>
      <div class="cardbox" id="f-wout" aria-live="polite"></div>
    </div>
    <div id="f-ai" class="grid2 even" hidden>
      <div class="panel">
        <div id="f-planform" class="panel">
          <label class="lab">Crop<input class="in" id="fp-crop" type="text" placeholder="Wheat"></label>
          <div class="row2">
            <label class="lab">Place<input class="in" id="fp-place" type="text" placeholder="Nagpur, Maharashtra"></label>
            <label class="lab">Season<select class="in" id="fp-season"><option>Kharif (monsoon)</option><option>Rabi (winter)</option><option>Zaid (summer)</option><option>Not sure</option></select></label>
            <label class="lab">Soil<input class="in" id="fp-soil" type="text" placeholder="Black cotton soil"></label>
            <label class="lab">Field size<input class="in" id="fp-area" type="text" placeholder="2 acres"></label>
          </div>
          <div><button class="primary" id="fp-go" type="button">Make a crop plan</button></div>
        </div>
        <div id="f-probform" class="panel" hidden>
          <label class="lab">Crop<input class="in" id="fq-crop" type="text" placeholder="Tomato"></label>
          <label class="lab">What do you see?<textarea class="in" id="fq-what" rows="5" placeholder="Yellow spots on lower leaves, spreading upward. Started 5 days ago."></textarea></label>
          <label class="lab">Weather and care lately (optional)<input class="in" id="fq-ctx" type="text" placeholder="Humid, rain last week, no spray yet"></label>
          <div><button class="primary" id="fq-go" type="button">Find likely causes</button></div>
        </div>
      </div>
      <div class="panel"><div class="outpane" id="f-out" aria-live="polite"><p class="muted">Your result appears here.</p></div></div>
    </div>
  </div>
</section>

<section class="view studio" id="v-planet">
  <div class="wrap">
    <h2 class="st-title">Planet</h2>
    <p class="st-sub">See weather warnings for the next 7 days, plan home energy and solar power, or estimate your carbon footprint.</p>
    <div class="seg" id="pmodes" role="group" aria-label="Planet tool" style="margin-bottom:22px">
      <button type="button" data-m="warn" aria-pressed="true">Weather warnings</button>
      <button type="button" data-m="energy" aria-pressed="false">Home energy</button>
      <button type="button" data-m="carbon" aria-pressed="false">Carbon footprint</button>
    </div>

    <div id="pl-warn">
      <form class="actions" id="pw-form" style="margin-bottom:14px;max-width:560px"><input class="in" id="pw-city" type="text" placeholder="Search a city, for example Nagpur" aria-label="City" style="flex:1"><button class="primary" type="submit">Check</button></form>
      <div class="seg" id="pw-places" style="margin-bottom:12px" hidden></div>
      <p class="note" id="pw-note" hidden></p>
      <div id="pw-result" hidden>
        <h3 class="ab-h" style="margin-top:6px" id="pw-title"></h3>
        <div class="alerts" id="pw-alerts"></div>
        <h3 class="ab-h">Next 7 days</h3>
        <div class="days" id="pw-days"></div>
        <div class="grid2 even" style="margin-top:22px"><div class="panel"><div><button class="btn" id="pw-explain" type="button">Explain and advise</button></div>
          <div class="outpane" id="pw-out" style="min-height:70px" aria-live="polite"><p class="muted">Advice appears here.</p></div></div></div>
      </div>
    </div>

    <div id="pl-energy" hidden class="grid2 even">
      <div class="panel">
        <div class="grp"><h3>Appliances</h3>
          <div class="apphead"><span>Appliance</span><span>Watts</span><span>Qty</span><span>Hours</span><span></span></div>
          <div class="apps" id="e-apps"></div>
          <div><button class="btn" id="e-add" type="button">Add appliance</button></div></div>
        <div class="row2">
          <label class="lab">Price per unit (₹/kWh)<input class="in" id="e-tariff" type="number" min="0" step="0.5" value="8"></label>
          <label class="lab">Sunny hours per day<input class="in" id="e-sun" type="number" min="1" max="9" step="0.5" value="5"></label>
          <label class="lab">Panel size (watts)<input class="in" id="e-panel" type="number" min="100" step="10" value="400"></label>
          <label class="lab">Solar cost (₹ per kW)<input class="in" id="e-cost" type="number" min="0" step="1000" value="50000"></label>
        </div>
        <p class="sub2">Prices are examples. Change them to match your area and your quotes.</p>
      </div>
      <div class="panel"><div class="cardbox" id="e-out" aria-live="polite"></div>
        <div><button class="btn" id="e-ai" type="button">Make a savings plan</button></div>
        <div class="outpane" id="e-aiout" style="min-height:70px"><p class="muted">The plan appears here.</p></div></div>
    </div>

    <div id="pl-carbon" hidden class="grid2 even">
      <div class="panel">
        <div class="row2">
          <label class="lab">Electricity (units per month)<input class="in" id="c-kwh" type="number" min="0" value="200"></label>
          <label class="lab">Vehicle (km per month)<input class="in" id="c-km" type="number" min="0" value="500"></label>
          <label class="lab">Fuel<select class="in" id="c-fuel"><option value="2.31">Petrol</option><option value="2.68">Diesel</option><option value="0">No vehicle or electric</option></select></label>
          <label class="lab">Vehicle mileage (km per litre)<input class="in" id="c-mil" type="number" min="1" value="15"></label>
          <label class="lab">LPG cylinders per year<input class="in" id="c-lpg" type="number" min="0" value="6"></label>
          <label class="lab">Flight hours per year<input class="in" id="c-fly" type="number" min="0" value="4"></label>
        </div>
        <p class="sub2">These are rough estimates from common emission factors, good for comparing choices.</p>
      </div>
      <div class="panel"><div class="cardbox" id="c-out" aria-live="polite"></div>
        <div><button class="btn" id="c-ai" type="button">Make a reduction plan</button></div>
        <div class="outpane" id="c-aiout" style="min-height:70px"><p class="muted">The plan appears here.</p></div></div>
    </div>
  </div>
</section>

<section class="view studio" id="v-about">
  <div class="wrap"><!--PROFILE--></div>
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
const VIEWS=['chat','code','photo','video','image','data','music','work','farm','planet','about'];
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
  if(v!=='music'&&typeof mAC!=='undefined'&&mAC)stopMusic();
  if(v==='music')drawRoll(-1);
  if(v==='data')drawChart();
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


/* ================= shared helpers for the new sections ================= */
async function askAI(task,text,extra){
  const r=await api('/ai',{task,text,extra:extra||''});
  const j=await r.json().catch(()=>({}));
  if(!r.ok||j.error)throw new Error(j.error||'The server did not answer.');
  return j.answer;
}
async function runAI(task,text,extra,out,btn){
  btn.disabled=true;out.innerHTML='<p class="muted">Working on it. This can take up to a minute.</p>';
  try{const a=await askAI(task,text,extra);out.innerHTML='<div class="body">'+md(a)+'</div>';return a}
  catch(e){out.innerHTML='<p class="note">Could not get an answer. '+esc(e.message)+'</p>';return null}
  finally{btn.disabled=false}
}
function segment(sel,cb){
  document.querySelectorAll(sel+' button').forEach(b=>b.onclick=()=>{
    document.querySelectorAll(sel+' button').forEach(x=>x.setAttribute('aria-pressed',x===b));cb(b);
  });
}
const fmtN=(n,d=2)=>Number.isFinite(n)?(Math.abs(n)>=1000?Math.round(n).toLocaleString('en-IN'):String(+n.toFixed(d))):'-';
const cssv=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const FONT='system-ui, -apple-system, "Segoe UI", sans-serif';
function setupCanvas(cv){
  const dpr=window.devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;
  cv.width=Math.max(1,Math.round(w*dpr));cv.height=Math.max(1,Math.round(h*dpr));
  const g=cv.getContext('2d');g.setTransform(dpr,0,0,dpr,0,0);g.clearRect(0,0,w,h);return {g,w,h};
}

/* ================= image ================= */
const ISTYLE={photo:', realistic photograph, natural light, fine detail',art:', digital illustration, rich colours, fine detail',
  three:', 3D render, soft studio lighting, fine detail',anime:', anime style, vibrant colours, clean lines',
  water:', watercolour painting, soft edges, paper texture',logo:', minimal flat vector logo, white background',none:''};
let istyle='photo',iurl='',iext='png',iseed=0;
const istat=m=>$('#istatus').textContent=m;
segment('#istyles',b=>istyle=b.dataset.s);
async function makeImage(fresh){
  const p=$('#i-prompt').value.trim();
  if(!p){istat('Describe the picture first.');$('#i-prompt').focus();return}
  if(fresh||!iseed)iseed=1+Math.floor(Math.random()*9999999);
  const [w,h]=$('#i-size').value.split('x').map(Number);
  $('#i-go').disabled=true;$('#i-again').disabled=true;istat('Making your picture. This can take up to 40 seconds.');
  try{
    const r=await api('/image',{prompt:p+ISTYLE[istyle],w,h,seed:iseed});
    if(!r.ok){const j=await r.json().catch(()=>({}));throw new Error(j.error||'Could not make the picture. Try again.')}
    const blob=await r.blob();
    iext=/jpe?g/.test(blob.type)?'jpg':/webp/.test(blob.type)?'webp':'png';
    if(iurl)URL.revokeObjectURL(iurl);iurl=URL.createObjectURL(blob);
    const img=$('#i-img');img.src=iurl;img.hidden=false;$('#idrop').hidden=true;$('#i-dl').disabled=false;istat('');
  }catch(e){istat(e.message)}
  $('#i-go').disabled=false;$('#i-again').disabled=false;
}
$('#i-go').onclick=()=>makeImage(false);
$('#i-again').onclick=()=>makeImage(true);
$('#i-better').onclick=async()=>{
  const t=$('#i-prompt').value.trim();if(!t){istat('Write a short idea first.');return}
  $('#i-better').disabled=true;istat('Improving your description...');
  try{$('#i-prompt').value=(await askAI('image_prompt',t)).trim();istat('')}catch(e){istat(e.message)}
  $('#i-better').disabled=false;
};
$('#i-dl').onclick=()=>{
  if(!iurl)return;const a=document.createElement('a');a.href=iurl;a.download='deep-oq-image.'+iext;
  document.body.append(a);a.click();a.remove();
};

/* ================= work ================= */
const WORK={
  email:{text:'What should the email say?',ph:'Tell Rahul the meeting moves to Friday 4 pm. Polite and short.',extra:'Tone (optional)',btn:'Write email'},
  summary:{text:'Text to summarize',ph:'Paste the text here',extra:'Length or focus (optional)',btn:'Summarize'},
  actions:{text:'Meeting notes',ph:'Paste your meeting notes here',extra:'People or deadlines to keep in mind (optional)',btn:'Find the actions'},
  report:{text:'What is the report about?',ph:'Topic, who will read it, and the main points',extra:'Length or format (optional)',btn:'Write report'},
  sql:{text:'What data do you need?',ph:'The top 5 customers by total order value in 2025',extra:'Database and tables, for example: PostgreSQL; customers(id, name), orders(id, customer_id, total, order_date)',btn:'Write SQL'},
  script:{text:'What is the video about?',ph:'A 60 second ad for a local bakery',extra:'Style, platform or length (optional)',btn:'Write script'}
};
let wkind='email';
function setWork(k){
  wkind=k;const w=WORK[k];
  $('#wtextL').textContent=w.text;$('#w-text').placeholder=w.ph;$('#wextraL').textContent=w.extra;$('#w-go').textContent=w.btn;
}
segment('#wkinds',b=>setWork(b.dataset.k));setWork('email');
$('#w-go').onclick=()=>{
  const t=$('#w-text').value.trim();
  if(!t){$('#w-out').innerHTML='<p class="note">Add some text first.</p>';$('#w-text').focus();return}
  runAI(wkind,t,$('#w-extra').value.trim(),$('#w-out'),$('#w-go'));
};

/* ================= robotics starters (in the Code section) ================= */
const ROBO=[
  ['Line follower','Arduino (C++)','A line following robot car with two IR sensors and an L298N motor driver. Include the pin definitions, sensor reading and smooth turning logic.'],
  ['Obstacle avoider','Arduino (C++)','An obstacle avoiding robot with an HC-SR04 ultrasonic sensor on a servo and an L298N motor driver. It should scan left and right and choose the clearer side.'],
  ['Servo arm','Arduino (C++)','A 3 servo robotic arm controlled by 3 potentiometers, with smooth movement and a safe angle limit on each servo.'],
  ['ROS 2 node','ROS 2 (Python)','A ROS 2 node that subscribes to /scan (LaserScan) and publishes /cmd_vel (Twist) so the robot stops when anything is closer than 0.5 m.'],
  ['Camera detection','Python','A script for a Jetson-style board that opens the camera with OpenCV, runs an object detection model, and draws boxes and labels on the live video.']
];
ROBO.forEach(r=>{
  const b=document.createElement('button');b.type='button';b.textContent=r[0];
  b.onclick=()=>{setMode('write');$('#clang').value=r[1];$('#ctask').value=r[2];$('#ctask').focus()};
  $('#crobo').append(b);
});

/* ================= data ================= */
const D={head:[],rows:[],cols:[]};let dkind='bar';
function parseCSV(text){
  const nl=text.indexOf('\n'),first=text.slice(0,nl>-1?nl:text.length);
  let delim=',',best=0;
  for(const c of [',',';','\t','|']){const n=first.split(c).length;if(n>best){best=n;delim=c}}
  const rows=[];let row=[],cur='',q=false;
  for(let i=0;i<text.length;i++){
    const ch=text[i];
    if(q){if(ch==='"'){if(text[i+1]==='"'){cur+='"';i++}else q=false}else cur+=ch}
    else if(ch==='"')q=true;
    else if(ch===delim){row.push(cur);cur=''}
    else if(ch==='\n'||ch==='\r'){
      if(ch==='\r'&&text[i+1]==='\n')i++;
      row.push(cur);cur='';
      if(row.length>1||row[0].trim()!=='')rows.push(row);
      row=[];
    }else cur+=ch;
  }
  if(cur!==''||row.length){row.push(cur);if(row.length>1||row[0].trim()!=='')rows.push(row)}
  return rows;
}
const numOf=v=>{const t=String(v==null?'':v).replace(/[,\s₹$€£%]/g,'');return t!==''&&Number.isFinite(+t)?+t:NaN};
function analyseCols(head,rows){
  return head.map((name,ci)=>{
    const vals=rows.map(r=>r[ci]).filter(v=>String(v).trim()!=='');
    const nums=vals.map(numOf).filter(Number.isFinite);
    const isNum=vals.length>0&&nums.length/vals.length>=.9;
    const c={name:name||('Column '+(ci+1)),idx:ci,filled:vals.length,type:isNum?'number':'text'};
    if(isNum){
      const s=[...nums].sort((a,b)=>a-b),n=s.length,sum=s.reduce((a,b)=>a+b,0);
      c.min=s[0];c.max=s[n-1];c.sum=sum;c.mean=sum/n;c.median=n%2?s[(n-1)/2]:(s[n/2-1]+s[n/2])/2;
    }else{
      const m=new Map();vals.forEach(v=>m.set(v,(m.get(v)||0)+1));
      c.uniq=m.size;c.top=[...m.entries()].sort((a,b)=>b[1]-a[1]).slice(0,3).map(e=>e[0]+' ('+e[1]+')');
    }
    return c;
  });
}
function colDetail(c){
  return c.type==='number'?'lowest '+fmtN(c.min)+', highest '+fmtN(c.max)+', average '+fmtN(c.mean)
    :c.uniq+' different values. Most common: '+c.top.join(', ');
}
function loadData(file){
  if(!file)return;const note=$('#dnote');note.hidden=true;
  if(file.size>8*1024*1024){note.textContent='That file is '+(file.size/1048576).toFixed(1)+' MB. The limit is 8 MB.';note.hidden=false;return}
  const fr=new FileReader();
  fr.onerror=()=>{note.textContent='This file could not be read.';note.hidden=false};
  fr.onload=()=>{
    const rows=parseCSV(String(fr.result).replace(/^﻿/,''));
    if(rows.length<2){note.textContent='No table found. Use a CSV file with a header row and at least one row of data.';note.hidden=false;return}
    const head=rows[0].map(h=>h.trim());
    D.head=head;D.rows=rows.slice(1).map(r=>head.map((_,i)=>r[i]===undefined?'':r[i]));D.cols=analyseCols(head,D.rows);
    D.name=file.name;showData();
  };
  fr.readAsText(file);
}
function showData(){
  $('#dwork').hidden=false;
  $('#dinfo').textContent=D.name+': '+D.rows.length.toLocaleString('en-IN')+' rows and '+D.head.length+' columns.';
  $('#dsum').innerHTML='<thead><tr><th>Column</th><th>Type</th><th>Filled</th><th>Details</th></tr></thead><tbody>'
    +D.cols.map(c=>'<tr><td><b>'+esc(c.name)+'</b></td><td>'+(c.type==='number'?'Number':'Text')+'</td><td>'+c.filled+' of '+D.rows.length+'</td><td>'+esc(colDetail(c))+'</td></tr>').join('')+'</tbody>';
  const all=D.cols.map(c=>'<option value="'+c.idx+'">'+esc(c.name)+'</option>').join('');
  const nums=D.cols.filter(c=>c.type==='number');
  $('#d-x').innerHTML=all;$('#d-y').innerHTML=nums.map(c=>'<option value="'+c.idx+'">'+esc(c.name)+'</option>').join('');
  const t=D.cols.find(c=>c.type==='text');$('#d-x').value=String(t?t.idx:0);
  $('#d-out').innerHTML='<p class="muted">The analysis appears here.</p>';
  setKind(dkind);
}
function setKind(k){
  dkind=k;
  document.querySelectorAll('#dkind button').forEach(b=>b.setAttribute('aria-pressed',b.dataset.k===k));
  $('#dxw').hidden=k!=='bar';$('#daggw').hidden=k!=='bar';
  drawChart();
}
function chartMsg(cv,text){const {g,w,h}=setupCanvas(cv);g.fillStyle=cssv('--ink-2');g.font='15px '+FONT;g.textAlign='center';g.fillText(text,w/2,h/2)}
function drawChart(){
  const cv=$('#dchart');if(!cv.clientWidth||!D.cols.length)return;
  const nums=D.cols.filter(c=>c.type==='number');
  const agg=$('#d-agg').value,xi=+$('#d-x').value,yv=$('#d-y').value,yi=yv===''?-1:+yv;
  if(dkind!=='bar'&&!nums.length)return chartMsg(cv,'This file has no number columns to chart.');
  if(dkind==='bar'&&agg!=='count'&&yi<0)return chartMsg(cv,'Choose "Count of rows", or a file with a number column.');
  const {g,w,h}=setupCanvas(cv),ink=cssv('--ink'),ink2=cssv('--ink-2'),line=cssv('--line'),col=cssv('--river');
  g.font='13px '+FONT;
  if(dkind==='bar'){
    const m=new Map();
    D.rows.forEach(r=>{
      const k=String(r[xi]).trim()||'(empty)',v=numOf(r[yi]),e=m.get(k)||{sum:0,n:0,rows:0};
      e.rows++;if(Number.isFinite(v)){e.sum+=v;e.n++}m.set(k,e);
    });
    let items=[...m.entries()].map(([k,e])=>[k,agg==='count'?e.rows:agg==='avg'?(e.n?e.sum/e.n:0):e.sum]);
    items.sort((a,b)=>b[1]-a[1]);items=items.slice(0,12);
    const left=Math.min(170,Math.max(60,...items.map(i=>g.measureText(i[0]).width+14))),right=70,top=8,rowH=(h-top-8)/items.length;
    const max=Math.max(...items.map(i=>Math.abs(i[1])),1e-9);
    items.forEach(([k,v],i)=>{
      const y=top+i*rowH,bh=Math.min(26,rowH*.66),bw=Math.max(2,(w-left-right)*Math.abs(v)/max);
      g.fillStyle=ink;g.textAlign='right';g.textBaseline='middle';
      let label=k;while(label.length>3&&g.measureText(label).width>left-12)label=label.slice(0,-2);if(label!==k)label+='…';
      g.fillText(label,left-10,y+rowH/2);
      g.fillStyle=col;g.fillRect(left,y+(rowH-bh)/2,bw,bh);
      g.fillStyle=ink2;g.textAlign='left';g.fillText(fmtN(v),left+bw+8,y+rowH/2);
    });
    return;
  }
  const c=nums.find(c=>c.idx===yi)||nums[0],vals=D.rows.map(r=>numOf(r[c.idx])).filter(Number.isFinite);
  if(!vals.length)return chartMsg(cv,'No numbers to draw.');
  const pad={l:46,r:14,t:16,b:30};
  if(dkind==='hist'){
    const bins=10,lo=Math.min(...vals),hi=Math.max(...vals),span=(hi-lo)||1,counts=new Array(bins).fill(0);
    vals.forEach(v=>counts[Math.min(bins-1,Math.floor((v-lo)/span*bins))]++);
    const mx=Math.max(...counts),bw=(w-pad.l-pad.r)/bins;
    counts.forEach((n,i)=>{
      const bh=(h-pad.t-pad.b)*n/mx,x=pad.l+i*bw;
      g.fillStyle=col;g.fillRect(x+2,h-pad.b-bh,bw-4,bh);
      g.fillStyle=ink2;g.textAlign='center';g.fillText(String(n),x+bw/2,h-pad.b-bh-8);
    });
    g.fillStyle=ink2;g.textAlign='left';g.fillText(fmtN(lo),pad.l,h-8);g.textAlign='right';g.fillText(fmtN(hi),w-pad.r,h-8);
    g.textAlign='center';g.fillText(c.name,w/2,h-8);
    return;
  }
  const step=Math.max(1,Math.ceil(vals.length/400)),pts=vals.filter((_,i)=>i%step===0);
  const lo=Math.min(...pts),hi=Math.max(...pts),span=(hi-lo)||1;
  g.strokeStyle=line;g.lineWidth=1;g.beginPath();g.moveTo(pad.l,h-pad.b);g.lineTo(w-pad.r,h-pad.b);g.stroke();
  g.strokeStyle=col;g.lineWidth=2;g.beginPath();
  pts.forEach((v,i)=>{const x=pad.l+(w-pad.l-pad.r)*(pts.length>1?i/(pts.length-1):.5),y=h-pad.b-(h-pad.t-pad.b)*(v-lo)/span;i?g.lineTo(x,y):g.moveTo(x,y)});
  g.stroke();
  g.fillStyle=ink2;g.textAlign='right';g.fillText(fmtN(hi),pad.l-6,pad.t+4);g.fillText(fmtN(lo),pad.l-6,h-pad.b);
  g.textAlign='left';g.fillText('first row',pad.l,h-8);g.textAlign='right';g.fillText('last row',w-pad.r,h-8);
  g.textAlign='center';g.fillText(c.name,w/2,h-8);
}
function dataSummary(){
  let s='File: '+D.name+'. Rows: '+D.rows.length+'. Columns: '+D.head.length+'.\n';
  D.cols.forEach(c=>{s+='- '+c.name+' ('+(c.type==='number'?'number':'text')+', '+c.filled+' filled): '+colDetail(c)+(c.type==='number'?', total '+fmtN(c.sum)+', median '+fmtN(c.median):'')+'\n'});
  s+='First rows:\n'+D.rows.slice(0,5).map(r=>r.map(x=>String(x).slice(0,30)).join(' ; ')).join('\n');
  return s.slice(0,6000);
}
segment('#dkind',b=>setKind(b.dataset.k));
['#d-x','#d-y','#d-agg'].forEach(s=>$(s).addEventListener('change',drawChart));
$('#dfile').onchange=e=>{loadData(e.target.files[0]);e.target.value=''};
dropZone($('#dstage'),loadData);
$('#d-go').onclick=()=>{
  const q=$('#d-q').value.trim();
  runAI('data',dataSummary()+(q?'\n\nQUESTION: '+q:'\n\nNo question. Give an overview.'),'',$('#d-out'),$('#d-go'));
};
let rz=0;window.addEventListener('resize',()=>{clearTimeout(rz);rz=setTimeout(()=>{if($('#v-data').classList.contains('on'))drawChart();if($('#v-music').classList.contains('on'))drawRoll(-1)},150)});

/* ================= music ================= */
const KEYS=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const MOOD={
  happy:{scale:[0,2,4,5,7,9,11],progs:[[0,4,5,3],[0,3,4,0],[0,5,3,4]],gate:.92},
  calm:{scale:[0,2,4,5,7,9,11],progs:[[0,5,3,4],[0,3,0,4]],gate:1},
  sad:{scale:[0,2,3,5,7,8,10],progs:[[0,5,2,6],[0,3,4,0],[0,6,5,4]],gate:1},
  epic:{scale:[0,2,3,5,7,8,10],progs:[[0,5,2,6],[0,6,2,4],[0,3,5,4]],gate:.95},
  mysterious:{scale:[0,1,3,5,7,8,10],progs:[[0,1,0,6],[0,5,1,0]],gate:1},
  playful:{scale:[0,2,4,5,7,9,11],progs:[[0,3,0,4],[0,4,3,4]],gate:.55}
};
const RHYTHMS=[[1,1,1,1],[1,.5,.5,1,1],[.5,.5,1,.5,.5,1],[2,1,1],[1.5,.5,1,1],[1,1,2],[.5,.5,.5,.5,1,1],[1,.5,.5,2],[.5,.5,1,2]];
function rng(seed){let a=seed>>>0;return()=>{a=(a+0x6D2B79F5)|0;let t=Math.imul(a^(a>>>15),1|a);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296}}

/* Builds the whole piece from a seed. Same seed and settings always give the same music. */
function composeSong(o){
  const R=rng(o.seed),M=MOOD[o.mood],base=60+KEYS.indexOf(o.key);
  const deg=i=>{const oc=Math.floor(i/7),r=((i%7)+7)%7;return base+12*oc+M.scale[r]};
  const pick=a=>a[Math.floor(R()*a.length)];
  const prog=pick(M.progs),ev=[];
  const block=[];let cur=7+pick([0,2,4]);
  for(let b=0;b<4;b++){
    const ch=prog[b],tones=[ch,ch+2,ch+4].flatMap(d=>[d,d+7,d-7]).filter(i=>i>=4&&i<=13);
    const pat=pick(RHYTHMS);let t=0;
    for(const dur of pat){
      const strong=t===0||t===2;
      if(strong){cur=tones.reduce((best,i)=>Math.abs(i-cur)<Math.abs(best-cur)?i:best,tones[0])}
      else{cur=Math.max(4,Math.min(13,cur+pick([-2,-1,-1,0,1,1,2])))}
      if(strong||R()>.12)block.push({t:b*4+t,d:dur*M.gate,i:cur,v:strong?92:78});
      t+=dur;
    }
  }
  const blocks=o.bars/4;
  for(let k=0;k<blocks;k++){
    block.forEach(n=>{
      let i=n.i;if(k>0&&R()<.2)i=Math.max(4,Math.min(13,i+pick([-1,1])));
      ev.push({tr:'mel',t:n.t+16*k,d:n.d,n:deg(i),v:n.v});
    });
  }
  const last=(o.bars-1)*4;
  for(let j=ev.length-1;j>=0;j--)if(ev[j].tr==='mel'&&ev[j].t>=last)ev.splice(j,1);
  ev.push({tr:'mel',t:last,d:4,n:deg(7),v:88});
  for(let b=0;b<o.bars;b++){
    const ch=b===o.bars-1?0:prog[b%4];
    [ch-7,ch-5,ch-3].forEach(i=>ev.push({tr:'chord',t:b*4,d:3.95,n:deg(i),v:50}));
    if(b===o.bars-1)ev.push({tr:'bass',t:b*4,d:4,n:deg(ch-14),v:82});
    else{ev.push({tr:'bass',t:b*4,d:1.9,n:deg(ch-14),v:84});ev.push({tr:'bass',t:b*4+2,d:1.9,n:deg(ch-14),v:76})}
    if(o.drums){
      if(b===o.bars-1)ev.push({tr:'drums',t:b*4,d:.2,n:36,v:100});
      else for(let q=0;q<8;q++){
        const t=b*4+q*.5;ev.push({tr:'drums',t,d:.1,n:42,v:q%2?38:55});
        if(q===0||q===4)ev.push({tr:'drums',t,d:.2,n:36,v:100});
        if(q===2||q===6)ev.push({tr:'drums',t,d:.2,n:38,v:88});
      }
    }
  }
  return {events:ev,bpm:o.bpm,bars:o.bars,total:o.bars*4,mood:o.mood,key:o.key};
}

/* ---- sound ---- */
const freqOf=n=>440*Math.pow(2,(n-69)/12);
function voice(ac,dest,n,t0,dur,vel,type){
  const f=freqOf(n),g=ac.createGain(),o=ac.createOscillator(),peak=(vel/127)*.32;
  o.type=type==='chip'?'square':type==='strings'?'sawtooth':type==='pad'?'sine':'triangle';
  o.frequency.value=f;o.connect(g);
  if(type==='piano'){const o2=ac.createOscillator(),g2=ac.createGain();o2.type='sine';o2.frequency.value=f*2;g2.gain.value=.22;o2.connect(g2).connect(g);o2.start(t0);o2.stop(t0+dur+.4)}
  let out=g;
  if(type==='strings'||type==='chip'){const lp=ac.createBiquadFilter();lp.type='lowpass';lp.frequency.value=type==='chip'?3200:1800;g.connect(lp);out=lp}
  out.connect(dest);
  const att=(type==='pad'||type==='strings')?.06:.008;
  g.gain.setValueAtTime(.0001,t0);g.gain.exponentialRampToValueAtTime(peak,t0+att);
  if(type==='piano'||type==='chip')g.gain.exponentialRampToValueAtTime(peak*.4,t0+Math.max(.12,dur*.8));
  g.gain.setTargetAtTime(.0001,t0+dur,.05);
  o.start(t0);o.stop(t0+dur+.4);
}
function drumHit(ac,dest,n,t0,vel){
  const v=vel/127;
  if(n===36){
    const o=ac.createOscillator(),g=ac.createGain();
    o.frequency.setValueAtTime(150,t0);o.frequency.exponentialRampToValueAtTime(45,t0+.12);
    g.gain.setValueAtTime(.9*v,t0);g.gain.exponentialRampToValueAtTime(.001,t0+.22);
    o.connect(g).connect(dest);o.start(t0);o.stop(t0+.25);return;
  }
  const len=n===38?.18:.05,buf=ac.createBuffer(1,Math.floor(ac.sampleRate*len),ac.sampleRate),d=buf.getChannelData(0);
  for(let i=0;i<d.length;i++)d[i]=Math.random()*2-1;
  const s=ac.createBufferSource(),hp=ac.createBiquadFilter(),g=ac.createGain();
  s.buffer=buf;hp.type='highpass';hp.frequency.value=n===38?1500:7000;
  g.gain.setValueAtTime((n===38?.5:.22)*v,t0);g.gain.exponentialRampToValueAtTime(.001,t0+len);
  s.connect(hp).connect(g).connect(dest);s.start(t0);
}

/* ---- state, controls, playback ---- */
let song=null,mAC=null,mRAF=0,mStart=0,mSeed=Math.floor(Math.random()*1e9);
const mset=m=>$('#m-status').textContent=m;
function mOptions(){
  return {mood:$('#m-mood').value,key:$('#m-key').value,bpm:+$('#m-tempo').value,bars:+$('#m-bars').value,drums:$('#m-drums').checked,seed:mSeed};
}
function rebuild(){stopMusic();song=composeSong(mOptions());drawRoll(-1);mset(song.bars+' bars in '+song.key+', '+song.mood+', '+song.bpm+' beats per minute.')}
function drawRoll(playBeat){
  const cv=$('#m-roll');if(!cv.clientWidth||!song)return;
  const {g,w,h}=setupCanvas(cv),pads={t:6,b:6},colors={mel:cssv('--accent'),chord:cssv('--river'),bass:cssv('--ink-2')};
  const mel=song.events.filter(e=>e.tr!=='drums'),lo=Math.min(...mel.map(e=>e.n))-1,hi=Math.max(...mel.map(e=>e.n))+1;
  const ph=h-pads.t-pads.b-14,py=n=>pads.t+ph*(1-(n-lo)/(hi-lo)),bh=Math.max(3,ph/(hi-lo)*.9);
  ['chord','bass','mel'].forEach(tr=>{g.fillStyle=colors[tr];song.events.filter(e=>e.tr===tr).forEach(e=>g.fillRect(e.t/song.total*w,py(e.n)-bh/2,Math.max(2,e.d/song.total*w-1),bh))});
  g.fillStyle=cssv('--ink-2');song.events.filter(e=>e.tr==='drums'&&e.n!==42).forEach(e=>g.fillRect(e.t/song.total*w,h-12,2,8));
  if(playBeat>=0){g.fillStyle=cssv('--ink');g.fillRect(Math.min(w-2,playBeat/song.total*w),0,2,h)}
}
function stopMusic(){
  cancelAnimationFrame(mRAF);
  if(mAC){try{mAC.close()}catch(e){}mAC=null}
  $('#m-play').textContent='Play';if(song)drawRoll(-1);
}
function playMusic(){
  stopMusic();if(!song)return;
  const AC=window.AudioContext||window.webkitAudioContext;if(!AC){mset('This browser cannot play sound. Download the MIDI file instead.');return}
  const ac=new AC();mAC=ac;
  const comp=ac.createDynamicsCompressor(),master=ac.createGain();master.gain.value=.85;comp.connect(master).connect(ac.destination);
  const snd=$('#m-sound').value,spb=60/song.bpm;mStart=ac.currentTime+.15;
  song.events.forEach(e=>{
    const t=mStart+e.t*spb,d=Math.max(.05,e.d*spb);
    if(e.tr==='drums')drumHit(ac,comp,e.n,t,e.v);else voice(ac,comp,e.n,t,d,e.v,e.tr==='mel'?snd:'pad');
  });
  $('#m-play').textContent='Stop';mset('Playing...');
  const total=song.total*spb;
  (function tick(){
    if(mAC!==ac)return;
    const el=ac.currentTime-mStart;
    if(el>total+.6){stopMusic();mset('Finished. Press Play to hear it again.');return}
    drawRoll(Math.max(0,el/spb));mRAF=requestAnimationFrame(tick);
  })();
}

/* ---- MIDI file ---- */
function vlq(n){const out=[n&127];while((n>>=7)>0)out.unshift((n&127)|128);return out}
function chunk(tag,bytes){const l=bytes.length;return [...tag].map(c=>c.charCodeAt(0)).concat([l>>>24&255,l>>>16&255,l>>>8&255,l&255],bytes)}
function trackBytes(list){
  list.sort((a,b)=>a.tick-b.tick||a.order-b.order);
  const out=[];let last=0;
  for(const e of list){out.push(...vlq(e.tick-last),...e.data);last=e.tick}
  out.push(0,0xFF,0x2F,0);return out;
}
function toMidi(sg,sound){
  const TPQ=480,prog={piano:0,pad:89,chip:80,strings:48},tracks=[];
  const us=Math.round(60000000/sg.bpm);
  tracks.push(trackBytes([{tick:0,order:-2,data:[0xFF,0x51,3,us>>>16&255,us>>>8&255,us&255]},{tick:0,order:-2,data:[0xFF,0x58,4,4,2,24,8]}]));
  [['mel',0,prog[sound]],['chord',1,89],['bass',2,33],['drums',9,null]].forEach(([tr,ch,pg])=>{
    const l=[];if(pg!==null)l.push({tick:0,order:-1,data:[0xC0|ch,pg]});
    sg.events.filter(e=>e.tr===tr).forEach(e=>{
      const a=Math.round(e.t*TPQ),b=a+Math.max(1,Math.round(e.d*TPQ));
      l.push({tick:a,order:1,data:[0x90|ch,e.n,e.v]},{tick:b,order:0,data:[0x80|ch,e.n,0]});
    });
    tracks.push(trackBytes(l));
  });
  const head=chunk('MThd',[0,1,0,tracks.length,TPQ>>8,TPQ&255]);
  return new Uint8Array(head.concat(...tracks.map(t=>chunk('MTrk',t))));
}

['m-mood','m-key','m-bars','m-drums'].forEach(id=>$('#'+id).addEventListener('change',rebuild));
$('#m-tempo').addEventListener('input',e=>{$('#m-tempoo').textContent=e.target.value+' bpm';rebuild()});
$('#m-sound').addEventListener('change',()=>{if(mAC)playMusic()});
$('#m-new').onclick=()=>{mSeed=Math.floor(Math.random()*1e9);rebuild();playMusic()};
$('#m-play').onclick=()=>{if(mAC){stopMusic();mset('Stopped.')}else playMusic()};
$('#m-dl').onclick=()=>{
  if(!song)return;
  const a=document.createElement('a'),url=URL.createObjectURL(new Blob([toMidi(song,$('#m-sound').value)],{type:'audio/midi'}));
  a.href=url;a.download='deep-oq-music.mid';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),4000);
};
$('#m-fromdesc').onclick=async()=>{
  const t=$('#m-desc').value.trim();if(!t){mset('Describe the music first.');return}
  $('#m-fromdesc').disabled=true;mset('Reading your description...');
  try{
    const a=await askAI('music_params',t),m=a.match(/\{[\s\S]*\}/);if(!m)throw new Error('Could not understand the reply. Try different words.');
    const p=JSON.parse(m[0]);
    if(MOOD[p.mood])$('#m-mood').value=p.mood;
    if(KEYS.includes(p.key))$('#m-key').value=p.key;
    if(Number.isFinite(+p.tempo)){const v=Math.max(60,Math.min(180,Math.round(+p.tempo)));$('#m-tempo').value=v;$('#m-tempoo').textContent=v+' bpm'}
    if([8,16,32].includes(+p.bars))$('#m-bars').value=String(+p.bars);
    if(['piano','pad','chip','strings'].includes(p.sound))$('#m-sound').value=p.sound;
    if(typeof p.drums==='boolean')$('#m-drums').checked=p.drums;
    mSeed=Math.floor(Math.random()*1e9);rebuild();playMusic();
  }catch(e){mset(e.message||'Could not use that description.')}
  $('#m-fromdesc').disabled=false;
};
rebuild();

/* ================= farm ================= */
/* Crop factors (Kc): start, middle and end of season, typical values from FAO guidance. */
const CROPS={Wheat:[.3,1.15,.4],Rice:[1.05,1.2,.9],Maize:[.3,1.2,.6],Sugarcane:[.4,1.25,.75],Cotton:[.35,1.15,.7],
  Soybean:[.4,1.15,.5],Groundnut:[.4,1.15,.6],Mustard:[.35,1.15,.35],Chickpea:[.4,1,.35],Potato:[.5,1.15,.75],
  Tomato:[.6,1.15,.8],Onion:[.7,1.05,.75],Banana:[.5,1.1,1]};
$('#f-crop').innerHTML=Object.keys(CROPS).map(c=>'<option>'+c+'</option>').join('');
function farmWater(){
  const c=CROPS[$('#f-crop').value],st=$('#f-stage').value;
  const kc=st==='ini'?c[0]:st==='dev'?(c[0]+c[1])/2:st==='mid'?c[1]:c[2];
  const et0=+$('#f-clim').value,rain=Math.max(0,+$('#f-rain').value||0),eff=+$('#f-eff').value;
  const area=Math.max(0,+$('#f-area').value||0)*(+$('#f-unit').value);
  const need=Math.max(0,kc*et0-rain),litres=need*area/eff;
  $('#f-wout').innerHTML='<div><div class="big">'+fmtN(litres,0)+' litres</div><p class="sub2">of water to apply per day for the whole field</p></div>'
    +'<dl class="kv"><dt>Crop factor (Kc)</dt><dd>'+fmtN(kc)+'</dd>'
    +'<dt>Crop water need</dt><dd>'+fmtN(need)+' mm per day</dd>'
    +'<dt>Field area</dt><dd>'+fmtN(area,0)+' m²</dd>'
    +'<dt>Per week</dt><dd>'+fmtN(litres*7,0)+' litres</dd>'
    +'<dt>Per hectare per day</dt><dd>'+fmtN(need*10000/eff,0)+' litres</dd></dl>'
    +'<p class="sub2">Water need = crop factor × weather demand − useful rain, divided by how well your system delivers water. Soil and local advice can change this.</p>';
}
['f-crop','f-stage','f-clim','f-eff','f-area','f-unit','f-rain'].forEach(id=>$('#'+id).addEventListener('input',farmWater));
farmWater();
segment('#fmodes',b=>{
  const m=b.dataset.m;
  $('#f-water').hidden=m!=='water';$('#f-ai').hidden=m==='water';
  $('#f-planform').hidden=m!=='plan';$('#f-probform').hidden=m!=='problem';
  $('#f-out').innerHTML='<p class="muted">Your result appears here.</p>';
});
$('#fp-go').onclick=()=>{
  const crop=$('#fp-crop').value.trim();
  if(!crop){$('#f-out').innerHTML='<p class="note">Enter the crop first.</p>';$('#fp-crop').focus();return}
  runAI('farm_plan','Crop: '+crop+'\nPlace: '+($('#fp-place').value.trim()||'not given')+'\nSeason: '+$('#fp-season').value
    +'\nSoil: '+($('#fp-soil').value.trim()||'not given')+'\nField size: '+($('#fp-area').value.trim()||'not given'),'',$('#f-out'),$('#fp-go'));
};
$('#fq-go').onclick=()=>{
  const crop=$('#fq-crop').value.trim(),what=$('#fq-what').value.trim();
  if(!crop||!what){$('#f-out').innerHTML='<p class="note">Enter the crop and what you see.</p>';return}
  runAI('farm_problem','Crop: '+crop+'\nWhat the farmer sees: '+what+'\nRecent weather and care: '+($('#fq-ctx').value.trim()||'not given'),'',$('#f-out'),$('#fq-go'));
};

/* ================= planet ================= */
segment('#pmodes',b=>{
  const m=b.dataset.m;
  $('#pl-warn').hidden=m!=='warn';$('#pl-energy').hidden=m!=='energy';$('#pl-carbon').hidden=m!=='carbon';
});
const WCODE={0:'Clear',1:'Mostly clear',2:'Partly cloudy',3:'Cloudy',45:'Fog',48:'Fog',51:'Light drizzle',53:'Drizzle',55:'Heavy drizzle',
  56:'Freezing drizzle',57:'Freezing drizzle',61:'Light rain',63:'Rain',65:'Heavy rain',66:'Freezing rain',67:'Freezing rain',71:'Light snow',
  73:'Snow',75:'Heavy snow',77:'Snow grains',80:'Rain showers',81:'Rain showers',82:'Violent showers',85:'Snow showers',86:'Snow showers',
  95:'Thunderstorm',96:'Thunderstorm with hail',99:'Thunderstorm with hail'};
function dayAlerts(d){
  const a=[],add=(level,text)=>a.push({level,text});
  if(d.max>=45)add('severe','Extreme heat, up to '+Math.round(d.max)+'°C');else if(d.max>=40)add('warn','Heatwave conditions, up to '+Math.round(d.max)+'°C');
  if(d.min<=0)add('warn','Frost risk, as low as '+Math.round(d.min)+'°C');
  if(d.rain>=100)add('severe','Extremely heavy rain, about '+Math.round(d.rain)+' mm');else if(d.rain>=50)add('warn','Heavy rain, about '+Math.round(d.rain)+' mm');
  if(d.wind>=80)add('severe','Damaging winds, up to '+Math.round(d.wind)+' km/h');else if(d.wind>=50)add('warn','Strong winds, up to '+Math.round(d.wind)+' km/h');
  if(d.uv>=11)add('severe','Extreme UV index '+Math.round(d.uv));else if(d.uv>=8)add('warn','Very high UV index '+Math.round(d.uv));
  if(d.code>=95)add('warn',WCODE[d.code]||'Thunderstorm');
  return a;
}
let pwText='';
const pwnote=m=>{const n=$('#pw-note');n.textContent=m;n.hidden=!m};
async function pickPlace(p){
  pwnote('');$('#pw-result').hidden=true;
  const name=[p.name,p.admin1,p.country].filter(Boolean).join(', ');
  try{
    const r=await fetch('https://api.open-meteo.com/v1/forecast?latitude='+p.latitude+'&longitude='+p.longitude
      +'&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,uv_index_max&timezone=auto&forecast_days=7');
    if(!r.ok)throw new Error('bad');
    const j=(await r.json()).daily,days=j.time.map((t,i)=>({date:t,max:j.temperature_2m_max[i],min:j.temperature_2m_min[i],
      rain:j.precipitation_sum[i]||0,wind:j.wind_speed_10m_max[i]||0,uv:j.uv_index_max[i]||0,code:j.weather_code[i]}));
    const label=d=>new Date(d+'T12:00:00').toLocaleDateString('en-IN',{weekday:'short',day:'numeric',month:'short'});
    const lines=[],html=[];
    days.forEach(d=>dayAlerts(d).forEach(x=>{lines.push(label(d.date)+': '+x.text);html.push('<div class="alert '+x.level+'"><b>'+esc(label(d.date))+'</b> '+esc(x.text)+'</div>')}));
    $('#pw-title').textContent=name;
    $('#pw-alerts').innerHTML=html.length?html.join(''):'<div class="alert ok">No weather warnings for the next 7 days.</div>';
    $('#pw-days').innerHTML=days.map(d=>'<div class="day"><b>'+esc(label(d.date))+'</b>'+esc(WCODE[d.code]||'')+'<br>'+Math.round(d.max)+'° / '+Math.round(d.min)+'°<br>Rain '+fmtN(d.rain,1)+' mm<br>Wind '+Math.round(d.wind)+' km/h<br>UV '+Math.round(d.uv)+'</div>').join('');
    pwText='Place: '+name+'\nWarnings in the next 7 days:\n'+(lines.join('\n')||'None');
    $('#pw-out').innerHTML='<p class="muted">Advice appears here.</p>';$('#pw-result').hidden=false;
  }catch(e){pwnote('Could not load the forecast. Check your internet and try again.')}
}
$('#pw-form').onsubmit=async e=>{
  e.preventDefault();const q=$('#pw-city').value.trim();if(!q)return;
  pwnote('');$('#pw-places').hidden=true;
  try{
    const r=await fetch('https://geocoding-api.open-meteo.com/v1/search?name='+encodeURIComponent(q)+'&count=5&language=en&format=json');
    const list=(await r.json()).results||[];
    if(!list.length){pwnote('No place found with that name. Try a nearby city.');$('#pw-result').hidden=true;return}
    const box=$('#pw-places');box.innerHTML='';
    if(list.length>1){
      list.forEach(p=>{const b=document.createElement('button');b.type='button';b.textContent=[p.name,p.admin1,p.country].filter(Boolean).join(', ');b.onclick=()=>pickPlace(p);box.append(b)});
      box.hidden=false;
    }
    pickPlace(list[0]);
  }catch(err){pwnote('Could not reach the weather data. Check your internet and try again.')}
};
$('#pw-explain').onclick=()=>runAI('planet_warning',pwText,'',$('#pw-out'),$('#pw-explain'));

/* ---- home energy and solar ---- */
const APPS=[['LED bulbs',9,6,5],['Ceiling fans',70,3,10],['Refrigerator',150,1,8],['Television',100,1,4],['Air conditioner',1500,1,6],['Laptop',60,1,6]];
function addApp(a){
  const r=document.createElement('div');r.className='approw';
  r.innerHTML='<input class="in" type="text" aria-label="Appliance" value="'+esc(a[0])+'"><input class="in" type="number" min="0" aria-label="Watts" value="'+a[1]+'">'
    +'<input class="in" type="number" min="0" aria-label="Quantity" value="'+a[2]+'"><input class="in" type="number" min="0" max="24" step="0.5" aria-label="Hours per day" value="'+a[3]+'">'
    +'<button class="x" type="button" aria-label="Remove appliance">×</button>';
  $('#e-apps').append(r);
}
APPS.forEach(addApp);
let eText='';
function energyCalc(){
  let daily=0;
  document.querySelectorAll('#e-apps .approw').forEach(r=>{const i=r.querySelectorAll('input');daily+=(+i[1].value||0)*(+i[2].value||0)*(+i[3].value||0)/1000});
  const tariff=+$('#e-tariff').value||0,sun=Math.max(1,+$('#e-sun').value||5),pw=Math.max(100,+$('#e-panel').value||400),cpk=+$('#e-cost').value||0;
  const monthly=daily*30,bill=monthly*tariff,needKw=daily/(sun*.75),panels=daily>0?Math.ceil(needKw*1000/pw):0,kw=panels*pw/1000;
  const gen=kw*sun*.75*365,use=daily*365,saved=Math.min(gen,use)*tariff,cost=kw*cpk,payback=saved>0?cost/saved:null;
  $('#e-out').innerHTML='<div><div class="big">'+fmtN(daily)+' kWh</div><p class="sub2">used per day</p></div>'
    +'<dl class="kv"><dt>Per month</dt><dd>'+fmtN(monthly,0)+' kWh</dd><dt>Monthly bill</dt><dd>₹'+fmtN(bill,0)+'</dd>'
    +'<dt>Solar system needed</dt><dd>'+panels+' panels ('+fmtN(kw)+' kW)</dd><dt>Roof space</dt><dd>about '+fmtN(panels*2,0)+' m²</dd>'
    +'<dt>Power made per year</dt><dd>'+fmtN(gen,0)+' kWh</dd><dt>Saving per year</dt><dd>₹'+fmtN(saved,0)+'</dd>'
    +'<dt>Estimated cost</dt><dd>₹'+fmtN(cost,0)+'</dd><dt>Pays back in</dt><dd>'+(payback?fmtN(payback,1)+' years':'-')+'</dd></dl>';
  const rows=[];document.querySelectorAll('#e-apps .approw').forEach(r=>{const i=r.querySelectorAll('input');rows.push(i[0].value+': '+i[1].value+' W x '+i[2].value+', '+i[3].value+' hours a day')});
  eText='Appliances:\n'+rows.join('\n')+'\nUse per day: '+fmtN(daily)+' kWh. Monthly bill: Rs '+fmtN(bill,0)+' at Rs '+tariff+' per kWh.\nSolar needed: '+panels+' panels of '+pw+' W ('+fmtN(kw)+' kW) with '+sun+' sunny hours a day. Cost about Rs '+fmtN(cost,0)+', saving Rs '+fmtN(saved,0)+' a year.';
}
$('#pl-energy').addEventListener('input',energyCalc);
$('#e-apps').addEventListener('click',e=>{if(e.target.closest('.x')){e.target.closest('.approw').remove();energyCalc()}});
$('#e-add').onclick=()=>{addApp(['New appliance',100,1,2]);energyCalc()};
$('#e-ai').onclick=()=>{energyCalc();runAI('planet_energy',eText,'',$('#e-aiout'),$('#e-ai'))};
energyCalc();

/* ---- carbon footprint (rough yearly estimate, kg CO2e) ---- */
let cText='';
function carbonCalc(){
  const kwh=+$('#c-kwh').value||0,km=+$('#c-km').value||0,fuel=+$('#c-fuel').value,mil=Math.max(1,+$('#c-mil').value||15),lpg=+$('#c-lpg').value||0,fly=+$('#c-fly').value||0;
  const parts=[['Electricity',kwh*12*.82],['Vehicle',fuel?(km*12/mil)*fuel:0],['Cooking gas',lpg*14.2*3],['Flights',fly*120]];
  const total=parts.reduce((a,p)=>a+p[1],0),mx=Math.max(...parts.map(p=>p[1]),1);
  $('#c-out').innerHTML='<div><div class="big">'+fmtN(total/1000,1)+' tonnes</div><p class="sub2">of CO₂ per year. Roughly 2 tonnes is the average in India and 4.7 tonnes worldwide.</p></div>'
    +'<div class="bars">'+parts.map(p=>'<div class="bar"><span>'+p[0]+'</span><i style="width:'+Math.round(p[1]/mx*100)+'%"></i><span>'+fmtN(p[1]/1000,2)+' t</span></div>').join('')+'</div>';
  cText='Yearly carbon footprint: '+fmtN(total/1000,1)+' tonnes CO2.\n'+parts.map(p=>p[0]+': '+fmtN(p[1]/1000,2)+' tonnes').join('\n');
}
$('#pl-carbon').addEventListener('input',carbonCalc);
$('#c-ai').onclick=()=>{carbonCalc();runAI('planet_carbon',cText,'',$('#c-aiout'),$('#c-ai'))};
carbonCalc();

showView(location.hash.slice(1));
</script>
</body>
</html>
"""


# ==========================================================================
# MORE ROUTES AND START
# ==========================================================================
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
