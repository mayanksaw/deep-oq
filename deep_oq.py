"""
Deep OQ v2 - Groq-powered AI with a large searchable knowledge database.

Per question:
  RETRIEVE -> search the database for relevant knowledge
  PLAN     -> break hard questions into steps
  SOLVE    -> answer using the plan + retrieved knowledge
  VERIFY   -> second pass checks and fixes the answer
Chats are saved in the database, so Deep OQ keeps conversation memory per session.

Run:
  pip install -r requirements.txt
  set GROQ_API_KEY=your_key        (Windows)  /  export GROQ_API_KEY=your_key
  python ingest.py my_knowledge_folder     (optional: fill the database)
  python deep_oq.py                        -> http://localhost:5000
"""
import os
import uuid
from flask import Flask, request, jsonify
from groq import Groq
import datetime
import kb
import tools

# Two-tier models: FAST for simple chat, SMART (120B reasoning model) for hard questions.
FAST_MODEL = os.environ.get("DEEP_OQ_MODEL", "llama-3.3-70b-versatile")
SMART_MODEL = os.environ.get("DEEP_OQ_SMART_MODEL", "openai/gpt-oss-120b")
client = Groq(api_key=os.environ["GROQ_API_KEY"])
ACCESS_KEY = os.environ.get("ACCESS_KEY", "")  # set on Render to lock the site
app = Flask(__name__)

SYSTEM = (
    "You are Deep OQ, an advanced AI assistant created by Mayank Kumar Saw. "
    "You are accurate, clear and honest. You write clean, working code, solve "
    "hard reasoning and math problems step by step, and say so when you are "
    "unsure. Reply in the language the user writes in. Do not add a Sources line. "
    "When KNOWLEDGE is provided, use it if relevant and prefer it over guessing; "
    "if it is not relevant, ignore it."
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
    hits = kb.search(question, k=6)
    if not hits:
        return ""
    parts = [f"[{i+1}] ({h['title']}) {h['content']}" for i, h in enumerate(hits)]
    return "KNOWLEDGE:\n" + "\n\n".join(parts)


def deep_oq(session, question):
    history = kb.load_history(session, 12)
    know = knowledge_block(question)
    tool_text, used = tools.run_tools(question, lambda m, t, k: chat(m, t, k))
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


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "page.html"),
          encoding="utf-8") as _f:
    PAGE = _f.read()


@app.get("/")
def index():
    return PAGE, 200, {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-cache"}


@app.get("/stats")
def stats():
    return jsonify(kb.stats())


def denied():
    return ACCESS_KEY and request.headers.get("X-Access-Key", "") != ACCESS_KEY


@app.post("/teach")
def teach():
    if denied():
        return jsonify(error="locked"), 401
    t = (request.get_json() or {}).get("text", "")
    n = kb.add_text(t, title="pasted note", source="pasted-" + uuid.uuid4().hex[:8])
    return jsonify(chunks_added=n)


@app.post("/ask")
def ask():
    if denied():
        return jsonify(error="locked"), 401
    data = request.get_json()
    session = data.get("session") or "default"
    try:
        answer = deep_oq(session, data["q"])
        kb.save_msg(session, "user", data["q"])
        kb.save_msg(session, "assistant", answer)
    except Exception as e:
        print("ask failed:", e)
        return jsonify(error=str(e)[:300]), 500
    return jsonify(answer=answer)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
