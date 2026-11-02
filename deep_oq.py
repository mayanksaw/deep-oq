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
from flask import Flask, request, jsonify, render_template_string
from groq import Groq
import kb

MODEL = os.environ.get("DEEP_OQ_MODEL", "llama-3.3-70b-versatile")
client = Groq(api_key=os.environ["GROQ_API_KEY"])
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


def chat(messages, temperature=0.3, max_tokens=2048):
    r = client.chat.completions.create(
        model=MODEL, messages=messages,
        temperature=temperature, max_tokens=max_tokens)
    return r.choices[0].message.content.strip()


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
    base = [{"role": "system", "content": SYSTEM}] + history
    user = (know + "\n\n" if know else "") + f"QUESTION:\n{question}"

    if not is_hard(question):
        return chat(base + [{"role": "user", "content": user}], 0.5)

    plan = chat(base + [{"role": "user", "content":
        f"{user}\n\nWrite a short numbered plan to answer this correctly. "
        "Do not answer yet."}], 0.2, 600)

    draft = chat(base + [{"role": "user", "content":
        f"{user}\n\nPLAN:\n{plan}\n\nFollow the plan and give the full answer. "
        "Do arithmetic and logic one step at a time."}], 0.2)

    return chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content":
        f"{user}\n\nDRAFT ANSWER:\n{draft}\n\nCheck the draft for mistakes (math, "
        "logic, code bugs, missed requirements). Output ONLY the corrected final "
        "answer for the user, with no mention of this review."}], 0.1)


PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Deep OQ</title>
<style>
body{margin:0;font:16px system-ui;background:#0f1115;color:#e8e8e8;display:flex;flex-direction:column;height:100vh}
header{display:flex;justify-content:space-between;align-items:center;padding:14px 18px;border-bottom:1px solid #222}
h1{margin:0;font-size:18px}#st{font-size:12px;color:#8a93a3}
#log{flex:1;overflow:auto;padding:18px}
.m{max-width:760px;margin:0 auto 14px;padding:12px 14px;border-radius:10px;white-space:pre-wrap;line-height:1.5}
.u{background:#1c2430}.a{background:#171a21}
form{display:flex;gap:8px;padding:12px;border-top:1px solid #222}
input{flex:1;padding:12px;border-radius:8px;border:1px solid #333;background:#15181e;color:#fff}
button{padding:12px 18px;border:0;border-radius:8px;background:#4f8cff;color:#fff}
details{padding:0 18px 10px}summary{cursor:pointer;color:#8a93a3;font-size:13px}
textarea{width:100%;height:70px;background:#15181e;color:#fff;border:1px solid #333;border-radius:8px;padding:8px;margin-top:6px}
</style>
<header><h1>Deep OQ</h1><span id=st></span></header>
<details><summary>Teach Deep OQ (add to knowledge database)</summary>
<textarea id=kt placeholder="Paste any text, notes, news, code docs..."></textarea>
<button onclick="teach()">Save to database</button></details>
<div id=log></div>
<form id=f><input id=q placeholder="Ask Deep OQ..." autocomplete=off><button>Send</button></form>
<script>
let sid=localStorage.sid||(localStorage.sid=crypto.randomUUID());
const log=document.getElementById('log');
function add(c,t){const d=document.createElement('div');d.className='m '+c;d.textContent=t;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
async function stats(){const j=await (await fetch('/stats')).json();st.textContent=j.chunks+' knowledge chunks'}
async function teach(){const t=kt.value.trim();if(!t)return;
await fetch('/teach',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});kt.value='';stats()}
f.onsubmit=async e=>{e.preventDefault();const q=document.getElementById('q').value.trim();if(!q)return;
document.getElementById('q').value='';add('u',q);const w=add('a','Thinking...');
const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({q,session:sid})});
w.textContent=(await r.json()).answer}
stats()
</script>"""


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/stats")
def stats():
    return jsonify(kb.stats())


@app.post("/teach")
def teach():
    t = (request.get_json() or {}).get("text", "")
    n = kb.add_text(t, title="pasted note", source="pasted-" + uuid.uuid4().hex[:8])
    return jsonify(chunks_added=n)


@app.post("/ask")
def ask():
    data = request.get_json()
    session = data.get("session") or "default"
    try:
        answer = deep_oq(session, data["q"])
        kb.save_msg(session, "user", data["q"])
        kb.save_msg(session, "assistant", answer)
    except Exception as e:
        answer = f"Error: {e}"
    return jsonify(answer=answer)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
