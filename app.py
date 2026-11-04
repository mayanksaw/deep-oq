"""
app.py - Deep OQ server (Flask + Groq)

Run locally:   python app.py
On Render:     gunicorn app:app
Needs env var: GROQ_API_KEY
"""

import os

import requests
from flask import Flask, jsonify, request

import ai_advanced as adv
import ai_tools

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")


def groq_llm(prompt, system="", image_url=None):
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return "[Deep OQ] GROQ_API_KEY is not set."
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if image_url:
        content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]
        model = VISION_MODEL
    else:
        content = prompt
        model = MODEL
    messages.append({"role": "user", "content": content})
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "messages": messages, "temperature": 0.6},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Deep OQ] Model error: {e}"


ai_tools.set_llm(groq_llm)
adv.set_models({"groq": groq_llm})

app = Flask(__name__)

from admin import admin
app.register_blueprint(admin)
PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deep OQ</title>
<style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:0 auto;padding:16px;background:#0f1117;color:#e8e8ea}
h1{margin:0 0 12px}#log{min-height:50vh;padding:8px 0}
.m{margin:10px 0;padding:10px 12px;border-radius:10px;white-space:pre-wrap;line-height:1.45}
.u{background:#2a2f45;margin-left:15%}.a{background:#1b1f2d;margin-right:15%}
form{display:flex;gap:8px}input{flex:1;padding:12px;border-radius:8px;border:1px solid #333;background:#171a24;color:inherit}
button{padding:12px 18px;border:0;border-radius:8px;background:#5b6cff;color:#fff;cursor:pointer}
select{padding:10px;border-radius:8px;background:#171a24;color:inherit;border:1px solid #333;margin-bottom:10px;max-width:100%}
</style></head><body>
<h1>Deep OQ</h1>
<select id="tool"><option value="">Auto (smart)</option></select>
<div id="log"></div>
<form id="f"><input id="q" placeholder="Ask anything..." autocomplete="off"><button>Send</button></form>
<script>
const log=document.getElementById('log'),q=document.getElementById('q'),sel=document.getElementById('tool');
fetch('/api/tools').then(r=>r.json()).then(d=>d.tools.forEach(t=>{
  const o=document.createElement('option');o.value=t.name;o.textContent=t.category+' / '+t.name;sel.appendChild(o);}));
function add(t,c){const d=document.createElement('div');d.className='m '+c;d.textContent=t;log.appendChild(d);d.scrollIntoView();return d;}
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault();const text=q.value.trim();if(!text)return;q.value='';add(text,'u');
  const w=add('Thinking...','a');
  const r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({message:text,tool:sel.value})});
  const d=await r.json();w.textContent=d.answer||d.error||'No answer';
};
</script></body></html>"""


@app.get("/")
def home():
    return PAGE


@app.get("/api/tools")
def tools():
    items = [{"name": n, "category": ai_tools.TOOLS[n]["category"], "description": ai_tools.TOOLS[n]["description"]}
             for n in ai_tools.list_tools()]
    return jsonify({"count": len(items), "tools": items})


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400
    tool = data.get("tool") or ""
    if tool:
        if tool not in ai_tools.TOOLS:
            return jsonify({"error": f"unknown tool: {tool}"}), 400
        main = next(iter(ai_tools.TOOLS[tool]["params"]), "text")
        return jsonify({"answer": ai_tools.run_tool(tool, **{main: message})})
    return jsonify({"answer": adv.smart(message)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
