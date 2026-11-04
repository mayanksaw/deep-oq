import os
import sqlite3

import requests
from flask import (Flask, jsonify, redirect, request, send_file, session,
                   render_template_string)
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-in-railway")
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 30)

DB = os.environ.get("DB_PATH", "users.db")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = (
    "You are Deep OQ, a smart, fast and accurate AI assistant created by "
    "Mayank Kumar Saw. Answer clearly and correctly. For math, work step by "
    "step and double-check the result. Do not add a 'Sources:' line."
)


# ---------- database ----------
def db():
    d = os.path.dirname(DB)
    if d:
        os.makedirs(d, exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS users("
              "email TEXT PRIMARY KEY, name TEXT, pw TEXT)")
    return c


# ---------- auth ----------
@app.get("/login")
def login_page():
    if "email" in session:
        return redirect("/")
    return send_file("login.html")


@app.post("/api/login")
def api_login():
    d = request.get_json(force=True, silent=True) or {}
    email = d.get("email", "").strip().lower()
    pw = d.get("password", "")
    if not email or len(pw) < 6:
        return jsonify(error="Enter a valid email and a 6+ character password."), 400
    c = db()
    if d.get("mode") == "signup":
        if c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            return jsonify(error="Account already exists. Please sign in."), 400
        name = d.get("name", "").strip() or email.split("@")[0]
        c.execute("INSERT INTO users VALUES(?,?,?)",
                  (email, name, generate_password_hash(pw)))
        c.commit()
    else:
        row = c.execute("SELECT name, pw FROM users WHERE email=?",
                        (email,)).fetchone()
        if not row or not check_password_hash(row[1], pw):
            return jsonify(error="Wrong email or password."), 401
        name = row[0]
    session.permanent = True
    session["email"] = email
    session["name"] = name
    return jsonify(name=name, email=email)


@app.get("/logout")
def logout():
    session.clear()
    return redirect("/login")


# ---------- chat ----------
@app.post("/api/chat")
def api_chat():
    if "email" not in session:
        return jsonify(error="Please log in"), 401
    if not GROQ_API_KEY:
        return jsonify(error="GROQ_API_KEY is not set on the server."), 500
    msgs = (request.get_json(force=True, silent=True) or {}).get("messages", [])
    msgs = [m for m in msgs[-20:]
            if m.get("role") in ("user", "assistant") and m.get("content")]
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": GROQ_MODEL, "temperature": 0.4,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + msgs},
            timeout=60)
        r.raise_for_status()
        return jsonify(reply=r.json()["choices"][0]["message"]["content"])
    except Exception as e:
        return jsonify(error=f"AI error: {e}"), 502


PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deep OQ</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Inter,system-ui,sans-serif;background:#05060f;color:#eef0ff;height:100vh;display:flex;flex-direction:column}
header{display:flex;justify-content:space-between;align-items:center;padding:14px 22px;border-bottom:1px solid rgba(255,255,255,.1)}
header b{font-size:20px;background:linear-gradient(90deg,#00d4ff,#7c5cff,#ff4fd8);-webkit-background-clip:text;background-clip:text;color:transparent}
header a{color:#9aa0c3;text-decoration:none;font-size:14px}
#chat{flex:1;overflow-y:auto;padding:22px;display:flex;flex-direction:column;gap:14px;max-width:820px;width:100%;margin:0 auto}
.m{padding:12px 16px;border-radius:16px;max-width:85%;white-space:pre-wrap;line-height:1.5;animation:in .3s}
@keyframes in{from{opacity:0;transform:translateY(8px)}to{opacity:1}}
.u{align-self:flex-end;background:linear-gradient(135deg,#7c5cff,#00d4ff)}
.a{align-self:flex-start;background:rgba(255,255,255,.08)}
form{display:flex;gap:10px;padding:16px 22px;max-width:820px;width:100%;margin:0 auto}
input{flex:1;padding:14px 16px;border-radius:14px;border:1px solid rgba(255,255,255,.15);background:rgba(255,255,255,.06);color:#fff;font-size:15px;outline:none}
input:focus{border-color:#7c5cff}
button{padding:0 22px;border:0;border-radius:14px;color:#fff;font-weight:600;cursor:pointer;background:linear-gradient(90deg,#7c5cff,#00d4ff)}
</style></head><body>
<header><b>Deep OQ</b><span><span style="color:#9aa0c3;font-size:14px">{{ name }}</span> &nbsp; <a href="/logout">Log out</a></span></header>
<div id="chat"><div class="m a">Hi {{ name }}! I'm Deep OQ. Ask me anything.</div></div>
<form id="f"><input id="q" placeholder="Message Deep OQ..." autocomplete="off" autofocus><button>Send</button></form>
<script>
const chat=document.getElementById("chat"),q=document.getElementById("q");let hist=[];
function add(t,c){const d=document.createElement("div");d.className="m "+c;d.textContent=t;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d}
document.getElementById("f").onsubmit=async e=>{
  e.preventDefault();const t=q.value.trim();if(!t)return;q.value="";
  add(t,"u");hist.push({role:"user",content:t});const w=add("Thinking...","a");
  try{
    const r=await fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({messages:hist})});
    if(r.status===401){location.href="/login";return}
    const j=await r.json();w.textContent=j.reply||j.error;if(j.reply)hist.push({role:"assistant",content:j.reply});
  }catch(err){w.textContent="Network error."}
  chat.scrollTop=chat.scrollHeight;
};
</script></body></html>"""


@app.get("/")
def home():
    if "email" not in session:
        return redirect("/login")
    return render_template_string(PAGE, name=session.get("name", "friend"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
