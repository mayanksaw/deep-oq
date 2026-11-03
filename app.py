"""
app.py - Deep OQ server (Flask + Groq + SQLite)

Run locally:   python app.py
Production:    gunicorn app:app
Env vars:      GROQ_API_KEY (required), DEEPOQ_DB (optional db path),
               ALLOWED_ORIGIN (optional, for a separate frontend)
"""

import os
import threading
import time
from collections import defaultdict, deque
from functools import wraps

import requests
from flask import Flask, g, jsonify, request

import ai_advanced as adv
import ai_tools
import db

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
        content, model = prompt, MODEL
    messages.append({"role": "user", "content": content})
    try:
        r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {key}"},
                          json={"model": model, "messages": messages, "temperature": 0.6}, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Deep OQ] Model error: {e}"


ai_tools.set_llm(groq_llm)
adv.set_models({"groq": groq_llm})
db.init()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1_000_000  # 1 MB request limit


# ---------------- security helpers ----------------

_hits = defaultdict(deque)
_lock = threading.Lock()


def rate_limited(key, limit, window=60):
    now = time.time()
    with _lock:
        q = _hits[key]
        while q and q[0] < now - window:
            q.popleft()
        if len(q) >= limit:
            return True
        q.append(now)
        return False


def client_ip():
    return (request.headers.get("X-Forwarded-For", request.remote_addr) or "?").split(",")[0].strip()


def token_from_request():
    h = request.headers.get("Authorization", "")
    return h[7:] if h.startswith("Bearer ") else None


def auth_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        user = db.user_from_token(token_from_request())
        if not user:
            return jsonify({"error": "Please log in."}), 401
        if rate_limited(("u", user["id"]), 30):
            return jsonify({"error": "Too many requests. Slow down."}), 429
        g.user = user
        return fn(*a, **kw)
    return wrapper


@app.after_request
def headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    origin = os.environ.get("ALLOWED_ORIGIN")
    if origin:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    return resp


# ---------------- auth routes ----------------

@app.post("/api/register")
def register():
    if rate_limited(("reg", client_ip()), 5, 3600):
        return jsonify({"error": "Too many sign-ups from this address."}), 429
    d = request.get_json(silent=True) or {}
    try:
        db.create_user(d.get("username", ""), d.get("password", ""))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"token": db.login(d["username"], d["password"])})


@app.post("/api/login")
def login():
    if rate_limited(("login", client_ip()), 10, 300):
        return jsonify({"error": "Too many attempts. Wait a few minutes."}), 429
    d = request.get_json(silent=True) or {}
    token = db.login(d.get("username", ""), d.get("password", ""))
    if not token:
        return jsonify({"error": "Wrong username or password."}), 401
    return jsonify({"token": token})


@app.post("/api/logout")
@auth_required
def logout():
    db.logout(token_from_request())
    return jsonify({"ok": True})


# ---------------- chat routes ----------------

@app.get("/api/tools")
def tools():
    items = [{"name": n, "category": ai_tools.TOOLS[n]["category"], "description": ai_tools.TOOLS[n]["description"]}
             for n in ai_tools.list_tools()]
    return jsonify({"count": len(items), "tools": items})


@app.get("/api/conversations")
@auth_required
def conversations():
    return jsonify(db.list_conversations(g.user["id"]))


@app.get("/api/conversations/<int:cid>")
@auth_required
def conversation(cid):
    if not db.owns_conversation(g.user["id"], cid):
        return jsonify({"error": "Not found."}), 404
    return jsonify(db.get_messages(cid))


@app.delete("/api/conversations/<int:cid>")
@auth_required
def delete_conversation(cid):
    db.delete_conversation(g.user["id"], cid)
    return jsonify({"ok": True})


@app.post("/api/chat")
@auth_required
def chat():
    d = request.get_json(silent=True) or {}
    message = (d.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400
    if len(message) > 20000:
        return jsonify({"error": "message too long"}), 400

    uid = g.user["id"]
    cid = d.get("conversation_id")
    if cid:
        if not db.owns_conversation(uid, int(cid)):
            return jsonify({"error": "Conversation not found."}), 404
    else:
        cid = db.new_conversation(uid, message[:60])

    tool = d.get("tool") or ""
    if tool and tool not in ai_tools.TOOLS:
        return jsonify({"error": f"unknown tool: {tool}"}), 400

    db.add_message(cid, "user", message, tool or None)

    # personal memory goes in as context so Deep OQ remembers this user
    facts = db.list_facts(uid)[:10]
    context = "\n".join(f"- {f['key']}: {f['value']}" for f in facts)
    prompt = f"Known about the user:\n{context}\n\nRequest: {message}" if context and not tool else message

    if tool:
        main = next(iter(ai_tools.TOOLS[tool]["params"]), "text")
        answer = ai_tools.run_tool(tool, **{main: message})
    else:
        answer = adv.smart(prompt)

    db.add_message(cid, "assistant", str(answer), tool or None)
    return jsonify({"answer": answer, "conversation_id": cid})


@app.get("/api/memory")
@auth_required
def memory_list():
    return jsonify(db.list_facts(g.user["id"]))


@app.post("/api/memory")
@auth_required
def memory_save():
    d = request.get_json(silent=True) or {}
    if not d.get("key") or not d.get("value"):
        return jsonify({"error": "key and value are required"}), 400
    db.save_fact(g.user["id"], d["key"], d["value"])
    return jsonify({"ok": True})


# ---------------- admin ----------------
# Set ADMIN_USERS on the server, e.g.  ADMIN_USERS=mayank,second_admin
# Only logged-in users whose username is in that list can use the admin page.
# The admin page shows counts and usernames only, never anyone's message text.

def admin_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        user = db.user_from_token(token_from_request())
        admins = {x.strip().lower() for x in os.environ.get("ADMIN_USERS", "").split(",") if x.strip()}
        if not user or user["username"] not in admins:
            return jsonify({"error": "Admins only."}), 403
        if rate_limited(("admin", user["id"]), 60):
            return jsonify({"error": "Too many requests."}), 429
        g.user = user
        return fn(*a, **kw)
    return wrapper


@app.get("/api/admin/overview")
@admin_required
def admin_overview():
    return jsonify({
        "stats": db.admin_stats(),
        "tools_loaded": len(ai_tools.TOOLS),
        "tool_usage": db.admin_tool_usage(),
        "daily": db.admin_daily(),
        "users": db.admin_users(),
        "groq_key_set": bool(os.environ.get("GROQ_API_KEY")),
        "model": MODEL,
    })


@app.delete("/api/admin/users/<int:uid>")
@admin_required
def admin_delete_user(uid):
    if uid == g.user["id"]:
        return jsonify({"error": "You cannot delete your own admin account here."}), 400
    if not db.username_of(uid):
        return jsonify({"error": "User not found."}), 404
    db.delete_user(uid)
    return jsonify({"ok": True})


ADMIN_PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deep OQ Admin</title>
<style>
:root{--bg:#0f1117;--card:#1b1f2d;--line:#2a2f45;--fg:#e8e8ea;--mut:#9aa0b5;--acc:#5b6cff;--bad:#ff6b6b}
body{font-family:system-ui,sans-serif;margin:0;padding:16px;background:var(--bg);color:var(--fg)}
.wrap{max-width:1000px;margin:0 auto}h1{margin:0 0 4px}h2{font-size:16px;margin:24px 0 8px;color:var(--mut);font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
.card b{display:block;font-size:24px}.card span{color:var(--mut);font-size:13px}
table{width:100%;border-collapse:collapse;background:var(--card);border-radius:10px;overflow:hidden}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);font-size:14px}th{color:var(--mut);font-weight:600}
.scroll{overflow-x:auto}input,button{padding:10px;border-radius:8px;border:1px solid var(--line);background:#171a24;color:inherit;font:inherit}
button{background:var(--acc);border:0;color:#fff;cursor:pointer}button.del{background:transparent;border:1px solid var(--bad);color:var(--bad);padding:4px 10px}
.bar{height:10px;background:var(--acc);border-radius:5px}.row{display:flex;align-items:center;gap:8px;margin:4px 0;font-size:13px}
.row .n{width:150px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.row .v{width:40px;text-align:right;color:var(--mut)}
.row .t{flex:1;background:#171a24;border-radius:5px}#login{display:flex;gap:8px;flex-wrap:wrap}#msg{color:var(--bad);margin:8px 0}
</style></head><body><div class="wrap">
<h1>Deep OQ Admin</h1><div style="color:var(--mut)" id="sub">Log in with an admin account</div>
<div id="login"><input id="u" placeholder="admin username"><input id="p" type="password" placeholder="password"><button onclick="signin()">Log in</button></div>
<div id="msg"></div><div id="main" hidden>
<div class="grid" id="cards"></div>
<h2>Messages per day (last 14 days)</h2><div class="card" id="daily"></div>
<h2>Most used tools</h2><div class="card" id="tools"></div>
<h2>Users</h2><div class="scroll"><table id="users"></table></div>
</div></div>
<script>
let token=sessionStorage.getItem('adm')||'';
const $=id=>document.getElementById(id);
const H=()=>({'Content-Type':'application/json','Authorization':'Bearer '+token});
const ago=t=>t?new Date(t*1000).toLocaleString():'never';
function el(tag,text,attrs){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;Object.assign(e,attrs||{});return e;}
function bars(box,items,label,val){box.replaceChildren();const max=Math.max(1,...items.map(i=>i[val]));
  if(!items.length)box.textContent='No data yet.';
  items.forEach(i=>{const r=el('div');r.className='row';r.append(el('span',i[label],{className:'n'}));
    const t=el('div');t.className='t';const b=el('div');b.className='bar';b.style.width=(i[val]/max*100)+'%';t.append(b);
    r.append(t,el('span',i[val],{className:'v'}));box.append(r);});}
async function signin(){
  const r=await fetch('/api/login',{method:'POST',headers:H(),body:JSON.stringify({username:$('u').value,password:$('p').value})});
  const d=await r.json();if(!d.token){$('msg').textContent=d.error;return;}
  token=d.token;sessionStorage.setItem('adm',token);load();}
async function load(){
  const r=await fetch('/api/admin/overview',{headers:H()});
  if(!r.ok){$('msg').textContent=r.status==403?'This account is not an admin. Set ADMIN_USERS on the server.':'Please log in.';return;}
  const d=await r.json();$('msg').textContent='';$('login').hidden=true;$('main').hidden=false;
  $('sub').textContent=d.tools_loaded+' tools loaded - model: '+d.model+' - Groq key '+(d.groq_key_set?'set':'MISSING');
  const s=d.stats,c=$('cards');c.replaceChildren();
  [['Users',s.users],['New (24h)',s.new_users_24h],['Active (24h)',s.active_users_24h],['Chats',s.conversations],['Messages',s.messages],['Messages (24h)',s.messages_24h],['DB size (KB)',s.db_size_kb]]
   .forEach(([k,v])=>{const x=el('div');x.className='card';x.append(el('b',v),el('span',k));c.append(x);});
  bars($('daily'),d.daily,'day','messages');bars($('tools'),d.tool_usage,'tool','uses');
  const t=$('users');t.replaceChildren();const hr=el('tr');['ID','Username','Joined','Chats','Messages','Last active',''].forEach(h=>hr.append(el('th',h)));t.append(hr);
  d.users.forEach(u=>{const tr=el('tr');[u.id,u.username,ago(u.created),u.chats,u.messages,ago(u.last_active)].forEach(v=>tr.append(el('td',v)));
    const td=el('td');const b=el('button','Delete',{className:'del'});b.onclick=()=>del(u);td.append(b);tr.append(td);t.append(tr);});}
async function del(u){
  if(!confirm('Delete user "'+u.username+'" and ALL their chats? This cannot be undone.'))return;
  const r=await fetch('/api/admin/users/'+u.id,{method:'DELETE',headers:H()});
  const d=await r.json();if(!r.ok)alert(d.error);load();}
if(token)load();
</script></body></html>"""


@app.get("/admin")
def admin_page():
    return ADMIN_PAGE


@app.get("/health")
def health():
    return jsonify({"status": "ok", "tools": len(ai_tools.TOOLS)})


# ---------------- simple web page ----------------

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deep OQ</title>
<style>
body{font-family:system-ui,sans-serif;max-width:760px;margin:0 auto;padding:16px;background:#0f1117;color:#e8e8ea}
h1{margin:0 0 12px}#log{min-height:45vh;padding:8px 0}
.m{margin:10px 0;padding:10px 12px;border-radius:10px;white-space:pre-wrap;line-height:1.45}
.u{background:#2a2f45;margin-left:15%}.a{background:#1b1f2d;margin-right:15%}
form{display:flex;gap:8px}input,select,button{padding:12px;border-radius:8px;border:1px solid #333;background:#171a24;color:inherit;font:inherit}
input{flex:1}button{background:#5b6cff;border:0;color:#fff;cursor:pointer}
#auth{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}#auth input{min-width:140px}
</style></head><body>
<h1>Deep OQ</h1>
<div id="auth"><input id="u" placeholder="username"><input id="p" type="password" placeholder="password (8+ chars)">
<button onclick="go('login')">Log in</button><button onclick="go('register')">Sign up</button><span id="who"></span></div>
<select id="tool"><option value="">Auto (smart)</option></select>
<div id="log"></div>
<form id="f"><input id="q" placeholder="Ask anything..." autocomplete="off"><button>Send</button></form>
<script>
let token=sessionStorage.getItem('t')||'',cid=null;
const log=document.getElementById('log'),q=document.getElementById('q'),sel=document.getElementById('tool'),who=document.getElementById('who');
const H=()=>({'Content-Type':'application/json','Authorization':'Bearer '+token});
function add(t,c){const d=document.createElement('div');d.className='m '+c;d.textContent=t;log.appendChild(d);d.scrollIntoView();return d;}
function show(){who.textContent=token?'Logged in':'';}show();
fetch('/api/tools').then(r=>r.json()).then(d=>d.tools.forEach(t=>{const o=document.createElement('option');o.value=t.name;o.textContent=t.category+' / '+t.name;sel.appendChild(o);}));
async function go(kind){
  const r=await fetch('/api/'+kind,{method:'POST',headers:H(),body:JSON.stringify({username:u.value,password:p.value})});
  const d=await r.json();if(d.token){token=d.token;sessionStorage.setItem('t',token);show();add('Logged in.','a');}else add(d.error,'a');}
document.getElementById('f').onsubmit=async e=>{
  e.preventDefault();const text=q.value.trim();if(!text)return;
  if(!token){add('Log in or sign up first.','a');return;}
  q.value='';add(text,'u');const w=add('Thinking...','a');
  const r=await fetch('/api/chat',{method:'POST',headers:H(),body:JSON.stringify({message:text,tool:sel.value,conversation_id:cid})});
  const d=await r.json();if(d.conversation_id)cid=d.conversation_id;w.textContent=d.answer||d.error||'No answer';};
</script></body></html>"""


@app.get("/")
def home():
    return PAGE


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
