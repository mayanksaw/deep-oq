"""
app.py - Deep OQ server (Flask + Groq + PostgreSQL)

Run locally:   python app.py
Production:    gunicorn app:app
Env vars:      GROQ_API_KEY (required), DATABASE_URL (PostgreSQL connection string),
               ALLOWED_ORIGIN (optional, for a separate frontend),
               ADMIN_USERS (comma separated admin usernames),
               TRUSTED_PROXIES (optional, number of reverse proxies in front
               of the app, default 1; set 0 if running without a proxy)
"""

import os
import threading
import time
from collections import defaultdict, deque
from functools import wraps

import requests
from flask import Flask, g, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix

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

# Trust only the real number of proxies in front of the app, so a client can't
# spoof X-Forwarded-For to get around the rate limits. Set TRUSTED_PROXIES=0
# if you run without a reverse proxy.
_proxies = int(os.environ.get("TRUSTED_PROXIES", "1"))
if _proxies > 0:
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=_proxies, x_proto=_proxies, x_host=_proxies)


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
    # ProxyFix has already put the real client address in remote_addr
    return request.remote_addr or "?"


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
        if not user or user["username"].lower() not in admins:
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
<form id="login"><input id="u" placeholder="admin username" autocomplete="username"><input id="p" type="password" placeholder="password" autocomplete="current-password"><button>Log in</button></form>
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
$('login').onsubmit=async e=>{
  e.preventDefault();
  const r=await fetch('/api/login',{method:'POST',headers:H(),body:JSON.stringify({username:$('u').value,password:$('p').value})});
  const d=await r.json();if(!d.token){$('msg').textContent=d.error;return;}
  token=d.token;sessionStorage.setItem('adm',token);load();};
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


# ---------------- web page (animated login + chat) ----------------

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deep OQ</title>
<style>
:root{--bg:#0b0d14;--card:#12151f;--line:#262b3d;--fg:#e8eaf2;--mut:#8a90a6;--acc:#6c6cff;--ok:#3ee0c3;--bad:#ff6b81}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--fg);min-height:100vh}
[hidden]{display:none!important}
button,input,select{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px}

/* ---------- animated login ---------- */
#net{position:fixed;inset:0;width:100%;height:100%;z-index:0}
#authView{position:relative;z-index:1;min-height:100vh;display:grid;place-items:center;padding:20px}
.card{width:min(400px,100%);background:rgba(18,21,31,.92);border:1px solid var(--line);border-radius:18px;padding:32px 28px 28px;
  box-shadow:0 30px 80px rgba(0,0,0,.45);animation:rise .9s cubic-bezier(.2,.8,.2,1) both}
@keyframes rise{from{opacity:0;transform:translateY(18px) scale(.98)}to{opacity:1;transform:none}}
.card.shake{animation:shake .45s}
@keyframes shake{20%,60%{transform:translateX(-8px)}40%,80%{transform:translateX(8px)}}
.brand{display:flex;align-items:center;gap:12px;margin-bottom:6px}
.lg{width:38px;height:38px;flex:none}
.lg .ring{transform-origin:50% 50%;animation:spin 14s linear infinite}
.lg .core{transform-origin:50% 50%;animation:pulse 2.4s ease-in-out infinite}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes pulse{50%{transform:scale(1.25);opacity:.7}}
.brand h1{margin:0;font-size:26px;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:14px;margin:0 0 22px;min-height:20px}
.tabs{position:relative;display:grid;grid-template-columns:1fr 1fr;background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:4px;margin-bottom:20px}
.tabs button{position:relative;z-index:1;background:none;border:0;color:var(--mut);font-weight:600;font-size:14px;padding:9px 0;cursor:pointer;border-radius:9px;transition:color .25s}
.tabs button[aria-selected=true]{color:var(--fg)}
.pill{position:absolute;top:4px;bottom:4px;left:4px;width:calc(50% - 4px);background:var(--line);border-radius:9px;transition:transform .35s cubic-bezier(.3,1.3,.5,1)}
.tabs[data-mode=register] .pill{transform:translateX(100%)}
.field{position:relative;margin-bottom:14px}
.field input{width:100%;background:var(--bg);border:1px solid var(--line);border-radius:12px;font-size:15px;padding:22px 60px 8px 14px;outline:none;transition:border-color .2s,box-shadow .2s}
.field input:focus{border-color:var(--acc);box-shadow:0 0 0 3px rgba(108,108,255,.22)}
.field label{position:absolute;left:14px;top:16px;color:var(--mut);font-size:15px;pointer-events:none;transform-origin:left top;transition:transform .2s,color .2s}
.field input:focus+label,.field input:not(:placeholder-shown)+label{transform:translateY(-9px) scale(.76)}
.field input:focus+label{color:var(--acc)}
.eye{position:absolute;right:8px;top:50%;transform:translateY(-50%);background:none;border:0;color:var(--mut);cursor:pointer;font-size:12px;padding:8px;border-radius:8px}
.eye:hover{color:var(--fg)}
#err{min-height:20px;font-size:13.5px;margin:2px 0 12px;color:var(--bad)}
#err.ok{color:var(--ok)}
.go{width:100%;min-height:46px;border:0;border-radius:12px;background:var(--acc);color:#fff;font-weight:700;font-size:15px;cursor:pointer;transition:transform .15s,filter .2s}
.go:hover{filter:brightness(1.1)}.go:active{transform:scale(.98)}.go:disabled{cursor:progress}
.go .spin{display:none;width:18px;height:18px;border:2px solid rgba(255,255,255,.35);border-top-color:#fff;border-radius:50%;margin:0 auto;animation:spin .7s linear infinite}
.go.loading #lbl{display:none}.go.loading .spin{display:block}
.go.done{background:var(--ok);color:#06251f}

/* ---------- chat ---------- */
#appView{max-width:760px;margin:0 auto;padding:16px;animation:rise .6s both}
.top{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px}
.top h1{margin:0;font-size:22px}
.top .right{display:flex;align-items:center;gap:10px;color:var(--mut);font-size:14px}
.ghost{background:transparent;border:1px solid var(--line);border-radius:8px;padding:7px 12px;cursor:pointer}
.ghost:hover{border-color:var(--acc)}
#log{min-height:45vh;padding:8px 0}
.m{margin:10px 0;padding:10px 12px;border-radius:10px;white-space:pre-wrap;line-height:1.45;overflow-wrap:anywhere}
.u{background:#2a2f45;margin-left:15%}.a{background:#1b1f2d;margin-right:15%}
#f{display:flex;gap:8px;margin-top:8px}
#f input{flex:1;padding:12px;border-radius:8px;border:1px solid var(--line);background:#171a24}
#f button{padding:12px 18px;border:0;border-radius:8px;background:var(--acc);color:#fff;cursor:pointer}
#tool{padding:10px;border-radius:8px;border:1px solid var(--line);background:#171a24;max-width:100%}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}
</style></head><body>
<canvas id="net" aria-hidden="true"></canvas>

<!-- LOGIN / SIGN UP -->
<div id="authView">
  <form class="card" id="authForm" novalidate>
    <div class="brand">
      <svg class="lg" viewBox="0 0 40 40" aria-hidden="true">
        <circle class="ring" cx="20" cy="20" r="16" fill="none" stroke="#6c6cff" stroke-width="2" stroke-dasharray="6 5"/>
        <circle cx="20" cy="20" r="9" fill="none" stroke="#3ee0c3" stroke-width="1.5" opacity=".7"/>
        <circle class="core" cx="20" cy="20" r="4" fill="#6c6cff"/>
      </svg>
      <h1>Deep OQ</h1>
    </div>
    <p class="sub" id="subtitle">Log in to pick up your chats.</p>

    <div class="tabs" id="tabs" data-mode="login" role="tablist">
      <span class="pill"></span>
      <button type="button" id="tabLogin" role="tab" aria-selected="true">Log in</button>
      <button type="button" id="tabReg" role="tab" aria-selected="false">Sign up</button>
    </div>

    <div class="field">
      <input id="u" name="username" placeholder=" " autocomplete="username" autocapitalize="none" spellcheck="false" required>
      <label for="u">Username</label>
    </div>
    <div class="field">
      <input id="p" name="password" type="password" placeholder=" " autocomplete="current-password" required>
      <label for="p">Password</label>
      <button type="button" class="eye" id="eye">Show</button>
    </div>

    <div id="err" role="alert" aria-live="polite"></div>
    <button class="go" id="submit"><span id="lbl">Log in</span><i class="spin"></i></button>
  </form>
</div>

<!-- CHAT -->
<div id="appView" hidden>
  <div class="top">
    <h1>Deep OQ</h1>
    <div class="right"><span id="who"></span><button class="ghost" id="logout" type="button">Log out</button></div>
  </div>
  <select id="tool"><option value="">Auto (smart)</option></select>
  <div id="log"></div>
  <form id="f"><input id="q" placeholder="Ask anything..." autocomplete="off"><button>Send</button></form>
</div>

<script>
const $=id=>document.getElementById(id);
let token=sessionStorage.getItem('t')||'', name=sessionStorage.getItem('n')||'', cid=null, mode='login';
const H=()=>({'Content-Type':'application/json','Authorization':'Bearer '+token});

/* ----- background network (runs only while the login screen is visible) ----- */
const cv=$('net'),ctx=cv.getContext('2d'),still=matchMedia('(prefers-reduced-motion: reduce)').matches;
let W,H2,nodes=[],mouse={x:-999,y:-999},boost=1,raf=0;
function resize(){
  const dpr=Math.min(devicePixelRatio||1,2);W=innerWidth;H2=innerHeight;
  cv.width=W*dpr;cv.height=H2*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
  const n=Math.min(90,Math.round(W*H2/14000));
  nodes=Array.from({length:n},()=>({x:Math.random()*W,y:Math.random()*H2,vx:(Math.random()-.5)*.35,vy:(Math.random()-.5)*.35,r:Math.random()*1.6+.8}));
}
addEventListener('resize',resize);
addEventListener('pointermove',e=>{mouse.x=e.clientX;mouse.y=e.clientY;});
addEventListener('pointerleave',()=>{mouse.x=mouse.y=-999;});
function frame(){
  raf=0;
  if($('authView').hidden){ctx.clearRect(0,0,W,H2);return;}
  ctx.clearRect(0,0,W,H2);boost+=(1-boost)*.03;
  for(const a of nodes){
    a.x+=a.vx*boost;a.y+=a.vy*boost;
    if(a.x<0||a.x>W)a.vx*=-1;if(a.y<0||a.y>H2)a.vy*=-1;
    const dm=Math.hypot(a.x-mouse.x,a.y-mouse.y);
    ctx.fillStyle=dm<160?'#3ee0c3':'rgba(138,144,166,.7)';
    ctx.beginPath();ctx.arc(a.x,a.y,a.r,0,6.283);ctx.fill();
    if(dm<160){ctx.strokeStyle='rgba(62,224,195,'+((1-dm/160)*.7)+')';ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(mouse.x,mouse.y);ctx.stroke();}
  }
  for(let i=0;i<nodes.length;i++)for(let j=i+1;j<nodes.length;j++){
    const a=nodes[i],b=nodes[j],d=Math.hypot(a.x-b.x,a.y-b.y);
    if(d<130){ctx.strokeStyle='rgba(108,108,255,'+((1-d/130)*.35)+')';ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}
  }
  if(!still)raf=requestAnimationFrame(frame);
}
function startNet(){if(!raf)raf=requestAnimationFrame(frame);}
resize();

/* ----- auth UI ----- */
function setMode(m){
  mode=m;const reg=m==='register';
  $('tabs').dataset.mode=m;
  $('tabLogin').setAttribute('aria-selected',!reg);$('tabReg').setAttribute('aria-selected',reg);
  $('subtitle').textContent=reg?'Create an account (password: 8+ characters).':'Log in to pick up your chats.';
  $('lbl').textContent=reg?'Create account':'Log in';
  $('p').autocomplete=reg?'new-password':'current-password';
  say('');
}
function say(t,ok){$('err').textContent=t;$('err').className=ok?'ok':'';}
function fail(t){say(t);const c=$('authForm');c.classList.remove('shake');void c.offsetWidth;c.classList.add('shake');}
$('tabLogin').onclick=()=>setMode('login');
$('tabReg').onclick=()=>setMode('register');
$('eye').onclick=()=>{const s=$('p').type==='password';$('p').type=s?'text':'password';$('eye').textContent=s?'Hide':'Show';};

function showApp(){
  const on=!!token;
  $('authView').hidden=on;$('appView').hidden=!on;
  $('who').textContent=on&&name?name:'';
  if(on)$('q').focus();else{$('u').focus();startNet();}
}
function signedOut(msg){
  token='';name='';cid=null;sessionStorage.removeItem('t');sessionStorage.removeItem('n');
  $('log').replaceChildren();$('p').value='';
  $('submit').disabled=false;$('submit').className='go';$('lbl').textContent=mode==='register'?'Create account':'Log in';
  showApp();if(msg)fail(msg);
}

$('authForm').onsubmit=async e=>{
  e.preventDefault();
  const username=$('u').value.trim(),password=$('p').value;
  if(!username||!password)return fail('Enter your username and password.');
  if(mode==='register'&&password.length<8)return fail('Password must be at least 8 characters.');
  say('');$('submit').disabled=true;$('submit').classList.add('loading');
  try{
    const r=await fetch('/api/'+mode,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username,password})});
    const d=await r.json().catch(()=>({}));
    if(d.token){
      token=d.token;name=username.toLowerCase();sessionStorage.setItem('t',token);sessionStorage.setItem('n',name);
      $('submit').classList.remove('loading');$('submit').classList.add('done');
      $('lbl').textContent=mode==='register'?'Account created':'Logged in';say('Opening your chats...',true);
      boost=6;setTimeout(()=>{$('p').value='';showApp();},800);
      return;
    }
    fail(d.error||'Something went wrong. Try again.');
  }catch(_){fail('Could not reach the server.');}
  $('submit').classList.remove('loading');$('submit').disabled=false;
};

$('logout').onclick=async()=>{
  try{await fetch('/api/logout',{method:'POST',headers:H()});}catch(_){}
  signedOut();
};

/* ----- chat ----- */
const log=$('log'),q=$('q'),sel=$('tool');
function add(t,c){const d=document.createElement('div');d.className='m '+c;d.textContent=t;log.appendChild(d);d.scrollIntoView();return d;}
fetch('/api/tools').then(r=>r.json()).then(d=>d.tools.forEach(t=>{const o=document.createElement('option');o.value=t.name;o.textContent=t.category+' / '+t.name;sel.appendChild(o);}));
$('f').onsubmit=async e=>{
  e.preventDefault();const text=q.value.trim();if(!text)return;
  q.value='';add(text,'u');const w=add('Thinking...','a');
  try{
    const r=await fetch('/api/chat',{method:'POST',headers:H(),body:JSON.stringify({message:text,tool:sel.value,conversation_id:cid})});
    if(r.status===401){signedOut('Your session expired. Please log in again.');return;}
    const d=await r.json();if(d.conversation_id)cid=d.conversation_id;
    w.textContent=d.answer||d.error||'No answer';
  }catch(_){w.textContent='Could not reach the server.';}
};

showApp();
</script></body></html>"""


@app.get("/")
def home():
    return PAGE


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
