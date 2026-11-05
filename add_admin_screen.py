"""
Adds an owner-only ADMIN screen to Deep OQ (page: /admin), with the new attractive design:
  - animated gradient background and glass cards, matching your login page
  - stat cards: people, chats, messages, active today
  - people with coloured avatars, Google / Password / Guest tags, search and sorting
  - chat bubbles, times shown as "5 min ago", Copy and Download buttons
  - works on phones
You can see every person, their chats and messages. Read only.
If your deep_oq.py already has the older admin screen, this script upgrades its look.

HOW TO USE
  1. Put this file in the same folder as deep_oq.py (the one that already has Google sign-in)
  2. Run:  python add_admin_screen.py
  3. Upload the updated deep_oq.py to GitHub
  4. On Render, check the environment variable ADMIN_EMAILS holds the Gmail you sign in with
     (several emails: separate them with commas)

A backup of your file is saved as deep_oq.py.bak2 (do not upload it).
If any step cannot find its place in your file, nothing is changed.
"""
import os
import shutil
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else "deep_oq.py"

EDITS = []


def edit(old, new):
    EDITS.append((old, new))


ADMIN_CODE = r'''# ==========================================================================
# ADMIN: owner-only screen to see people, chats and messages (read only)
# ==========================================================================
def _name_set(var):
    return {x.strip().lower() for x in os.environ.get(var, "").split(",") if x.strip()}


def is_admin():
    me = login_session.get("user") or ""
    if not me:
        return False
    if me.startswith("google:"):  # the email was verified by Google
        return me[7:] in _name_set("ADMIN_EMAILS")
    return me in _name_set("ADMIN_USERS")


def _adm_guard():
    if not is_admin():
        return jsonify(error="Not allowed."), 403
    if denied():
        return jsonify(error="locked"), 401
    return None


def _adm_q(sqlite_sql, pg_sql, params=()):
    if USE_PG:
        return _pg(pg_sql, params, "all")
    return DB.execute(sqlite_sql, params).fetchall()


def _adm_names():
    rows = _adm_q("SELECT username FROM users", "SELECT username FROM oq_users")
    return sorted((r[0] for r in rows), key=len, reverse=True)


def _adm_owner(session, names):
    return next((u for u in names if session.startswith(u + ":")), "")


def _adm_sessions():
    return _adm_q("SELECT session, COUNT(*), MAX(ts) FROM messages GROUP BY session",
                  "SELECT session, COUNT(*), MAX(ts) FROM oq_messages GROUP BY session")


@app.get("/admin")
def admin_page():
    if not is_admin():
        return "Not found", 404
    return ADMIN_PAGE, 200, {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store"}


@app.get("/api/admin/users")
def admin_users():
    bad = _adm_guard()
    if bad:
        return bad
    users = _adm_q("SELECT username, created FROM users", "SELECT username, created FROM oq_users")
    names = sorted((u[0] for u in users), key=len, reverse=True)
    stat = {u[0]: {"username": u[0], "created": str(u[1] or ""), "sessions": 0, "messages": 0, "last": ""}
            for u in users}
    guests = {"username": "(guests)", "created": "", "sessions": 0, "messages": 0, "last": ""}
    total = 0
    for s, n, last in _adm_sessions():
        owner = _adm_owner(s, names)
        row = stat[owner] if owner else guests
        row["sessions"] += 1
        row["messages"] += n
        row["last"] = max(row["last"], str(last or ""))
        total += n
    out = list(stat.values())
    if guests["sessions"]:
        out.append(guests)
    out.sort(key=lambda r: r["last"], reverse=True)
    return jsonify(users=out, total_messages=total)


@app.get("/api/admin/sessions")
def admin_sessions():
    bad = _adm_guard()
    if bad:
        return bad
    want = request.args.get("user", "")
    names = _adm_names()
    owner_want = "" if (want == "(guests)" and want not in names) else want
    mine = [(s, n, str(last or "")) for s, n, last in _adm_sessions() if _adm_owner(s, names) == owner_want]
    mine.sort(key=lambda x: x[2], reverse=True)
    out = []
    for s, n, last in mine[:200]:
        r = _adm_q("SELECT content FROM messages WHERE session=? AND role='user' ORDER BY id LIMIT 1",
                   "SELECT content FROM oq_messages WHERE session=%s AND role='user' ORDER BY id LIMIT 1", (s,))
        out.append({"session": s, "count": n, "last": last, "preview": ((r[0][0] if r else "") or "")[:90]})
    return jsonify(sessions=out)


@app.get("/api/admin/messages")
def admin_messages():
    bad = _adm_guard()
    if bad:
        return bad
    s = request.args.get("session", "")[:600]
    if not s:
        return jsonify(error="No chat chosen."), 400
    rows = _adm_q("SELECT role, content, ts FROM messages WHERE session=? ORDER BY id LIMIT 500",
                  "SELECT role, content, ts FROM oq_messages WHERE session=%s ORDER BY id LIMIT 500", (s,))
    return jsonify(messages=[{"role": r[0], "content": (r[1] or "")[:8000], "ts": str(r[2] or "")} for r in rows])


ADMIN_PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Deep OQ Admin</title>
<style>
:root{--bg:#04050d;--glass:rgba(18,20,42,.72);--line:rgba(255,255,255,.10);--text:#eef0ff;--muted:#8e94b8;--a:#7c5cff;--b:#00d4ff;--c:#ff4fd8;--ok:#33e6a1}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;overflow:hidden}
.aurora{position:fixed;inset:0;z-index:0;opacity:.8;pointer-events:none;background:radial-gradient(40% 35% at 15% 20%,rgba(124,92,255,.5),transparent 70%),radial-gradient(35% 40% at 85% 15%,rgba(0,212,255,.35),transparent 70%),radial-gradient(40% 40% at 75% 90%,rgba(255,79,216,.28),transparent 70%)}
.app{position:relative;z-index:1;display:flex;flex-direction:column;height:100%;padding:16px clamp(12px,2vw,28px);gap:14px;min-height:0}
header{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.logo{width:40px;height:40px;border-radius:13px;background:linear-gradient(135deg,var(--a),var(--b),var(--c));display:grid;place-items:center;font-weight:800;font-size:18px;box-shadow:0 8px 24px rgba(124,92,255,.4)}
h1{margin:0;font-size:21px;letter-spacing:-.4px;flex:1;min-width:160px}
h1 span{background:linear-gradient(90deg,var(--b),var(--a),var(--c));-webkit-background-clip:text;background-clip:text;color:transparent}
h1 small{display:block;font-size:12px;font-weight:400;color:var(--muted);letter-spacing:0}
.live{font-size:12px;color:var(--ok);display:flex;align-items:center;gap:6px}
.live i{width:8px;height:8px;border-radius:50%;background:var(--ok);box-shadow:0 0 10px var(--ok)}
.btn{color:var(--text);background:rgba(255,255,255,.07);border:1px solid var(--line);border-radius:12px;padding:8px 14px;cursor:pointer;font:inherit;font-size:14px;text-decoration:none;transition:.2s}
.btn:hover{background:rgba(255,255,255,.15);transform:translateY(-1px)}
.btn.pri{background:linear-gradient(90deg,var(--a),var(--b));border:0;font-weight:600}
.card{background:var(--glass);border:1px solid var(--line);border-radius:20px;-webkit-backdrop-filter:blur(14px);backdrop-filter:blur(14px)}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.stat{padding:14px 18px}
.stat b{display:block;font-size:30px;line-height:1.1;letter-spacing:-1px;background:linear-gradient(90deg,var(--b),var(--a));-webkit-background-clip:text;background-clip:text;color:transparent}
.stat span{color:var(--muted);font-size:13px}
.panes{flex:1;min-height:0;display:grid;grid-template-columns:320px 330px minmax(0,1fr);gap:12px}
.pane{display:flex;flex-direction:column;min-height:0;overflow:hidden}
.ph{display:flex;align-items:center;gap:8px;padding:14px 16px 10px}
.ph h2{font-size:12px;letter-spacing:1.3px;text-transform:uppercase;color:var(--muted);margin:0;flex:1}
.tools{display:flex;gap:8px;padding:0 12px 10px}
.in{flex:1;min-width:0;padding:9px 12px;border-radius:12px;border:1px solid var(--line);background:rgba(255,255,255,.05);color:var(--text);font:inherit;font-size:14px;outline:none}
.in:focus{border-color:var(--a);box-shadow:0 0 0 3px rgba(124,92,255,.22)}
select.in{flex:0 0 auto;max-width:120px;cursor:pointer}
select.in option{background:#12142a}
.scroll{overflow:auto;padding:0 10px 10px;flex:1;min-height:0}
.item{display:flex;gap:12px;align-items:center;width:100%;text-align:left;background:none;border:1px solid transparent;color:var(--text);border-radius:14px;padding:10px;margin-bottom:6px;cursor:pointer;font:inherit;transition:.15s}
.item:hover{background:rgba(255,255,255,.06)}
.item.on{background:linear-gradient(135deg,rgba(124,92,255,.28),rgba(0,212,255,.14));border-color:rgba(124,92,255,.7)}
.item .t{min-width:0;flex:1}
.item b{display:block;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.item small{display:block;color:var(--muted);font-size:12px;margin-top:2px}
.av{flex:none;width:40px;height:40px;border-radius:50%;display:grid;place-items:center;font-weight:700;font-size:15px;color:#fff}
.tag{flex:none;font-size:10.5px;letter-spacing:.5px;text-transform:uppercase;padding:3px 8px;border-radius:99px;border:1px solid var(--line);color:var(--muted)}
.tag.g{color:var(--b);border-color:rgba(0,212,255,.4)}
.tag.p{color:var(--c);border-color:rgba(255,79,216,.4)}
.empty{display:grid;place-items:center;text-align:center;color:var(--muted);padding:40px 16px;gap:8px}
.empty i{font-style:normal;font-size:34px;opacity:.7}
.mh{padding:0 16px 10px;color:var(--muted);font-size:13px;display:flex;align-items:center;gap:10px;min-height:20px}
.mh b{color:var(--text)}
.row{display:flex;gap:10px;margin:0 0 14px;align-items:flex-end}
.row.user{flex-direction:row-reverse}
.row .av{width:30px;height:30px;font-size:12px}
.bub{max-width:min(80%,680px);padding:10px 14px;border-radius:18px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.55}
.row.user .bub{background:linear-gradient(135deg,var(--a),#4b54f0);border-bottom-right-radius:6px}
.row.assistant .bub{background:rgba(255,255,255,.07);border:1px solid var(--line);border-bottom-left-radius:6px}
.bub time{display:block;font-size:11px;opacity:.65;margin-top:5px}
.back-m{display:none}
@media(max-width:900px){
  body{overflow:auto}
  .app{height:auto;min-height:100%}
  .cards{grid-template-columns:repeat(2,1fr)}
  .panes{grid-template-columns:1fr;height:78vh}
  .pane{display:none}
  body[data-pane=people] #p1,body[data-pane=chats] #p2,body[data-pane=msgs] #p3{display:flex}
  .back-m{display:inline-block}
}
</style></head><body data-pane="people">
<div class="aurora"></div>
<div class="app">
  <header>
    <div class="logo">O</div>
    <h1><span>Deep OQ</span> Admin<small>People, chats and messages (read only)</small></h1>
    <span class="live"><i></i>Live</span>
    <a class="btn" href="/">Back to app</a>
    <button class="btn pri" id="ref" type="button">Refresh</button>
  </header>
  <div class="cards">
    <div class="card stat"><b id="s-people">0</b><span>People</span></div>
    <div class="card stat"><b id="s-chats">0</b><span>Chats</span></div>
    <div class="card stat"><b id="s-msgs">0</b><span>Messages</span></div>
    <div class="card stat"><b id="s-active">0</b><span>Active today</span></div>
  </div>
  <div class="panes">
    <section class="card pane" id="p1">
      <div class="ph"><h2>People</h2></div>
      <div class="tools"><input class="in" id="q" placeholder="Search people" aria-label="Search people">
        <select class="in" id="sort" aria-label="Sort"><option value="recent">Recent</option><option value="messages">Most messages</option><option value="name">A to Z</option></select></div>
      <div class="scroll" id="users"></div>
    </section>
    <section class="card pane" id="p2">
      <div class="ph"><button class="btn back-m" id="b2" type="button">Back</button><h2>Chats</h2></div>
      <div class="scroll" id="sess"></div>
    </section>
    <section class="card pane" id="p3">
      <div class="ph"><button class="btn back-m" id="b3" type="button">Back</button><h2>Messages</h2>
        <button class="btn" id="copy" type="button">Copy</button><button class="btn" id="dl" type="button">Download</button></div>
      <div class="mh" id="mh"></div>
      <div class="scroll" id="msgs"></div>
    </section>
  </div>
</div>
<script>
const $=s=>document.querySelector(s);
const el=(t,c,x)=>{const e=document.createElement(t);if(c)e.className=c;if(x!==undefined)e.textContent=x;return e};
async function api(u){
  const go=()=>fetch(u,{headers:{'X-Access-Key':localStorage.ak||''}});
  let r=await go();
  if(r.status===401){localStorage.ak=prompt('Enter the access key')||'';r=await go()}
  const j=await r.json().catch(()=>({}));
  if(!r.ok)throw new Error(j.error||'Request failed');
  return j;
}
function toDate(t){
  if(!t)return null;
  let s=String(t).replace(' ','T');
  if(!/([zZ]|[+-]\d\d:?\d\d)$/.test(s))s+='Z';
  const d=new Date(s);return isNaN(d)?null:d;
}
function ago(t){
  const d=toDate(t);if(!d)return 'never';
  const s=Math.max(0,(Date.now()-d)/1000);
  if(s<60)return 'just now';
  if(s<3600)return Math.floor(s/60)+' min ago';
  if(s<86400)return Math.floor(s/3600)+' h ago';
  if(s<604800)return Math.floor(s/86400)+' d ago';
  return d.toLocaleDateString();
}
function stamp(t){
  const d=toDate(t);
  return d?d.toLocaleString([],{day:'numeric',month:'short',hour:'numeric',minute:'2-digit'}):'';
}
function who(n){
  if(n==='(guests)')return {name:'Guests',kind:'Guest'};
  if(n.startsWith('google:'))return {name:n.slice(7),kind:'Google'};
  return {name:n,kind:'Password'};
}
function avatar(name){
  const a=el('div','av',(name.replace(/[^a-z0-9]/gi,'')[0]||'?').toUpperCase());
  let h=0;for(const ch of name)h=(h*31+ch.charCodeAt(0))%360;
  a.style.background='linear-gradient(135deg,hsl('+h+',80%,58%),hsl('+((h+50)%360)+',80%,46%))';
  return a;
}
function empty(box,icon,text){
  const d=el('div','empty');d.append(el('i','',icon),el('span','',text));
  box.replaceChildren(d);
}
const pane=p=>{document.body.dataset.pane=p};
let users=[],total=0,curUser=null,curSess=null,curMsgs=[];

function drawStats(){
  const today=new Date().toDateString();
  $('#s-people').textContent=users.length;
  $('#s-chats').textContent=users.reduce((a,u)=>a+u.sessions,0);
  $('#s-msgs').textContent=total;
  $('#s-active').textContent=users.filter(u=>{const d=toDate(u.last);return d&&d.toDateString()===today}).length;
}
async function loadUsers(){
  try{
    const j=await api('/api/admin/users');
    users=j.users;total=j.total_messages;
    drawStats();drawUsers();
  }catch(e){empty($('#users'),'!',e.message)}
}
function drawUsers(){
  const f=$('#q').value.trim().toLowerCase(),box=$('#users'),mode=$('#sort').value;
  let list=users.filter(u=>who(u.username).name.toLowerCase().includes(f));
  if(mode==='messages')list.sort((a,b)=>b.messages-a.messages);
  else if(mode==='name')list.sort((a,b)=>who(a.username).name.localeCompare(who(b.username).name));
  else list.sort((a,b)=>String(b.last).localeCompare(String(a.last)));
  box.replaceChildren();
  if(!list.length){empty(box,'?','No one found.');return}
  list.forEach(u=>{
    const w=who(u.username);
    const b=el('button','item'+(u.username===curUser?' on':''));b.type='button';
    const t=el('div','t');
    t.append(el('b','',w.name),el('small','',u.messages+' messages \u00b7 '+u.sessions+' chats \u00b7 '+ago(u.last)));
    b.append(avatar(w.name),t,el('span','tag '+(w.kind==='Google'?'g':w.kind==='Password'?'p':''),w.kind));
    b.onclick=()=>{curUser=u.username;curSess=null;curMsgs=[];drawUsers();loadSessions();empty($('#msgs'),'\u2709','Choose a chat.');$('#mh').replaceChildren();pane('chats')};
    box.append(b);
  });
}
async function loadSessions(){
  const box=$('#sess');empty(box,'\u2026','Loading...');
  try{
    const j=await api('/api/admin/sessions?user='+encodeURIComponent(curUser));
    if(!j.sessions.length){empty(box,'\u2709','No chats yet.');return}
    box.replaceChildren();
    j.sessions.forEach(s=>{
      const b=el('button','item'+(s.session===curSess?' on':''));b.type='button';
      const t=el('div','t');
      t.append(el('b','',s.preview||'(no text)'),el('small','',s.count+' messages \u00b7 '+ago(s.last)));
      b.append(t);
      b.onclick=()=>{curSess=s.session;box.querySelectorAll('.item').forEach(x=>x.classList.remove('on'));b.classList.add('on');loadMsgs();pane('msgs')};
      box.append(b);
    });
  }catch(e){empty(box,'!',e.message)}
}
async function loadMsgs(){
  const box=$('#msgs');empty(box,'\u2026','Loading...');
  try{
    const j=await api('/api/admin/messages?session='+encodeURIComponent(curSess));
    curMsgs=j.messages;
    if(!curMsgs.length){empty(box,'\u2709','Empty chat.');return}
    const w=who(curUser);
    const mh=$('#mh');mh.replaceChildren();
    const nm=el('b','',w.name);mh.append(nm,el('span','',curMsgs.length+' messages'));
    box.replaceChildren();
    curMsgs.forEach(m=>{
      const isU=m.role==='user';
      const row=el('div','row '+(isU?'user':'assistant'));
      const bub=el('div','bub',m.content);
      bub.append(el('time','',stamp(m.ts)));
      row.append(avatar(isU?w.name:'Deep OQ'),bub);
      box.append(row);
    });
    box.scrollTop=0;
  }catch(e){empty(box,'!',e.message)}
}
function chatText(){
  const w=who(curUser).name;
  return curMsgs.map(m=>'['+stamp(m.ts)+'] '+(m.role==='user'?w:'Deep OQ')+':\n'+m.content).join('\n\n');
}
$('#copy').onclick=()=>{
  if(!curMsgs.length)return;
  navigator.clipboard.writeText(chatText()).then(()=>{const b=$('#copy');b.textContent='Copied';setTimeout(()=>b.textContent='Copy',1400)});
};
$('#dl').onclick=()=>{
  if(!curMsgs.length)return;
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([chatText()],{type:'text/plain'}));
  a.download='chat-'+who(curUser).name.replace(/[^a-z0-9]+/gi,'_')+'.txt';
  document.body.append(a);a.click();a.remove();
};
$('#q').addEventListener('input',drawUsers);
$('#sort').addEventListener('change',drawUsers);
$('#b2').onclick=()=>pane('people');
$('#b3').onclick=()=>pane('chats');
$('#ref').onclick=()=>{loadUsers();if(curUser!==null)loadSessions();if(curSess)loadMsgs()};
empty($('#sess'),'\u263a','Choose a person.');
empty($('#msgs'),'\u2709','Choose a chat.');
loadUsers();
</script>
</body></html>
"""


'''

# 1. admin routes and page
edit('''@app.get("/stats")''', ADMIN_CODE + '''@app.get("/stats")''')

# 2. an Admin button on the main page, only for the owner
edit('''    pill = LOGOUT_PILL % _html.escape(login_session.get("user", ""))''',
     '''    pill = LOGOUT_PILL % _html.escape(login_session.get("user", ""))
    if is_admin():
        pill += ('<a href="/admin" style="position:fixed;left:12px;bottom:48px;z-index:99999;padding:7px 13px;'
                 'border-radius:99px;font:600 12px system-ui,sans-serif;color:#04122b;background:#33e6a1;'
                 'text-decoration:none">Admin</a>')''')

# 3. honest notice on the login page
edit('''<div class="ok" id="ok">''',
     '''<p class="sub" style="font-size:12px;text-align:center;margin:14px 0 0;min-height:0">Chats are saved to your account and the site owner can access them.</p>

      <div class="ok" id="ok">''')


START = 'ADMIN_PAGE = r"""'
END = '</body></html>\n"""'


def main():
    if not os.path.exists(TARGET):
        sys.exit("Could not find %s. Put this script in the same folder as deep_oq.py." % TARGET)
    with open(TARGET, "r", encoding="utf-8") as fh:
        src = fh.read()
    if '"/api/admin/users"' in src:
        # the admin screen is already there: only upgrade the look of the page
        if 'id="s-people"' in src:
            sys.exit("The admin screen with the new design is already in this file. Nothing to do.")
        if src.count(START) != 1:
            sys.exit("Could not find the admin page in this file. Nothing was changed.")
        s = src.index(START)
        e = src.find(END, s)
        if e == -1:
            sys.exit("The admin page looks different from the one I expected. Nothing was changed.")
        block = ADMIN_CODE[ADMIN_CODE.index(START):ADMIN_CODE.index(END) + len(END)]
        src = src[:s] + block + src[e + len(END):]
        msg = "Done. Admin page upgraded to the new design in %s (backup: %s.bak2)." % (TARGET, TARGET)
    else:
        for i, (old, new) in enumerate(EDITS, 1):
            n = src.count(old)
            if n != 1:
                sys.exit("Step %d: found %d matching places (expected 1). Nothing was changed. "
                         "Your file may be different from the one I saw." % (i, n))
            src = src.replace(old, new, 1)
        msg = "Done. Admin screen added to %s (backup: %s.bak2)." % (TARGET, TARGET)
    shutil.copyfile(TARGET, TARGET + ".bak2")
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(src)
    print(msg)
    print("Next: upload deep_oq.py to GitHub, and check ADMIN_EMAILS on Render.")


if __name__ == "__main__":
    main()
