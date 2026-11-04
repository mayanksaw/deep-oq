"""
Adds an owner-only ADMIN screen to Deep OQ (page: /admin).
You can see every person, their chats and messages. Read only.

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
:root{--bg:#04050d;--line:rgba(255,255,255,.12);--text:#eef0ff;--muted:#8e94b8;--a:#7c5cff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px system-ui,-apple-system,"Segoe UI",sans-serif;height:100vh;display:flex;flex-direction:column}
header{display:flex;gap:12px;align-items:center;padding:12px 18px;border-bottom:1px solid var(--line)}
header h1{font-size:18px;margin:0;flex:1}
a.back,button.ref{color:var(--text);background:rgba(255,255,255,.07);border:1px solid var(--line);border-radius:10px;padding:8px 14px;text-decoration:none;cursor:pointer;font:inherit}
.stats{color:var(--muted);font-size:13px}
main{flex:1;display:grid;grid-template-columns:300px 320px 1fr;min-height:0}
@media(max-width:900px){main{grid-template-columns:1fr;grid-auto-rows:minmax(240px,1fr);overflow:auto}}
.col{border-right:1px solid var(--line);overflow:auto;min-height:0;padding:10px}
.col h2{font-size:12px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);margin:4px 6px 10px}
.item{display:block;width:100%;text-align:left;background:none;border:1px solid transparent;color:var(--text);border-radius:12px;padding:10px 12px;margin-bottom:6px;cursor:pointer;font:inherit}
.item:hover{background:rgba(255,255,255,.06)}
.item.on{background:rgba(124,92,255,.22);border-color:var(--a)}
.item b{font-weight:600;overflow-wrap:anywhere}
.item small{display:block;color:var(--muted);font-size:12px;margin-top:3px}
input.s{width:100%;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:rgba(255,255,255,.05);color:var(--text);margin-bottom:10px;font:inherit}
.msg{margin:0 0 12px;padding:10px 14px;border-radius:14px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5;max-width:92%}
.msg.user{background:#2a3192;margin-left:auto}
.msg.assistant{background:rgba(255,255,255,.07)}
.msg time{display:block;color:var(--muted);font-size:11px;margin-top:6px}
.empty{color:var(--muted);padding:10px 6px;margin:0}
</style></head><body>
<header><a class="back" href="/">&larr; Deep OQ</a><h1>Admin</h1><span class="stats" id="stats"></span><button class="ref" id="ref" type="button">Refresh</button></header>
<main>
<section class="col"><h2>People</h2><input class="s" id="q" placeholder="Search people" aria-label="Search people"><div id="users"></div></section>
<section class="col"><h2>Chats</h2><div id="sess"><p class="empty">Choose a person.</p></div></section>
<section class="col" style="border-right:0"><h2>Messages</h2><div id="msgs"><p class="empty">Choose a chat.</p></div></section>
</main>
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
const label=n=>n.startsWith('google:')?n.slice(7)+' (Google)':n;
const when=t=>t?t.replace('T',' ').slice(0,16):'never';
let users=[],curUser=null,curSess=null;
function note(box,t){box.replaceChildren(el('p','empty',t))}
async function loadUsers(){
  try{
    const j=await api('/api/admin/users');
    users=j.users;
    $('#stats').textContent=j.users.length+' people, '+j.total_messages+' messages';
    drawUsers();
  }catch(e){note($('#users'),e.message)}
}
function drawUsers(){
  const f=$('#q').value.trim().toLowerCase(),box=$('#users');box.replaceChildren();
  const list=users.filter(u=>label(u.username).toLowerCase().includes(f));
  if(!list.length){note(box,'No one found.');return}
  list.forEach(u=>{
    const b=el('button','item'+(u.username===curUser?' on':''));b.type='button';
    b.append(el('b','',label(u.username)),el('small','',u.messages+' messages, '+u.sessions+' chats, last '+when(u.last)));
    b.onclick=()=>{curUser=u.username;curSess=null;drawUsers();loadSessions();note($('#msgs'),'Choose a chat.')};
    box.append(b);
  });
}
async function loadSessions(){
  const box=$('#sess');note(box,'Loading...');
  try{
    const j=await api('/api/admin/sessions?user='+encodeURIComponent(curUser));
    box.replaceChildren();
    if(!j.sessions.length){note(box,'No chats.');return}
    j.sessions.forEach(s=>{
      const b=el('button','item'+(s.session===curSess?' on':''));b.type='button';
      b.append(el('b','',s.preview||'(no text)'),el('small','',s.count+' messages, '+when(s.last)));
      b.onclick=()=>{curSess=s.session;box.querySelectorAll('.item').forEach(x=>x.classList.remove('on'));b.classList.add('on');loadMsgs()};
      box.append(b);
    });
  }catch(e){note(box,e.message)}
}
async function loadMsgs(){
  const box=$('#msgs');note(box,'Loading...');
  try{
    const j=await api('/api/admin/messages?session='+encodeURIComponent(curSess));
    box.replaceChildren();
    if(!j.messages.length){note(box,'Empty chat.');return}
    j.messages.forEach(m=>{
      const d=el('div','msg '+(m.role==='user'?'user':'assistant'),m.content);
      d.append(el('time','',when(m.ts)));
      box.append(d);
    });
    box.parentElement.scrollTop=0;
  }catch(e){note(box,e.message)}
}
$('#q').addEventListener('input',drawUsers);
$('#ref').onclick=()=>{loadUsers();if(curUser!==null)loadSessions();if(curSess)loadMsgs()};
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


def main():
    if not os.path.exists(TARGET):
        sys.exit("Could not find %s. Put this script in the same folder as deep_oq.py." % TARGET)
    with open(TARGET, "r", encoding="utf-8") as fh:
        src = fh.read()
    if '"/api/admin/users"' in src:
        sys.exit("The admin screen is already in this file. Nothing to do.")
    for i, (old, new) in enumerate(EDITS, 1):
        n = src.count(old)
        if n != 1:
            sys.exit("Step %d: found %d matching places (expected 1). Nothing was changed. "
                     "Your file may be different from the one I saw." % (i, n))
        src = src.replace(old, new, 1)
    shutil.copyfile(TARGET, TARGET + ".bak2")
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(src)
    print("Done. Admin screen added to %s (backup: %s.bak2)." % (TARGET, TARGET))
    print("Next: upload deep_oq.py to GitHub, and check ADMIN_EMAILS on Render.")


if __name__ == "__main__":
    main()
