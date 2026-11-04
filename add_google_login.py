"""
Adds "Continue with Google" to Deep OQ.

HOW TO USE
  1. Put this file in the same folder as deep_oq.py
  2. Run:  python add_google_login.py
  3. Upload the updated deep_oq.py to GitHub
  4. On your host (Render / Railway) set the environment variable GOOGLE_CLIENT_ID

A backup of your old file is saved as deep_oq.py.bak
If any step cannot find its place in your file, nothing is changed.
"""
import os
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else "deep_oq.py"

EDITS = []


def edit(old, new):
    EDITS.append((old, new))


# 1. setting
edit('''ACCESS_KEY = os.environ.get("ACCESS_KEY", "")  # set on Render to lock the site''',
     '''ACCESS_KEY = os.environ.get("ACCESS_KEY", "")  # set on Render to lock the site
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()  # "Continue with Google" needs this''')

# 2. allow the Google route without a login
edit('''p.startswith(("/api/login", "/api/register", "/vr"))''',
     '''p.startswith(("/api/login", "/api/register", "/api/google", "/vr"))''')

# 3. nobody can register a fake Google account name
edit('''    if not 1 <= len(name) <= 40 or any(ord(ch) < 32 for ch in name):''',
     '''    if name.startswith("google:"):
        return jsonify(error="That username is not allowed."), 400
    if not 1 <= len(name) <= 40 or any(ord(ch) < 32 for ch in name):''')

# 4. the Google sign-in route
edit('''@app.get("/logout")''',
     '''@app.post("/api/google")
def api_google():
    if not GOOGLE_CLIENT_ID:
        return jsonify(error="Google sign-in is not set up yet."), 503
    if _too_many_fails():
        return jsonify(error="Too many tries. Wait 10 minutes and try again."), 429
    d = request.get_json(silent=True) or {}
    token = str(d.get("credential", ""))[:4000]
    if not token:
        return jsonify(error="No Google token received."), 400
    try:
        info = _get_json("https://oauth2.googleapis.com/tokeninfo?id_token=" + urllib.parse.quote(token))
    except Exception:
        _login_fails.setdefault(_client_ip(), []).append(time.time())
        return jsonify(error="Google could not verify this sign-in."), 401
    good = (info.get("aud") == GOOGLE_CLIENT_ID
            and info.get("iss") in ("accounts.google.com", "https://accounts.google.com")
            and str(info.get("email_verified")).lower() == "true" and info.get("email"))
    if not good:
        _login_fails.setdefault(_client_ip(), []).append(time.time())
        return jsonify(error="This Google account could not be verified."), 401
    name = "google:" + info["email"].strip().lower()
    try:
        if not user_get(name):
            user_add(name, generate_password_hash(secrets.token_urlsafe(32)))
    except Exception as e:
        print("google login failed:", e)
        return jsonify(error="Could not sign in. Try again."), 500
    _start_login(name, d.get("remember", True))
    return jsonify(username=info.get("name") or info["email"])


@app.get("/logout")''')

# 5. give the Client ID to the login page
edit('''    return LOGIN_PAGE, 200, {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-cache"}''',
     '''    return (LOGIN_PAGE.replace("__GOOGLE_CLIENT_ID__", GOOGLE_CLIENT_ID), 200,
            {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-cache"})''')

# 6. Google script in the login page head
edit('''</head>
<body>
<div class="aurora"></div>''',
     '''<script src="https://accounts.google.com/gsi/client" async defer></script>
</head>
<body>
<div class="aurora"></div>''')

# 7. the Google button
edit('''<p class="switch" id="switch">''',
     '''<div id="gwrap" hidden>
        <div class="divider">or</div>
        <div id="gbtn" style="display:flex;justify-content:center"></div>
      </div>

      <p class="switch" id="switch">''')

# 8. the Google JavaScript
edit('''/* ===== Form logic ===== */''',
     '''/* ===== Google sign-in ===== */
const GID = "__GOOGLE_CLIENT_ID__";
async function onGoogle(resp){
  document.getElementById("msg").textContent = "";
  try{
    const r = await fetch("/api/google", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({credential: resp.credential, remember: document.getElementById("remember").checked})});
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || "Google sign-in failed.");
    document.getElementById("okTitle").textContent = "Welcome, " + (j.username || "") + "!";
    document.getElementById("ok").classList.add("show");
    setTimeout(() => location.href = AFTER_LOGIN_URL, 1800);
  } catch(err){ fail(err.message); }
}
function initGoogle(){
  if (!GID || !window.google || !google.accounts) return;
  google.accounts.id.initialize({client_id: GID, callback: onGoogle});
  google.accounts.id.renderButton(document.getElementById("gbtn"), {theme:"filled_black", size:"large",
    shape:"pill", text:"continue_with", logo_alignment:"left", width:300});
  document.getElementById("gwrap").hidden = false;
}
window.addEventListener("load", initGoogle);

/* ===== Form logic ===== */''')


def main():
    if not os.path.exists(TARGET):
        sys.exit("Could not find %s. Put this script in the same folder as deep_oq.py." % TARGET)
    with open(TARGET, "r", encoding="utf-8") as fh:
        src = fh.read()
    if '"/api/google"' in src:
        sys.exit("Google sign-in is already in this file. Nothing to do.")
    for i, (old, new) in enumerate(EDITS, 1):
        n = src.count(old)
        if n != 1:
            sys.exit("Step %d: found %d matching places (expected 1). Nothing was changed. "
                     "Your file may be different from the one I saw." % (i, n))
        src = src.replace(old, new, 1)
    shutil.copyfile(TARGET, TARGET + ".bak")
    with open(TARGET, "w", encoding="utf-8") as fh:
        fh.write(src)
    print("Done. Google sign-in added to %s (backup: %s.bak)." % (TARGET, TARGET))
    print("Next: set the GOOGLE_CLIENT_ID environment variable, then upload deep_oq.py to GitHub.")


import shutil

if __name__ == "__main__":
    main()
