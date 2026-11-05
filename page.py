"""
Applies the dark sidebar theme to deep_oq.py.

  1. Put this file in the same folder as deep_oq.py
  2. Run:  python apply_theme.py
  3. Restart your app. A backup is saved as deep_oq.py.bak

Safe to run twice: it skips if the theme is already applied.
"""
import os
import shutil
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else "deep_oq.py"
MARK = "/* ===== Deep OQ dark sidebar theme ===== */"

CSS = r'''
/* ===== Deep OQ dark sidebar theme ===== */
:root,:root:root{
  --bg:#0f1034;--surface:#171a52;--ink:#e9ebff;--ink-2:#8d93c9;--line:#2a3070;
  --accent:#8a5a45;--accent-ink:#fff;--river:#14c8b0;
  --bubble:#2a3192;--bubble-ink:#fff;--code-bg:#090c26;--code-line:#222962;--err:#ff8d78;
  --side:#141a4a;
}
body{grid-template-columns:210px minmax(0,1fr)!important;grid-template-rows:minmax(0,1fr)!important}
header.top{grid-column:1;grid-row:1;flex-direction:column;flex-wrap:nowrap;align-items:stretch;gap:6px;
  background:var(--side);padding:16px 10px;overflow-y:auto}
.views{grid-column:2;grid-row:1}
.brand{font-size:17px;padding:0 8px 10px;order:0}
header.top .brand svg{width:20px;height:20px}
#chatTools,body:not([data-view=chat]) #chatTools{display:contents!important}
#newBtn{order:1;background:#dcdcf0;color:#1a1d4a;border:0;border-radius:10px;padding:10px 14px;justify-content:flex-start;font-weight:600}
#newBtn:hover{background:#fff}
.tabs{order:2;flex-direction:column;margin:6px 0 0;padding:0;border:0;border-radius:0;gap:2px}
#kbBtn{order:3}#histBtn{order:4}
.tab{text-align:left;padding:10px 12px;border-radius:10px;font-size:14.5px;color:var(--ink);width:100%}
.tab:hover{background:rgba(255,255,255,.07)}
.tab[aria-current=page]{background:#f2f2ff;color:#14173f}
a.tab[href="/vr"]{margin-top:2px;color:#04122b!important}
#kbBtn,#histBtn{border:0;border-radius:10px;justify-content:space-between;padding:10px 12px;width:100%}
#kbBtn:hover,#histBtn:hover{background:rgba(255,255,255,.07)}
.badge{background:#fff;color:#14173f}
.acct{order:9;margin-top:auto!important;flex-wrap:wrap;border-top:1px solid var(--line);padding-top:12px}
.acct .who{max-width:100%!important;flex-basis:100%}
.acct a{padding:5px 12px!important;font-size:12px!important}

/* greeting and centred composer on the empty screen */
body:not(.chatting) #v-chat.on{display:flex;flex-direction:column;justify-content:center;align-items:center;gap:6px}
body:not(.chatting) #scroll{flex:none;overflow:visible;width:100%}
body:not(.chatting) .composer{width:100%;max-width:640px}
.hero{padding:0;display:flex;justify-content:center;align-items:center;gap:14px}
.hero p{display:none}
.hero h1{margin:0;font-family:var(--read);font-weight:400;font-size:clamp(26px,3.4vw,38px);letter-spacing:0;order:1}
#heroMark{order:0}
#heroMark svg{width:30px;height:30px}
.field{background:#14185a;border-color:#2f6f86;box-shadow:0 0 0 2px rgba(20,200,176,.14)}
.field:focus-within{border-color:var(--river)}
.hint{font-size:11px}
.ideas{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;border:0;margin-top:18px}
.ideas li{border:0}
.ideas button{width:auto;padding:7px 14px;font-size:13px;border:1px solid var(--line);border-radius:999px;color:#cfd2f3}
.ideas button:hover{padding-left:14px;border-color:var(--river);color:#fff}
.chatting .ideas{display:none}
.chatting #scroll .col{max-width:760px}

@media (max-width:720px){
  body{grid-template-columns:1fr!important;grid-template-rows:auto minmax(0,1fr)!important}
  header.top{grid-column:1;grid-row:1;flex-direction:row;flex-wrap:wrap;overflow:visible;padding:10px 14px}
  .views{grid-column:1;grid-row:2}
  #chatTools,body:not([data-view=chat]) #chatTools{display:flex!important;margin-left:auto}
  .tabs{flex-direction:row;overflow-x:auto;width:100%;order:3}
  .tab{width:auto;white-space:nowrap}
  .acct{margin-top:0!important;border:0;padding:0}
  #newBtn,#kbBtn,#histBtn{width:auto}
}
'''

JS = r'''
/* greeting with the user's name, suggestion chips under the input box */
(function(){
  const h=new Date().getHours(),g=h<12?'Good morning':h<17?'Good afternoon':'Good evening';
  const w=document.querySelector('.acct .who');
  let n=w?w.textContent.trim().split(/[@\s]/)[0].replace(/\d+$/,''):'';
  n=n?n[0].toUpperCase()+n.slice(1):'';
  document.querySelector('.hero h1').textContent=n?g+', '+n:g;
  document.querySelector('.composer .col').appendChild(document.querySelector('.ideas'));
})();
'''

CSS_ANCHOR = "[hidden]{display:none!important}\n</style>"
JS_ANCHOR = "showView(location.hash.slice(1));\n</script>"


def main():
    if not os.path.exists(TARGET):
        sys.exit("Could not find " + TARGET + ". Put this script next to deep_oq.py.")
    with open(TARGET, "r", encoding="utf-8", newline="") as f:
        src = f.read().replace("\r\n", "\n")
    if MARK in src:
        sys.exit("The theme is already applied. Nothing changed.")
    for name, anchor in (("CSS", CSS_ANCHOR), ("JS", JS_ANCHOR)):
        n = src.count(anchor)
        if n != 1:
            sys.exit("Could not place the %s patch (found %d matching spots, expected 1). "
                     "Your file may differ from the one I saw. Nothing changed." % (name, n))
    shutil.copyfile(TARGET, TARGET + ".bak")
    src = src.replace(CSS_ANCHOR, "[hidden]{display:none!important}" + CSS + "</style>")
    src = src.replace(JS_ANCHOR, JS + "showView(location.hash.slice(1));\n</script>")
    with open(TARGET, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    print("Done. Theme applied. Backup saved as " + TARGET + ".bak")
    print("Restart the app and reload the page.")


if __name__ == "__main__":
    main()
