"""
Fixes the mobile bug in deep_oq.py: closing the sidebar also closed the Memory panel.
Run it in the same folder as deep_oq.py:   python patch_memory_fix.py
A backup is saved as deep_oq.py.bak first.
"""
import shutil
import sys

FILE = sys.argv[1] if len(sys.argv) > 1 else "deep_oq.py"

OLD1 = "function memOpen(o){memDrawer.hidden=!o;if(o)loadMem();else $('#memBtn').focus()}"
NEW1 = "function memOpen(o){memDrawer.hidden=!o;if(o){sbClose();loadMem()}else if(innerWidth>860)$('#memBtn').focus()}"

OLD2 = "document.addEventListener('click',e=>{if(!memDrawer.hidden&&!e.target.closest('#memDrawer,#memBtn'))memOpen(false)});"
NEW2 = "document.addEventListener('click',e=>{if(!memDrawer.hidden&&!e.target.closest('#memDrawer,#memBtn,#sbShade,#sbToggle,header.top'))memOpen(false)});"

OLD3 = ".sbtoggle,.sbshade{display:none}"
NEW3 = "#memDrawer{z-index:9600}\n.sbtoggle,.sbshade{display:none}"

with open(FILE, "r", encoding="utf-8") as f:
    src = f.read()

if "#memDrawer{z-index:9600}" in src:
    sys.exit("Already patched. Nothing to do.")

for i, (old, new) in enumerate(((OLD1, NEW1), (OLD2, NEW2), (OLD3, NEW3)), 1):
    if src.count(old) != 1:
        sys.exit("Could not find change %d exactly once in %s. Nothing was changed." % (i, FILE))
    src = src.replace(old, new)

shutil.copyfile(FILE, FILE + ".bak")
with open(FILE, "w", encoding="utf-8") as f:
    f.write(src)
print("Patched %s (backup: %s.bak). Redeploy to Render." % (FILE, FILE))
