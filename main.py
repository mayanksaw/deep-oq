"""
Deep OQ + Debugger  (main.py)
Upload this next to your existing deep_oq.py. It loads Deep OQ unchanged and adds:
  - the /debug engine (static checks, optional sandboxed run, diagnose, fix, verify, retry with other models)
  - a "Debug" button and error boxes in the Code tab (injected automatically, no edits to deep_oq.py)

Render start command :  gunicorn main:app --workers 1 --threads 4 --timeout 300
Render build command :  pip install -r requirements.txt
Optional settings    :  DEBUG_RUN = (empty) never run code | admin = owner only | all = everyone (risky)
                        DEBUG_ROUNDS = fix attempts, default 3
Local                :  python main.py
"""
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile

from flask import jsonify, request

from deep_oq import app, chat, LANGS, denied, is_admin, EXAM_MODEL, EXAM_CHECK_MODEL

DEBUG_RUN = os.environ.get("DEBUG_RUN", "").strip().lower()
DEBUG_ROUNDS = max(1, min(5, int(os.environ.get("DEBUG_ROUNDS", "3") or 3)))
RUN_SECONDS = 8


def _extract_code(text):
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text or "", re.S)
    return max(blocks, key=len).strip("\n") if blocks else ""


def _after_code(text):
    """Text after the last closing fence (the CHANGES notes)."""
    parts = (text or "").rsplit("```", 1)
    return parts[1].strip() if len(parts) == 2 else ""


def _numbered(code, limit=900):
    lines = code.split("\n")[:limit]
    return "\n".join(f"{i + 1:>4}| {l}" for i, l in enumerate(lines))


class _FlakeReport:
    def __init__(self):
        self.out = []

    def unexpectedError(self, filename, msg):
        self.out.append(str(msg))

    def syntaxError(self, filename, msg, lineno, offset, text):
        self.out.append(f"SyntaxError on line {lineno}: {msg}")

    def flake(self, m):
        text = m.message % m.message_args
        if "imported but unused" in text or "assigned to but never used" in text:
            return  # style noise, not a bug
        self.out.append(f"line {m.lineno}: {text}")


def static_check(lang, code):
    """Returns a list of problems found without running anything. Empty list = nothing found."""
    probs = []
    if lang == "Python":
        try:
            ast.parse(code)
        except SyntaxError as e:
            return [f"SyntaxError on line {e.lineno}: {e.msg}"]
        try:
            from pyflakes.api import check as _flake
            rep = _FlakeReport()
            _flake(code, "code.py", rep)
            probs += rep.out[:15]
        except ImportError:
            pass
        except Exception as e:
            print("pyflakes failed:", e)
    elif lang == "JavaScript" and shutil.which("node"):
        work = tempfile.mkdtemp(prefix="oqchk_")
        try:
            path = os.path.join(work, "main.js")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(code)
            p = subprocess.run(["node", "--check", path], capture_output=True, text=True, timeout=15)
            if p.returncode != 0:
                probs.append((p.stderr or "JavaScript syntax error").strip()[-600:])
        except Exception as e:
            print("node check failed:", e)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    elif lang == "PHP" and shutil.which("php"):
        work = tempfile.mkdtemp(prefix="oqchk_")
        try:
            path = os.path.join(work, "main.php")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(code)
            p = subprocess.run(["php", "-l", path], capture_output=True, text=True, timeout=15)
            if p.returncode != 0:
                probs.append((p.stdout + p.stderr).strip()[-600:])
        except Exception as e:
            print("php check failed:", e)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return probs


def run_sandboxed(lang, code):
    """Runs Python or JavaScript with hard limits. Returns a dict, or None when running is not possible."""
    ext = {"Python": ".py", "JavaScript": ".js"}.get(lang)
    if not ext:
        return None
    exe = sys.executable if lang == "Python" else shutil.which("node")
    if not exe:
        return None
    work = tempfile.mkdtemp(prefix="oqdbg_")

    def limits():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (RUN_SECONDS, RUN_SECONDS))
            resource.setrlimit(resource.RLIMIT_FSIZE, (5 << 20, 5 << 20))
            if lang == "Python":  # node reserves a lot of virtual memory, so only cap Python
                resource.setrlimit(resource.RLIMIT_AS, (768 << 20, 768 << 20))
        except Exception:
            pass

    try:
        path = os.path.join(work, "main" + ext)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(code)
        env = {"PATH": os.environ.get("PATH", ""), "HOME": work, "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONIOENCODING": "utf-8"}
        p = subprocess.run([exe, path], input="", capture_output=True, text=True, timeout=RUN_SECONDS,
                           cwd=work, env=env, preexec_fn=limits if os.name == "posix" else None)
        return {"code": p.returncode, "out": p.stdout[-3000:], "err": p.stderr[-3000:], "timeout": False}
    except subprocess.TimeoutExpired:
        return {"code": -1, "out": "", "err": f"Stopped after {RUN_SECONDS} seconds.", "timeout": True}
    except Exception as e:
        print("sandbox run failed:", e)
        return None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _norm_out(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def _verify(lang, code, expected, can_run):
    """Returns (problems, how_checked). problems empty means the fix passed every check available."""
    problems = static_check(lang, code)
    how = ["static check passed"] if not problems else []
    ran = None
    if not problems and can_run:
        ran = run_sandboxed(lang, code)
        if ran is not None:
            if ran["timeout"]:
                how.append("run hit the time limit (inconclusive, may be a server or long loop)")
            elif ran["code"] != 0 and "EOFError" not in ran["err"]:
                problems.append("The program crashed when run:\n" + ran["err"][-900:])
            else:
                how.append("ran without errors")
                if expected and _norm_out(expected) not in _norm_out(ran["out"]) \
                        and _norm_out(ran["out"]) != _norm_out(expected):
                    problems.append("It ran, but the output did not match the expected output.\n"
                                    "EXPECTED:\n" + expected[:600] + "\nGOT:\n" + ran["out"][:600])
                elif expected:
                    how.append("output matched what you expected")
    return problems, how


DBG_SYS = ("You are Deep OQ's debugging engine, a principal engineer who has fixed every kind of bug. "
           "Find the REAL root cause, not just the symptom. Never invent library functions, flags or APIs. "
           "Never use tables, horizontal lines (---) or HTML tags. Be concise and exact.")


def debug_code(lang, code, error="", expected="", can_run=False):
    evidence = []
    base_static = static_check(lang, code)
    if base_static:
        evidence.append("AUTOMATIC STATIC CHECK FOUND:\n" + "\n".join(base_static[:15]))
    base_run = run_sandboxed(lang, code) if can_run else None
    if base_run:
        if base_run["timeout"]:
            evidence.append("RUNNING THE ORIGINAL CODE: " + base_run["err"] + " (possible infinite loop or hang)")
        elif base_run["code"] != 0:
            evidence.append("RUNNING THE ORIGINAL CODE FAILED:\n" + base_run["err"][-1500:])
        else:
            evidence.append("RUNNING THE ORIGINAL CODE: exit 0. Output:\n" + (base_run["out"][:800] or "(none)"))
    if error:
        evidence.append("ERROR OR SYMPTOM REPORTED BY THE USER:\n" + error)
    if expected:
        evidence.append("EXPECTED OUTPUT OR BEHAVIOUR:\n" + expected)
    ev = "\n\n".join(evidence) or ("No error text was given and the automatic checks found nothing, so hunt for "
                                   "logic bugs, wrong edge cases, off-by-one errors, bad types and unsafe code.")

    # ---- stage 1: diagnose
    diag = chat([{"role": "system", "content": DBG_SYS},
                 {"role": "user", "content":
                  f"LANGUAGE: {lang}\n\nCODE (with line numbers):\n{_numbered(code)}\n\nEVIDENCE:\n{ev}\n\n"
                  "Write exactly these parts:\nROOT CAUSE: 1 to 3 sentences.\n"
                  "ALL BUGS: a numbered list; each item has the line number, what is wrong, and why it breaks. "
                  "Include bugs that have not triggered yet.\n"
                  "MISSING: only if something you need is not in the code or evidence (a file, a version, the full "
                  "traceback), say exactly what; otherwise omit this part. Do not write code here."}],
                0.1, 2500, smart=True)
    diag = diag.replace("```", "")

    # ---- stages 2-4: fix, verify, retry with a different model each round
    prefers = [None, EXAM_CHECK_MODEL, EXAM_MODEL, None, EXAM_CHECK_MODEL]
    feedback, fixed, notes, problems, how = "", "", "", ["no attempt was made"], []
    rounds_used = 0
    for rnd in range(DEBUG_ROUNDS):
        rounds_used = rnd + 1
        retry = ("\n\nYOUR PREVIOUS FIX FAILED ITS CHECK. Read this failure carefully and fix the real cause:\n"
                 + feedback + "\n\nPREVIOUS FIX:\n```\n" + fixed + "\n```") if feedback else ""
        try:
            raw = chat([{"role": "system", "content": DBG_SYS},
                        {"role": "user", "content":
                         f"LANGUAGE: {lang}\n\nORIGINAL CODE:\n```\n{code}\n```\n\nDIAGNOSIS:\n{diag}\n\n"
                         f"EVIDENCE:\n{ev}{retry}\n\n"
                         "Write the COMPLETE corrected code in ONE fenced code block: no placeholders, no '...', "
                         "keep the same features, names and structure, change only what is needed, and fix every "
                         "bug in the diagnosis. After the block write 'CHANGES:' and at most 6 short bullets "
                         "(line, before, after, why), then one line 'TEST:' saying how to confirm it works."}],
                       0.15, 6144, smart=True, prefer=prefers[rnd % len(prefers)])
        except Exception as e:
            print("debug fix call failed:", e)
            feedback = "The model call failed: " + str(e)[:200]
            continue
        new = _extract_code(raw)
        if not new:
            feedback = "You did not return a fenced code block. Return the full corrected code in one block."
            continue
        fixed, notes = new, _after_code(raw)
        problems, how = _verify(lang, fixed, expected, can_run)
        if not problems:
            break
        feedback = "\n".join(problems)[:1800]

    # ---- report
    verified = bool(fixed) and not problems
    if verified:
        state = "**Verified:** " + ", ".join(how or ["no automatic check is available for this language"]) + "."
        if not how:
            state = ("**Not machine-verified:** no automatic checker exists for " + lang +
                     " on this server, so this fix is reviewed by the AI only. Run it and test it.")
    elif fixed:
        state = ("**Not fully verified.** This is the best fix after " + str(rounds_used) + " rounds, but it still "
                 "fails a check:\n\n" + "\n".join("- " + p.replace("\n", " ")[:300] for p in problems[:3]) +
                 "\n\nSend me the full error message, your " + lang + " version and any file or input the code "
                 "uses, and I will take it from there.")
    else:
        state = ("**I could not produce a working fix yet.** Send the complete error message (the whole "
                 "traceback), your " + lang + " version, and the input you used, then try again.")
    out = ["## Debug report", state, "### Root cause and bugs found", diag]
    if fixed:
        out += ["### Fixed code", "```" + lang.lower().split("/")[0] + "\n" + fixed + "\n```"]
        if notes:
            out += [notes]
    return "\n\n".join(out), verified, rounds_used


@app.post("/debug")
def debug_route():
    if denied():
        return jsonify(error="locked"), 401
    d = request.get_json(silent=True) or {}
    lang = d.get("lang", "Python")
    code = str(d.get("code", "")).strip()[:20000]
    error = str(d.get("error", "")).strip()[:6000]
    expected = str(d.get("expected", "")).strip()[:3000]
    if lang not in LANGS:
        return jsonify(error="Pick a language from the list."), 400
    if not code:
        return jsonify(error="Paste the code first."), 400
    can_run = DEBUG_RUN == "all" or (DEBUG_RUN == "admin" and is_admin())
    try:
        answer, verified, rounds = debug_code(lang, code, error, expected, can_run)
        return jsonify(answer=answer, verified=verified, rounds=rounds)
    except Exception as e:
        print("debug failed:", e)
        return jsonify(error=str(e)[:300]), 500


# ==========================================================================
# DEBUG TAB: injected into the home page, so deep_oq.py needs no changes
# ==========================================================================
DEBUG_UI = r"""<script id="oq-debug-ui">
(function(){
  if(typeof MODES==='undefined'||!document.querySelector('#cmodes'))return;
  MODES.debug={btn:'Debug code',task:'Anything else I should know? (optional)',ph:'It worked yesterday',code:'Your code'};
  const modes=document.querySelector('#cmodes'),b=document.createElement('button');
  b.type='button';b.dataset.m='debug';b.setAttribute('aria-pressed','false');b.textContent='Debug';
  b.onclick=()=>setMode('debug');modes.appendChild(b);

  const wrap=document.createElement('div');wrap.id='cdbgW';wrap.hidden=true;wrap.className='panel';
  wrap.innerHTML='<label class="lab">Error message or what goes wrong (paste the full traceback)'
    +'<textarea class="in mono" id="cerr" rows="4" spellcheck="false" style="min-height:90px"></textarea></label>'
    +'<label class="lab">Expected output (optional)'
    +'<textarea class="in mono" id="cexp" rows="2" spellcheck="false" style="min-height:60px"></textarea></label>';
  const runRow=document.querySelector('#crun').parentElement;runRow.parentElement.insertBefore(wrap,runRow);

  const old=setMode;
  setMode=function(m){old(m);wrap.hidden=m!=='debug';};

  document.addEventListener('click',async e=>{
    if(cmode!=='debug'||!e.target.closest('#crun'))return;
    e.stopImmediatePropagation();e.preventDefault();
    const out=document.querySelector('#cout'),btn=document.querySelector('#crun');
    const code=document.querySelector('#ccode').value;
    if(!code.trim()){out.innerHTML='<p class="note">Paste the code first.</p>';document.querySelector('#ccode').focus();return}
    const extra=document.querySelector('#ctask').value.trim();
    const body={lang:document.querySelector('#clang').value,code:code,
      error:document.querySelector('#cerr').value+(extra?'\n'+extra:''),expected:document.querySelector('#cexp').value};
    btn.disabled=true;document.querySelector('#crunbar').hidden=true;
    document.querySelector('#cframe').hidden=true;document.querySelector('#cconsole').hidden=true;
    out.innerHTML='<p class="muted">Debugging: checking, diagnosing, fixing and verifying. This can take a minute or two.</p>';
    try{
      const r=await api('/debug',body),j=await r.json().catch(()=>({}));
      if(!r.ok||j.error)throw new Error(j.error||'The server did not answer.');
      out.innerHTML='<div class="body">'+md(j.answer)+'</div>';
      const m=j.answer.match(/```[^\n]*\n([\s\S]*?)```/);
      lastCode=m?m[1]:'';lastLang=body.lang;
      if(lastCode&&(lastLang==='HTML/CSS'||lastLang==='JavaScript')){
        document.querySelector('#cpreview').textContent=lastLang==='HTML/CSS'?'Preview':'Run';
        document.querySelector('#crunbar').hidden=false;
      }
    }catch(err){out.innerHTML='<p class="note">Could not debug this. '+esc(err.message)+'</p>'}
    btn.disabled=false;
  },true);
})();
</script>"""


@app.after_request
def _inject_debug_ui(resp):
    try:
        if request.path == "/" and resp.status_code == 200 and resp.mimetype == "text/html":
            html = resp.get_data(as_text=True)
            i = html.rfind("</body>")
            if i > -1 and "oq-debug-ui" not in html:
                resp.set_data(html[:i] + DEBUG_UI + html[i:])
    except Exception as e:
        print("debug UI injection failed:", e)
    return resp


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
