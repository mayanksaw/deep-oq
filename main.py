"""
Deep OQ + Debugger  (main.py, version 3)
Upload this next to your existing deep_oq.py. It loads Deep OQ unchanged and adds:
  - a team of AI agents that diagnose, fix, check and review code (/debug)
  - REAL compiler / checker support for many languages (whatever is installed on the server)
  - MULTI-FILE debugging: pick several files and they are checked and fixed together
  - a TEST-CASE LOOP: run the fix against your test cases, feed failures back, fix again, run again
  - a "Debug" button and boxes in the Code tab (injected automatically, deep_oq.py needs no edits)

Render start command :  gunicorn main:app --workers 1 --threads 4 --timeout 300
Settings (Render Environment, all optional):
  DEBUG_RUN     (empty) never run code on the server | admin = owner only | all = every user (risky)
  DEBUG_ROUNDS  fix attempts, default 3 (one extra round is added when you give test cases)
  DEBUG_AGENTS  full (default, 7 agents) | lite (fewer AI calls, faster)
Local test:  python main.py
"""
import ast
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
from html.parser import HTMLParser

from flask import jsonify, request

from deep_oq import app, chat, LANGS, denied, is_admin, EXAM_MODEL, EXAM_CHECK_MODEL

DEBUG_RUN = os.environ.get("DEBUG_RUN", "").strip().lower()
DEBUG_ROUNDS = max(1, min(5, int(os.environ.get("DEBUG_ROUNDS", "3") or 3)))
FULL = os.environ.get("DEBUG_AGENTS", "full").strip().lower() != "lite"
RUN_SECONDS = 8
BUILD_SECONDS = 25
MAX_FILES = 8
MAX_CHARS = 40000

# ==========================================================================
# LANGUAGE TABLES
# ==========================================================================
EXTS = {"Python": {".py"}, "JavaScript": {".js", ".mjs", ".cjs", ".jsx"}, "TypeScript": {".ts", ".tsx"},
        "Java": {".java"}, "C++": {".cpp", ".cc", ".cxx", ".hpp", ".h"}, "C": {".c", ".h"}, "C#": {".cs"},
        "Go": {".go"}, "Rust": {".rs"}, "PHP": {".php"}, "Kotlin": {".kt"}, "Swift": {".swift"},
        "SQL": {".sql"}, "HTML/CSS": {".html", ".htm", ".css"}, "Bash": {".sh", ".bash"}, "Dart": {".dart"}}
DEFAULT_NAME = {"Python": "main.py", "JavaScript": "main.js", "TypeScript": "main.ts", "Java": "Main.java",
                "C++": "main.cpp", "C": "main.c", "C#": "Program.cs", "Go": "main.go", "Rust": "main.rs",
                "PHP": "main.php", "Kotlin": "main.kt", "Swift": "main.swift", "SQL": "query.sql",
                "HTML/CSS": "index.html", "Bash": "script.sh", "Dart": "main.dart"}


def _has(*tools):
    return all(shutil.which(t) for t in tools)


def has_checker(lang):
    """True when this server can really check the language (not just ask the AI)."""
    return {"Python": True, "SQL": True, "HTML/CSS": True, "JavaScript": _has("node"),
            "TypeScript": _has("tsc"), "Java": _has("javac"), "C": _has("gcc"), "C++": _has("g++"),
            "Go": _has("go") or _has("gofmt"), "Rust": _has("rustc"), "PHP": _has("php"),
            "Bash": _has("bash")}.get(lang, False)


def can_execute(lang):
    return {"Python": True, "JavaScript": _has("node"), "Java": _has("javac", "java"), "C": _has("gcc"),
            "C++": _has("g++"), "Go": _has("go"), "Rust": _has("rustc"), "PHP": _has("php"),
            "Bash": _has("bash")}.get(lang, False)


def checker_report():
    """Printed in the server log so you can see what is installed."""
    return {l: ("check" if has_checker(l) else "AI only") + (" + run" if can_execute(l) else "") for l in sorted(EXTS)}


print("Debugger tools:", checker_report())


# ==========================================================================
# FILE HELPERS
# ==========================================================================
def _safe_name(name, default=""):
    n = str(name or "").replace("\\", "/").strip().lstrip("/")
    parts = [p for p in n.split("/") if p not in ("", ".")]
    if not parts or ".." in parts or not re.fullmatch(r"[\w./ +-]{1,100}", "/".join(parts)):
        return default
    return "/".join(parts)


def _write_files(work, files):
    for name, content in files.items():
        path = os.path.join(work, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)


def _src(lang, files):
    ok = EXTS.get(lang, ())
    return {n: c for n, c in files.items() if os.path.splitext(n)[1].lower() in ok}


def _env(work):
    return {"PATH": os.environ.get("PATH", ""), "HOME": work, "TMPDIR": work, "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8", "GOCACHE": os.path.join(work, ".gocache"),
            "GOPATH": os.path.join(work, ".gopath"), "GOFLAGS": "-mod=mod", "CGO_ENABLED": "0"}


def _cmd(argv, work, secs=BUILD_SECONDS):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=secs, cwd=work, env=_env(work))
        return p.returncode, (p.stdout + p.stderr).replace(work + "/", "")
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    except Exception as e:
        return 127, str(e)


def _extract_code(text):
    blocks = re.findall(r"```[^\n]*\n(.*?)```", text or "", re.S)
    return max(blocks, key=len).strip("\n") if blocks else ""


def _after_code(text):
    parts = (text or "").rsplit("```", 1)
    return parts[1].strip() if len(parts) == 2 else ""


def _numbered(code, limit=900):
    lines = code.split("\n")[:limit]
    return "\n".join(f"{i + 1:>4}| {l}" for i, l in enumerate(lines))


def _proj_text(files, cap=1500):
    if len(files) == 1:
        return _numbered(next(iter(files.values())), cap)
    out, used = [], 0
    for n, c in files.items():
        lines = c.split("\n")[:max(50, cap - used)]
        used += len(lines)
        out.append(f"=== FILE: {n} ===\n" + "\n".join(f"{i + 1:>4}| {l}" for i, l in enumerate(lines)))
    return "\n\n".join(out)


def _files_block(files):
    if len(files) == 1:
        return "```\n" + next(iter(files.values())) + "\n```"
    return "\n\n".join(f"FILE: {n}\n```\n{c}\n```" for n, c in files.items())


def _parse_files(raw, files):
    """Reads the fixer's answer: 'FILE: path' + a code block per changed file (or one block for one file)."""
    changed = {}
    for m in re.finditer(r"(?im)^\W*FILE:\s*`?([^\s`]+)`?\s*\n```[^\n]*\n(.*?)```", raw or "", re.S):
        name = _safe_name(m.group(1))
        if name:
            changed[name] = m.group(2).strip("\n")
    if not changed and len(files) == 1:
        code = _extract_code(raw)
        if code:
            changed[next(iter(files))] = code
    return changed


# ==========================================================================
# STATIC CHECKS: real tools where installed
# ==========================================================================
class _FlakeReport:
    def __init__(self, name):
        self.out, self.name = [], name

    def unexpectedError(self, filename, msg):
        self.out.append(f"{self.name}: {msg}")

    def syntaxError(self, filename, msg, lineno, offset, text):
        self.out.append(f"{self.name} SyntaxError on line {lineno}: {msg}")

    def flake(self, m):
        text = m.message % m.message_args
        if "imported but unused" in text or "assigned to but never used" in text:
            return
        self.out.append(f"{self.name} line {m.lineno}: {text}")


def _py_cross(files):
    """Checks that 'from x import y' between project files really finds y in x."""
    trees = {}
    for n, c in files.items():
        if n.endswith(".py"):
            try:
                trees[n[:-3].replace("/", ".")] = ast.parse(c)
            except SyntaxError:
                pass

    def names(tree):
        found = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    found.update(x.id for x in ast.walk(t) if isinstance(x, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                found.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                found.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                found.add("*")  # conditional definitions: do not guess
        return found

    probs = []
    for mod, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in trees:
                have = names(trees[node.module])
                for a in node.names:
                    if a.name != "*" and "*" not in have and a.name not in have \
                            and (node.module + "." + a.name) not in trees:
                        probs.append(f"{mod.replace('.', '/')}.py line {node.lineno}: cannot import name "
                                     f"'{a.name}' from '{node.module}' (it is not defined there)")
    return probs


def _js_cross(files):
    probs = []
    pat = re.compile(r"""(?:require\(\s*|from\s+|import\s*\(\s*|import\s+)['"](\.{1,2}/[^'"]+)['"]""")
    for n, c in files.items():
        base = os.path.dirname(n)
        for m in pat.finditer(c):
            target = os.path.normpath(os.path.join(base, m.group(1))).replace("\\", "/")
            cands = [target] + [target + e for e in (".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".json")] \
                + [target + "/index" + e for e in (".js", ".ts")]
            if not any(x in files for x in cands):
                line = c[:m.start()].count("\n") + 1
                probs.append(f"{n} line {line}: cannot find the file '{m.group(1)}' that is imported")
    return probs


class _Tags(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
            "track", "wbr"}
    OPT = {"p", "li", "td", "th", "tr", "dt", "dd", "option", "thead", "tbody", "tfoot", "colgroup", "body", "html"}

    def __init__(self):
        super().__init__()
        self.stack, self.errs = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append((tag, self.getpos()[0]))

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if self.stack and self.stack[-1][0] == tag:
            self.stack.pop()
        elif any(t == tag for t, _ in self.stack):
            while self.stack and self.stack[-1][0] != tag:
                t, line = self.stack.pop()
                if t not in self.OPT:
                    self.errs.append(f"line {line}: <{t}> is never closed")
            self.stack.pop()
        elif tag not in self.OPT:
            self.errs.append(f"line {self.getpos()[0]}: stray closing </{tag}>")

    def close(self):
        super().close()
        for t, line in self.stack:
            if t not in self.OPT:
                self.errs.append(f"line {line}: <{t}> is never closed")


def _css_braces(css, label):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    return [f"{label}: {abs(css.count('{') - css.count('}'))} unbalanced curly brace(s) in the CSS"] \
        if css.count("{") != css.count("}") else []


def _html_check(src):
    probs = []
    for n, c in src.items():
        if n.lower().endswith(".css"):
            probs += _css_braces(c, n)
            continue
        p = _Tags()
        try:
            p.feed(c)
            p.close()
        except Exception as e:
            probs.append(f"{n}: could not parse ({e})")
        probs += [f"{n} {e}" for e in p.errs[:6]]
        for i, st in enumerate(re.findall(r"<style[^>]*>(.*?)</style>", c, re.S | re.I)):
            probs += _css_braces(st, f"{n} <style> #{i + 1}")
        if shutil.which("node"):
            for i, js in enumerate(re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", c, re.S | re.I)):
                if js.strip():
                    work = tempfile.mkdtemp(prefix="oqchk_")
                    try:
                        _write_files(work, {"s.js": js})
                        rc, out = _cmd(["node", "--check", "s.js"], work, 15)
                        if rc != 0:
                            probs.append(f"{n} <script> #{i + 1}: " + out.strip()[-300:])
                    finally:
                        shutil.rmtree(work, ignore_errors=True)
    return probs


def _sql_check(src):
    """Syntax only, SQLite dialect. Nothing is executed (statements are only compiled with EXPLAIN)."""
    probs = []
    for n, c in src.items():
        conn = sqlite3.connect(":memory:")
        conn.set_authorizer(lambda *a: sqlite3.SQLITE_DENY)
        buf = ""
        for line in c.splitlines(True):
            buf += line
            if sqlite3.complete_statement(buf):
                try:
                    conn.execute("EXPLAIN " + buf)
                except sqlite3.Error as e:
                    if "syntax error" in str(e).lower():
                        probs.append(f"{n}: SQL syntax error near statement starting '{buf.strip()[:60]}': {e}")
                buf = ""
        if buf.strip():
            probs.append(f"{n}: the last statement is not finished (missing ';'?): '{buf.strip()[:60]}'")
        conn.close()
    return probs


def _tool_check(lang, src, work):
    probs = []
    names = list(src)
    if lang == "JavaScript":
        if shutil.which("node"):
            for n in names:
                if n.endswith((".js", ".mjs", ".cjs")):
                    rc, out = _cmd(["node", "--check", n], work, 15)
                    if rc != 0:
                        probs.append(f"{n}: " + out.strip()[-500:])
        probs += _js_cross(src)
    elif lang == "TypeScript":
        probs += _js_cross(src)
        if shutil.which("tsc"):
            rc, out = _cmd(["tsc", "--noEmit", "--skipLibCheck", "--target", "es2020"] + names, work)
            if rc != 0:
                probs += [l for l in out.splitlines() if "error TS" in l and "error TS5" not in l][:12]
    elif lang == "Java" and shutil.which("javac"):
        rc, out = _cmd(["javac", "-d", "out"] + names, work)
        if rc != 0:
            probs.append(out.strip()[-900:])
    elif lang in ("C", "C++") and shutil.which("gcc" if lang == "C" else "g++"):
        cc = ["gcc"] if lang == "C" else ["g++", "-std=c++17"]
        for n in names:
            if n.endswith((".h", ".hpp")):
                continue
            rc, out = _cmd(cc + ["-O1", "-Wall", "-Wextra", "-I.", "-c", "-o", os.devnull, n], work)
            for l in out.splitlines():
                low = l.lower()
                if "error" in low or ("warning:" in low and re.search(
                        r"uninitiali|format|return|null|bound|overflow|dangling|use-after|stringop|array|"
                        r"comparison|sequence-point|address", low)):
                    probs.append(l.strip()[:250])
    elif lang == "Go":
        if shutil.which("go"):
            if "go.mod" not in os.listdir(work):
                _write_files(work, {"go.mod": "module dbgproj\n\ngo 1.21\n"})
            rc, out = _cmd(["go", "vet", "./..."], work)
            if rc != 0:
                probs.append(out.strip()[-900:])
        elif shutil.which("gofmt"):
            for n in names:
                rc, out = _cmd(["gofmt", "-e", n], work, 15)
                if rc != 0:
                    probs.append(f"{n}: " + out.strip()[-400:])
    elif lang == "Rust" and shutil.which("rustc"):
        main = "main.rs" if "main.rs" in src else names[0]
        rc, out = _cmd(["rustc", "--edition", "2021", "--emit=metadata", "--out-dir", "out", main], work)
        if rc != 0:
            probs.append(out.strip()[-900:])
    elif lang == "PHP" and shutil.which("php"):
        for n in names:
            rc, out = _cmd(["php", "-l", n], work, 15)
            if rc != 0:
                probs.append(f"{n}: " + out.strip()[-300:])
    elif lang == "Bash" and shutil.which("bash"):
        for n in names:
            rc, out = _cmd(["bash", "-n", n], work, 15)
            if rc != 0:
                probs.append(f"{n}: " + out.strip()[-300:])
    return probs


def check_files(lang, files):
    """Problems found WITHOUT running the program. Empty list = nothing found."""
    src = _src(lang, files)
    if not src:
        return []
    probs = []
    if lang == "Python":
        for n, c in src.items():
            try:
                ast.parse(c)
            except SyntaxError as e:
                probs.append(f"{n} SyntaxError on line {e.lineno}: {e.msg}")
                continue
            try:
                from pyflakes.api import check as _flake
                rep = _FlakeReport(n)
                _flake(c, n, rep)
                probs += rep.out[:10]
            except ImportError:
                pass
            except Exception as e:
                print("pyflakes failed:", e)
        if not probs:
            probs += _py_cross(src)
    elif lang == "HTML/CSS":
        probs = _html_check(src)
    elif lang == "SQL":
        probs = _sql_check(src)
    else:
        work = tempfile.mkdtemp(prefix="oqchk_")
        try:
            _write_files(work, files)
            probs = _tool_check(lang, src, work)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return probs[:20]


# ==========================================================================
# RUNNING: build once, run once per test case (only when DEBUG_RUN allows it)
# ==========================================================================
def pick_entry(lang, files, entry=""):
    src = _src(lang, files)
    if entry in src:
        return entry
    pats = {"Python": r"__name__\s*==\s*['\"]__main__['\"]", "Java": r"static\s+void\s+main",
            "C": r"\bint\s+main\s*\(", "C++": r"\bint\s+main\s*\(", "Go": r"func\s+main\s*\(",
            "Rust": r"fn\s+main\s*\("}
    if lang in pats:
        for n, c in src.items():
            if re.search(pats[lang], c):
                return n
    for n in src:
        if re.match(r"(main|index|app|server|program)\.", os.path.basename(n), re.I):
            return n
    return sorted(src)[0] if src else ""


def _build(lang, src, entry, work):
    """Returns (argv to run, build error text or None)."""
    names = [n for n in src if not n.endswith((".h", ".hpp"))]
    prog = os.path.join(work, "prog")
    if lang == "Python":
        return [sys.executable, entry], None
    if lang == "JavaScript":
        return ["node", entry], None
    if lang == "PHP":
        return ["php", entry], None
    if lang == "Bash":
        return ["bash", entry], None
    if lang == "Java":
        rc, out = _cmd(["javac", "-d", "out"] + list(src), work)
        if rc != 0:
            return None, out.strip()[-1200:]
        text = src[entry]
        cls = re.search(r"(?:public\s+)?class\s+(\w+)[^{]*\{[\s\S]*?static\s+void\s+main", text)
        pkg = re.search(r"^\s*package\s+([\w.]+)\s*;", text, re.M)
        name = (cls.group(1) if cls else os.path.splitext(os.path.basename(entry))[0])
        return ["java", "-Xmx256m", "-cp", "out", (pkg.group(1) + "." if pkg else "") + name], None
    if lang in ("C", "C++"):
        cc = ["gcc"] if lang == "C" else ["g++", "-std=c++17"]
        rc, out = _cmd(cc + ["-O0", "-I.", "-o", prog] + names + ["-lm"], work)
        return ([prog], None) if rc == 0 else (None, out.strip()[-1200:])
    if lang == "Go":
        if "go.mod" not in os.listdir(work):
            _write_files(work, {"go.mod": "module dbgproj\n\ngo 1.21\n"})
        rc, out = _cmd(["go", "build", "-o", prog, "."], work)
        return ([prog], None) if rc == 0 else (None, out.strip()[-1200:])
    if lang == "Rust":
        rc, out = _cmd(["rustc", "--edition", "2021", "-o", prog, entry], work)
        return ([prog], None) if rc == 0 else (None, out.strip()[-1200:])
    return None, "This language cannot be run here."


def _limits(lang):
    def apply():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (RUN_SECONDS * 2, RUN_SECONDS * 2))
            resource.setrlimit(resource.RLIMIT_FSIZE, (5 << 20, 5 << 20))
            if lang in ("Python", "C", "C++", "Rust"):  # node, java and go reserve huge virtual memory
                resource.setrlimit(resource.RLIMIT_AS, (768 << 20, 768 << 20))
        except Exception:
            pass
    return apply


def _run_one(argv, work, stdin, lang):
    try:
        p = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=RUN_SECONDS, cwd=work,
                           env=_env(work), preexec_fn=_limits(lang) if os.name == "posix" else None)
        return {"code": p.returncode, "out": p.stdout[-3000:], "err": p.stderr[-3000:], "timeout": False}
    except subprocess.TimeoutExpired:
        return {"code": -1, "out": "", "err": f"Stopped after {RUN_SECONDS} seconds.", "timeout": True}
    except Exception as e:
        return {"code": -1, "out": "", "err": str(e)[:300], "timeout": False}


def execute(lang, files, entry, stdins):
    """Builds the project and runs it once per stdin text. Returns (build_error, results).
    (None, None) means this language cannot be run on this server."""
    src = _src(lang, files)
    if not src or not can_execute(lang):
        return None, None
    entry = pick_entry(lang, files, entry)
    work = tempfile.mkdtemp(prefix="oqdbg_")
    try:
        _write_files(work, files)
        argv, err = _build(lang, src, entry, work)
        if err:
            return err, []
        return None, [_run_one(argv, work, s, lang) for s in stdins]
    except Exception as e:
        print("execute failed:", e)
        return None, None
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_sandboxed(lang, code):
    """One run of a single piece of code (used by the test writer agent)."""
    _, res = execute(lang, {DEFAULT_NAME.get(lang, "main.txt"): code}, "", [""])
    return res[0] if res else None


def run_suite(lang, files):
    """Runs the project's OWN tests when it has them (pytest/unittest for Python, node --test for JS)."""
    names = list(files)
    if lang == "Python" and any(re.search(r"(^|/)(test_[^/]*|[^/]*_test)\.py$", n) for n in names):
        if importlib.util.find_spec("pytest"):
            argv = [sys.executable, "-m", "pytest", "-x", "-q", "--no-header", "-p", "no:cacheprovider"]
        else:
            argv = [sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", "*test*.py"]
    elif lang == "JavaScript" and _has("node") and any(re.search(r"\.(test|spec)\.[cm]?js$", n) for n in names):
        argv = ["node", "--test"]
    else:
        return None
    work = tempfile.mkdtemp(prefix="oqsuite_")
    try:
        _write_files(work, files)
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=RUN_SECONDS * 3, cwd=work,
                               env=_env(work), preexec_fn=_limits(lang) if os.name == "posix" else None)
            return {"code": p.returncode, "out": (p.stdout + p.stderr)[-1500:], "timeout": False}
        except subprocess.TimeoutExpired:
            return {"code": -1, "out": "", "timeout": True}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def parse_cases(text):
    """Test cases. Lines before '=>' are the input, lines after it the expected output; '###' separates cases."""
    cases = []
    for block in re.split(r"(?m)^\s*###\s*$", text or ""):
        parts = re.split(r"(?m)^\s*=>\s*$", block, maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            stdin = parts[0].strip("\n")
            cases.append(((stdin + "\n") if stdin.strip() else "", parts[1].strip()))
    return cases[:20]


def _norm_out(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def evaluate(lang, files, entry, cases, expected, can_run):
    """Checks a version of the project. Returns (problems, how, info). No problems = it passed every check."""
    problems = check_files(lang, files)
    how, info = [], ""
    if not problems and has_checker(lang):
        how.append("static/compile check passed")
    if problems or not can_run:
        return problems, how, info
    stdins = [c[0] for c in cases] or [""]
    build_err, results = execute(lang, files, entry, stdins)
    if build_err:
        problems.append("The program failed to build:\n" + build_err)
    elif results:
        if cases:
            bad = []
            for i, (case, res) in enumerate(zip(cases, results), 1):
                shown = repr(case[0][:150])
                if res["timeout"]:
                    bad.append(f"Test case {i} timed out (possible infinite loop). Input: {shown}")
                elif res["code"] != 0 and "EOFError" not in res["err"]:
                    bad.append(f"Test case {i} crashed. Input: {shown}\n{res['err'][-500:]}")
                elif _norm_out(res["out"]) != _norm_out(case[1]):
                    bad.append(f"Test case {i} gave the wrong output. Input: {shown}\n"
                               f"EXPECTED: {case[1][:300]}\nGOT: {res['out'][:300]}")
            if bad:
                problems += bad[:4]
            else:
                how.append(f"all {len(cases)} test cases passed")
        else:
            res = results[0]
            if res["timeout"]:
                info = "RUNNING THE PROGRAM: " + res["err"] + " (possible infinite loop or hang)"
                how.append("run hit the time limit (inconclusive, may be a server or long loop)")
            elif res["code"] != 0 and "EOFError" not in res["err"]:
                info = "RUNNING THE PROGRAM FAILED:\n" + res["err"][-1500:]
                problems.append("The program crashed when run:\n" + res["err"][-900:])
            else:
                info = "RUNNING THE PROGRAM: exit 0. Output:\n" + (res["out"][:800] or "(none)")
                how.append("ran without errors")
                if expected and _norm_out(expected) not in _norm_out(res["out"]):
                    problems.append("It ran, but the output did not match the expected output.\nEXPECTED:\n"
                                    + expected[:600] + "\nGOT:\n" + res["out"][:600])
                elif expected:
                    how.append("output matched what you expected")
    suite = run_suite(lang, files)
    if suite:
        if suite["timeout"]:
            how.append("the project's test suite timed out (inconclusive)")
        elif suite["code"] == 0:
            how.append("the project's own test suite passed")
        else:
            problems.append("The project's own tests failed:\n" + suite["out"][-900:])
    return problems, how, info


# ==========================================================================
# THE AGENTS
# ==========================================================================
DBG_SYS = ("You are Deep OQ's debugging engine, a principal engineer who has fixed every kind of bug. "
           "Find the REAL root cause, not just the symptom. Never invent library functions, flags or APIs. "
           "Never use tables, horizontal lines (---) or HTML tags. Be concise and exact.")


def _json_obj(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except ValueError:
        return {}


def _parallel(jobs):
    res = [None] * len(jobs)

    def work(i):
        try:
            res[i] = jobs[i]()
        except Exception as e:
            print("agent failed:", str(e)[:160])
    ths = [threading.Thread(target=work, args=(i,)) for i in range(len(jobs))]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return res


def agent_triage(lang, ptext, ev):
    raw = chat([{"role": "system", "content": DBG_SYS},
                {"role": "user", "content":
                 f"LANGUAGE: {lang}\n\nCODE:\n{ptext}\n\nEVIDENCE:\n{ev}\n\n"
                 'Reply with ONLY JSON: {"kind":"syntax|runtime|logic|performance|security|environment|none",'
                 '"already_correct":true or false,"summary":"one short sentence"}. '
                 "already_correct is true ONLY if you read the code line by line and find no bug at all."}],
               0.0, 400)
    return _json_obj(raw)


def agent_critic(lang, original, fixed, diag):
    try:
        raw = chat([{"role": "system", "content": DBG_SYS},
                    {"role": "user", "content":
                     f"You are a strict independent code reviewer. LANGUAGE: {lang}\n\nORIGINAL:\n{original}"
                     f"\n\nDIAGNOSIS:\n{diag}\n\nFIXED:\n{fixed}\n\n"
                     "Check: is every bug from the diagnosis really fixed? Is the original behaviour kept? Any new "
                     "bug, missing import, wrong edge case or leftover placeholder? "
                     'Reply with ONLY JSON: {"ok":true or false,"issues":["short issue with line number"]}. '
                     "Say ok=true when the fix is correct; do not nitpick style."}],
                   0.0, 800, smart=True, prefer=EXAM_MODEL)
        j = _json_obj(raw)
        return bool(j.get("ok", True)), [str(x)[:250] for x in (j.get("issues") or [])][:4]
    except Exception as e:
        print("critic failed:", str(e)[:160])
        return True, []


def agent_tests(lang, fixed):
    if lang not in ("Python", "JavaScript"):
        return ""
    style = ("plain assert statements and print('ALL TESTS PASSED') at the end" if lang == "Python" else
             "if(!(condition)) throw new Error('...') checks and console.log('ALL TESTS PASSED') at the end")
    raw = chat([{"role": "system", "content": DBG_SYS},
                {"role": "user", "content":
                 f"Write 3 to 6 SHORT test checks for this {lang} code, to be appended directly after it. Use {style}. "
                 "Cover a normal case, an edge case (empty or zero) and the bug that was fixed. No input(), no files, "
                 "no network, do not redefine the code. Output ONE fenced code block and nothing else.\n\n"
                 f"```\n{fixed}\n```"}], 0.1, 1200)
    return _extract_code(raw)


def agent_review(lang, ptext):
    raw = chat([{"role": "system", "content": DBG_SYS},
                {"role": "user", "content":
                 f"Review this {lang} code for security holes (injection, unsafe eval/exec, secrets), serious "
                 "performance problems, and missing error handling that could crash it. List at most 4 short "
                 "bullets, each with a file and line number and a concrete fix. If there is nothing important, "
                 f"reply exactly NONE.\n\n{ptext}"}], 0.1, 700).strip()
    return "" if raw.upper().startswith("NONE") else raw.replace("```", "")


def _run_tests(lang, fixed, tests):
    ran = run_sandboxed(lang, fixed + "\n\n" + tests + "\n")
    if ran is None:
        return "not run", ""
    if ran["timeout"]:
        return "inconclusive", "the tests hit the time limit"
    if ran["code"] == 0 and "ALL TESTS PASSED" in ran["out"]:
        return "passed", ""
    return "failed", (ran["err"] or ran["out"])[-500:]


# ==========================================================================
# THE TEAM
# ==========================================================================
def debug_code(lang, files, error="", expected="", can_run=False, cases=None, entry=""):
    cases = cases or []
    multi = len(files) > 1
    agents = ["Checker"]
    evidence = []
    base_problems, _, base_info = evaluate(lang, files, entry, cases, expected, can_run)
    if base_problems:
        evidence.append("CHECKS ON THE ORIGINAL CODE FOUND:\n" + "\n".join(base_problems[:8]))
    if base_info:
        evidence.append(base_info)
    if cases:
        evidence.append(f"THE USER GAVE {len(cases)} TEST CASES that the code must pass"
                        + ("" if can_run else " (they cannot be run on this server, so reason through them by hand)")
                        + ":\n" + "\n".join(f"input {c[0]!r} => expected {c[1][:150]!r}" for c in cases[:6]))
    if error:
        evidence.append("ERROR OR SYMPTOM REPORTED BY THE USER:\n" + error)
    if expected:
        evidence.append("EXPECTED OUTPUT OR BEHAVIOUR:\n" + expected)
    ev = "\n\n".join(evidence) or ("No error text was given and the automatic checks found nothing, so hunt for "
                                   "logic bugs, wrong edge cases, off-by-one errors, bad types and unsafe code.")
    ptext = _proj_text(files)

    # ---- agent 1: triage
    triage = {}
    try:
        triage = agent_triage(lang, ptext, ev)
        agents.append("Triage")
    except Exception as e:
        print("triage failed:", str(e)[:160])
    if triage.get("already_correct") is True and not error and not expected and not base_problems:
        notes = ""
        if FULL:
            try:
                notes = agent_review(lang, ptext)
                agents.append("Security reviewer")
            except Exception as e:
                print("review failed:", str(e)[:160])
        out = ["## Debug report", "**No bugs found.** " + str(triage.get("summary", ""))[:300] +
               " The automatic checks found nothing wrong.", "**Agents used:** " + ", ".join(agents) + "."]
        if notes:
            out += ["### Suggestions (not bugs)", notes]
        out.append("If something still goes wrong when you use it, paste the exact error or the output you "
                   "expected and I will dig deeper.")
        return "\n\n".join(out), True, 0

    # ---- agent 2: root-cause analyst
    kind = str(triage.get("kind", ""))
    diag = chat([{"role": "system", "content": DBG_SYS},
                 {"role": "user", "content":
                  f"LANGUAGE: {lang}\n" + (f"TRIAGE SAYS THIS IS A {kind.upper()} PROBLEM.\n" if kind and kind != "none" else "")
                  + f"\nCODE (with line numbers):\n{ptext}\n\nEVIDENCE:\n{ev}\n\n"
                  "Write exactly these parts:\nROOT CAUSE: 1 to 3 sentences.\n"
                  "ALL BUGS: a numbered list; each item has the file and line number, what is wrong, and why it "
                  "breaks. Include bugs that have not triggered yet and problems BETWEEN files.\n"
                  "MISSING: only if something you need is not in the code or evidence (a file, a version, the full "
                  "traceback), say exactly what; otherwise omit this part. Do not write code here."}],
                0.1, 2500, smart=True).replace("```", "")
    agents.append("Root-cause analyst")

    rule = ("For EACH file you change write a line 'FILE: <path>' and then that file's COMPLETE new content in ONE "
            "fenced code block. Do not output files you did not change. " if multi else
            "Write the COMPLETE corrected code in ONE fenced code block: no placeholders, no '...'. ")

    def fix_job(prefer, retry):
        def job():
            return chat([{"role": "system", "content": DBG_SYS},
                         {"role": "user", "content":
                          f"LANGUAGE: {lang}\n\nORIGINAL CODE:\n{_files_block(files)}\n\nDIAGNOSIS:\n{diag}\n\n"
                          f"EVIDENCE:\n{ev}{retry}\n\n{rule}Keep the same features, names and structure, change only "
                          "what is needed, and fix every bug in the diagnosis. After the code write 'CHANGES:' and at "
                          "most 6 short bullets (file, line, before, after, why), then one line 'TEST:' saying how to "
                          "confirm it works."}], 0.15, 6144, smart=True, prefer=prefer)
        return job

    # ---- agents 3 and 4 + verifier: fix, run the checks and test cases, feed failures back, repeat
    prefers = [None, EXAM_CHECK_MODEL, EXAM_MODEL, None, EXAM_CHECK_MODEL, EXAM_MODEL]
    rounds = min(6, DEBUG_ROUNDS + (1 if cases else 0))
    feedback, changed, notes, problems, how = "", {}, "", ["no attempt was made"], []
    final = dict(files)
    used = 0
    for rnd in range(rounds):
        used = rnd + 1
        retry = ("\n\nYOUR PREVIOUS FIX FAILED ITS CHECKS. Read this failure carefully and fix the real cause:\n"
                 + feedback + "\n\nPREVIOUS FIX:\n" + _files_block(changed)) if feedback and changed else \
                ("\n\nYOUR PREVIOUS ANSWER WAS NOT USABLE: " + feedback) if feedback else ""
        prefs = [None, EXAM_CHECK_MODEL] if (rnd == 0 and FULL) else [prefers[rnd % len(prefers)]]
        raws = _parallel([fix_job(p, retry) for p in prefs])
        cands = []
        for raw in raws:
            ch = _parse_files(raw or "", files)
            if ch:
                new_files = {**files, **ch}
                pr, hw, _ = evaluate(lang, new_files, entry, cases, expected, can_run)
                cands.append((pr, hw, ch, _after_code(raw or ""), new_files))
        if not cands:
            feedback = ("No fixer returned usable code. " + ("Use 'FILE: path' lines, each followed by a fenced code block."
                        if multi else "Return the full corrected code in one fenced block."))
            continue
        cands.sort(key=lambda c: len(c[0]))
        problems, how, changed, notes, final = cands[0]
        for a in (["Fixer" + (" x2 (two models)" if len(prefs) > 1 else ""), "Verifier"]
                  + (["Test-case runner"] if cases and can_run else [])):
            if a not in agents:
                agents.append(a)
        if not problems and FULL:
            ok, issues = agent_critic(lang, _files_block({n: files[n] for n in changed if n in files}) or "(new file)",
                                      _files_block(changed), diag)
            if "Independent critic" not in agents:
                agents.append("Independent critic")
            if ok:
                how = how + ["independent reviewer approved"]
            else:
                problems = ["The independent reviewer found: " + "; ".join(issues or ["a problem in the fix"])]
        if not problems:
            break
        feedback = "\n".join(problems)[:1800]

    verified = bool(changed) and not problems

    # ---- agents 6 and 7 together
    tests, tests_note, review = "", "", ""
    if changed and verified and FULL:
        single = next(iter(changed.values())) if not multi else ""
        jobs = [lambda: agent_review(lang, _proj_text(changed))]
        if not multi:
            jobs.insert(0, lambda: agent_tests(lang, single))
        res = _parallel(jobs)
        if multi:
            review = res[0] or ""
        else:
            tests, review = res[0] or "", res[1] or ""
            if tests:
                agents.append("Test writer")
        agents.append("Security reviewer")
        if tests and can_run and any("ran without errors" in h for h in how):
            label, detail = _run_tests(lang, single, tests)
            tests_note = {"passed": "**Auto-tests passed** when run on the server.",
                          "failed": "**Auto-tests failed** (the tests are AI-written, so they can be wrong too):\n\n" + detail,
                          "inconclusive": "Auto-tests were inconclusive: " + detail + "."}.get(label, "")

    # ---- report
    if verified:
        state = "**Verified:** " + ", ".join(how) + "." if how else (
            "**Not machine-verified:** this server has no checker for " + lang + ", so this fix is reviewed by the "
            "AI only. Run it and test it.")
    elif changed:
        state = ("**Not fully verified.** This is the best fix after " + str(used) + " rounds, but it still "
                 "fails a check:\n\n" + "\n".join("- " + p.replace("\n", " ")[:300] for p in problems[:3]) +
                 "\n\nSend me the full error message, your " + lang + " version and any file or input the code "
                 "uses, and I will take it from there.")
    else:
        state = ("**I could not produce a working fix yet.** Send the complete error message (the whole "
                 "traceback), your " + lang + " version, and the input you used, then try again.")
    fence = lang.lower().split("/")[0]
    out = ["## Debug report", state, "**Agents used:** " + ", ".join(agents) + ".",
           "### Root cause and bugs found", diag]
    if changed:
        if multi:
            out.append("### Fixed files")
            for n, c in changed.items():
                out += [f"**{n}**", "```" + fence + "\n" + c + "\n```"]
            same = [n for n in files if n not in changed]
            if same:
                out.append("Unchanged: " + ", ".join(same))
        else:
            out += ["### Fixed code", "```" + fence + "\n" + next(iter(changed.values())) + "\n```"]
        if notes:
            out.append(notes)
    if tests:
        out += ["### Tests", tests_note or "Suggested tests (not run on the server). Add them below your code:",
                "```" + fence + "\n" + tests + "\n```"]
    if review:
        out += ["### Other things to improve", review]
    return "\n\n".join(out), verified, used


@app.post("/debug")
def debug_route():
    if denied():
        return jsonify(error="locked"), 401
    d = request.get_json(silent=True) or {}
    lang = d.get("lang", "Python")
    if lang not in LANGS:
        return jsonify(error="Pick a language from the list."), 400
    files = {}
    if isinstance(d.get("files"), list):
        for f in d["files"][:MAX_FILES]:
            if isinstance(f, dict):
                n, c = _safe_name(f.get("name")), str(f.get("content", ""))
                if n and c.strip():
                    files[n] = c
    code = str(d.get("code", "")).strip("\n")
    if not files:
        if not code.strip():
            return jsonify(error="Paste the code first, or choose your files."), 400
        name = DEFAULT_NAME.get(lang, "main.txt")
        if lang == "Java":
            m = re.search(r"public\s+(?:final\s+|abstract\s+)?(?:class|interface|enum)\s+(\w+)", code)
            name = (m.group(1) if m else "Main") + ".java"
        files = {name: code}
    if sum(len(c) for c in files.values()) > MAX_CHARS:
        return jsonify(error=f"That is too much code for one go (limit {MAX_CHARS} characters). "
                             "Send only the files that matter."), 400
    error = str(d.get("error", "")).strip()[:6000]
    expected = str(d.get("expected", "")).strip()[:3000]
    cases = parse_cases(str(d.get("cases", ""))[:8000])
    entry = _safe_name(d.get("entry"))
    can_run = DEBUG_RUN == "all" or (DEBUG_RUN == "admin" and is_admin())
    try:
        answer, verified, rounds = debug_code(lang, files, error, expected, can_run, cases, entry)
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
  MODES.debug={btn:'Debug code',task:'Anything else I should know? (optional)',ph:'It worked yesterday',code:'Your code (or choose project files below)'};
  const Q=s=>document.querySelector(s);
  const modes=Q('#cmodes'),b=document.createElement('button');
  b.type='button';b.dataset.m='debug';b.setAttribute('aria-pressed','false');b.textContent='Debug';
  b.onclick=()=>setMode('debug');modes.appendChild(b);

  const wrap=document.createElement('div');wrap.id='cdbgW';wrap.hidden=true;wrap.className='panel';
  wrap.innerHTML=
    '<label class="lab">Error message or what goes wrong (paste the full traceback)'
    +'<textarea class="in mono" id="cerr" rows="4" spellcheck="false" style="min-height:90px"></textarea></label>'
    +'<label class="lab">Expected output (optional)'
    +'<textarea class="in mono" id="cexp" rows="2" spellcheck="false" style="min-height:60px"></textarea></label>'
    +'<label class="lab">Test cases (optional). Input first, then a line with =&gt; then the expected output. Separate cases with ###'
    +'<textarea class="in mono" id="dcases" rows="5" spellcheck="false" style="min-height:100px" placeholder="2 3\n=&gt;\n5\n###\n10 20\n=&gt;\n30"></textarea></label>'
    +'<label class="lab">Project files (optional, up to 8 files). Pick all the files that belong together'
    +'<input class="in" type="file" id="dfiles" multiple></label>'
    +'<div class="muted" id="dflist"></div>'
    +'<label class="lab">Main file to run (optional)<input class="in" id="dentry" placeholder="main.py"></label>';
  const runRow=Q('#crun').parentElement;runRow.parentElement.insertBefore(wrap,runRow);

  let files=[];
  Q('#dfiles').onchange=async e=>{
    const list=[...e.target.files].slice(0,8);files=[];let note='';
    for(const f of list){
      if(f.size>200000){note+=' Skipped '+f.name+' (over 200 KB).';continue}
      try{const t=await f.text();if(t.indexOf('\u0000')>-1){note+=' Skipped '+f.name+' (not a text file).';continue}
        files.push({name:f.name,content:t})}catch(x){note+=' Could not read '+f.name+'.'}
    }
    Q('#dflist').textContent=files.length?(files.length+' file(s): '+files.map(f=>f.name).join(', ')+'.'+note):note;
    if(!Q('#dentry').value&&files.length){const m=files.find(f=>/^(main|app|index|server|program)\./i.test(f.name));if(m)Q('#dentry').placeholder=m.name}
  };

  const old=setMode;
  setMode=function(m){old(m);wrap.hidden=m!=='debug';};

  document.addEventListener('click',async e=>{
    if(cmode!=='debug'||!e.target.closest('#crun'))return;
    e.stopImmediatePropagation();e.preventDefault();
    const out=Q('#cout'),btn=Q('#crun'),code=Q('#ccode').value;
    if(!code.trim()&&!files.length){out.innerHTML='<p class="note">Paste the code first, or choose your project files.</p>';Q('#ccode').focus();return}
    const extra=Q('#ctask').value.trim();
    const body={lang:Q('#clang').value,code:code,files:files,entry:Q('#dentry').value,cases:Q('#dcases').value,
      error:Q('#cerr').value+(extra?'\n'+extra:''),expected:Q('#cexp').value};
    btn.disabled=true;Q('#crunbar').hidden=true;Q('#cframe').hidden=true;Q('#cconsole').hidden=true;
    out.innerHTML='<p class="muted">Debugging: checking, diagnosing, fixing, running your tests and verifying. This can take a minute or two.</p>';
    try{
      const r=await api('/debug',body),j=await r.json().catch(()=>({}));
      if(!r.ok||j.error)throw new Error(j.error||'The server did not answer.');
      out.innerHTML='<div class="body">'+md(j.answer)+'</div>';
      const m=j.answer.match(/```[^\n]*\n([\s\S]*?)```/);
      lastCode=m?m[1]:'';lastLang=body.lang;
      if(lastCode&&!files.length&&(lastLang==='HTML/CSS'||lastLang==='JavaScript')){
        Q('#cpreview').textContent=lastLang==='HTML/CSS'?'Preview':'Run';Q('#crunbar').hidden=false;
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
