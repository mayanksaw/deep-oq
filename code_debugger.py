#!/usr/bin/env python3
"""
code_debugger.py - AI-assisted bug finder for YOUR OWN code.

Scans a file or folder of source code, runs standard static-analysis tools
where available (pylint, bandit, mypy, eslint...), then asks an LLM (Gemini
or Claude, same setup as exam_ai.py) to read the code plus those findings
and explain each bug in plain language with a suggested fix.

This tool only reads files you point it at and writes a report. It does not
scan other people's servers, does not exploit anything, and does not modify
your code unless you pass --apply (which only ever touches files inside the
folder you scanned).

SETUP (Gemini - default)
  pip install google-genai
  Get a free key at https://aistudio.google.com/apikey, then EITHER:
    - paste the key into a file named gemini_key.txt next to this script, OR
    - export GEMINI_API_KEY="your-key"

SETUP (Claude - alternative)
  pip install anthropic
  export ANTHROPIC_API_KEY="your-key"

OPTIONAL (better results, not required):
  pip install pylint bandit mypy radon
  npm install -g eslint          # for JS/TS projects

USAGE
  python code_debugger.py scan myfile.py
  python code_debugger.py scan ./my_project --lang python
  python code_debugger.py scan ./my_project --security      # extra focus on security bugs
  python code_debugger.py scan ./my_project --deep          # strongest model, exhaustive review, no truncation
  python code_debugger.py scan ./my_project --apply         # let the AI write fixes to disk (asks first per file)
  python code_debugger.py watch ./my_project                # re-scan changed files automatically
"""

import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPORT_DIR = Path("debug_reports")
KEY_FILE = Path("gemini_key.txt")
# Fast, cheap default models for ordinary scans.
GEMINI_MODEL = os.environ.get("EXAM_AI_GEMINI_MODEL", "gemini-2.5-flash")
CLAUDE_MODEL = os.environ.get("EXAM_AI_MODEL", "claude-sonnet-5-5")
# Strongest available models, used only with --deep: slower and more expensive
# per file, but reason harder about subtle bugs (concurrency, edge cases,
# multi-step exploit chains). Override with env vars if your account's names differ.
GEMINI_DEEP_MODEL = os.environ.get("EXAM_AI_GEMINI_DEEP_MODEL", "gemini-2.5-pro")
CLAUDE_DEEP_MODEL = os.environ.get("EXAM_AI_DEEP_MODEL", "claude-opus-5-5")
# --deep sends far more of the file and asks for far more reasoning per bug.
DEEP_CODE_CHARS = 60_000
NORMAL_CODE_CHARS = 16_000

EXT_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cs": "csharp", ".go": "go",
    ".rb": "ruby", ".php": "php", ".rs": "rust", ".swift": "swift", ".kt": "kotlin", ".sh": "bash",
    ".sql": "sql", ".html": "html", ".css": "css",
}
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build", ".next", "vendor"}
MAX_FILE_BYTES = 200_000  # skip huge generated/minified files

# --------------------------------------------------------------------------
# LLM WRAPPER (same pattern as exam_ai.py)
# --------------------------------------------------------------------------
_client = None
_provider = None


def _gemini_key():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    return key


def _init_client():
    global _client, _provider
    wanted = os.environ.get("EXAM_AI_PROVIDER", "").lower()
    gkey = _gemini_key()
    if wanted == "gemini" or (not wanted and gkey):
        if not gkey:
            sys.exit("No Gemini key found. Put it in gemini_key.txt or set GEMINI_API_KEY.")
        try:
            from google import genai
        except ImportError:
            sys.exit("Install the SDK first:  pip install google-genai")
        _client, _provider = genai.Client(api_key=gkey), "gemini"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
        except ImportError:
            sys.exit("Install the SDK first:  pip install anthropic")
        _client, _provider = anthropic.Anthropic(), "claude"
    else:
        sys.exit("No API key found. Put your Gemini key in gemini_key.txt "
                 "(or set GEMINI_API_KEY / ANTHROPIC_API_KEY).")


def llm(system, prompt, max_tokens=4000, temperature=0.0, deep=False):
    if _client is None:
        _init_client()
    if _provider == "gemini":
        from google.genai import types
        model = GEMINI_DEEP_MODEL if deep else GEMINI_MODEL
        resp = _client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                                max_output_tokens=max_tokens * 4),
        )
        return resp.text or ""
    model = CLAUDE_DEEP_MODEL if deep else CLAUDE_MODEL
    resp = _client.messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        system=system, messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def parse_json(text):
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

# --------------------------------------------------------------------------
# FILE DISCOVERY
# --------------------------------------------------------------------------
def find_files(target, lang=None):
    target = Path(target)
    if target.is_file():
        return [target]
    files = []
    for root, dirs, names in os.walk(target):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in names:
            p = Path(root) / name
            ext = p.suffix.lower()
            if ext not in EXT_LANG:
                continue
            if lang and EXT_LANG[ext] != lang:
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append(p)
    return sorted(files)

# --------------------------------------------------------------------------
# STATIC ANALYZERS (used if installed; skipped quietly if not)
# --------------------------------------------------------------------------
def _run(cmd, cwd=None):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120)
        return r.stdout + r.stderr
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def static_findings(path):
    """Run whichever linters are installed; return {tool: output}. Never fails the scan if tools are missing."""
    ext = path.suffix.lower()
    lang = EXT_LANG.get(ext)
    out = {}
    if lang == "python":
        for tool, cmd in [
            ("pylint", ["pylint", "--disable=all", "--enable=E,W", "-sn", str(path)]),
            ("bandit", ["bandit", "-q", "-f", "txt", str(path)]),
            ("mypy", ["mypy", "--no-error-summary", "--ignore-missing-imports", str(path)]),
        ]:
            res = _run(cmd)
            if res and res.strip():
                out[tool] = res.strip()[:4000]
    elif lang in ("javascript", "typescript"):
        res = _run(["npx", "--yes", "eslint", "--no-eslintrc", "--env", "es2021,browser,node",
                    "--parser-options=ecmaVersion:2021", str(path)])
        if res and res.strip():
            out["eslint"] = res.strip()[:4000]
    return out

# --------------------------------------------------------------------------
# AI REVIEW
# --------------------------------------------------------------------------
REVIEW_SYSTEM = (
    "You are a senior software engineer doing a careful code review on code the user owns and asked you to "
    "debug. Find real bugs: logic errors, crashes, off-by-one errors, race conditions, resource leaks, null/"
    "undefined handling, and (if asked) security issues such as injection, unsafe deserialization, path "
    "traversal, and secrets in code. Do not invent problems that aren't there. Be specific: point to the "
    "exact line and explain the failing input or scenario."
)

DEEP_SYSTEM_ADDENDUM = (
    "\nThis is a DEEP review: take as much reasoning as you need. Beyond obvious bugs, also look for: "
    "race conditions and non-atomic check-then-act patterns; integer overflow/underflow; incorrect error "
    "handling that swallows exceptions; resource/memory leaks across all code paths including exception "
    "paths; algorithmic complexity problems that cause slowness or DoS at scale; subtle off-by-one and "
    "boundary errors; incorrect assumptions about input encoding, locale, or timezone; thread-safety of "
    "shared state; and multi-step bug chains where two individually-minor issues combine into something "
    "serious. For each bug, think through at least one concrete failing scenario before writing it down."
)

REVIEW_PROMPT = """File: {path}
Language: {lang}
{security_note}

STATIC ANALYZER OUTPUT (may be empty, may contain noise - use your own judgement too):
{static}

SOURCE CODE (line numbers added):
{code}

Reply with ONLY a JSON array, one object per bug found (empty array [] if none):
[
  {{
    "line": <int, best-guess line number>,
    "severity": "critical" | "high" | "medium" | "low" | "style",
    "title": "<short name of the bug>",
    "explanation": "<what is wrong and why, 1-3 sentences>",
    "failure_case": "<a concrete input/scenario that triggers it>",
    "fix": "<the corrected code for just the relevant lines>"
  }}
]"""


def numbered(code):
    return "\n".join(f"{i+1:>4}| {line}" for i, line in enumerate(code.splitlines()))


def review_file(path, security=False, deep=False):
    try:
        code = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return [], f"Could not read file: {e}"
    if not code.strip():
        return [], None

    lang = EXT_LANG.get(path.suffix.lower(), "text")
    static = static_findings(path)
    static_text = "\n\n".join(f"--- {t} ---\n{o}" for t, o in static.items()) or "(no static analyzer output)"
    note = "Pay special attention to SECURITY bugs (injection, unsafe eval, hardcoded secrets, path traversal, SSRF, unsafe deserialization)." if security else ""

    system = REVIEW_SYSTEM + (DEEP_SYSTEM_ADDENDUM if deep else "")
    char_limit = DEEP_CODE_CHARS if deep else NORMAL_CODE_CHARS
    prompt = REVIEW_PROMPT.format(path=path, lang=lang, security_note=note,
                                  static=static_text[:6000], code=numbered(code)[:char_limit])
    try:
        raw = llm(system, prompt, max_tokens=(8000 if deep else 4000), deep=deep)
    except Exception as e:
        return [], f"AI review failed: {e}"
    bugs = parse_json(raw)
    if bugs is None:
        return [], "AI did not return valid JSON (showing raw reply below)\n" + raw[:1500]
    return bugs, None

# --------------------------------------------------------------------------
# APPLY FIXES
# --------------------------------------------------------------------------
def apply_fix(path, bug):
    """Show a diff for one suggested fix and, if the user agrees, write it. Never auto-applies."""
    code = path.read_text(encoding="utf-8", errors="replace")
    lines = code.splitlines(keepends=True)
    ln = max(1, min(bug.get("line", 1), len(lines)))
    print(f"\n--- Suggested fix for '{bug['title']}' near line {ln} in {path} ---")
    print(bug["fix"])
    ans = input("Apply this fix? This replaces that single line with the text above. [y/N] ").strip().lower()
    if ans != "y":
        print("Skipped.")
        return False
    lines[ln - 1] = bug["fix"].rstrip("\n") + "\n"
    path.write_text("".join(lines), encoding="utf-8")
    print("Applied.")
    return True

# --------------------------------------------------------------------------
# REPORTING
# --------------------------------------------------------------------------
SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "style": 4}
SEV_ICON = {"critical": "[CRITICAL]", "high": "[HIGH]", "medium": "[MEDIUM]", "low": "[LOW]", "style": "[STYLE]"}


def print_bug(path, bug):
    sev = bug.get("severity", "medium")
    print(f"\n{SEV_ICON.get(sev, '[?]')} {path}:{bug.get('line', '?')} - {bug.get('title', 'Untitled issue')}")
    print(f"    {bug.get('explanation', '')}")
    if bug.get("failure_case"):
        print(f"    Triggered by: {bug['failure_case']}")
    if bug.get("fix"):
        print(f"    Suggested fix:\n      " + bug["fix"].replace("\n", "\n      "))


def write_report(all_bugs, target):
    REPORT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = REPORT_DIR / f"report_{stamp}.md"
    lines = [f"# Debug report for {target}", f"Generated: {time.strftime('%Y-%m-%d %H:%M')}", ""]
    total = sum(len(b) for _, b in all_bugs)
    lines.append(f"**{total} issue(s) found across {len(all_bugs)} file(s).**\n")
    for path, bugs in all_bugs:
        if not bugs:
            continue
        lines.append(f"## {path}\n")
        for b in sorted(bugs, key=lambda x: SEV_ORDER.get(x.get("severity", "medium"), 2)):
            lines.append(f"### {SEV_ICON.get(b.get('severity','medium'),'')} Line {b.get('line','?')}: {b.get('title','')}")
            lines.append(f"{b.get('explanation','')}\n")
            if b.get("failure_case"):
                lines.append(f"**Triggered by:** {b['failure_case']}\n")
            if b.get("fix"):
                lines.append(f"**Suggested fix:**\n```\n{b['fix']}\n```\n")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out

# --------------------------------------------------------------------------
# COMMANDS
# --------------------------------------------------------------------------
def scan(target, lang=None, security=False, apply_fixes=False, quiet=False, deep=False):
    files = find_files(target, lang)
    if not files:
        print("No matching source files found.")
        return
    print(f"Scanning {len(files)} file(s)" + (" in DEEP mode (slower, stronger model)..." if deep else "..."))
    all_bugs = []
    for path in files:
        if not quiet:
            print(f"\n[scanning] {path}")
        bugs, error = review_file(path, security=security, deep=deep)
        if error:
            print(f"   note: {error}")
        all_bugs.append((path, bugs))
        for b in bugs:
            print_bug(path, b)
        if apply_fixes and bugs:
            for b in sorted(bugs, key=lambda x: SEV_ORDER.get(x.get("severity", "medium"), 2)):
                apply_fix(path, b)

    report = write_report(all_bugs, target)
    total = sum(len(b) for _, b in all_bugs)
    print(f"\n{'='*60}\n{total} issue(s) found. Full report saved to: {report}\n{'='*60}")


def watch(target, lang=None, security=False, interval=3, deep=False):
    print(f"Watching {target} for changes (Ctrl+C to stop)...")
    seen = {}

    def hash_file(p):
        try:
            return hashlib.md5(p.read_bytes()).hexdigest()
        except OSError:
            return None

    try:
        while True:
            for path in find_files(target, lang):
                h = hash_file(path)
                if h and seen.get(path) != h:
                    seen[path] = h
                    print(f"\n[changed] {path}")
                    bugs, error = review_file(path, security=security, deep=deep)
                    if error:
                        print(f"   note: {error}")
                    for b in bugs:
                        print_bug(path, b)
                    if not bugs and not error:
                        print("   no issues found")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


def main():
    p = argparse.ArgumentParser(description="AI-assisted debugger for your own code")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="scan a file or folder once")
    s.add_argument("target")
    s.add_argument("--lang", help="limit to one language, e.g. python")
    s.add_argument("--security", action="store_true", help="extra focus on security bugs")
    s.add_argument("--deep", action="store_true",
                   help=f"strongest model ({CLAUDE_DEEP_MODEL} / {GEMINI_DEEP_MODEL}), exhaustive review, up to {DEEP_CODE_CHARS:,} chars/file")
    s.add_argument("--apply", action="store_true", help="interactively apply suggested fixes")
    s.add_argument("--quiet", action="store_true", help="less progress output")

    w = sub.add_parser("watch", help="re-scan files as they change")
    w.add_argument("target")
    w.add_argument("--lang")
    w.add_argument("--security", action="store_true")
    w.add_argument("--deep", action="store_true", help="strongest model, exhaustive review")

    args = p.parse_args()
    if args.cmd == "scan":
        scan(args.target, args.lang, args.security, args.apply, args.quiet, args.deep)
    elif args.cmd == "watch":
        watch(args.target, args.lang, args.security, deep=args.deep)


if __name__ == "__main__":
    main()
