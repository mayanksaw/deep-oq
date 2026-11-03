"""
ai_advanced.py - Deep OQ advanced intelligence layer
=====================================================
Sits on top of ai_tools.py and adds the things that make an AI system
much stronger than a plain chatbot:

  * agent()             - think -> pick a tool -> read result -> repeat (ReAct loop)
  * solve_complex()     - plan the goal, solve each step, then verify the result
  * self_consistent()   - ask several times and vote (big accuracy boost for math/logic)
  * reflect()           - draft -> harsh critique -> rewrite, repeated
  * verify()            - independent check of an answer
  * ensemble()          - ask several models, then a judge merges the best answer
  * KnowledgeBase       - local RAG (TF-IDF search over your own documents)
  * remember()/recall() - persistent long-term memory in a JSON file
  * research()          - search the web, read pages, write a cited answer
  * web_fetch()         - safe page reader (blocks localhost / private networks)
  * long_summarize()    - map-reduce summary for very long text
  * structured()        - JSON output with automatic retry
  * run_python()        - run code (OFF by default, see warning below)
  * reasoning_gate()    - send exact math / counting to real code instead of guessing
  * smart()             - one entry point that picks the best strategy

Setup (same model function you gave ai_tools):

    import ai_tools, ai_advanced as adv
    ai_tools.set_llm(my_llm)
    adv.set_search(my_search)          # optional: fn(query) -> [{"title","url","snippet"}]
    adv.set_models({"main": my_llm})   # optional: add more models for ensemble()

    adv.smart("What is 1234*5678?")                # exact, via calculator
    adv.agent("Plan a 3-acre tomato farm budget")  # multi-step with tools
    adv.solve_complex("Design a trading bot app")  # plan -> solve -> verify

Honest note: these techniques make any model noticeably better, but the base
model is still the biggest factor. Use the strongest model you can afford.
"""

import hashlib
import ipaddress
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from urllib.parse import urljoin, urlparse

import ai_tools as T

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------

# WARNING: run_python executes code on YOUR server. It is not a real sandbox.
# Leave False unless the server is isolated (container, no secrets, no network).
ALLOW_CODE_EXEC = False

MEMORY_FILE = "deep_oq_memory.json"

_SEARCH = None
_MODELS = {}
_CACHE = {}


def set_search(fn):
    """fn(query) -> list of {"title": str, "url": str, "snippet": str}"""
    global _SEARCH
    _SEARCH = fn


def set_models(models):
    """models = {"name": fn(prompt, system='', image_url=None) -> str}"""
    _MODELS.clear()
    _MODELS.update(models)


def ask(prompt, system="", image_url=None):
    return T._ask(prompt, system=system, image_url=image_url)


def cached_ask(prompt, system="", ttl=600):
    """Same question twice = instant answer, no API cost."""
    key = hashlib.sha256((system + "\x00" + prompt).encode()).hexdigest()
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    out = ask(prompt, system=system)
    _CACHE[key] = (time.time(), out)
    return out


# ----------------------------------------------------------------------
# Structured output
# ----------------------------------------------------------------------

def extract_json(text):
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    if not m:
        raise ValueError("no JSON found")
    return json.loads(m.group(0))


def structured(prompt, shape, retries=2, system=""):
    """Get valid JSON from the model, retrying with a correction if it fails."""
    last = ""
    for _ in range(retries + 1):
        out = ask(prompt + f"\n\nReply with ONLY valid JSON shaped like: {shape}", system)
        try:
            return extract_json(out)
        except Exception as e:
            last = str(e)
            prompt += "\n\nYour previous reply was not valid JSON. Reply with JSON only."
    raise ValueError(f"Model did not return valid JSON ({last})")


# ----------------------------------------------------------------------
# Accuracy boosters
# ----------------------------------------------------------------------

def self_consistent(question, n=5):
    """Solve n times independently, return the majority answer."""
    answers = []
    for _ in range(int(n)):
        out = ask(question, "Solve carefully step by step. End with one line: FINAL: <answer>")
        found = re.findall(r"FINAL:\s*(.+)", out)
        if found:
            answers.append(found[-1].strip().rstrip("."))
    if not answers:
        return ask(question)
    best, count = Counter(answers).most_common(1)[0]
    return f"{best}  (agreement {count}/{len(answers)})"


def reflect(task, rounds=2):
    """Draft -> critique -> rewrite. Stops early if the critic is satisfied."""
    draft = ask(task, "Answer as accurately and completely as you can.")
    for _ in range(int(rounds)):
        critique = ask(
            f"Task:\n{task}\n\nDraft:\n{draft}",
            "You are a harsh reviewer. List concrete errors, gaps and improvements. "
            "If it is already excellent, reply with exactly: OK",
        )
        if critique.strip().upper().startswith("OK"):
            break
        draft = ask(
            f"Task:\n{task}\n\nDraft:\n{draft}\n\nReviewer notes:\n{critique}",
            "Rewrite the draft and fix every note. Output only the improved answer.",
        )
    return draft


def verify(question, answer):
    """Independent second opinion on an answer."""
    return ask(
        f"Question:\n{question}\n\nProposed answer:\n{answer}",
        "You are an independent verifier. Re-derive the answer yourself. "
        "Say CORRECT or INCORRECT first, then explain, then give the corrected answer if needed.",
    )


def ensemble(prompt, system=""):
    """Ask every connected model, then let a judge merge the best answer."""
    if not _MODELS:
        return ask(prompt, system)
    drafts = {}
    for name, fn in _MODELS.items():
        try:
            drafts[name] = fn(prompt, system=system)
        except Exception as e:
            drafts[name] = f"[{name} failed: {e}]"
    if len(drafts) == 1:
        return next(iter(drafts.values()))
    joined = "\n\n".join(f"### Answer from {n}\n{a}" for n, a in drafts.items())
    judge = next(iter(_MODELS.values()))
    return judge(
        f"Question:\n{prompt}\n\n{joined}",
        system="You are a judge. Compare the answers, discard mistakes, and write the single best, "
               "most accurate final answer, combining the strongest parts.",
    )


def long_summarize(text, chunk_words=1200, _depth=0):
    """Map-reduce summary so even a book-length text fits the model."""
    words = text.split()
    if len(words) <= chunk_words or _depth >= 3:
        return ask(text, "Summarize faithfully with the key points.")
    parts = [
        ask(" ".join(words[i:i + chunk_words]), "Summarize this section faithfully in 5-8 bullets.")
        for i in range(0, len(words), chunk_words)
    ]
    combined = "\n\n".join(parts)
    if len(combined.split()) > chunk_words:
        return long_summarize(combined, chunk_words, _depth + 1)
    return ask(combined, "Combine these section summaries into one coherent summary with key points.")


# ----------------------------------------------------------------------
# Reasoning gate: exact answers for math and counting
# ----------------------------------------------------------------------

def reasoning_gate(message):
    """Return an exact answer for plain math / counting questions, else None."""
    m = message.strip().rstrip("=?").strip()
    if re.fullmatch(r"[\d\s.+\-*/()%^,]+", m) and re.search(r"\d", m):
        try:
            return T.run_tool("calculator", expression=m.replace("^", "**").replace(",", ""))
        except Exception:
            return None
    q = re.search(r"[\"“'`](.+)[\"”'`]", message, re.S)
    if q and re.search(r"how many (words|characters|letters|sentences)|count (the )?(words|characters|letters)",
                       message, re.I):
        text = q.group(1)
        stats = json.loads(T.run_tool("word_count", text=text))
        if re.search(r"letter", message, re.I):
            stats["letters"] = len(re.findall(r"[A-Za-z]", text))
        return json.dumps(stats)
    return None


# ----------------------------------------------------------------------
# Long-term memory
# ----------------------------------------------------------------------

def _load_mem():
    try:
        with open(MEMORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def remember(key, value):
    mem = _load_mem()
    mem[key] = {"value": value, "time": time.strftime("%Y-%m-%d %H:%M")}
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(mem, f, ensure_ascii=False, indent=2)
    return f"Remembered '{key}'."


def recall(query):
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    for k, v in _load_mem().items():
        words = set(re.findall(r"[a-z0-9]+", (k + " " + str(v["value"])).lower()))
        overlap = len(q & words)
        if overlap:
            scored.append((overlap, k, v["value"]))
    scored.sort(reverse=True)
    return "\n".join(f"{k}: {val}" for _, k, val in scored[:5]) or "Nothing found."


# ----------------------------------------------------------------------
# Local RAG: answer from your own documents
# ----------------------------------------------------------------------

def _tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class KnowledgeBase:
    """TF-IDF search over chunks of your documents. No extra libraries needed."""

    def __init__(self):
        self.chunks = []

    def add(self, doc_id, text, size=120, overlap=30):
        words = text.split()
        step = max(1, size - overlap)
        for i in range(0, max(1, len(words)), step):
            piece = " ".join(words[i:i + size])
            if piece:
                self.chunks.append({"doc": doc_id, "text": piece, "tf": Counter(_tok(piece))})
            if i + size >= len(words):
                break
        return len(self.chunks)

    def search(self, query, k=4):
        n = len(self.chunks)
        if not n:
            return []
        df = Counter()
        for c in self.chunks:
            df.update(c["tf"].keys())

        def vec(tf):
            return {t: (1 + math.log(f)) * (math.log((n + 1) / (df.get(t, 0) + 1)) + 1) for t, f in tf.items()}

        qv = vec(Counter(_tok(query)))
        qn = math.sqrt(sum(v * v for v in qv.values())) or 1
        scored = []
        for c in self.chunks:
            cv = vec(c["tf"])
            cn = math.sqrt(sum(v * v for v in cv.values())) or 1
            dot = sum(w * cv.get(t, 0) for t, w in qv.items())
            scored.append((dot / (qn * cn), c))
        scored.sort(key=lambda x: -x[0])
        return [(s, c) for s, c in scored[:int(k)] if s > 0]

    def answer(self, question, k=4):
        hits = self.search(question, k)
        if not hits:
            return "I could not find this in the documents."
        context = "\n\n".join(f"[{c['doc']}] {c['text']}" for _, c in hits)
        return ask(
            f"Question: {question}\n\nDocuments:\n{context}",
            "Answer using ONLY the documents. Cite the [doc] names. If the answer is not there, say so.",
        )


KB = KnowledgeBase()


# ----------------------------------------------------------------------
# Web: safe fetch + research
# ----------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def _public_url(url):
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    try:
        for info in socket.getaddrinfo(p.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
    except Exception:
        return False
    return True


def web_fetch(url, max_chars=8000):
    """Read a public web page as plain text. Blocks internal addresses and re-checks redirects."""
    opener = urllib.request.build_opener(_NoRedirect)
    for _ in range(4):
        if not _public_url(url):
            return "ERROR: blocked or invalid URL."
        req = urllib.request.Request(url, headers={"User-Agent": "DeepOQ/1.0"})
        try:
            with opener.open(req, timeout=10) as r:
                raw = r.read(600_000).decode("utf-8", errors="replace")
            break
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                url = urljoin(url, e.headers["Location"])
                continue
            return f"ERROR: HTTP {e.code}"
        except Exception as e:
            return f"ERROR: {e}"
    else:
        return "ERROR: too many redirects."
    raw = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", text).strip()[:int(max_chars)]


def research(question, max_pages=4):
    """Search, read the best pages, and write a cited answer."""
    if _SEARCH is None:
        return "No search connected. Call ai_advanced.set_search(fn) first."
    queries = structured(f"Question: {question}", '["search query 1", "search query 2", "search query 3"]')
    seen, sources = set(), []
    for q in queries[:3]:
        for r in _SEARCH(q)[:3]:
            if r.get("url") and r["url"] not in seen:
                seen.add(r["url"])
                sources.append(r)
    notes = []
    for i, s in enumerate(sources[:int(max_pages)], 1):
        page = web_fetch(s["url"], 6000)
        if page.startswith("ERROR"):
            page = s.get("snippet", "")
        if page:
            notes.append(f"[{i}] {s.get('title', s['url'])}\n" + ask(
                f"Question: {question}\n\nPage text:\n{page}",
                "Extract only the facts relevant to the question. Be exact with numbers. Short bullets."))
    if not notes:
        return "I could not find usable sources."
    answer = ask(
        f"Question: {question}\n\nNotes from sources:\n\n" + "\n\n".join(notes),
        "Write a clear, accurate answer using ONLY these notes. Cite sources as [1], [2]. "
        "If sources disagree or are weak, say so.",
    )
    refs = "\n".join(f"[{i}] {s.get('title', '')} - {s['url']}" for i, s in enumerate(sources[:int(max_pages)], 1))
    return f"{answer}\n\nSources:\n{refs}"


# ----------------------------------------------------------------------
# Code execution (off by default)
# ----------------------------------------------------------------------

def run_python(code, timeout=10):
    if not ALLOW_CODE_EXEC:
        return ("Code execution is disabled. Set ai_advanced.ALLOW_CODE_EXEC = True only on an "
                "isolated server (container, no secrets).")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "main.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            r = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True,
                               timeout=int(timeout), cwd=d)
        except subprocess.TimeoutExpired:
            return "Timed out."
        return (r.stdout + r.stderr)[:4000] or "(no output)"


# ----------------------------------------------------------------------
# Agents
# ----------------------------------------------------------------------

_AGENT_SYSTEM = """You are the Deep OQ agent. Solve the goal step by step using tools.
Each turn reply with ONLY one JSON object:
{"thought": "short reasoning", "action": "<tool name or finish>", "args": {"param": "value"}, "answer": "final answer when action is finish"}
Rules: use calculator or word_count for exact numbers, never guess them. Use web/research tools for fresh facts.
Finish as soon as you can answer well."""


def _catalogue(allowed):
    lines = []
    for n in allowed:
        t = T.TOOLS[n]
        lines.append(f"- {n}({', '.join(t['params'])}): {t['description']}")
    return "\n".join(lines)


def agent(goal, max_steps=8, allowed=None):
    """ReAct loop: the model picks tools, sees results, and continues until done."""
    allowed = [n for n in (allowed or T.TOOLS) if n in T.TOOLS and n != "agent"]
    system = _AGENT_SYSTEM + "\n\nTools:\n" + _catalogue(allowed)
    history = []
    for _ in range(int(max_steps)):
        out = ask(f"Goal: {goal}\n\n" + "\n".join(history), system)
        try:
            msg = extract_json(out)
        except Exception:
            history.append("Observation: reply was not valid JSON. Reply with one JSON object only.")
            continue
        action = str(msg.get("action", "")).strip()
        if action == "finish":
            return msg.get("answer", "")
        args = msg.get("args") or {}
        if action not in allowed or not isinstance(args, dict):
            obs = f"Unknown tool or bad args: {action}"
        else:
            obs = str(T.run_tool(action, **args))[:3000]
        history.append(f"Thought: {msg.get('thought', '')}\nAction: {action} {json.dumps(args, ensure_ascii=False)}\nObservation: {obs}")
    return ask(f"Goal: {goal}\n\nWork so far:\n" + "\n".join(history),
               "Give the best final answer from the work so far. Say what is uncertain.")


def plan(goal):
    steps = structured(f"Goal: {goal}", '{"steps": ["step 1", "step 2"]}')
    return steps.get("steps", [])


def solve_complex(goal, verify_result=True):
    """Plan -> solve each step with context -> combine -> verify."""
    steps = plan(goal)[:8]
    context = []
    for i, step in enumerate(steps, 1):
        out = ask(
            f"Overall goal: {goal}\n\nDone so far:\n" + "\n\n".join(context) + f"\n\nNow do step {i}: {step}",
            "Complete this step fully and accurately. Output the result of the step only.",
        )
        context.append(f"Step {i} ({step}):\n{out}")
    final = ask(
        f"Goal: {goal}\n\nResults:\n" + "\n\n".join(context),
        "Combine the step results into one complete, well-organised final deliverable.",
    )
    if verify_result:
        check = verify(goal, final)
        if check.strip().upper().startswith("INCORRECT"):
            final = ask(f"Goal: {goal}\n\nDraft:\n{final}\n\nReviewer:\n{check}",
                        "Fix the draft using the reviewer notes. Output only the corrected deliverable.")
    return final


def smart(message):
    """One entry point: exact tools first, then the best strategy for the question."""
    exact = reasoning_gate(message)
    if exact is not None:
        return exact
    hits = KB.search(message, 3)
    if hits and hits[0][0] > 0.35:
        return KB.answer(message)
    verdict = ask(
        message,
        "Classify the request. Reply with ONE word: SIMPLE (chat, short answer), MATH (calculation or logic puzzle), "
        "COMPLEX (needs planning or many parts), TOOL (best done with a specialised tool).",
    ).strip().upper()
    if verdict.startswith("MATH"):
        return self_consistent(message, 3)
    if verdict.startswith("COMPLEX"):
        return solve_complex(message)
    if verdict.startswith("TOOL"):
        return T.auto(message)
    return reflect(message, rounds=1)


# ----------------------------------------------------------------------
# Register everything as tools so run_tool / the agent / function-calling can use them
# ----------------------------------------------------------------------

@T.tool("agent", "advanced", "Multi-step agent that uses tools to reach a goal.", {"goal": "string"})
def _t_agent(goal):
    return agent(goal)


@T.tool("solve_complex", "advanced", "Plan, solve step by step, and verify a big task.", {"goal": "string"})
def _t_solve(goal):
    return solve_complex(goal)


@T.tool("self_consistent", "advanced", "Solve several times and vote for the most reliable answer.", {"question": "string"})
def _t_sc(question):
    return self_consistent(question)


@T.tool("reflect", "advanced", "Draft, self-critique and improve an answer.", {"task": "string"})
def _t_reflect(task):
    return reflect(task)


@T.tool("verify", "advanced", "Independently check whether an answer is correct.", {"question": "string", "answer": "string"})
def _t_verify(question, answer):
    return verify(question, answer)


@T.tool("ensemble", "advanced", "Ask multiple models and merge the best answer.", {"prompt": "string"})
def _t_ensemble(prompt):
    return ensemble(prompt)


@T.tool("long_summarize", "advanced", "Summarize very long text.", {"text": "string"})
def _t_long(text):
    return long_summarize(text)


@T.tool("research", "advanced", "Search the web, read pages, answer with citations.", {"question": "string"})
def _t_research(question):
    return research(question)


@T.tool("web_fetch", "advanced", "Read a public web page as text.", {"url": "string"})
def _t_fetch(url):
    return web_fetch(url)


@T.tool("kb_add", "advanced", "Add a document to the knowledge base.", {"doc_id": "string", "text": "string"})
def _t_kb_add(doc_id, text):
    return f"Knowledge base now has {KB.add(doc_id, text)} chunks."


@T.tool("kb_ask", "advanced", "Answer a question from the knowledge base documents.", {"question": "string"})
def _t_kb_ask(question):
    return KB.answer(question)


@T.tool("remember", "advanced", "Save a fact to long-term memory.", {"key": "string", "value": "string"})
def _t_remember(key, value):
    return remember(key, value)


@T.tool("recall", "advanced", "Look up facts from long-term memory.", {"query": "string"})
def _t_recall(query):
    return recall(query)


@T.tool("run_python", "advanced", "Run Python code (only if enabled on the server).", {"code": "string"})
def _t_run(code):
    return run_python(code)


@T.tool("plan", "advanced", "Break a goal into ordered steps.", {"goal": "string"})
def _t_plan(goal):
    return json.dumps(plan(goal), ensure_ascii=False)


if __name__ == "__main__":
    print(f"Deep OQ total tools: {len(T.TOOLS)}")
    print("Advanced:", ", ".join(T.list_tools('advanced')))
