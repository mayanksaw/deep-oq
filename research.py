"""
research.py - deep research for Deep OQ.

Idea: instead of one giant prompt (which Groq's free tier rejects with 413),
split the job into many SMALL requests and stitch the answers together:

  1. PLAN      - model breaks the topic into sub-questions
  2. GATHER    - search + read web pages for each sub-question
  3. EXTRACT   - model turns each pile of pages into dense notes
  4. WRITE     - model writes one long section per sub-question
  5. SUMMARY   - model writes an executive summary

Every request stays under ~5k tokens, but the final report can be 10,000+
tokens long. A rate limiter keeps you under the per-minute token limit.

Env vars:
  GROQ_KEYS       comma-separated Groq keys (key1,key2,...)
  GROQ_MODELS     comma-separated model IDs, tried in order
                  (default: openai/gpt-oss-20b,llama-3.3-70b-versatile)
                  Check real IDs: GET https://api.groq.com/openai/v1/models
  GEMINI_API_KEY  optional backup provider
  GEMINI_MODEL    default: gemini-2.0-flash
  TPM_BUDGET      tokens per minute to allow (default 6000; raise it on a paid tier)

Use:
  from research import research
  report = research("impact of AI on healthcare in India", sections=6)
"""
import html as _html
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

GROQ_KEYS = [k.strip() for k in os.environ.get("GROQ_KEYS", "").split(",") if k.strip()]
GROQ_MODELS = [
    m.strip()
    for m in os.environ.get("GROQ_MODELS", "openai/gpt-oss-20b,llama-3.3-70b-versatile").split(",")
    if m.strip()
]
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

TPM_BUDGET = int(os.environ.get("TPM_BUDGET", "6000"))  # stay under Groq's 8000
MAX_IN_TOKENS = 3500   # max prompt size per request
NOTE_TOKENS = 1200     # output size when extracting notes
SECTION_TOKENS = 1500  # output size per report section

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"


# ---------------------------------------------------------------- rate limiter
_lock = threading.Lock()
_window = []  # (timestamp, tokens) for the last 60 seconds


def est_tokens(text):
    return len(text) // 4 + 1


def _wait_for_budget(tokens):
    """Block until sending `tokens` keeps us under TPM_BUDGET for the last 60s."""
    while True:
        with _lock:
            now = time.time()
            _window[:] = [(t, n) for t, n in _window if now - t < 60]
            used = sum(n for _, n in _window)
            if not _window or used + tokens <= TPM_BUDGET:
                _window.append((now, tokens))
                return
            wait = 60 - (now - _window[0][0]) + 0.5
        time.sleep(max(wait, 1))


# ------------------------------------------------------------------- LLM calls
def _post(url, payload, headers, timeout=90):
    req = urllib.request.Request(
        url,
        json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "deep-oq/1.0", **headers},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _groq(messages, max_out):
    for model in GROQ_MODELS:
        for key in GROQ_KEYS:
            payload = {"model": model, "messages": messages, "max_tokens": max_out, "temperature": 0.3}
            if "gpt-oss" in model:
                payload["reasoning_effort"] = "low"  # keeps tokens for the real answer
            try:
                data = _post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    payload,
                    {"Authorization": "Bearer " + key},
                )
                text = data["choices"][0]["message"].get("content") or ""
                if text.strip():
                    return text
            except urllib.error.HTTPError as e:
                if e.code in (404, 413):  # bad model ID / too big: other keys fail too
                    print(f"[research] {model}: HTTP {e.code}, skipping model")
                    break
                if e.code == 429:         # rate limit: try the next key
                    time.sleep(1)
                    continue
                print(f"[research] {model}: HTTP {e.code}")
            except Exception as e:
                print(f"[research] {model}: {e}")
    return None


def _gemini(system, prompt, max_out):
    if not GEMINI_KEY:
        return None
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_out, "temperature": 0.3},
    }
    try:
        data = _post(url, payload, {})
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"[research] gemini failed: {e}")
        return None


def llm(prompt, system="You are a careful, thorough research assistant.", max_out=SECTION_TOKENS):
    prompt = prompt[: MAX_IN_TOKENS * 4]
    _wait_for_budget(est_tokens(prompt) + est_tokens(system) + max_out)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    for attempt in range(3):
        text = _groq(messages, max_out) or _gemini(system, prompt, max_out)
        if text:
            return text
        time.sleep(20 * (attempt + 1))  # everything busy: wait, then retry
    raise RuntimeError("All providers are busy. Try again in a minute.")


# ------------------------------------------------------------------ web access
def search_urls(query, n=3):
    """Default search: DuckDuckGo HTML. Swap this for your own browser/search tool."""
    try:
        req = urllib.request.Request(
            "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query),
            headers={"User-Agent": UA},
        )
        page = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
    except Exception as e:
        print(f"[research] search failed: {e}")
        return []
    urls = []
    for href in re.findall(r'class="result__a"[^>]*href="([^"]+)"', page):
        m = re.search(r"uddg=([^&]+)", href)
        u = urllib.parse.unquote(m.group(1)) if m else href
        if u.startswith("//"):
            u = "https:" + u
        if u.startswith("http") and u not in urls:
            urls.append(u)
        if len(urls) >= n:
            break
    return urls


def fetch_text(url, limit=5000):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        raw = urllib.request.urlopen(req, timeout=20).read(400_000).decode("utf-8", "ignore")
    except Exception:
        return ""
    raw = re.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", " ", raw)
    text = _html.unescape(re.sub(r"(?s)<[^>]+>", " ", raw))
    return re.sub(r"\s+", " ", text).strip()[:limit]


def gather(question, pages=3):
    out = []
    for url in search_urls(question, pages):
        txt = fetch_text(url)
        if txt:
            out.append(f"[{url}]\n{txt}")
    return out


# -------------------------------------------------------------------- pipeline
def _parse_list(text):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        return [str(x).strip() for x in json.loads(m.group(0)) if str(x).strip()]
    except Exception:
        return []


def research(topic, sections=6, progress=print):
    """Return a long, structured research report (markdown) for `topic`."""
    progress(f"Planning research on: {topic}")
    plan = llm(
        f"Topic: {topic}\n\nBreak this into {sections} distinct, specific research "
        "sub-questions that together cover the topic completely. "
        "Return ONLY a JSON array of strings.",
        max_out=600,
    )
    questions = _parse_list(plan)[:sections] or [topic]

    notes = []
    for i, q in enumerate(questions, 1):
        progress(f"[{i}/{len(questions)}] Researching: {q}")
        pages = gather(q)
        sources = "\n\n".join(pages)[: MAX_IN_TOKENS * 4 - 900] or "(no web pages found - use your own knowledge)"
        note = llm(
            f"Question: {q}\n\nSources:\n{sources}\n\n"
            "Extract every important fact, number, name, date, example and disagreement "
            "relevant to the question. Be dense, no filler. Put the source URL after facts "
            "taken from a page.",
            max_out=NOTE_TOKENS,
        )
        notes.append((q, note))

    parts = []
    for i, (q, note) in enumerate(notes, 1):
        progress(f"[{i}/{len(notes)}] Writing section: {q}")
        sec = llm(
            f"Overall topic: {topic}\nSection question: {q}\n\nResearch notes:\n{note}\n\n"
            "Write a detailed, well-structured report section of 600-900 words. Use the notes "
            "only for facts, explain clearly, include concrete numbers and examples, and keep "
            "source URLs where given. Do not repeat the question as a heading.",
            max_out=SECTION_TOKENS,
        )
        parts.append(f"## {q}\n\n{sec}")

    progress("Writing executive summary")
    digest = "\n\n".join(f"{q}\n{n[:700]}" for q, n in notes)
    summary = llm(
        f"Topic: {topic}\n\nKey findings:\n{digest}\n\n"
        "Write an executive summary (about 250 words) with the main conclusions and "
        "3-5 actionable takeaways.",
        max_out=700,
    )
    return f"# {topic}\n\n## Executive summary\n\n{summary}\n\n" + "\n\n".join(parts)


if __name__ == "__main__":
    if not GROQ_KEYS and not GEMINI_KEY:
        sys.exit("Set GROQ_KEYS (and/or GEMINI_API_KEY) first.")
    print(research(" ".join(sys.argv[1:]) or "state of renewable energy in India"))
