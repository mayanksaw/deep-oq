"""
Run in the same folder as deep_oq.py:

    python upgrade_smart.py

Makes a backup (deep_oq.py.bak3), applies the upgrade, and checks the file still compiles.
Safe to run twice.
"""
import py_compile
import shutil
import sys

PATH = "deep_oq.py"

with open(PATH, encoding="utf-8") as f:
    src = f.read()
orig = src
done = []


def once(old, new, label, already):
    global src
    if already in src:
        done.append("skipped (already applied): " + label)
        return
    n = src.count(old)
    if n != 1:
        sys.exit("STOPPED: '%s' expected 1 match but found %d. Nothing was changed." % (label, n))
    src = src.replace(old, new)
    done.append("applied: " + label)


# 1. chat_stream can now think at a chosen level
once(
    "def chat_stream(messages, temperature, max_tokens, on_text):",
    "def chat_stream(messages, temperature, max_tokens, on_text, effort=None):",
    "1. streaming answers can use high thinking",
    "def chat_stream(messages, temperature, max_tokens, on_text, effort=None):",
)
once(
    '            kw = _call_kw(m, messages, temperature, max_tokens)\n            kw["stream"] = True',
    '            kw = _call_kw(m, messages, temperature, max_tokens, effort)\n            kw["stream"] = True',
    "1b. pass the thinking level to the model",
    "_call_kw(m, messages, temperature, max_tokens, effort)",
)

# 2. a real question gets the highest thinking level; greetings stay fast
once(
    "def _deep_oq_raw(session, question, force_exam=False, emit=None):",
    '''def is_substantive(text):
    """False for greetings and tiny messages, True for a real question or request."""
    t = (text or "").strip()
    if _MEM_SKIP.match(t) and len(t) < 40:
        return False
    return len(t.split()) >= 4 or "?" in t


def _deep_oq_raw(session, question, force_exam=False, emit=None):''',
    "2. helper that spots real questions",
    "def is_substantive(text):",
)
once(
    '        return chat_stream(base + [{"role": "user", "content": user}], 0.5, OQ_OUT, lambda t: emit("text", t))',
    '        return chat_stream(base + [{"role": "user", "content": user}], 0.4, OQ_OUT, lambda t: emit("text", t),\n'
    '                           effort="high" if is_substantive(question) else None)',
    "2b. real questions use high thinking and still stream",
    'effort="high" if is_substantive(question) else None',
)

if src == orig:
    print("Nothing to change. Already applied.")
else:
    shutil.copyfile(PATH, PATH + ".bak3")
    with open(PATH, "w", encoding="utf-8") as f:
        f.write(src)
    try:
        py_compile.compile(PATH, doraise=True)
    except py_compile.PyCompileError as e:
        shutil.copyfile(PATH + ".bak3", PATH)
        sys.exit("The patched file did not compile, so your original was restored.\n" + str(e))

print("\n".join(done))
print("Done. Backup saved as deep_oq.py.bak3")
