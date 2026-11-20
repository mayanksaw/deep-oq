"""
Run in the same folder as deep_oq.py:

    python upgrade_answers.py

Makes a backup (deep_oq.py.bak2), applies the upgrades, and checks the file still compiles.
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


# ---------------------------------------------------------------------------
# 1. Better thinking effort for everyday chat (set DEEP_OQ_FAST_EFFORT=low for more speed)
# ---------------------------------------------------------------------------
once(
    'OQ_OUT = _env_int("DEEP_OQ_MAX_OUT", 12288)',
    'FAST_EFFORT = os.environ.get("DEEP_OQ_FAST_EFFORT", "medium").strip() or "medium"  # low, medium or high\n'
    'OQ_OUT = _env_int("DEEP_OQ_MAX_OUT", 12288)',
    "1. everyday chat thinks harder (medium effort)",
    "FAST_EFFORT = os.environ.get",
)
once(
    'kw["reasoning_effort"] = effort or "low"',
    'kw["reasoning_effort"] = effort or FAST_EFFORT',
    "1b. use that effort setting",
    "effort or FAST_EFFORT",
)

# ---------------------------------------------------------------------------
# 2. Answer-writing rules + a full self-introduction
# ---------------------------------------------------------------------------
STYLE_BLOCK = r'''
# ==========================================================================
# ANSWER QUALITY: how Deep OQ understands a question and writes the answer
# ==========================================================================
ANSWER_STYLE = (
    " ANSWER QUALITY. Before writing, silently work out what the person really wants, what they probably "
    "already know, and what would make the answer genuinely useful. Then: "
    "(1) Answer the real question first, in one or two sentences. "
    "(2) Give the reasoning, steps or explanation that matter, in a logical order where each step builds on the last. "
    "(3) Use one concrete example or a short analogy when an idea is abstract. "
    "(4) Point out the one or two things most likely to go wrong (a common mistake, an edge case, a caveat). "
    "(5) Stop when the question is answered: no filler, no repeating the question, no 'I hope this helps'. "
    "Match depth to the question: a greeting or a simple fact gets one or two sentences; a how or why question "
    "gets a clear explanation; a hard or multi-part problem gets a full structured solution. "
    "If a question can mean two things, answer the most likely meaning and state your assumption in one short line. "
    "Never leave an answer half-finished and never repeat a sentence or a paragraph. "
    "For code: working code first, then how to run it and the key idea. "
    "For maths and science: what is given, the method, steps with the reason for each, a check, then the final answer in bold. "
    "For comparisons: say which is better for what, and why. "
    "For advice: give a clear recommendation, not a list of options with no choice. "
    "These rules override any earlier instruction about length."
)
SYSTEM += ANSWER_STYLE

SELF_Q = re.compile(r"\b(who are you|what are you|tell me about (you|yourself)|about yourself|introduce yourself|"
                    r"your introduction|full introduction|what can you do|your features|your capabilities)\b", re.I)
SELF_INFO = (
    " ABOUT YOU (use this when asked who you are or what you can do; never name any underlying model or company): "
    "You are Deep OQ, created by Mayank Kumar Saw. In this app you can: answer questions and explain ideas; solve "
    "maths, science and exam problems step by step with a second-model check; use Research mode to search many sites "
    "and write a cited answer; read pasted links, PDFs, photos and files; remember facts about the user in Memory; "
    "use Knowledge notes the user adds. Code: write, debug, fix, explain, convert and review code in 16 languages, "
    "and run Python and JavaScript in the browser. Photo and Video editors, a video maker, and an image maker. "
    "Lab: photo vision, data analysis, business help, robot, storyboards, music, farming and energy. "
    "News and daily current affairs. Music. A VR guide for Meta Quest with games, chess, YouTube and movies. "
    "Write a warm, well organised introduction with short headings or bullets, about 150 to 250 words, "
    "and end by asking what the person wants to do."
)


'''
once(
    "def deep_oq(session, question, force_exam=False, emit=None):\n"
    "    return plain_math(_deep_oq_raw(session, question, force_exam, emit))",
    STYLE_BLOCK.lstrip("\n") + "def deep_oq(session, question, force_exam=False, emit=None):\n"
    "    return plain_math(_deep_oq_raw(session, question, force_exam, emit))",
    "2. answer-writing rules and full self-introduction",
    "ANSWER_STYLE = (",
)
once(
    '(REASON_RULES if rigorous else "")',
    '(REASON_RULES if rigorous else "") + (SELF_INFO if SELF_Q.search(question) else "")',
    "2b. self-introduction is used when asked about Deep OQ",
    "SELF_INFO if SELF_Q.search",
)

# ---------------------------------------------------------------------------
# 3. Understand first, then write (the hard-question path)
# ---------------------------------------------------------------------------
once(
    '''    plan = chat(base + [{"role": "user", "content":
        f"{user}\\n\\nWrite a short numbered plan to answer this correctly. "
        "Do not answer yet."}], 0.2, 600, smart=True)

    draft = chat(base + [{"role": "user", "content":
        f"{user}\\n\\nPLAN:\\n{plan}\\n\\nFollow the plan and give the full answer. "
        "Do arithmetic and logic one step at a time."}], 0.2, smart=True)''',
    '''    plan = chat(base + [{"role": "user", "content":
        f"{user}\\n\\nBefore answering, think like an expert teacher. Write, in at most 12 short lines: "
        "UNDERSTANDING: what the person really wants and anything unclear or assumed; "
        "KEY POINTS: the facts, steps or ideas the answer must contain; "
        "PITFALLS: the mistakes or edge cases to watch for; "
        "SHAPE: how to lay out the answer. Do not write the answer yet."}], 0.2, 900, smart=True)

    draft = chat(base + [{"role": "user", "content":
        f"{user}\\n\\nYOUR PRIVATE NOTES (never show or mention them):\\n{plan}\\n\\nNow write the final answer for the "
        "person: the direct answer first, then the explanation in clear steps, with an example where it helps and the "
        "main pitfall. Do arithmetic and logic one step at a time."}], 0.2, smart=True)''',
    "3. understand-first planning for hard questions",
    "YOUR PRIVATE NOTES (never show or mention them)",
)

if src == orig:
    print("Nothing to change. All upgrades were already applied.")
else:
    shutil.copyfile(PATH, PATH + ".bak2")
    with open(PATH, "w", encoding="utf-8") as f:
        f.write(src)
    try:
        py_compile.compile(PATH, doraise=True)
    except py_compile.PyCompileError as e:
        shutil.copyfile(PATH + ".bak2", PATH)
        sys.exit("The patched file did not compile, so your original was restored.\n" + str(e))

print("\n".join(done))
print("Done. Backup saved as deep_oq.py.bak2")
