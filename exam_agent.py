#!/usr/bin/env python3
"""
exam_agent
==========
An AI agent that solves JEE / NEET / board style questions by combining:

  * a language model (Claude, through the Anthropic API),
  * a concept database for every chapter (concepts_db.py), and
  * calculation tools that give exact, checkable results (agent_tools.py).

The model reads the question (text or a photo of it), looks up the chapter's method, does every
calculation with the tools instead of by guesswork, checks the result, and gives a step-by-step answer.

Setup
-----
    pip install anthropic sympy
    export ANTHROPIC_API_KEY="your key"            # Windows PowerShell: $env:ANTHROPIC_API_KEY="your key"

Use
---
    python exam_agent.py "Find the area under min{f,g} where f = x^2+5/12 ..."
    python exam_agent.py --image question.png                    # photo or screenshot of a question
    python exam_agent.py --image question.png "Solve part (b) only"
    python exam_agent.py --chat                                  # keep asking follow-up questions
    python exam_agent.py --selftest                              # offline test, no API key needed

Optional: set EXAM_AGENT_MODEL to use a different model name (default below).

Honest limits
-------------
The agent is much more reliable than a model working alone, but it can still make mistakes, misread a
blurry image, or choose the wrong method. Always check the final answer, and compare with your own working.
Using the API costs money and sends your question (and image) to Anthropic's servers.
"""

import argparse
import base64
import json
import mimetypes
import os
import sys

from agent_tools import TOOL_SCHEMAS, run_tool

DEFAULT_MODEL = os.environ.get("EXAM_AGENT_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You are an expert exam tutor and problem solver for JEE (Main and Advanced), NEET, board exams and similar tests in Maths, Physics, Chemistry and Biology.

Work like this on every question:
1. Restate what is given and what is asked. If a formula, bracket or fraction in the question could be read in more than one way (for example (x^2+5)/12 versus x^2+5/12), say how you read it. Solve the literal reading. If the other reading would give a very different or much cleaner answer, mention it briefly at the end.
2. Call lookup_concept with the chapter's topic words to recall the core concept, formulas, method and common trap. Skip this only for trivial arithmetic.
3. Plan the method, then do every non-trivial calculation with the tools (math_tool, calc, molar_mass, ph_calc). Do not trust mental arithmetic for integrals, roots, areas, matrices or long numerics. Tool output is exact; use it.
4. Check your answer: units, sign, size, domain of every root, and a second method or numeric check when possible. Clean answer test: exam numerical answers are usually clean numbers. If your result is an ugly irrational number, re-read the question for a misreading before answering.
5. Never state an answer from memory of an answer key. Derive it. If you remember a published answer that disagrees with your derivation, say so and explain.
6. Never stop halfway. Always finish with a final answer.

Format the reply in plain text that reads well in a terminal. Use this structure:
GIVEN / TO FIND
CONCEPT
SOLUTION (numbered steps, each with the formula, then the substitution)
FINAL ANSWER: <the answer, with units>
CHECK (one or two lines)

If a question is ambiguous, unreadable, or needs data you do not have, say exactly what is missing instead of guessing. For Biology and theory questions, answer from the concept notes and your knowledge, state the key reason, and say when you are unsure. Note that syllabi differ between exams, so mention it if an answer depends on the syllabus."""


# --------------------------------------------------------------------------- helpers

def image_block(path):
    """Build an image content block from a file path."""
    media = mimetypes.guess_type(path)[0] or "image/png"
    if media not in ("image/png", "image/jpeg", "image/gif", "image/webp"):
        raise ValueError(f"Unsupported image type {media}. Use PNG, JPEG, GIF or WebP.")
    with open(path, "rb") as fh:
        data = base64.standard_b64encode(fh.read()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}


def _blocks_to_dicts(content):
    """Keep only the text and tool_use blocks, as plain dicts."""
    out = []
    for b in content:
        if b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return out


def make_client():
    try:
        import anthropic
    except ImportError:
        sys.exit("The 'anthropic' package is missing. Install it with:  pip install anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Get a key from the Anthropic Console and set it as an "
                 "environment variable, then run again.")
    return anthropic.Anthropic()


# --------------------------------------------------------------------------- agent

class ExamAgent:
    def __init__(self, client=None, model=DEFAULT_MODEL, max_steps=15, max_tokens=4096, verbose=True):
        self.client = client if client is not None else make_client()
        self.model = model
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.messages = []

    def _log(self, text):
        if self.verbose:
            print(text, file=sys.stderr, flush=True)

    def ask(self, question, image_path=None):
        """Send a question (and optional image), run tools until the model finishes, return the answer text."""
        content = []
        if image_path:
            content.append(image_block(image_path))
        content.append({"type": "text", "text": question or "Solve the question shown in the image."})
        self.messages.append({"role": "user", "content": content})

        for step in range(1, self.max_steps + 1):
            response = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS, messages=self.messages)
            self.messages.append({"role": "assistant", "content": _blocks_to_dicts(response.content)})

            if response.stop_reason != "tool_use":
                text = "".join(b.text for b in response.content if b.type == "text").strip()
                if response.stop_reason == "max_tokens":
                    text += "\n\n[The answer was cut off because of the length limit. Ask me to continue.]"
                return text

            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                self._log(f"[step {step}] tool: {block.name} {json.dumps(block.input)[:200]}")
                output = run_tool(block.name, block.input)
                self._log(f"          -> {output[:200].replace(chr(10), ' | ')}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
            self.messages.append({"role": "user", "content": results})

        return ("I used the maximum number of tool steps without finishing. Try asking again with the question split "
                "into smaller parts, or raise --max-steps.")


# --------------------------------------------------------------------------- self-test

def selftest():
    """Offline test of the agent loop using a scripted fake model (no network, no API key)."""
    import tempfile
    from types import SimpleNamespace as NS

    class FakeClient:
        def __init__(self, script):
            self.script = list(script)
            self.calls = []
            self.messages = NS(create=self._create)

        def _create(self, **kwargs):
            self.calls.append(json.loads(json.dumps(kwargs["messages"], default=str)))
            return self.script.pop(0)

    def text(t):
        return NS(type="text", text=t)

    def tool(i, name, args):
        return NS(type="tool_use", id=i, name=name, input=args)

    bad = 0

    def check(label, ok, detail=""):
        nonlocal bad
        print(("PASS " if ok else "FAIL ") + label + ("" if ok else f"  {detail}"))
        bad += 0 if ok else 1

    # 1. A tool call, then a final answer.
    jee_args = {"operation": "area_under_min", "f": "x**2 + 5/12", "g": "2 - 8*x/3", "lower": "0",
                "upper": "3/4", "even": True, "multiplier": "9"}
    fake = FakeClient([
        NS(stop_reason="tool_use", content=[text("Looking up."), tool("t1", "math_tool", jee_args)]),
        NS(stop_reason="end_turn", content=[text("FINAL ANSWER: 6")]),
    ])
    agent = ExamAgent(client=fake, verbose=False)
    answer = agent.ask("JEE area question")
    second_call_last = fake.calls[1][-1]["content"][0]
    check("agent returns final text", answer == "FINAL ANSWER: 6", answer)
    check("tool result is sent back to the model",
          second_call_last["type"] == "tool_result" and "9 x area: 6" in second_call_last["content"], str(second_call_last))
    check("tool_use id is matched", second_call_last["tool_use_id"] == "t1")

    # 2. Follow-up keeps the history (chat mode).
    fake2 = FakeClient([NS(stop_reason="end_turn", content=[text("ok")])])
    agent.client = fake2
    agent.ask("follow-up")
    check("history is kept for follow-ups", len(fake2.calls[0]) >= 4, str(len(fake2.calls[0])))

    # 3. Runaway tool loop is stopped.
    loop = [NS(stop_reason="tool_use", content=[tool(f"t{i}", "calc", {"expression": "1+1"})]) for i in range(5)]
    guard = ExamAgent(client=FakeClient(loop), verbose=False, max_steps=3)
    check("runaway loop is stopped", "maximum number of tool steps" in guard.ask("loop"))

    # 4. A bad tool call returns an error text instead of crashing.
    err = FakeClient([
        NS(stop_reason="tool_use", content=[tool("e1", "math_tool", {"operation": "evaluate", "expression": "__import__('os')"})]),
        NS(stop_reason="end_turn", content=[text("done")]),
    ])
    ExamAgent(client=err, verbose=False).ask("unsafe")
    check("unsafe tool input becomes an error message", "Error" in err.calls[1][-1]["content"][0]["content"])

    # 5. Image input is encoded.
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        fh.write(png)
        path = fh.name
    blk = image_block(path)
    os.unlink(path)
    check("image becomes a base64 block", blk["type"] == "image" and blk["source"]["media_type"] == "image/png")

    # 6. Tool schemas are well formed.
    ok = all({"name", "description", "input_schema"} <= set(t) for t in TOOL_SCHEMAS)
    check("tool schemas have the required fields", ok)

    print("\nAll agent tests passed." if bad == 0 else f"\n{bad} test(s) failed.")
    print("Note: this tests the agent logic offline. A real run needs the anthropic package and an API key.")
    return 1 if bad else 0


# --------------------------------------------------------------------------- command line

def main(argv=None):
    ap = argparse.ArgumentParser(description="AI agent that solves exam questions with tools.")
    ap.add_argument("question", nargs="*", help="the question text")
    ap.add_argument("--image", help="path to a photo/screenshot of the question (PNG, JPEG, GIF, WebP)")
    ap.add_argument("--chat", action="store_true", help="interactive mode with follow-up questions")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"model name (default {DEFAULT_MODEL})")
    ap.add_argument("--max-steps", type=int, default=15, help="maximum tool-use rounds per question")
    ap.add_argument("--quiet", action="store_true", help="hide the tool-call log")
    ap.add_argument("--selftest", action="store_true", help="offline test, no API key needed")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    question = " ".join(args.question).strip()
    if not (question or args.image or args.chat):
        ap.print_help()
        return 1

    agent = ExamAgent(model=args.model, max_steps=args.max_steps, verbose=not args.quiet)

    if args.chat:
        print("Exam agent. Type a question, 'image <path>' to send a picture, or 'q' to quit.\n")
        pending_image = args.image
        if question or pending_image:
            print(agent.ask(question, pending_image), "\n")
        while True:
            try:
                line = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if line.lower() in ("q", "quit", "exit"):
                return 0
            if not line:
                continue
            img = None
            if line.lower().startswith("image "):
                img, _, line = line[6:].strip().partition(" ")
            try:
                print("\n" + agent.ask(line, img) + "\n")
            except Exception as exc:
                print(f"Error: {exc}\n")
    else:
        print(agent.ask(question, args.image))
    return 0


if __name__ == "__main__":
    sys.exit(main())
