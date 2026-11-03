"""
test_deepoq.py - check that Deep OQ really works.

Run:  python test_deepoq.py                       (tests local code, no key needed)
      python test_deepoq.py https://deep-oq.onrender.com   (tests your live site)

Part 1 checks the exact-answer tools (must be 100% correct).
Part 2 (needs GROQ_API_KEY or a live URL) compares a plain model answer
against the advanced pipeline on questions models often get wrong.
"""

import json
import os
import sys
import tempfile

LIVE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else None
os.environ.setdefault("DEEPOQ_DB", os.path.join(tempfile.mkdtemp(), "test.db"))

passed = failed = 0


def check(name, got, expected):
    global passed, failed
    ok = str(expected) in str(got)
    passed += ok
    failed += not ok
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got={str(got)[:80]!r} wanted={expected!r}"))


# ---------------- Part 1: exact tools ----------------
import ai_advanced as adv
import ai_tools as T

print("== Part 1: exact tools ==")
check("calculator", T.run_tool("calculator", expression="1234*5678"), "7006652")
check("calculator pow", T.run_tool("calculator", expression="2**20"), "1048576")
check("calculator blocks code", T.run_tool("calculator", expression="__import__('os').system('echo hi')"), "failed")
check("gate math", adv.reasoning_gate("987654 * 123456 ="), "121931812224")
check("gate words", adv.reasoning_gate('How many words in "the quick brown fox jumps"?'), '"words": 5')
check("gate letters", adv.reasoning_gate('How many letters in "strawberry"? count letters'), '"letters": 10')
check("loan emi", T.run_tool("loan_emi", principal="500000", annual_rate="9", months="60"), "10,379.18")
check("risk reward", T.run_tool("risk_reward", entry="100", stop_loss="95", target="115"), "1:3.00")
check("position size", T.run_tool("position_size", capital="100000", risk_percent="1", entry="100", stop_loss="98"), "500.00 units")
check("scale notes", T.run_tool("scale_notes", root="C", mode="major"), "C D E F G A B")
check("web blocks localhost", adv.web_fetch("http://localhost:5000"), "blocked")
check("web blocks private ip", adv.web_fetch("http://192.168.1.1"), "blocked")
check("code exec off by default", adv.run_python("print(1)"), "disabled")

# ---------------- Part 1b: database + server ----------------
print("\n== Part 1b: database and server ==")
import db

db.init()
uid = db.create_user("tester", "password123")
check("login ok", bool(db.login("tester", "password123")), "True")
check("login wrong password", db.login("tester", "wrongpass1"), "None")
try:
    db.create_user("tester", "password123")
    check("duplicate username blocked", "allowed", "blocked")
except ValueError:
    check("duplicate username blocked", "blocked", "blocked")
cid = db.new_conversation(uid, "t")
db.add_message(cid, "user", "hello")
check("messages saved", db.get_messages(cid)[0]["content"], "hello")
other = db.create_user("someone_else", "password123")
check("other user cannot read chat", db.owns_conversation(other, cid), "False")

if not LIVE:
    T.set_llm(lambda prompt, system="", image_url=None: "stub answer")
    import app as server

    c = server.app.test_client()
    check("chat needs login", c.post("/api/chat", json={"message": "hi"}).status_code, "401")
    tok = c.post("/api/login", json={"username": "tester", "password": "password123"}).get_json()["token"]
    h = {"Authorization": f"Bearer {tok}"}
    check("exact math through server", c.post("/api/chat", json={"message": "12*12"}, headers=h).get_json()["answer"], "144")
    check("health", c.get("/health").get_json()["tools"], "141")

# ---------------- Part 2: real model quality ----------------
print("\n== Part 2: plain model vs Deep OQ ==")

QUESTIONS = [
    ("What is 48271 * 9613?", "464,", lambda a: "464045" in a.replace(",", "") or "464,0" in a),
    ('How many words are in "I love to build my own artificial intelligence"?', "8", lambda a: "8" in a),
    ("If a bat and ball cost $1.10 total and the bat costs $1 more than the ball, how many cents is the ball?", "5",
     lambda a: "5" in a and "10 cents" not in a.lower()),
]

if LIVE:
    import requests

    print("Live mode: register a test user on your site first, then paste the token below.")
    token = input("Token (or Enter to skip): ").strip()
    if token:
        for q, _, ok in QUESTIONS:
            r = requests.post(f"{LIVE}/api/chat", json={"message": q}, headers={"Authorization": f"Bearer {token}"}, timeout=120)
            a = r.json().get("answer", "")
            check(f"live: {q[:45]}", "OK" if ok(a) else a, "OK")
elif os.environ.get("GROQ_API_KEY"):
    import app as server

    T.set_llm(server.groq_llm)
    for q, _, ok in QUESTIONS:
        plain = server.groq_llm(q)
        deep = adv.smart(q)
        print(f"\nQ: {q}\n  plain model : {'RIGHT' if ok(plain) else 'WRONG'}  {plain[:70]!r}\n  Deep OQ     : {'RIGHT' if ok(deep) else 'WRONG'}  {str(deep)[:70]!r}")
        check(f"Deep OQ: {q[:45]}", "OK" if ok(deep) else deep, "OK")
else:
    print("Skipped (set GROQ_API_KEY to compare a plain model with Deep OQ).")

print(f"\nResult: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
