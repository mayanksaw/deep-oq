"""Renames the database tables in deep_oq_v2.py from oq_* to v2_* so they no longer clash with your old tables.
Put this file in the same folder as deep_oq_v2.py, then run:  python fix.py
"""
import sys

P = "deep_oq_v2.py"
OLD_NEW = [
    ("oq_conversations", "v2_conversations"),
    ("oq_messages", "v2_messages"),
    ("oq_facts", "v2_facts"),
    ("oq_usage", "v2_usage"),
    ("oq_idx_", "v2_idx_"),
    ("oq_uq_facts", "v2_uq_facts"),
]

try:
    s = open(P, encoding="utf-8").read()
except FileNotFoundError:
    sys.exit(f"Cannot find {P}. Put fix.py in the same folder as {P} and try again.")

if "v2_messages" in s and "oq_messages" not in s:
    sys.exit("Already fixed. Nothing to do.")

total = 0
for old, new in OLD_NEW:
    n = s.count(old)
    total += n
    s = s.replace(old, new)

if total == 0:
    sys.exit("Found no table names to rename. Is this the right file?")

open(P, "w", encoding="utf-8").write(s)
print(f"done: {total} replacements made in {P}")
print("Now run:  python deep_oq_v2.py selftest")
