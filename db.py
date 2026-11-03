"""
db.py - Deep OQ database (SQLite)

SQLite is a real database stored in ONE file (deep_oq.db). No separate
database server to install, and it handles thousands of users. When you
outgrow it, the same tables move to PostgreSQL.

Tables:
  users          - accounts (password stored as a salted hash, never plain text)
  sessions       - login tokens
  conversations  - one per chat
  messages       - every user / assistant message
  facts          - long-term memory per user
"""

import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager

DB_PATH = os.environ.get("DEEPOQ_DB", "deep_oq.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    pw_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'New chat',
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool TEXT,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created REAL NOT NULL,
    UNIQUE(user_id, key)
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id);
"""


@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    c.execute("PRAGMA journal_mode = WAL")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init():
    with conn() as c:
        c.executescript(SCHEMA)


# ---------------- users & login ----------------

def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()


def create_user(username, password):
    username = username.strip().lower()
    if not (3 <= len(username) <= 32) or not username.replace("_", "").isalnum():
        raise ValueError("Username must be 3-32 letters, numbers or underscore.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    salt = secrets.token_hex(16)
    try:
        with conn() as c:
            cur = c.execute(
                "INSERT INTO users(username, pw_hash, salt, created) VALUES (?,?,?,?)",
                (username, _hash(password, salt), salt, time.time()),
            )
            return cur.lastrowid
    except sqlite3.IntegrityError:
        raise ValueError("Username already taken.")


def login(username, password, days=30):
    with conn() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username.strip().lower(),)).fetchone()
        if not row or not hmac.compare_digest(row["pw_hash"], _hash(password, row["salt"])):
            return None
        token = secrets.token_urlsafe(32)
        c.execute("INSERT INTO sessions(token, user_id, expires) VALUES (?,?,?)",
                  (token, row["id"], time.time() + days * 86400))
        return token


def user_from_token(token):
    if not token:
        return None
    with conn() as c:
        c.execute("DELETE FROM sessions WHERE expires < ?", (time.time(),))
        row = c.execute(
            "SELECT u.id, u.username FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token=?",
            (token,),
        ).fetchone()
        return dict(row) if row else None


def logout(token):
    with conn() as c:
        c.execute("DELETE FROM sessions WHERE token=?", (token,))


# ---------------- conversations & messages ----------------

def new_conversation(user_id, title="New chat"):
    with conn() as c:
        return c.execute("INSERT INTO conversations(user_id, title, created) VALUES (?,?,?)",
                         (user_id, title[:80], time.time())).lastrowid


def owns_conversation(user_id, conv_id):
    with conn() as c:
        return c.execute("SELECT 1 FROM conversations WHERE id=? AND user_id=?",
                         (conv_id, user_id)).fetchone() is not None


def list_conversations(user_id):
    with conn() as c:
        rows = c.execute("SELECT id, title, created FROM conversations WHERE user_id=? ORDER BY id DESC LIMIT 100",
                         (user_id,)).fetchall()
        return [dict(r) for r in rows]


def delete_conversation(user_id, conv_id):
    with conn() as c:
        c.execute("DELETE FROM conversations WHERE id=? AND user_id=?", (conv_id, user_id))


def add_message(conv_id, role, content, tool=None):
    with conn() as c:
        c.execute("INSERT INTO messages(conversation_id, role, content, tool, created) VALUES (?,?,?,?,?)",
                  (conv_id, role, content, tool, time.time()))


def get_messages(conv_id, limit=200):
    with conn() as c:
        rows = c.execute(
            "SELECT role, content, tool, created FROM messages WHERE conversation_id=? ORDER BY id LIMIT ?",
            (conv_id, limit)).fetchall()
        return [dict(r) for r in rows]


# ---------------- long-term memory ----------------

def save_fact(user_id, key, value):
    with conn() as c:
        c.execute(
            "INSERT INTO facts(user_id, key, value, created) VALUES (?,?,?,?) "
            "ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value, created=excluded.created",
            (user_id, key.strip()[:100], value.strip()[:2000], time.time()))


def list_facts(user_id):
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT key, value FROM facts WHERE user_id=? ORDER BY id DESC LIMIT 200", (user_id,))]


# ---------------- admin (counts only, never message text) ----------------

def admin_stats():
    day = time.time() - 86400
    with conn() as c:
        one = lambda q, *a: c.execute(q, a).fetchone()[0]
        return {
            "users": one("SELECT COUNT(*) FROM users"),
            "new_users_24h": one("SELECT COUNT(*) FROM users WHERE created > ?", day),
            "conversations": one("SELECT COUNT(*) FROM conversations"),
            "messages": one("SELECT COUNT(*) FROM messages"),
            "messages_24h": one("SELECT COUNT(*) FROM messages WHERE created > ?", day),
            "active_users_24h": one(
                "SELECT COUNT(DISTINCT cv.user_id) FROM messages m JOIN conversations cv ON cv.id = m.conversation_id "
                "WHERE m.created > ?", day),
            "db_size_kb": round(os.path.getsize(DB_PATH) / 1024) if os.path.exists(DB_PATH) else 0,
        }


def admin_users():
    with conn() as c:
        rows = c.execute(
            "SELECT u.id, u.username, u.created, "
            "(SELECT COUNT(*) FROM conversations WHERE user_id=u.id) AS chats, "
            "(SELECT COUNT(*) FROM messages m JOIN conversations cv ON cv.id=m.conversation_id WHERE cv.user_id=u.id) AS messages, "
            "(SELECT MAX(m.created) FROM messages m JOIN conversations cv ON cv.id=m.conversation_id WHERE cv.user_id=u.id) AS last_active "
            "FROM users u ORDER BY u.id DESC LIMIT 500").fetchall()
        return [dict(r) for r in rows]


def admin_tool_usage(limit=15):
    with conn() as c:
        rows = c.execute(
            "SELECT COALESCE(tool, 'auto (smart)') AS tool, COUNT(*) AS uses FROM messages "
            "WHERE role='user' GROUP BY tool ORDER BY uses DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]


def admin_daily(days=14):
    with conn() as c:
        rows = c.execute(
            "SELECT date(created, 'unixepoch') AS day, COUNT(*) AS messages FROM messages "
            "WHERE role='user' AND created > ? GROUP BY day ORDER BY day", (time.time() - days * 86400,)).fetchall()
        return [dict(r) for r in rows]


def delete_user(user_id):
    with conn() as c:
        c.execute("DELETE FROM users WHERE id=?", (user_id,))


def username_of(user_id):
    with conn() as c:
        row = c.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()
        return row["username"] if row else None
