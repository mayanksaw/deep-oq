"""
db.py - Deep OQ database (PostgreSQL)

Reads the connection string from the DATABASE_URL environment variable
(Render sets this for you when you link a PostgreSQL database).

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
import time
from contextlib import contextmanager

import subprocess
import sys

try:
    import psycopg  # noqa: F401
    import psycopg_pool  # noqa: F401
except ImportError:
    # installs the PostgreSQL driver automatically on first start
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           "psycopg[binary,pool]"])

from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Link your Render PostgreSQL database.")

# Some providers hand out "postgres://" URLs; psycopg wants "postgresql://".
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]

SESSION_TTL = 60 * 60 * 24 * 30  # 30 days
PBKDF2_ITERATIONS = 200_000

_pool = ConnectionPool(
    DATABASE_URL,
    min_size=1,
    max_size=5,
    kwargs={"row_factory": dict_row},
    open=False,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT,
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS facts (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    fact TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id);
"""

_initialized = False


# ---------------------------------------------------------------------------
# Connection / setup
# ---------------------------------------------------------------------------

@contextmanager
def get_conn():
    """Yield a pooled connection. Commits on success, rolls back on error."""
    with _pool.connection() as conn:
        yield conn


def init_db():
    """Open the pool and create tables. Safe to call more than once."""
    global _initialized
    if _initialized:
        return
    _pool.open(wait=True)
    with get_conn() as conn:
        conn.execute(SCHEMA)
    _initialized = True


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def _hash_password(password, salt_hex):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        PBKDF2_ITERATIONS,
    ).hex()


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def create_user(username, password):
    """Create an account. Returns the new user id, or None if the name is taken."""
    username = username.strip()
    salt = secrets.token_hex(16)
    pw_hash = _hash_password(password, salt)
    try:
        with get_conn() as conn:
            row = conn.execute(
                "INSERT INTO users (username, password_hash, salt, created_at) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (username, pw_hash, salt, time.time()),
            ).fetchone()
        return row["id"]
    except UniqueViolation:
        return None


def get_user_by_username(username):
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, username, created_at FROM users WHERE username = %s",
            (username.strip(),),
        ).fetchone()


def get_user(user_id):
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, username, created_at FROM users WHERE id = %s",
            (user_id,),
        ).fetchone()


def verify_user(username, password):
    """Check credentials. Returns the user dict (id, username) or None."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, salt FROM users WHERE username = %s",
            (username.strip(),),
        ).fetchone()
    if not row:
        # burn similar time so response timing doesn't reveal valid usernames
        _hash_password(password, secrets.token_hex(16))
        return None
    candidate = _hash_password(password, row["salt"])
    if hmac.compare_digest(candidate, row["password_hash"]):
        return {"id": row["id"], "username": row["username"]}
    return None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def create_session(user_id, ttl=SESSION_TTL):
    """Create a login token for the user and return it."""
    token = secrets.token_urlsafe(32)
    now = time.time()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) "
            "VALUES (%s, %s, %s, %s)",
            (token, user_id, now, now + ttl),
        )
    return token


def get_user_by_token(token):
    """Return the user dict for a valid, unexpired token, else None."""
    if not token:
        return None
    with get_conn() as conn:
        return conn.execute(
            "SELECT u.id, u.username FROM sessions s "
            "JOIN users u ON u.id = s.user_id "
            "WHERE s.token = %s AND s.expires_at > %s",
            (token, time.time()),
        ).fetchone()


def delete_session(token):
    with get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token = %s", (token,))


def delete_expired_sessions():
    with get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at <= %s", (time.time(),))


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def create_conversation(user_id, title=None):
    now = time.time()
    with get_conn() as conn:
        row = conn.execute(
            "INSERT INTO conversations (user_id, title, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (user_id, title, now, now),
        ).fetchone()
    return row["id"]


def list_conversations(user_id):
    """Newest activity first."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE user_id = %s ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()


def get_conversation(user_id, conversation_id):
    """Returns the conversation only if it belongs to this user."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE id = %s AND user_id = %s",
            (conversation_id, user_id),
        ).fetchone()


def rename_conversation(user_id, conversation_id, title):
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET title = %s WHERE id = %s AND user_id = %s",
            (title, conversation_id, user_id),
        )


def delete_conversation(user_id, conversation_id):
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM conversations WHERE id = %s AND user_id = %s",
            (conversation_id, user_id),
        )


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def add_message(conversation_id, role, content):
    now = time.time()
    with get_conn() as conn:
        row = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (conversation_id, role, content, now),
        ).fetchone()
        conn.execute(
            "UPDATE conversations SET updated_at = %s WHERE id = %s",
            (now, conversation_id),
        )
    return row["id"]


def get_messages(conversation_id, limit=None):
    """Oldest first. If limit is given, returns the most recent `limit` messages."""
    with get_conn() as conn:
        if limit:
            rows = conn.execute(
                "SELECT id, role, content, created_at FROM messages "
                "WHERE conversation_id = %s ORDER BY id DESC LIMIT %s",
                (conversation_id, limit),
            ).fetchall()
            return list(reversed(rows))
        return conn.execute(
            "SELECT id, role, content, created_at FROM messages "
            "WHERE conversation_id = %s ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()


# ---------------------------------------------------------------------------
# Facts (long-term memory)
# ---------------------------------------------------------------------------

def add_fact(user_id, fact):
    """Store a fact for the user. Skips exact duplicates. Returns the fact id."""
    fact = fact.strip()
    if not fact:
        return None
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM facts WHERE user_id = %s AND fact = %s",
            (user_id, fact),
        ).fetchone()
        if existing:
            return existing["id"]
        row = conn.execute(
            "INSERT INTO facts (user_id, fact, created_at) VALUES (%s, %s, %s) "
            "RETURNING id",
            (user_id, fact, time.time()),
        ).fetchone()
    return row["id"]


def list_facts(user_id, limit=None):
    with get_conn() as conn:
        if limit:
            return conn.execute(
                "SELECT id, fact, created_at FROM facts "
                "WHERE user_id = %s ORDER BY id DESC LIMIT %s",
                (user_id, limit),
            ).fetchall()
        return conn.execute(
            "SELECT id, fact, created_at FROM facts "
            "WHERE user_id = %s ORDER BY id ASC",
            (user_id,),
        ).fetchall()


def delete_fact(user_id, fact_id):
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM facts WHERE id = %s AND user_id = %s",
            (fact_id, user_id),
        )


def clear_facts(user_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM facts WHERE user_id = %s", (user_id,))


# Create tables automatically when the module is imported (idempotent).
init_db()
