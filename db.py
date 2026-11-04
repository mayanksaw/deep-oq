"""
db.py - Deep OQ database (PostgreSQL)

Reads the connection string from the DATABASE_URL environment variable
(Render sets this for you when you link a PostgreSQL database).

Tables:
  users          - accounts (password stored as a salted hash, never plain text)
  sessions       - login tokens (only a hash of each token is stored)
  conversations  - one per chat
  messages       - every user / assistant message (+ which tool was used)
  facts          - long-term memory per user (key / value)
"""

import hashlib
import hmac
import os
import re
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
USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,32}$")

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
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_active DOUBLE PRECISION;

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
ALTER TABLE messages ADD COLUMN IF NOT EXISTS tool TEXT;

-- An earlier version of this file made facts(fact TEXT). It was never used,
-- so replace it with the key/value layout the server expects.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema = current_schema()
                 AND table_name = 'facts' AND column_name = 'fact') THEN
        DROP TABLE facts;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS facts (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,
    UNIQUE (user_id, key)
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
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


def init():
    """Open the pool and create/upgrade tables. Safe to call more than once."""
    global _initialized
    if _initialized:
        return
    _pool.open(wait=True)
    with get_conn() as conn:
        conn.execute(SCHEMA)
    _initialized = True


init_db = init  # old name, kept in case anything still uses it


# ---------------------------------------------------------------------------
# Passwords and tokens
# ---------------------------------------------------------------------------

def _hash_password(password, salt_hex):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), PBKDF2_ITERATIONS
    ).hex()


def _token_hash(token):
    # Only the hash is stored, so a leaked database can't be used to log in.
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Users and sessions
# ---------------------------------------------------------------------------

def create_user(username, password):
    """Create an account. Raises ValueError with a user-facing message if invalid."""
    username = (username or "").strip().lower()
    password = password or ""
    if not USERNAME_RE.match(username):
        raise ValueError("Username must be 3-32 characters: letters, numbers, _ . or -")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if len(password) > 128:
        raise ValueError("Password is too long (128 characters max).")
    salt = secrets.token_hex(16)
    try:
        with get_conn() as conn:
            row = conn.execute(
                "INSERT INTO users (username, password_hash, salt, created_at) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (username, _hash_password(password, salt), salt, time.time()),
            ).fetchone()
    except UniqueViolation:
        raise ValueError("That username is taken.")
    return row["id"]


def login(username, password):
    """Check credentials. Returns a new session token, or None if wrong."""
    username = (username or "").strip().lower()
    password = (password or "")[:128]
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, password_hash, salt FROM users WHERE username = %s", (username,)
        ).fetchone()
    if not row:
        # spend similar time so response timing doesn't reveal valid usernames
        _hash_password(password, secrets.token_hex(16))
        return None
    if not hmac.compare_digest(_hash_password(password, row["salt"]), row["password_hash"]):
        return None
    token = secrets.token_urlsafe(32)
    now = time.time()
    with get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at <= %s", (now,))
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) "
            "VALUES (%s, %s, %s, %s)",
            (_token_hash(token), row["id"], now, now + SESSION_TTL),
        )
        conn.execute("UPDATE users SET last_active = %s WHERE id = %s", (now, row["id"]))
    return token


def user_from_token(token):
    """Return {id, username} for a valid, unexpired token, else None."""
    if not token:
        return None
    now = time.time()
    with get_conn() as conn:
        user = conn.execute(
            "SELECT u.id, u.username, u.last_active FROM sessions s "
            "JOIN users u ON u.id = s.user_id "
            "WHERE s.token = %s AND s.expires_at > %s",
            (_token_hash(token), now),
        ).fetchone()
        if user and (user["last_active"] or 0) < now - 300:
            conn.execute("UPDATE users SET last_active = %s WHERE id = %s", (now, user["id"]))
    return {"id": user["id"], "username": user["username"]} if user else None


def logout(token):
    if token:
        with get_conn() as conn:
            conn.execute("DELETE FROM sessions WHERE token = %s", (_token_hash(token),))


def username_of(user_id):
    with get_conn() as conn:
        row = conn.execute("SELECT username FROM users WHERE id = %s", (user_id,)).fetchone()
    return row["username"] if row else None


def delete_user(user_id):
    """Deletes the user and, through cascades, their sessions, chats and memory."""
    with get_conn() as conn:
        conn.execute("DELETE FROM users WHERE id = %s", (user_id,))


# ---------------------------------------------------------------------------
# Conversations and messages
# ---------------------------------------------------------------------------

def new_conversation(user_id, title=None):
    now = time.time()
    with get_conn() as conn:
        row = conn.execute(
            "INSERT INTO conversations (user_id, title, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (user_id, (title or "New chat")[:80], now, now),
        ).fetchone()
    return row["id"]


def list_conversations(user_id):
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE user_id = %s ORDER BY updated_at DESC LIMIT 200",
            (user_id,),
        ).fetchall()


def owns_conversation(user_id, conversation_id):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id = %s AND user_id = %s",
            (conversation_id, user_id),
        ).fetchone()
    return row is not None


def delete_conversation(user_id, conversation_id):
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM conversations WHERE id = %s AND user_id = %s",
            (conversation_id, user_id),
        )


def add_message(conversation_id, role, content, tool=None):
    now = time.time()
    with get_conn() as conn:
        row = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, tool, created_at) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (conversation_id, role, content, tool, now),
        ).fetchone()
        conn.execute("UPDATE conversations SET updated_at = %s WHERE id = %s", (now, conversation_id))
    return row["id"]


def get_messages(conversation_id):
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, role, content, tool, created_at FROM messages "
            "WHERE conversation_id = %s ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()


# ---------------------------------------------------------------------------
# Facts (long-term memory)
# ---------------------------------------------------------------------------

def save_fact(user_id, key, value):
    """Save or update one remembered fact for the user."""
    now = time.time()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO facts (user_id, key, value, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (user_id, key) DO UPDATE "
            "SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at",
            (user_id, str(key).strip()[:100], str(value).strip()[:2000], now, now),
        )


def list_facts(user_id):
    """Most recently updated first."""
    with get_conn() as conn:
        return conn.execute(
            "SELECT key, value, updated_at FROM facts "
            "WHERE user_id = %s ORDER BY updated_at DESC LIMIT 200",
            (user_id,),
        ).fetchall()


# ---------------------------------------------------------------------------
# Admin (counts and usernames only, never message text)
# ---------------------------------------------------------------------------

def admin_stats():
    day_ago = time.time() - 86400
    with get_conn() as conn:
        def one(sql, *params):
            return list(conn.execute(sql, params).fetchone().values())[0]
        return {
            "users": one("SELECT COUNT(*) FROM users"),
            "new_users_24h": one("SELECT COUNT(*) FROM users WHERE created_at > %s", day_ago),
            "active_users_24h": one("SELECT COUNT(*) FROM users WHERE last_active > %s", day_ago),
            "conversations": one("SELECT COUNT(*) FROM conversations"),
            "messages": one("SELECT COUNT(*) FROM messages"),
            "messages_24h": one("SELECT COUNT(*) FROM messages WHERE created_at > %s", day_ago),
            "db_size_kb": int(one("SELECT pg_database_size(current_database()) / 1024")),
        }


def admin_tool_usage():
    with get_conn() as conn:
        return conn.execute(
            "SELECT tool, COUNT(*) AS uses FROM messages "
            "WHERE tool IS NOT NULL AND role = 'user' "
            "GROUP BY tool ORDER BY uses DESC LIMIT 15"
        ).fetchall()


def admin_daily(days=14):
    since = time.time() - days * 86400
    with get_conn() as conn:
        return conn.execute(
            "SELECT to_char(to_timestamp(created_at) AT TIME ZONE 'UTC', 'YYYY-MM-DD') AS day, "
            "COUNT(*) AS messages FROM messages WHERE created_at > %s "
            "GROUP BY day ORDER BY day",
            (since,),
        ).fetchall()


def admin_users():
    with get_conn() as conn:
        return conn.execute(
            "SELECT u.id, u.username, u.created_at AS created, u.last_active, "
            "(SELECT COUNT(*) FROM conversations c WHERE c.user_id = u.id) AS chats, "
            "(SELECT COUNT(*) FROM messages m JOIN conversations c ON c.id = m.conversation_id "
            " WHERE c.user_id = u.id) AS messages "
            "FROM users u ORDER BY u.id DESC LIMIT 500"
        ).fetchall()


# Create/upgrade tables when the module is imported (safe to repeat).
init()
