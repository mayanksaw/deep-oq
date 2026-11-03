"""
db.py - Deep OQ database layer (PostgreSQL, built for Render)

Env vars:
  DATABASE_URL  (required) the connection string from your Render Postgres page.
                Use the "Internal Database URL" when the app runs on Render,
                and the "External Database URL" when running on your own computer.
  DB_SSLMODE    (optional) e.g. "require". Needed for the External URL.
  DB_POOL_MAX   (optional) max connections, default 10.
"""

import hashlib
import hmac
import os
import re
import secrets
import threading
import time
from contextlib import contextmanager

import psycopg2
from psycopg2 import errors, pool
from psycopg2.extras import RealDictCursor

SESSION_TTL = 30 * 24 * 3600          # sessions last 30 days
USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,30}$")
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1

_pool = None
_pool_lock = threading.Lock()


# ---------------- connection helpers ----------------

def _get_pool():
    global _pool
    with _pool_lock:
        if _pool is None:
            url = os.environ.get("DATABASE_URL")
            if not url:
                raise RuntimeError("DATABASE_URL is not set. Copy it from your Render Postgres page.")
            kwargs = {}
            if os.environ.get("DB_SSLMODE"):
                kwargs["sslmode"] = os.environ["DB_SSLMODE"]
            _pool = pool.ThreadedConnectionPool(1, int(os.environ.get("DB_POOL_MAX", "10")), url, **kwargs)
        return _pool


@contextmanager
def cur():
    """Yield a dict cursor; commit on success, roll back on error."""
    p = _get_pool()
    conn = p.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            yield c
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


def init():
    with cur() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          BIGSERIAL PRIMARY KEY,
            username    TEXT UNIQUE NOT NULL,
            pw_hash     TEXT NOT NULL
        );
        ALTER TABLE users ADD COLUMN IF NOT EXISTS created BIGINT NOT NULL DEFAULT 0;
        ALTER TABLE users ADD COLUMN IF NOT EXISTS last_active BIGINT;

        CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id    BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE
        );
        ALTER TABLE sessions ADD COLUMN IF NOT EXISTS created BIGINT NOT NULL DEFAULT 0;
        ALTER TABLE sessions ADD COLUMN IF NOT EXISTS expires BIGINT NOT NULL DEFAULT 0;
        CREATE INDEX IF NOT EXISTS sessions_user_idx ON sessions(user_id);
        
        CREATE TABLE IF NOT EXISTS conversations (
            id      BIGSERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title   TEXT NOT NULL
        );
        ALTER TABLE conversations ADD COLUMN IF NOT EXISTS created BIGINT NOT NULL DEFAULT 0;
        ALTER TABLE conversations ADD COLUMN IF NOT EXISTS updated BIGINT NOT NULL DEFAULT 0;
        CREATE INDEX IF NOT EXISTS conv_user_idx ON conversations(user_id, updated DESC);
        
        CREATE TABLE IF NOT EXISTS messages (
            id              BIGSERIAL PRIMARY KEY,
            conversation_id BIGINT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            tool            TEXT
        );
        ALTER TABLE messages ADD COLUMN IF NOT EXISTS created BIGINT NOT NULL DEFAULT 0;
        CREATE INDEX IF NOT EXISTS msg_conv_idx ON messages(conversation_id, id);
        CREATE INDEX IF NOT EXISTS msg_created_idx ON messages(created);
        
        CREATE TABLE IF NOT EXISTS facts (
            user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            key     TEXT NOT NULL,
            value   TEXT NOT NULL,
            PRIMARY KEY (user_id, key)
        );
        ALTER TABLE facts ADD COLUMN IF NOT EXISTS updated BIGINT NOT NULL DEFAULT 0;
        """)


# ---------------- passwords and tokens ----------------

def _hash_password(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${h.hex()}"


def _check_password(password, stored):
    try:
        _, n, r, p, salt_hex, hash_hex = stored.split("$")
        h = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex),
                           n=int(n), r=int(r), p=int(p), dklen=len(hash_hex) // 2)
        return hmac.compare_digest(h.hex(), hash_hex)
    except Exception:
        return False


_DUMMY_HASH = _hash_password("not-a-real-password")  # keeps timing similar for unknown users


def _token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


# ---------------- accounts ----------------

def create_user(username, password):
    username = (username or "").strip().lower()
    password = password or ""
    if not USERNAME_RE.match(username):
        raise ValueError("Username must be 3-30 characters: letters, numbers, . _ -")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    if len(password) > 200:
        raise ValueError("Password is too long.")
    try:
        with cur() as c:
            c.execute("INSERT INTO users (username, pw_hash, created) VALUES (%s, %s, %s)",
                      (username, _hash_password(password), int(time.time())))
    except errors.UniqueViolation:
        raise ValueError("That username is already taken.")


def login(username, password):
    """Return a new session token, or None if the credentials are wrong."""
    username = (username or "").strip().lower()
    password = password or ""
    with cur() as c:
        c.execute("SELECT id, pw_hash FROM users WHERE username = %s", (username,))
        row = c.fetchone()
        if not row:
            _check_password(password, _DUMMY_HASH)
            return None
        if not _check_password(password, row["pw_hash"]):
            return None
        now = int(time.time())
        token = secrets.token_urlsafe(32)
        c.execute("DELETE FROM sessions WHERE expires < %s", (now,))
        c.execute("INSERT INTO sessions (token_hash, user_id, created, expires) VALUES (%s, %s, %s, %s)",
                  (_token_hash(token), row["id"], now, now + SESSION_TTL))
        c.execute("UPDATE users SET last_active = %s WHERE id = %s", (now, row["id"]))
        return token


def logout(token):
    if not token:
        return
    with cur() as c:
        c.execute("DELETE FROM sessions WHERE token_hash = %s", (_token_hash(token),))


def user_from_token(token):
    if not token:
        return None
    now = int(time.time())
    with cur() as c:
        c.execute("""SELECT u.id, u.username FROM sessions s
                     JOIN users u ON u.id = s.user_id
                     WHERE s.token_hash = %s AND s.expires > %s""", (_token_hash(token), now))
        row = c.fetchone()
        if not row:
            return None
        c.execute("UPDATE users SET last_active = %s WHERE id = %s", (now, row["id"]))
        return {"id": row["id"], "username": row["username"]}


def username_of(uid):
    with cur() as c:
        c.execute("SELECT username FROM users WHERE id = %s", (uid,))
        row = c.fetchone()
        return row["username"] if row else None


def delete_user(uid):
    with cur() as c:
        c.execute("DELETE FROM users WHERE id = %s", (uid,))  # chats, messages, sessions, facts cascade


# ---------------- conversations ----------------

def new_conversation(uid, title):
    now = int(time.time())
    with cur() as c:
        c.execute("INSERT INTO conversations (user_id, title, created, updated) VALUES (%s, %s, %s, %s) RETURNING id",
                  (uid, (title or "New chat")[:80], now, now))
        return c.fetchone()["id"]


def owns_conversation(uid, cid):
    with cur() as c:
        c.execute("SELECT 1 FROM conversations WHERE id = %s AND user_id = %s", (cid, uid))
        return c.fetchone() is not None


def list_conversations(uid):
    with cur() as c:
        c.execute("SELECT id, title, created, updated FROM conversations WHERE user_id = %s ORDER BY updated DESC LIMIT 200",
                  (uid,))
        return c.fetchall()


def get_messages(cid):
    # callers must check owns_conversation() first (app.py does)
    with cur() as c:
        c.execute("SELECT role, content, tool, created FROM messages WHERE conversation_id = %s ORDER BY id", (cid,))
        return c.fetchall()


def add_message(cid, role, content, tool=None):
    now = int(time.time())
    with cur() as c:
        c.execute("INSERT INTO messages (conversation_id, role, content, tool, created) VALUES (%s, %s, %s, %s, %s)",
                  (cid, role, content, tool, now))
        c.execute("UPDATE conversations SET updated = %s WHERE id = %s", (now, cid))


def delete_conversation(uid, cid):
    with cur() as c:
        c.execute("DELETE FROM conversations WHERE id = %s AND user_id = %s", (cid, uid))


# ---------------- memory ----------------

def list_facts(uid):
    with cur() as c:
        c.execute("SELECT key, value FROM facts WHERE user_id = %s ORDER BY updated DESC LIMIT 100", (uid,))
        return c.fetchall()


def save_fact(uid, key, value):
    with cur() as c:
        c.execute("""INSERT INTO facts (user_id, key, value, updated) VALUES (%s, %s, %s, %s)
                     ON CONFLICT (user_id, key) DO UPDATE SET value = EXCLUDED.value, updated = EXCLUDED.updated""",
                  (uid, str(key)[:100], str(value)[:2000], int(time.time())))


# ---------------- admin (counts only, never message text) ----------------

def admin_stats():
    now = int(time.time())
    day = now - 86400
    with cur() as c:
        c.execute("""SELECT
            (SELECT COUNT(*) FROM users)                            AS users,
            (SELECT COUNT(*) FROM users WHERE created > %(d)s)            AS new_users_24h,
            (SELECT COUNT(*) FROM users WHERE last_active > %(d)s)        AS active_users_24h,
            (SELECT COUNT(*) FROM conversations)                        AS conversations,
            (SELECT COUNT(*) FROM messages)                             AS messages,
            (SELECT COUNT(*) FROM messages WHERE created > %(d)s)         AS messages_24h,
            (pg_database_size(current_database()) / 1024)               AS db_size_kb""", {"d": day})
        return {k: int(v) for k, v in c.fetchone().items()}


def admin_tool_usage():
    with cur() as c:
        c.execute("""SELECT tool, COUNT(*) AS uses FROM messages
                     WHERE tool IS NOT NULL AND role = 'user'
                     GROUP BY tool ORDER BY uses DESC LIMIT 15""")
        return c.fetchall()


def admin_daily():
    since = int(time.time()) - 14 * 86400
    with cur() as c:
        c.execute("""SELECT to_char(to_timestamp(created) AT TIME ZONE 'UTC', 'YYYY-MM-DD') AS day, COUNT(*) AS messages
                     FROM messages WHERE created > %s GROUP BY day ORDER BY day""", (since,))
        return c.fetchall()


def admin_users():
    with cur() as c:
        c.execute("""SELECT u.id, u.username, u.created, u.last_active,
                            (SELECT COUNT(*) FROM conversations c WHERE c.user_id = u.id) AS chats,
                            (SELECT COUNT(*) FROM messages m JOIN conversations c ON c.id = m.conversation_id
                             WHERE c.user_id = u.id) AS messages
                     FROM users u ORDER BY u.id DESC LIMIT 500""")
        return [{"id": r["id"], "username": r["username"], "created": r["created"],
                 "chats": int(r["chats"]), "messages": int(r["messages"]),
                 "last_active": r["last_active"]} for r in c.fetchall()]
