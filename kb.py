"""Deep OQ knowledge base: a large searchable database on SQLite (FTS5).
SQLite handles millions of chunks on one machine and needs no extra install."""
import os
import re
import sqlite3

DB_PATH = os.environ.get("DEEP_OQ_DB", "deep_oq.db")
CHUNK = 900      # characters per chunk
OVERLAP = 150


def conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5("
              "content, title UNINDEXED, source UNINDEXED, tokenize='porter unicode61')")
    c.execute("CREATE TABLE IF NOT EXISTS sources(source TEXT PRIMARY KEY, chunks INT)")
    c.execute("CREATE TABLE IF NOT EXISTS messages("
              "id INTEGER PRIMARY KEY, session TEXT, role TEXT, content TEXT, "
              "ts DATETIME DEFAULT CURRENT_TIMESTAMP)")
    return c


DB = conn()


def chunks(text):
    text = re.sub(r"\s+", " ", text).strip()
    i = 0
    while i < len(text):
        yield text[i:i + CHUNK]
        i += CHUNK - OVERLAP


def add_text(text, title="note", source="manual"):
    n = 0
    with DB:
        for ch in chunks(text):
            DB.execute("INSERT INTO docs(content,title,source) VALUES(?,?,?)",
                       (ch, title, source))
            n += 1
        DB.execute("INSERT INTO sources(source,chunks) VALUES(?,?) "
                   "ON CONFLICT(source) DO UPDATE SET chunks=chunks+excluded.chunks",
                   (source, n))
    return n


def search(query, k=6):
    words = re.findall(r"\w+", query.lower())
    if not words:
        return []
    q = " OR ".join(f'"{w}"' for w in words[:20])
    try:
        rows = DB.execute(
            "SELECT content,title,source FROM docs WHERE docs MATCH ? "
            "ORDER BY bm25(docs) LIMIT ?", (q, k)).fetchall()
    except sqlite3.OperationalError:
        return []
    return [{"content": r[0], "title": r[1], "source": r[2]} for r in rows]


def save_msg(session, role, content):
    with DB:
        DB.execute("INSERT INTO messages(session,role,content) VALUES(?,?,?)",
                   (session, role, content))


def load_history(session, n=12):
    rows = DB.execute("SELECT role,content FROM messages WHERE session=? "
                      "ORDER BY id DESC LIMIT ?", (session, n)).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def stats():
    d = DB.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    s = DB.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    return {"chunks": d, "sources": s}
