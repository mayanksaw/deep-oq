"""
admin.py - admin pages for Deep OQ (Flask blueprint).

SETUP
1. Put this file next to app.py (src/admin.py).
2. In app.py:
       from admin import admin
       app.register_blueprint(admin)
3. In Render, set env var:  ADMIN_EMAILS=you@gmail.com,other@gmail.com
4. Open /admin while logged in with one of those Google accounts.

ASSUMPTIONS (edit if your app differs)
- Login stores the user's email in flask.session["email"]  (see SESSION_EMAIL_KEY)
- Tables/columns:
    users(id, email, created_at)
    conversations(id, user_id, created_at)
    messages(id, conversation_id, role, content, created_at)
"""
import os
from functools import wraps

import psycopg2
import psycopg2.extras
from flask import Blueprint, abort, render_template_string, session

SESSION_EMAIL_KEY = "email"

admin = Blueprint("admin", __name__, url_prefix="/admin")

ADMIN_EMAILS = {
    e.strip().lower()
    for e in os.environ.get("ADMIN_EMAILS", "").split(",")
    if e.strip()
}


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        email = (session.get(SESSION_EMAIL_KEY) or "").lower()
        if not email or email not in ADMIN_EMAILS:
            abort(404)  # 404 so the page's existence isn't revealed
        return f(*args, **kwargs)

    return wrapper


def query(sql, params=None):
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or ())
            return cur.fetchall()
    finally:
        conn.close()


BASE = """
<!doctype html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deep OQ Admin</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #f6f7f9; color: #1a1a1a; }
  header { background: #1a1a1a; color: #fff; padding: 12px 20px; display: flex; gap: 20px; align-items: center; }
  header a { color: #fff; text-decoration: none; }
  main { max-width: 1000px; margin: 20px auto; padding: 0 16px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .card { background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .card b { display: block; font-size: 28px; }
  table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; }
  th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #eee; font-size: 14px; vertical-align: top; }
  th { background: #fafafa; }
  .msg { background: #fff; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; white-space: pre-wrap; }
  .user { border-left: 4px solid #4a7cff; }
  .assistant { border-left: 4px solid #2da44e; }
  .meta { color: #777; font-size: 12px; margin-bottom: 4px; }
  .scroll { overflow-x: auto; }
</style>
</head><body>
<header>
  <strong>Deep OQ Admin</strong>
  <a href="{{ url_for('admin.dashboard') }}">Dashboard</a>
  <a href="{{ url_for('admin.users') }}">Users</a>
  <a href="{{ url_for('admin.conversations') }}">Conversations</a>
</header>
<main>{% block body %}{% endblock %}</main>
</body></html>
"""

DASHBOARD = BASE.replace(
    "{% block body %}{% endblock %}",
    """
<h2>Overview</h2>
<div class="cards">
  <div class="card"><b>{{ s.users }}</b>Users</div>
  <div class="card"><b>{{ s.conversations }}</b>Conversations</div>
  <div class="card"><b>{{ s.messages }}</b>Messages</div>
  <div class="card"><b>{{ s.messages_24h }}</b>Messages (24h)</div>
</div>
<h2>Recent messages</h2>
<div class="scroll"><table>
  <tr><th>Time</th><th>Role</th><th>Content</th><th></th></tr>
  {% for m in recent %}
  <tr>
    <td>{{ m.created_at }}</td><td>{{ m.role }}</td>
    <td>{{ m.content[:120] }}{% if m.content|length > 120 %}…{% endif %}</td>
    <td><a href="{{ url_for('admin.conversation', conv_id=m.conversation_id) }}">open</a></td>
  </tr>
  {% endfor %}
</table></div>
""",
)

USERS = BASE.replace(
    "{% block body %}{% endblock %}",
    """
<h2>Users ({{ rows|length }})</h2>
<div class="scroll"><table>
  <tr><th>ID</th><th>Email</th><th>Signed up</th><th>Conversations</th></tr>
  {% for u in rows %}
  <tr><td>{{ u.id }}</td><td>{{ u.email }}</td><td>{{ u.created_at }}</td><td>{{ u.conv_count }}</td></tr>
  {% endfor %}
</table></div>
""",
)

CONVERSATIONS = BASE.replace(
    "{% block body %}{% endblock %}",
    """
<h2>Recent conversations</h2>
<div class="scroll"><table>
  <tr><th>ID</th><th>User</th><th>Started</th><th>Messages</th><th></th></tr>
  {% for c in rows %}
  <tr>
    <td>{{ c.id }}</td><td>{{ c.email }}</td><td>{{ c.created_at }}</td><td>{{ c.msg_count }}</td>
    <td><a href="{{ url_for('admin.conversation', conv_id=c.id) }}">view</a></td>
  </tr>
  {% endfor %}
</table></div>
""",
)

CONVERSATION = BASE.replace(
    "{% block body %}{% endblock %}",
    """
<h2>Conversation {{ conv_id }}</h2>
<p><a href="{{ url_for('admin.conversations') }}">&larr; back</a></p>
{% for m in msgs %}
  <div class="msg {{ m.role }}">
    <div class="meta">{{ m.role }} &middot; {{ m.created_at }}</div>
    {{ m.content }}
  </div>
{% else %}
  <p>No messages found.</p>
{% endfor %}
""",
)


@admin.route("/")
@admin_required
def dashboard():
    s = query(
        """
        SELECT
          (SELECT COUNT(*) FROM users)         AS users,
          (SELECT COUNT(*) FROM conversations) AS conversations,
          (SELECT COUNT(*) FROM messages)      AS messages,
          (SELECT COUNT(*) FROM messages
             WHERE created_at > NOW() - INTERVAL '24 hours') AS messages_24h
        """
    )[0]
    recent = query(
        "SELECT conversation_id, role, content, created_at "
        "FROM messages ORDER BY created_at DESC LIMIT 20"
    )
    return render_template_string(DASHBOARD, s=s, recent=recent)


@admin.route("/users")
@admin_required
def users():
    rows = query(
        """
        SELECT u.id, u.email, u.created_at, COUNT(c.id) AS conv_count
        FROM users u
        LEFT JOIN conversations c ON c.user_id = u.id
        GROUP BY u.id
        ORDER BY u.created_at DESC
        LIMIT 500
        """
    )
    return render_template_string(USERS, rows=rows)


@admin.route("/conversations")
@admin_required
def conversations():
    rows = query(
        """
        SELECT c.id, u.email, c.created_at, COUNT(m.id) AS msg_count
        FROM conversations c
        LEFT JOIN users u ON u.id = c.user_id
        LEFT JOIN messages m ON m.conversation_id = c.id
        GROUP BY c.id, u.email
        ORDER BY c.created_at DESC
        LIMIT 100
        """
    )
    return render_template_string(CONVERSATIONS, rows=rows)


@admin.route("/conversations/<conv_id>")
@admin_required
def conversation(conv_id):
    msgs = query(
        "SELECT role, content, created_at FROM messages "
        "WHERE conversation_id = %s ORDER BY created_at ASC",
        (conv_id,),
    )
    return render_template_string(CONVERSATION, conv_id=conv_id, msgs=msgs)
