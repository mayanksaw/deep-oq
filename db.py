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
