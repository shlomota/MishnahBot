"""Optional email-based sync: links a browser's cookie id to an email so its
progress/prefs are shared with every other browser linked to that same email
(one email -> many devices). Unverified by design - this is a low-stakes
reading tracker, not an account system, so linking is just "type an email
you'll remember," not proof of ownership.
"""
import datetime
import re
import sqlite3
from contextlib import contextmanager

import progress_store as ps
import user_prefs as up

DB_PATH = ps.DB_PATH
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_links (
                cookie_user_id TEXT PRIMARY KEY,
                email TEXT NOT NULL
            )
            """
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


def is_valid_email(email):
    return bool(EMAIL_RE.match(email.strip()))


def get_linked_email(cookie_user_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT email FROM user_links WHERE cookie_user_id = ?", (cookie_user_id,)
        ).fetchone()
    return row[0] if row else None


def effective_id(cookie_user_id):
    """The id progress/prefs should actually be stored/read under for this browser."""
    return get_linked_email(cookie_user_id) or cookie_user_id


def link_email(cookie_user_id, email):
    """Link this browser to an email, merging any progress it already has into it."""
    email = email.strip().lower()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO user_links (cookie_user_id, email) VALUES (?, ?)
            ON CONFLICT(cookie_user_id) DO UPDATE SET email = excluded.email
            """,
            (cookie_user_id, email),
        )

    # Union this device's completions into the email identity - a day already
    # marked done on another linked device is left alone, not overwritten.
    now_str = datetime.datetime.now().isoformat()
    for cycle, day_num in ps.get_all_completed(cookie_user_id):
        ps.set_day_completed(email, cycle, day_num, True, now_str)

    # Prefs: only adopt this device's prefs if the email has none of its own yet.
    if up.get_prefs(email) == up.DEFAULT_PREFS:
        local_prefs = up.get_prefs(cookie_user_id)
        if local_prefs != up.DEFAULT_PREFS:
            up.set_prefs(email, local_prefs["language_mode"], local_prefs["commentary"], local_prefs["font_size"])

    return email


def unlink(cookie_user_id):
    with _connect() as conn:
        conn.execute("DELETE FROM user_links WHERE cookie_user_id = ?", (cookie_user_id,))
