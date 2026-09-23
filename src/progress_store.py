"""Per-user (cookie-identified, no login) progress tracking for the daily Mishnah calendar."""
import datetime
import os
import sqlite3
import uuid
from contextlib import contextmanager

import extra_streamlit_components as stx
import streamlit as st

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mishnah_progress.db")
COOKIE_NAME = "mishnah_user_id"
COOKIE_LIFETIME = datetime.timedelta(days=3650)


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS progress (
                user_id TEXT NOT NULL,
                cycle TEXT NOT NULL,
                day_num INTEGER NOT NULL,
                completed_at TEXT NOT NULL,
                PRIMARY KEY (user_id, cycle, day_num)
            )
            """
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_user_id():
    """Return a stable anonymous ID for this browser, persisted via a cookie.

    The cookie-manager component returns a placeholder empty dict on the very
    first script run of a session (before its frontend has reported the
    browser's real cookies back over the websocket) - acting on that as "no
    cookie exists" would silently mint a fresh id on every page load/refresh,
    orphaning previously-saved progress. So on that first empty read we stop
    and let Streamlit's automatic rerun (triggered once the component's real
    value arrives) supply the truth before we decide anything.
    """
    cookie_manager = stx.CookieManager(key="mishnah_cookie_manager")
    cookies = cookie_manager.cookies

    if not cookies and not st.session_state.get("_cookie_manager_ready"):
        st.session_state["_cookie_manager_ready"] = True
        st.stop()

    existing = cookies.get(COOKIE_NAME)
    if existing:
        return existing

    if "pending_user_id" not in st.session_state:
        st.session_state.pending_user_id = str(uuid.uuid4())
    cookie_manager.set(
        COOKIE_NAME,
        st.session_state.pending_user_id,
        key="set_mishnah_user_id",
        expires_at=datetime.datetime.now() + COOKIE_LIFETIME,
    )
    return st.session_state.pending_user_id


def get_completed_days(user_id, cycle):
    with _connect() as conn:
        rows = conn.execute(
            "SELECT day_num FROM progress WHERE user_id = ? AND cycle = ?", (user_id, cycle)
        ).fetchall()
    return {row[0] for row in rows}


def set_day_completed(user_id, cycle, day_num, completed, now_str):
    with _connect() as conn:
        if completed:
            conn.execute(
                """
                INSERT INTO progress (user_id, cycle, day_num, completed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, cycle, day_num) DO NOTHING
                """,
                (user_id, cycle, day_num, now_str),
            )
        else:
            conn.execute(
                "DELETE FROM progress WHERE user_id = ? AND cycle = ? AND day_num = ?",
                (user_id, cycle, day_num),
            )
