"""Per-user (cookie-identified, no login) progress tracking for the daily Mishnah calendar."""
import os
import sqlite3
import uuid
from contextlib import contextmanager

import extra_streamlit_components as stx
import streamlit as st

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mishnah_progress.db")
COOKIE_NAME = "mishnah_user_id"


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
    """Return a stable anonymous ID for this browser, persisted via a cookie."""
    cookie_manager = stx.CookieManager(key="mishnah_cookie_manager")
    existing = cookie_manager.get(COOKIE_NAME)
    if existing:
        return existing

    if "pending_user_id" not in st.session_state:
        st.session_state.pending_user_id = str(uuid.uuid4())
    cookie_manager.set(
        COOKIE_NAME,
        st.session_state.pending_user_id,
        key="set_mishnah_user_id",
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
