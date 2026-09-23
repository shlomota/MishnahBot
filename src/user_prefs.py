"""Per-user (cookie-identified) display preferences for the Daily Mishnah tab."""
import sqlite3
from contextlib import contextmanager

from progress_store import DB_PATH as PROGRESS_DB_PATH

DB_PATH = PROGRESS_DB_PATH  # share the same small local db as progress tracking

LANGUAGE_MODES = ["Hebrew", "English", "Bilingual"]
DEFAULT_PREFS = {"language_mode": "Bilingual", "commentary": "None"}


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prefs (
                user_id TEXT PRIMARY KEY,
                language_mode TEXT NOT NULL,
                commentary TEXT NOT NULL
            )
            """
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_prefs(user_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT language_mode, commentary FROM prefs WHERE user_id = ?", (user_id,)
        ).fetchone()
    if row is None:
        return dict(DEFAULT_PREFS)
    return {"language_mode": row[0], "commentary": row[1]}


def set_prefs(user_id, language_mode, commentary):
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO prefs (user_id, language_mode, commentary) VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET language_mode = excluded.language_mode, commentary = excluded.commentary
            """,
            (user_id, language_mode, commentary),
        )
