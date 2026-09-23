"""Per-user (cookie-identified) display preferences for the Daily Mishnah tab."""
import sqlite3
from contextlib import contextmanager

from progress_store import DB_PATH as PROGRESS_DB_PATH

DB_PATH = PROGRESS_DB_PATH  # share the same small local db as progress tracking

LANGUAGE_MODES = ["Hebrew", "English", "Bilingual"]
FONT_SIZES = ["Small", "Medium", "Large", "Extra Large"]
FONT_SIZE_REM = {"Small": 1.8, "Medium": 2.1, "Large": 2.5, "Extra Large": 3.1}
DEFAULT_PREFS = {"language_mode": "Bilingual", "commentary": "None", "font_size": "Medium"}


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prefs (
                user_id TEXT PRIMARY KEY,
                language_mode TEXT NOT NULL,
                commentary TEXT NOT NULL,
                font_size TEXT NOT NULL DEFAULT 'Medium'
            )
            """
        )
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(prefs)")}
        if "font_size" not in existing_cols:
            conn.execute("ALTER TABLE prefs ADD COLUMN font_size TEXT NOT NULL DEFAULT 'Medium'")
        yield conn
        conn.commit()
    finally:
        conn.close()


def get_prefs(user_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT language_mode, commentary, font_size FROM prefs WHERE user_id = ?", (user_id,)
        ).fetchone()
    if row is None:
        return dict(DEFAULT_PREFS)
    return {"language_mode": row[0], "commentary": row[1], "font_size": row[2]}


def set_prefs(user_id, language_mode, commentary, font_size):
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO prefs (user_id, language_mode, commentary, font_size) VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                language_mode = excluded.language_mode,
                commentary = excluded.commentary,
                font_size = excluded.font_size
            """,
            (user_id, language_mode, commentary, font_size),
        )
