"""Fetches Mishnah/commentary text from Sefaria's API, cached permanently in SQLite.

Sefaria's texts don't change, so once a ref is fetched it's cached forever.
This makes repeat views instant/offline-independent of Sefaria's own uptime,
and lets us pre-warm upcoming days ahead of time.
"""
import json
import os
import sqlite3
from contextlib import contextmanager

import requests

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sefaria_cache.db")
API_BASE = "https://www.sefaria.org/api"
REQUEST_TIMEOUT = 10


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                cache_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            )
            """
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


def _cache_get(key):
    with _connect() as conn:
        row = conn.execute("SELECT payload FROM cache WHERE cache_key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def _cache_set(key, payload, now_str):
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache (cache_key, payload, fetched_at) VALUES (?, ?, ?)",
            (key, json.dumps(payload), now_str),
        )


def _get_json(path, now_str):
    key = f"GET {path}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    resp = requests.get(f"{API_BASE}{path}", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    _cache_set(key, data, now_str)
    return data


def fetch_text(ref, now_str):
    """Return {'he': [...], 'text': [...], 'heTitle': str} for a ref, cached forever.

    'he'/'text' are returned exactly as Sefaria nests them (a flat per-mishnah
    list for a single chapter, a list-of-chapters-of-mishnayot for a chapter
    range) - callers that need to know which mishnah is which should use that
    structure directly rather than a pre-flattened one.

    Returns None if the ref can't be fetched (network error, doesn't exist, etc).
    """
    try:
        data = _get_json(f"/texts/{ref}", now_str)
    except (requests.RequestException, ValueError):
        return None
    if data.get("error"):
        return None
    return {
        "he": data.get("he") or [],
        "text": data.get("text") or [],
        "heTitle": data.get("heTitle") or "",
    }


def hebrew_tractate_name(he_title):
    """Strip Sefaria's 'משנה '/'משניות ' prefix, leaving the bare tractate name."""
    for prefix in ("משניות ", "משנה "):
        if he_title.startswith(prefix):
            return he_title[len(prefix):]
    return he_title


_HEBREW_HUNDREDS = [(400, "ת"), (300, "ש"), (200, "ר"), (100, "ק")]
_HEBREW_TENS = [(90, "צ"), (80, "פ"), (70, "ע"), (60, "ס"), (50, "נ"), (40, "מ"), (30, "ל"), (20, "כ"), (10, "י")]
_HEBREW_ONES = [(9, "ט"), (8, "ח"), (7, "ז"), (6, "ו"), (5, "ה"), (4, "ד"), (3, "ג"), (2, "ב"), (1, "א")]


def hebrew_numeral(n):
    """Convert a positive integer to a Hebrew gematria numeral (e.g. 11 -> יא).

    15 and 16 use the traditional טו/טז substitution rather than יה/יו, which
    resemble forms of the divine name.
    """
    if n <= 0:
        return str(n)
    result = ""
    remaining = n
    for value, letter in _HEBREW_HUNDREDS:
        while remaining >= value:
            result += letter
            remaining -= value
    if remaining == 15:
        result += "טו"
        remaining = 0
    elif remaining == 16:
        result += "טז"
        remaining = 0
    else:
        for value, letter in _HEBREW_TENS:
            if remaining >= value:
                result += letter
                remaining -= value
                break
        for value, letter in _HEBREW_ONES:
            if remaining >= value:
                result += letter
                remaining -= value
                break
    return result


def available_commentaries(ref, now_str):
    """Return sorted list of commentator display names available for a ref."""
    try:
        data = _get_json(f"/related/{ref}", now_str)
    except (requests.RequestException, ValueError):
        return []
    names = set()
    index_titles = {}
    for link in data.get("links", []):
        if link.get("category") != "Commentary":
            continue
        collective = link.get("collectiveTitle") or {}
        name = collective.get("en")
        if not name:
            continue
        names.add(name)
        index_titles.setdefault(name, link.get("index_title"))
    return sorted((name, index_titles[name]) for name in names)


def flatten_to_paragraphs(nested):
    """Recursively join an arbitrarily-nested list of HTML strings into a flat list.

    Some commentaries nest more deeply than the base Mishnah text (a list of
    per-lemma fragments within each mishnah), so depth isn't predictable -
    just collect every leaf string in order.
    """
    if not nested:
        return []
    if isinstance(nested, str):
        return [nested] if nested else []
    out = []
    for item in nested:
        out.extend(flatten_to_paragraphs(item))
    return out


def commentary_ref(index_title, chapter_num, mishnah_num):
    """A commentary ref scoped to exactly one mishnah (e.g. Bartenura_on_Mishnah_Beitzah.4.1).

    Fetching per-mishnah rather than per-chapter-range sidesteps a real
    inconsistency in Sefaria's range API: a commentary's own segment count
    for a chapter doesn't reliably match that chapter's mishnah count within
    a multi-chapter range fetch (most visibly on the non-first chapter), so
    positional alignment there isn't trustworthy. Per-mishnah refs have no
    such ambiguity - each fetch is already scoped to exactly the right unit.
    """
    return f"{index_title.replace(' ', '_')}.{chapter_num}.{mishnah_num}"
