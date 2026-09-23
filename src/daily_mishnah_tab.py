"""Streamlit UI for the Daily Mishnah calendar tab."""
import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

import mishnah_calendar as mc
import progress_store as ps
import sefaria_client as sc
import user_prefs as up

# The Hebrew day used for "today" is computed in this timezone rather than the
# server's local time, since the Jewish calendar day is a real-world calendar
# concept tied to where learners actually are, not to server infrastructure.
APP_TIMEZONE = ZoneInfo("America/New_York")

TEXT_CSS = """
<style>
.mishnah-he { direction: rtl; text-align: right; font-size: 1.15rem; line-height: 1.9; margin-bottom: 0.4rem; }
.mishnah-en { direction: ltr; text-align: left; line-height: 1.7; margin-bottom: 0.9rem; color: var(--text-color, inherit); opacity: 0.92; }
.mishnah-num { font-weight: 600; opacity: 0.55; font-size: 0.85rem; margin-top: 0.6rem; }
.commentary-block { border-left: 3px solid rgba(128,128,128,0.35); padding-left: 0.8rem; margin-top: 0.5rem; }
</style>
"""


def _render_segments(he_segments, en_segments, language_mode):
    count = max(len(he_segments), len(en_segments))
    for i in range(count):
        he = he_segments[i] if i < len(he_segments) else ""
        en = en_segments[i] if i < len(en_segments) else ""
        st.markdown(f"<div class='mishnah-num'>{i + 1}</div>", unsafe_allow_html=True)
        if language_mode in ("Hebrew", "Bilingual") and he:
            st.markdown(f"<div class='mishnah-he'>{he}</div>", unsafe_allow_html=True)
        if language_mode in ("English", "Bilingual") and en:
            st.markdown(f"<div class='mishnah-en'>{en}</div>", unsafe_allow_html=True)


def _render_reading(reading, language_mode, commentary_choice, now_str):
    st.markdown(f"#### {reading.label}")
    data = sc.fetch_text(reading.sefaria_ref, now_str)
    if data is None or not (data["he"] or data["text"]):
        st.warning("Couldn't load this text right now.")
        st.link_button("Open on Sefaria ↗", reading.sefaria_url, width="stretch")
        return

    _render_segments(data["he"], data["text"], language_mode)

    if commentary_choice and commentary_choice != "None":
        # Look up this reading's own index_title for the chosen commentator name
        # (it varies per tractate, e.g. "Bartenura on Mishnah Beitzah") rather
        # than guessing a "{name} on {book}" pattern, which isn't universal.
        available = dict(sc.available_commentaries(reading.sefaria_ref, now_str))
        index_title = available.get(commentary_choice)
        c_data = None
        if index_title:
            c_ref = sc.commentary_ref(index_title, reading.chapter_start, reading.chapter_end)
            c_data = sc.fetch_text(c_ref, now_str)
        he_paragraphs = sc.flatten_to_paragraphs(c_data["he"]) if c_data else []
        en_paragraphs = sc.flatten_to_paragraphs(c_data["text"]) if c_data else []
        if he_paragraphs or en_paragraphs:
            with st.container():
                st.markdown(f"**{commentary_choice}**")
                st.markdown("<div class='commentary-block'>", unsafe_allow_html=True)
                if language_mode in ("Hebrew", "Bilingual") and he_paragraphs:
                    st.markdown(f"<div class='mishnah-he'>{'<br><br>'.join(he_paragraphs)}</div>", unsafe_allow_html=True)
                if language_mode in ("English", "Bilingual") and en_paragraphs:
                    st.markdown(f"<div class='mishnah-en'>{'<br><br>'.join(en_paragraphs)}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.caption(f"{commentary_choice} isn't available for this chapter.")

    st.link_button("Open on Sefaria ↗", reading.sefaria_url, width="stretch")


def _render_settings(user_id, prefs, available_commentary_names):
    with st.expander("Display settings", icon="⚙️"):
        language_mode = st.radio(
            "Language",
            up.LANGUAGE_MODES,
            index=up.LANGUAGE_MODES.index(prefs["language_mode"]),
            horizontal=True,
            key="pref_language_mode",
        )
        options = ["None"] + available_commentary_names
        current = prefs["commentary"] if prefs["commentary"] in options else "None"
        commentary = st.selectbox(
            "Commentary",
            options,
            index=options.index(current),
            key="pref_commentary",
        )
        if (language_mode, commentary) != (prefs["language_mode"], prefs["commentary"]):
            up.set_prefs(user_id, language_mode, commentary)
            st.rerun()
    return language_mode, commentary


def render_daily_mishnah_tab():
    st.markdown(TEXT_CSS, unsafe_allow_html=True)
    st.title("Daily Mishnah")
    st.caption(
        "Following R. Ethan Tucker's Hebrew-year-aligned calendar to finish all of Shishah Sedarim in one year. "
        "Your progress and preferences are tracked for this browser only — no login needed."
    )

    today = datetime.datetime.now(APP_TIMEZONE).date()
    cycle_id, variant_name, days, today_day = mc.calendar_for_date(today)
    days_by_num = {d.day_num: d for d in days}
    total_days = len(days)
    today_day_num = today_day.day_num if today_day else 1

    user_id = ps.get_user_id()
    completed_days = ps.get_completed_days(user_id, cycle_id)
    prefs = up.get_prefs(user_id)

    state_key = f"daily_mishnah_day_{cycle_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = today_day_num

    def _clamp(day_num):
        return max(1, min(total_days, day_num))

    nav_cols = st.columns([1, 1, 2, 1])
    if nav_cols[0].button("Prev", width="stretch"):
        st.session_state[state_key] = _clamp(st.session_state[state_key] - 1)
    if nav_cols[1].button("Today", width="stretch"):
        st.session_state[state_key] = today_day_num
    picked = nav_cols[2].number_input(
        "Jump to day #",
        min_value=1,
        max_value=total_days,
        value=st.session_state[state_key],
        label_visibility="collapsed",
    )
    if int(picked) != st.session_state[state_key]:
        st.session_state[state_key] = int(picked)
    if nav_cols[3].button("Next", width="stretch"):
        st.session_state[state_key] = _clamp(st.session_state[state_key] + 1)

    day = days_by_num[st.session_state[state_key]]
    is_today = day.day_num == today_day_num
    now_str = datetime.datetime.now().isoformat()

    # Discover which commentaries actually exist for today's reading(s), so the
    # settings dropdown never offers a commentary that isn't covered.
    commentary_names = set()
    for reading in day.readings:
        for name, _index_title in sc.available_commentaries(reading.sefaria_ref, now_str):
            commentary_names.add(name)
    language_mode, commentary = _render_settings(user_id, prefs, sorted(commentary_names))

    header = f"Day {day.day_num} of {total_days} — {day.hebrew_date} ({cycle_id})"
    st.subheader(header + " · today" if is_today else header)
    if day.parsha:
        st.caption(day.parsha)

    if day.is_siyum:
        st.info("No new chapters today — a siyum/review day in the calendar.")
    elif not day.readings:
        st.info("No new chapters scheduled for this day (Shabbat/holiday).")
    else:
        for reading in day.readings:
            with st.container(border=True):
                _render_reading(reading, language_mode, commentary, now_str)

    if day.completes:
        st.success(f"Completes: {day.completes}")

    done = day.day_num in completed_days
    new_done = st.checkbox("Mark this day as learned", value=done, key=f"done_{cycle_id}_{day.day_num}")
    if new_done != done:
        ps.set_day_completed(user_id, cycle_id, day.day_num, new_done, now_str)
        st.rerun()

    st.divider()
    pct = len(completed_days) / total_days
    st.progress(pct, text=f"{len(completed_days)} / {total_days} days completed ({pct:.0%})")

    with st.expander("View full schedule & progress log"):
        rows = [
            {
                "Day": d.day_num,
                "Hebrew Date": d.hebrew_date,
                "Schedule": d.raw_schedule or ("Siyum" if d.is_siyum else "—"),
                "Done": d.day_num in completed_days,
            }
            for d in days
        ]
        df = pd.DataFrame(rows)
        edited = st.data_editor(
            df,
            hide_index=True,
            width="stretch",
            disabled=["Day", "Hebrew Date", "Schedule"],
            key=f"progress_editor_{cycle_id}",
        )
        changed = edited[edited["Done"] != df["Done"]]
        if len(changed):
            for _, row in changed.iterrows():
                ps.set_day_completed(user_id, cycle_id, int(row["Day"]), bool(row["Done"]), now_str)
            st.rerun()
