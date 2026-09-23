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


def _text_css(font_size_rem):
    # overflow-wrap/word-break/max-width guarantee text reflows to the
    # viewport at any font size instead of forcing horizontal scroll -
    # Streamlit's own viewport meta tag blocks native pinch-zoom, so this
    # in-app size control is the reliable way to let someone "zoom in".
    return f"""
<style>
.mishnah-he {{
    direction: rtl; text-align: right; font-size: {font_size_rem}rem; line-height: 1.9;
    margin-bottom: 0.4rem; overflow-wrap: break-word; word-break: break-word; max-width: 100%;
}}
.mishnah-en {{
    direction: ltr; text-align: left; font-size: {font_size_rem}rem; line-height: 1.7;
    margin-bottom: 0.9rem; opacity: 0.92; overflow-wrap: break-word; word-break: break-word; max-width: 100%;
}}
.mishnah-header {{
    direction: rtl; text-align: right; font-weight: 600; opacity: 0.6;
    font-size: {font_size_rem * 0.6}rem; margin-top: 0.9rem;
}}
.commentary-block {{ border-left: 3px solid rgba(128,128,128,0.35); padding-left: 0.8rem; margin: 0.3rem 0 0.7rem 0; }}
</style>
"""


def _render_commentary_block(label, he, en, language_mode):
    with st.container():
        st.markdown("<div class='commentary-block'>", unsafe_allow_html=True)
        st.caption(label)
        if language_mode in ("Hebrew", "Bilingual") and he:
            st.markdown(f"<div class='mishnah-he'>{he}</div>", unsafe_allow_html=True)
        if language_mode in ("English", "Bilingual") and en:
            st.markdown(f"<div class='mishnah-en'>{en}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def _render_interleaved(reading, data, c_data, commentary_choice, language_mode):
    """Render each mishnah followed immediately by its own commentary, if any.

    Sefaria doesn't reliably segment every commentary one-per-mishnah within a
    multi-chapter range fetch (a commentary's own natural section count for a
    chapter can differ from that chapter's mishnah count, and in practice the
    non-first chapter of a range often collapses to a single combined slot).
    So per-mishnah interleaving is only used when a chapter's commentary
    segment count actually matches its mishnah count; otherwise that whole
    chapter's commentary is shown as one block after its last mishnah, rather
    than silently mis-attributing or dropping it.
    """
    is_range = reading.chapter_end != reading.chapter_start
    he_by_chapter = data["he"] if is_range else [data["he"]]
    en_by_chapter = data["text"] if is_range else [data["text"]]

    he_com_by_chapter = []
    en_com_by_chapter = []
    if c_data:
        he_com_by_chapter = c_data["he"] if is_range else [c_data["he"]]
        en_com_by_chapter = c_data["text"] if is_range else [c_data["text"]]

    for chapter_offset in range(max(len(he_by_chapter), len(en_by_chapter))):
        chapter_num = reading.chapter_start + chapter_offset
        he_chapter = he_by_chapter[chapter_offset] if chapter_offset < len(he_by_chapter) else []
        en_chapter = en_by_chapter[chapter_offset] if chapter_offset < len(en_by_chapter) else []
        he_com_chapter = he_com_by_chapter[chapter_offset] if chapter_offset < len(he_com_by_chapter) else []
        en_com_chapter = en_com_by_chapter[chapter_offset] if chapter_offset < len(en_com_by_chapter) else []

        mishnah_count = max(len(he_chapter), len(en_chapter))
        com_aligned = commentary_choice and commentary_choice != "None" and mishnah_count and (
            len(he_com_chapter) == mishnah_count or len(en_com_chapter) == mishnah_count
        )

        for mishnah_offset in range(mishnah_count):
            he = he_chapter[mishnah_offset] if mishnah_offset < len(he_chapter) else ""
            en = en_chapter[mishnah_offset] if mishnah_offset < len(en_chapter) else ""
            header = f"פרק {sc.hebrew_numeral(chapter_num)} משנה {sc.hebrew_numeral(mishnah_offset + 1)}"
            st.markdown(f"<div class='mishnah-header'>{header}</div>", unsafe_allow_html=True)
            if language_mode in ("Hebrew", "Bilingual") and he:
                st.markdown(f"<div class='mishnah-he'>{he}</div>", unsafe_allow_html=True)
            if language_mode in ("English", "Bilingual") and en:
                st.markdown(f"<div class='mishnah-en'>{en}</div>", unsafe_allow_html=True)

            if com_aligned:
                he_com_raw = he_com_chapter[mishnah_offset] if mishnah_offset < len(he_com_chapter) else None
                en_com_raw = en_com_chapter[mishnah_offset] if mishnah_offset < len(en_com_chapter) else None
                he_com = "<br><br>".join(sc.flatten_to_paragraphs(he_com_raw)) if he_com_raw else ""
                en_com = "<br><br>".join(sc.flatten_to_paragraphs(en_com_raw)) if en_com_raw else ""
                if he_com or en_com:
                    _render_commentary_block(commentary_choice, he_com, en_com, language_mode)

        if commentary_choice and commentary_choice != "None" and not com_aligned and (he_com_chapter or en_com_chapter):
            he_com = "<br><br>".join(sc.flatten_to_paragraphs(he_com_chapter))
            en_com = "<br><br>".join(sc.flatten_to_paragraphs(en_com_chapter))
            if he_com or en_com:
                _render_commentary_block(f"{commentary_choice} — chapter {chapter_num}", he_com, en_com, language_mode)


def _render_reading(reading, language_mode, commentary_choice, now_str):
    st.markdown(f"#### {reading.label}")
    data = sc.fetch_text(reading.sefaria_ref, now_str)
    if data is None or not (data["he"] or data["text"]):
        st.warning("Couldn't load this text right now.")
        st.link_button("Open on Sefaria ↗", reading.sefaria_url, width="stretch")
        return

    c_data = None
    if commentary_choice and commentary_choice != "None":
        # Look up this reading's own index_title for the chosen commentator name
        # (it varies per tractate, e.g. "Bartenura on Mishnah Beitzah") rather
        # than guessing a "{name} on {book}" pattern, which isn't universal.
        available = dict(sc.available_commentaries(reading.sefaria_ref, now_str))
        index_title = available.get(commentary_choice)
        if index_title:
            c_ref = sc.commentary_ref(index_title, reading.chapter_start, reading.chapter_end)
            c_data = sc.fetch_text(c_ref, now_str)
        if not c_data or not (c_data["he"] or c_data["text"]):
            st.caption(f"{commentary_choice} isn't available for this chapter.")
            c_data = None

    _render_interleaved(reading, data, c_data, commentary_choice, language_mode)

    st.link_button("Open on Sefaria ↗", reading.sefaria_url, width="stretch")


def _render_settings(user_id, prefs, available_commentary_names, state_key, current_day, total_days):
    with st.expander("Settings & jump to day", icon="⚙️"):
        picked = st.number_input(
            "Jump to day #",
            min_value=1,
            max_value=total_days,
            value=current_day,
        )
        if int(picked) != current_day:
            st.session_state[state_key] = int(picked)
            st.rerun()

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
        font_size = st.select_slider(
            "Text size",
            up.FONT_SIZES,
            value=prefs["font_size"],
            key="pref_font_size",
        )
        if (language_mode, commentary, font_size) != (prefs["language_mode"], prefs["commentary"], prefs["font_size"]):
            up.set_prefs(user_id, language_mode, commentary, font_size)
            st.rerun()
    return language_mode, commentary, font_size


def render_daily_mishnah_tab():
    # Resolved first and before any rendering: if the cookie manager needs a
    # beat to report the browser's real cookies, we stop cleanly here rather
    # than mid-render (see progress_store.get_user_id).
    user_id = ps.get_user_id()

    prefs = up.get_prefs(user_id)
    st.markdown(_text_css(up.FONT_SIZE_REM[prefs["font_size"]]), unsafe_allow_html=True)
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

    completed_days = ps.get_completed_days(user_id, cycle_id)

    state_key = f"daily_mishnah_day_{cycle_id}"
    if state_key not in st.session_state:
        st.session_state[state_key] = today_day_num

    def _clamp(day_num):
        return max(1, min(total_days, day_num))

    NAV_PREV, NAV_TODAY, NAV_NEXT = "◀ Prev", "Today", "Next ▶"
    on_today = st.session_state[state_key] == today_day_num
    nav_options = [NAV_PREV, NAV_NEXT] if on_today else [NAV_PREV, NAV_TODAY, NAV_NEXT]
    nav_choice = st.segmented_control("Navigate", nav_options, label_visibility="collapsed", key="nav_seg")
    if nav_choice is not None:
        if nav_choice == NAV_PREV:
            st.session_state[state_key] = _clamp(st.session_state[state_key] - 1)
        elif nav_choice == NAV_NEXT:
            st.session_state[state_key] = _clamp(st.session_state[state_key] + 1)
        elif nav_choice == NAV_TODAY:
            st.session_state[state_key] = today_day_num
        st.session_state.nav_seg = None
        st.rerun()

    day = days_by_num[st.session_state[state_key]]
    is_today = day.day_num == today_day_num
    now_str = datetime.datetime.now().isoformat()

    # Discover which commentaries actually exist for today's reading(s), so the
    # settings dropdown never offers a commentary that isn't covered.
    commentary_names = set()
    for reading in day.readings:
        for name, _index_title in sc.available_commentaries(reading.sefaria_ref, now_str):
            commentary_names.add(name)
    language_mode, commentary, _font_size = _render_settings(
        user_id, prefs, sorted(commentary_names), state_key, st.session_state[state_key], total_days
    )

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
        st.caption("Select a row to jump to that day's reading.")
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
        event = st.dataframe(
            df,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            key=f"progress_log_{cycle_id}",
        )
        selected_rows = event["selection"]["rows"] if event else []
        if selected_rows:
            selected_day_num = int(df.iloc[selected_rows[0]]["Day"])
            if selected_day_num != st.session_state[state_key]:
                st.session_state[state_key] = selected_day_num
                st.rerun()
