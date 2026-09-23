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


HEADER_RATIO = 1.15
# rem is tied to the browser's root font-size, not viewport width, so without
# this the same size preference renders at the identical physical size on a
# 27" monitor as on a phone - which reads as oversized on desktop. Scale down
# above a phone-width breakpoint instead.
DESKTOP_BREAKPOINT_PX = 768
DESKTOP_SCALE = 0.62


def _text_css(font_size_rem):
    desktop_rem = font_size_rem * DESKTOP_SCALE
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
    direction: rtl; text-align: right; font-weight: 700; opacity: 1;
    font-size: {font_size_rem * HEADER_RATIO}rem; margin-top: 1.1rem;
}}
.commentary-block {{ border-left: 3px solid rgba(128,128,128,0.35); padding-left: 0.8rem; margin: 0.3rem 0 0.7rem 0; }}
@media (min-width: {DESKTOP_BREAKPOINT_PX}px) {{
    .mishnah-he {{ font-size: {desktop_rem}rem; }}
    .mishnah-en {{ font-size: {desktop_rem}rem; }}
    .mishnah-header {{ font-size: {desktop_rem * HEADER_RATIO}rem; }}
}}
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


def _open_progress_dialog(user_id, cycle_id, days, completed_days, now_str, state_key):
    day_nums_by_tractate = {}
    for d in days:
        for r in d.readings:
            day_nums_by_tractate.setdefault(r.tractate, set()).add(d.day_num)
    tractates = list(day_nums_by_tractate)  # chronological, by first appearance

    jump_placeholder = "— pick a day —"
    day_num_by_label = {}
    for d in days:
        content = d.raw_schedule or ("Siyum" if d.is_siyum else "No reading")
        day_num_by_label[f"Day {d.day_num} · {d.hebrew_date} · {content}"] = d.day_num

    @st.dialog("Full schedule & progress", width="large")
    def _dialog():
        jump_choice = st.selectbox(
            "Jump to a day's text (type to search by tractate)",
            [jump_placeholder] + list(day_num_by_label),
            key=f"jump_select_{cycle_id}",
        )
        if jump_choice != jump_placeholder:
            st.session_state[state_key] = day_num_by_label[jump_choice]
            st.rerun()

        mark_col, btn_col = st.columns([2, 1])
        chosen_tractate = mark_col.selectbox(
            "Mark a whole tractate as done", tractates, key=f"mark_tractate_{cycle_id}"
        )
        if btn_col.button("Mark all", key=f"mark_tractate_btn_{cycle_id}"):
            for day_num in day_nums_by_tractate[chosen_tractate]:
                ps.set_day_completed(user_id, cycle_id, day_num, True, now_str)
            st.rerun()

        st.caption("Or check off any day directly here — changes save immediately.")
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

    _dialog()


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

    # A widget's session_state key can't be reassigned in the same run it was
    # instantiated in, so clearing the segmented control's selection (it's a
    # momentary action, not a persistent choice) has to happen at the top of
    # the *next* run, before the widget below re-instantiates with that key.
    if st.session_state.get("_reset_nav_seg"):
        st.session_state.nav_seg = None
        st.session_state._reset_nav_seg = False

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
        st.session_state._reset_nav_seg = True
        st.rerun()

    day = days_by_num[st.session_state[state_key]]
    is_today = day.day_num == today_day_num
    now_str = datetime.datetime.now().isoformat()

    if st.button("📋 Full schedule & progress"):
        _open_progress_dialog(user_id, cycle_id, days, completed_days, now_str, state_key)

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
    # A full-width button rather than st.checkbox: the checkbox's own tap
    # target is only ~13px, well under a usable mobile touch target.
    btn_label = "✅ Learned — tap to unmark" if done else "☐ Mark this day as learned"
    if st.button(btn_label, width="stretch", type="secondary" if done else "primary"):
        ps.set_day_completed(user_id, cycle_id, day.day_num, not done, now_str)
        st.rerun()

    st.divider()
    pct = len(completed_days) / total_days
    st.progress(pct, text=f"{len(completed_days)} / {total_days} days completed ({pct:.0%})")
