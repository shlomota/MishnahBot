"""Streamlit UI for the Daily Mishnah calendar tab."""
import datetime
from zoneinfo import ZoneInfo

import streamlit as st
import streamlit.components.v1 as components

import mishnah_calendar as mc
import progress_store as ps
import sefaria_client as sc
import user_links as ul
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
.commentary-label {{
    direction: ltr; text-align: left; font-weight: 700; opacity: 1;
    font-size: {font_size_rem * HEADER_RATIO}rem; margin-top: 0.3rem;
}}
@media (min-width: {DESKTOP_BREAKPOINT_PX}px) {{
    .mishnah-he {{ font-size: {desktop_rem}rem; }}
    .mishnah-en {{ font-size: {desktop_rem}rem; }}
    .mishnah-header {{ font-size: {desktop_rem * HEADER_RATIO}rem; }}
    .commentary-label {{ font-size: {desktop_rem * HEADER_RATIO}rem; }}
}}
</style>
"""


def _render_commentary_block(label, he, en, language_mode):
    with st.container():
        st.markdown("<div class='commentary-block'>", unsafe_allow_html=True)
        st.markdown(f"<div class='commentary-label'>{label}</div>", unsafe_allow_html=True)
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


def _toggle_day_done(effective_id, cycle_id, day_num, currently_done):
    # An on_click callback rather than a plain "if button: ...st.rerun()":
    # an explicit st.rerun() from inside a dialog closes it, but a widget's
    # own natural rerun (which a callback still goes through) leaves it
    # open - confirmed empirically - so repeatedly marking days here doesn't
    # kick you out of the list after every tap.
    ps.set_day_completed(effective_id, cycle_id, day_num, not currently_done, datetime.datetime.now().isoformat())


def _mark_tractate_done(effective_id, cycle_id, day_nums):
    now_str = datetime.datetime.now().isoformat()
    for day_num in day_nums:
        ps.set_day_completed(effective_id, cycle_id, day_num, True, now_str)


def _open_progress_dialog(effective_id, cycle_id, days, state_key, current_day_num, today_day_num):
    day_nums_by_tractate = {}
    for d in days:
        for r in d.readings:
            day_nums_by_tractate.setdefault(r.tractate, set()).add(d.day_num)
    tractates = list(day_nums_by_tractate)  # chronological, by first appearance
    total_days = len(days)
    hebrew_year = int(cycle_id)

    @st.dialog("Full schedule & progress", width="large")
    def _dialog():
        completed_days = ps.get_completed_days(effective_id, cycle_id)
        pct = len(completed_days) / total_days
        st.progress(pct, text=f"{len(completed_days)} / {total_days} days completed ({pct:.0%})")

        mark_col, btn_col = st.columns([2, 1])
        chosen_tractate = mark_col.selectbox(
            "Mark a whole tractate as done", tractates, key=f"mark_tractate_{cycle_id}"
        )
        btn_col.button(
            "Mark all",
            key=f"mark_tractate_btn_{cycle_id}",
            on_click=_mark_tractate_done,
            args=(effective_id, cycle_id, day_nums_by_tractate[chosen_tractate]),
        )

        st.caption("Scroll and tap a day to jump to its text, or tap the checkmark to mark it done.")
        current_row_id = f"cal-row-{cycle_id}-{current_day_num}"
        with st.container(height=450):
            for d in days:
                st.markdown(f"<span id='cal-row-{cycle_id}-{d.day_num}'></span>", unsafe_allow_html=True)
                row_done, row_go = st.columns([1, 6])
                is_done = d.day_num in completed_days
                row_done.button(
                    "✅" if is_done else "⬜",
                    key=f"cal_toggle_{cycle_id}_{d.day_num}",
                    on_click=_toggle_day_done,
                    args=(effective_id, cycle_id, d.day_num, is_done),
                )
                content = d.raw_schedule or ("Siyum" if d.is_siyum else "No reading")
                greg = mc.gregorian_date_for(hebrew_year, d.hebrew_date)
                greg_str = greg.strftime("%b %d") if greg else ""
                today_marker = "📍 " if d.day_num == today_day_num else ""
                label = f"{today_marker}{d.hebrew_date} · {greg_str} · {content}"
                is_current = d.day_num == current_day_num
                if row_go.button(
                    label,
                    key=f"cal_goto_{cycle_id}_{d.day_num}",
                    width="stretch",
                    type="primary" if is_current else "secondary",
                ):
                    st.session_state[state_key] = d.day_num
                    st.rerun()

        # st.container(height=...) has no native "default scroll position" API,
        # so scroll the current day's row into view via its own invisible
        # anchor span once the dialog's DOM has settled.
        components.html(
            f"""
            <script>
            setTimeout(function() {{
                var el = window.parent.document.getElementById('{current_row_id}');
                if (el) {{ el.scrollIntoView({{block: 'center', behavior: 'instant'}}); }}
            }}, 200);
            </script>
            """,
            height=0,
        )

    _dialog()


def _save_all_prefs(effective_id):
    # An on_change callback (see _toggle_day_done for why) rather than a
    # plain "if changed: ...st.rerun()", so changing one setting doesn't
    # close the dialog before you've had a chance to change another.
    up.set_prefs(
        effective_id,
        st.session_state.pref_language_mode,
        st.session_state.pref_commentary,
        st.session_state.pref_font_size,
    )


def _open_settings_dialog(cookie_user_id, effective_id, prefs, available_commentary_names):
    @st.dialog("Settings", icon="⚙️")
    def _dialog():
        st.radio(
            "Language",
            up.LANGUAGE_MODES,
            index=up.LANGUAGE_MODES.index(prefs["language_mode"]),
            horizontal=True,
            key="pref_language_mode",
            on_change=_save_all_prefs,
            args=(effective_id,),
        )
        options = ["None"] + available_commentary_names
        current = prefs["commentary"] if prefs["commentary"] in options else "None"
        st.selectbox(
            "Commentary",
            options,
            index=options.index(current),
            key="pref_commentary",
            on_change=_save_all_prefs,
            args=(effective_id,),
        )
        st.select_slider(
            "Text size",
            up.FONT_SIZES,
            value=prefs["font_size"],
            key="pref_font_size",
            on_change=_save_all_prefs,
            args=(effective_id,),
        )

        st.divider()
        linked_email = ul.get_linked_email(cookie_user_id)
        if linked_email:
            st.caption(
                f"🔗 Synced as **{linked_email}** — enter it on other devices to share "
                "progress and these settings there too."
            )
            if st.button("Unlink this browser", key="unlink_email_btn"):
                ul.unlink(cookie_user_id)
                st.rerun()
        else:
            st.caption(
                "Sync progress and display settings across devices (optional): "
                "enter the same email on each one."
            )
            email_input = st.text_input("Email", key="link_email_input", placeholder="you@example.com")
            if st.button("Sync this browser", key="link_email_btn"):
                if ul.is_valid_email(email_input):
                    ul.link_email(cookie_user_id, email_input)
                    st.rerun()
                else:
                    st.error("Enter a valid email address.")

    _dialog()


def render_daily_mishnah_tab():
    # Resolved first and before any rendering: if the cookie manager needs a
    # beat to report the browser's real cookies, we stop cleanly here rather
    # than mid-render (see progress_store.get_user_id).
    user_id = ps.get_user_id()
    effective_id = ul.effective_id(user_id)  # the linked email, once synced; else this browser's own id

    prefs = up.get_prefs(effective_id)
    st.markdown(_text_css(up.FONT_SIZE_REM[prefs["font_size"]]), unsafe_allow_html=True)
    st.title("Daily Mishnah")
    st.caption(
        "Following R. Ethan Tucker's Hebrew-year-aligned calendar to finish all of Shishah Sedarim in one year. "
        "Texts are retrieved from [Sefaria](https://www.sefaria.org). "
        "Optionally sync your progress and settings across devices with an email in Settings."
    )

    today = datetime.datetime.now(APP_TIMEZONE).date()
    cycle_id, variant_name, days, today_day = mc.calendar_for_date(today)
    days_by_num = {d.day_num: d for d in days}
    total_days = len(days)
    today_day_num = today_day.day_num if today_day else 1

    completed_days = ps.get_completed_days(effective_id, cycle_id)

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

    NAV_PREV, NAV_TODAY, NAV_NEXT, NAV_PROGRESS, NAV_SETTINGS = "◀ Prev", "Today", "Next ▶", "📋 Progress", "⚙️"
    on_today = st.session_state[state_key] == today_day_num
    nav_options = [NAV_PREV] + ([] if on_today else [NAV_TODAY]) + [NAV_NEXT, NAV_PROGRESS, NAV_SETTINGS]
    nav_choice = st.segmented_control("Navigate", nav_options, label_visibility="collapsed", key="nav_seg")
    if nav_choice is not None:
        if nav_choice == NAV_PREV:
            st.session_state[state_key] = _clamp(st.session_state[state_key] - 1)
        elif nav_choice == NAV_NEXT:
            st.session_state[state_key] = _clamp(st.session_state[state_key] + 1)
        elif nav_choice == NAV_TODAY:
            st.session_state[state_key] = today_day_num
        elif nav_choice == NAV_PROGRESS:
            st.session_state["_pending_dialog"] = "progress"
        elif nav_choice == NAV_SETTINGS:
            st.session_state["_pending_dialog"] = "settings"
        st.session_state._reset_nav_seg = True
        st.rerun()

    day = days_by_num[st.session_state[state_key]]
    is_today = day.day_num == today_day_num
    now_str = datetime.datetime.now().isoformat()
    language_mode, commentary = prefs["language_mode"], prefs["commentary"]

    # Discover which commentaries actually exist for this day's reading(s), so
    # the settings dropdown never offers a commentary that isn't covered.
    commentary_names = set()
    for reading in day.readings:
        for name, _index_title in sc.available_commentaries(reading.sefaria_ref, now_str):
            commentary_names.add(name)

    pending_dialog = st.session_state.pop("_pending_dialog", None)
    if pending_dialog == "progress":
        _open_progress_dialog(effective_id, cycle_id, days, state_key, day.day_num, today_day_num)
    elif pending_dialog == "settings":
        _open_settings_dialog(user_id, effective_id, prefs, sorted(commentary_names))

    greg_date = mc.gregorian_date_for(int(cycle_id), day.hebrew_date)
    greg_str = f" · {greg_date.strftime('%B %d, %Y')}" if greg_date else ""
    header = f"{day.hebrew_date} ({cycle_id}){greg_str}"
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
        ps.set_day_completed(effective_id, cycle_id, day.day_num, not done, now_str)
        st.rerun()

    st.divider()
    pct = len(completed_days) / total_days
    st.progress(pct, text=f"{len(completed_days)} / {total_days} days completed ({pct:.0%})")
