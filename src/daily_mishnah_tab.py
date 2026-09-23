"""Streamlit UI for the Daily Mishnah calendar tab."""
import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import mishnah_calendar as mc
import progress_store as ps

# The Hebrew day used for "today" is computed in this timezone rather than the
# server's local time, since the Jewish calendar day is a real-world calendar
# concept tied to where learners actually are, not to server infrastructure.
APP_TIMEZONE = ZoneInfo("America/New_York")


def render_daily_mishnah_tab():
    st.title("Daily Mishnah")
    st.caption(
        "Following R. Ethan Tucker's Hebrew-year-aligned calendar to finish all of Shishah Sedarim in one year. "
        "Your progress is tracked for this browser only — no login needed."
    )

    today = datetime.datetime.now(APP_TIMEZONE).date()
    cycle_id, variant_name, days, today_day = mc.calendar_for_date(today)
    days_by_num = {d.day_num: d for d in days}
    total_days = len(days)
    today_day_num = today_day.day_num if today_day else 1

    user_id = ps.get_user_id()
    completed_days = ps.get_completed_days(user_id, cycle_id)

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
                st.markdown(f"#### {reading.label}")
                link_col, toggle_col = st.columns(2)
                link_col.link_button("Open on Sefaria ↗", reading.sefaria_url, width="stretch")
                show_key = f"show_inline_{cycle_id}_{day.day_num}_{reading.sefaria_ref}"
                if toggle_col.toggle("Show text here", key=show_key):
                    components.iframe(reading.sefaria_url, height=700, scrolling=True)

    if day.completes:
        st.success(f"Completes: {day.completes}")

    done = day.day_num in completed_days
    new_done = st.checkbox("Mark this day as learned", value=done, key=f"done_{cycle_id}_{day.day_num}")
    if new_done != done:
        ps.set_day_completed(user_id, cycle_id, day.day_num, new_done, datetime.datetime.now().isoformat())
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
            now_str = datetime.datetime.now().isoformat()
            for _, row in changed.iterrows():
                ps.set_day_completed(user_id, cycle_id, int(row["Day"]), bool(row["Done"]), now_str)
            st.rerun()
