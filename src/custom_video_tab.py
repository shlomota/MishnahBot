"""Streamlit UI for generating an explainer video from any user-supplied text,
using the same pipeline as the Daily Mishnah videos (see video_scene)."""
import os

import streamlit as st

import video_scene as vs

LENGTH_OPTIONS = [15, 30, 45, 60, 90, 120, 180]


def _query_param(name, default=""):
    return st.query_params.get(name, default)


def _initial_length():
    # ?length=... lets a link pre-fill the form; snap to the nearest option.
    try:
        requested = int(_query_param("length", vs.DEFAULT_TARGET_SECONDS))
    except ValueError:
        requested = vs.DEFAULT_TARGET_SECONDS
    return min(LENGTH_OPTIONS, key=lambda s: abs(s - requested))


def render_custom_video_tab():
    st.title("🎬 Explainer video from your text")
    st.caption(
        "Paste any text, look up a classical reference, or type a topic, and get a narrated, "
        "hand-drawn-style whiteboard explainer video in the same style as the Daily Mishnah videos."
    )

    if not st.session_state.get("video_gen_unlocked"):
        # Same PIN and unlock flag as the Daily Mishnah tab - this page has no
        # login, and each generation spends real OpenRouter credits.
        pin = st.text_input("PIN to generate a video", type="password", key="pin_custom")
        if pin:
            if vs.VIDEO_GEN_PIN and pin == vs.VIDEO_GEN_PIN:
                st.session_state["video_gen_unlocked"] = True
                st.rerun()
            else:
                st.error("Incorrect PIN.")
        return

    text_key = "custom_video_text"
    if text_key not in st.session_state:
        st.session_state[text_key] = _query_param("text")

    ref_col, btn_col = st.columns([4, 1])
    ref = ref_col.text_input(
        "Or: a reference or topic", value=_query_param("ref"),
        placeholder='e.g. "Shabbat 2:3", "ג ד בבא קמא", or "history of sourdough starter"',
    )
    btn_col.markdown("<div style='height: 1.9rem'></div>", unsafe_allow_html=True)
    if btn_col.button("Look up", disabled=not ref.strip()):
        with st.spinner("Looking up..."):
            looked_up, source = vs.fetch_reference_text(ref)
        if looked_up:
            st.session_state[text_key] = looked_up
            if source == "model":
                st.info("Not a Sefaria citation - this is the model's own research on the topic, not a sourced text.")
        else:
            st.error(f'Couldn\'t find or research "{ref}" - try rephrasing.')

    text = st.text_area("Text", height=200, key=text_key,
                        placeholder="Paste a passage, a verse, a mishnah, a story... or look up a reference above")
    length = st.select_slider("Length (seconds)", options=LENGTH_OPTIONS, value=_initial_length())
    style = st.text_input(
        "Visual style (optional)", value=_query_param("style"),
        placeholder="Default: flat vector children's-book illustration, warm earthy palette",
    )
    tone = st.text_input(
        "Narration tone (optional)", value=_query_param("tone"),
        placeholder="Default: deadpan, absurdist comedy bit",
    )

    st.caption(
        "Takes a couple of minutes and costs real money (well under $0.20) - use deliberately. "
        "Identical requests reuse the saved video."
    )

    key = vs.custom_video_key(text, length, style, tone)
    path = vs.custom_video_path(key)
    if st.button("Generate video", disabled=not text.strip()):
        if not os.path.exists(path):
            with st.spinner("Generating explainer video (script, animation, narration) — this takes a couple of minutes..."):
                try:
                    vs.generate_video_for_text(text, length, style, tone)
                except Exception as e:
                    st.error(f"Video generation failed: {e}")
                    return
        st.session_state["custom_video_key"] = key

    shown = st.session_state.get("custom_video_key")
    if shown and os.path.exists(vs.custom_video_path(shown)):
        shown_path = vs.custom_video_path(shown)
        st.video(shown_path)
        with open(shown_path, "rb") as f:
            st.download_button("Download video", f, file_name=f"{shown}.mp4", mime="video/mp4")
