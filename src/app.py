import streamlit as st

from custom_video_tab import render_custom_video_tab
from daily_mishnah_tab import render_daily_mishnah_tab
from rag_tab import render_rag_tab

st.set_page_config(page_title="MishnahBot: A Cross-Lingual RAG Application", initial_sidebar_state="collapsed")

pages = st.navigation(
    [
        st.Page(render_rag_tab, title="Ask the Mishnah", url_path="", default=True),
        st.Page(render_daily_mishnah_tab, title="Daily Mishnah", url_path="daily"),
        st.Page(render_custom_video_tab, title="Explainer Video", url_path="video"),
    ]
)
pages.run()
