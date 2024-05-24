import streamlit as st
from bidi.algorithm import get_display
from datetime import datetime
from langchain_heb import SimpleQAChainWithTranslation as HebQAChain, hebrew_llm_chain, translation_chain
from langchain_eng import SimpleQAChain as EngQAChain, english_llm_chain
from chroma import simple_retriever
import os


# JavaScript for redirecting based on URL
redirect_script = """
<script type="text/javascript">
    const currentUrl = window.location.href;
    if (currentUrl.includes("taami.us")) {
        window.location.replace("http://taami.us:8501");
    }
</script>
"""

GA_TRACKING_CODE = """
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-8Y74XE23K4"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-8Y74XE23K4');
</script>
"""

# Function to read the request count and date from a file
def read_request_count():
    if not os.path.exists("request_count.txt"):
        return 100, str(datetime.today().date())
    with open("request_count.txt", "r") as f:
        lines = f.readlines()
        count = int(lines[0].strip())
        last_update_date = lines[1].strip()
        return count, last_update_date

# Function to write the request count and date to a file
def write_request_count(count, date):
    with open("request_count.txt", "w") as f:
        f.write(f"{count}\n{date}")


# Streamlit app
st.set_page_config(page_title="MishnahBot: A Cross-Lingual RAG Application")
# Insert the redirect script
st.markdown(redirect_script, unsafe_allow_html=True)
st.markdown(GA_TRACKING_CODE, unsafe_allow_html=True)

# Page title
st.title("MishnahBot: A Cross-Lingual RAG Application")

# Load request count
requests_left, last_update_date = read_request_count()
current_date = str(datetime.today().date())
if current_date != last_update_date:
    requests_left = 100
    last_update_date = current_date
    write_request_count(requests_left, last_update_date)

requests_counter = st.empty()
requests_counter.markdown(f"Requests left today: {requests_left}")

# Language selection
language = st.radio("Choose your language:", ('Hebrew', 'English'))

# Clear the text input when language changes
if "language" in st.session_state and st.session_state.language != language:
    st.session_state.query_input = ""
st.session_state.language = language

# Set text area direction based on language
if language == 'Hebrew':
    st.markdown(
        """
        <style>
        .stTextArea textarea {
            direction: rtl;
        }
        .rtl {
            direction: rtl;
            text-align: right;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .stTextArea textarea {
            direction: ltr;
        }
        .ltr {
            direction: ltr;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Predefined queries
if language == 'Hebrew':
    #default_queries = [
    #    get_display("מהו סוג המלאכה השלישי האסור בשבת?"),
    #    get_display("ממתי אפשר לקרוא את שמע בבוקר?"),
    #    get_display("מי מסר את התורה לאנשי כנסת הגדולה?")
    #]
    default_queries = ["מהו סוג המלאכה השלישי האסור בשבת?", "האם מותר לבנות סוכה על הגב של גמל?", "מי מסר את התורה לאנשי כנסת הגדולה?"]
else:
    default_queries = ["What is the third type of work forbidden on Shabbat?", "Is it permitted to build a sukkah on the back of a camel?", "Who transmitted the Torah to the Men of the Great Assembly?"]

st.subheader("Default queries:")
col1, col2, col3 = st.columns(3)
if col1.button(default_queries[0]):
    st.session_state.query_input = default_queries[0]
if col2.button(default_queries[1]):
    st.session_state.query_input = default_queries[1]
if col3.button(default_queries[2]):
    st.session_state.query_input = default_queries[2]

# Ensure the session state for query_input exists before rendering the text area
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# Input text area
query_input = st.text_area("Enter your question:", value=st.session_state.query_input, key="query_input")

# Submit button
if st.button("Submit") and query_input:
    if requests_left > 0:
        # Adjust the QA chain based on the selected language
        if language == 'Hebrew':
            qa_chain = HebQAChain(translation_chain, simple_retriever, hebrew_llm_chain)
        else:
            qa_chain = EngQAChain(simple_retriever, english_llm_chain)
        
        response, sources = qa_chain({"query": query_input})
        
        # Update the request count
        requests_left -= 1
        write_request_count(requests_left, last_update_date)
        requests_counter.markdown(f"Requests left today: {requests_left}")
        
        st.markdown(f"<h3 class='rtl'>תשובה:</h3>" if language == 'Hebrew' else "<h3>Answer:</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='rtl'>{response}</div>" if language == 'Hebrew' else f"<div class='ltr'>{response}</div>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='rtl'>מקורות:</h3>" if language == 'Hebrew' else "<h3>Sources:</h3>", unsafe_allow_html=True)
        for source in sources:
            with st.expander(source['name']):
                st.markdown(f"#### {source['name']}")
                st.markdown(f"<div class='rtl'>{source['text']}</div>" if language == 'Hebrew' else f"<div class='ltr'>{source['text']}</div>", unsafe_allow_html=True)
    else:
        st.warning("No requests left for today. Please try again tomorrow.")

# Add footer with hyperlink
st.markdown(
    """
    <div style='text-align: center; padding-top: 20px;'>
        Made by <a href="https://www.linkedin.com/in/shlomo-tannor-aa967a1a8/" target="_blank">Shlomo Tannor</a> |
        <a href="https://medium.com/@stannor/exploring-rag-applications-across-languages-conversing-with-the-mishnah-16615c30f780" target="_blank">Read more on Medium</a>
    </div>
    """,
    unsafe_allow_html=True
)
