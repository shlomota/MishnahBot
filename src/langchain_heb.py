import re
import json
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from chroma import simple_retriever, remove_vowels_hebrew

translation_llm = ChatBedrock(
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    model_kwargs={"temperature": 0.0, "max_tokens": 100}
)

generation_llm = ChatBedrock(
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    model_kwargs={"max_tokens": 1500, "temperature": 0.3},
)

translation_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""Translate the following Hebrew text to English, take into account that it relates to the mishnah.
Provide only the resulting English query NO OTHER TEXT.
Input text: {text}
Translation: """
)

hebrew_rerank_prompt_template = PromptTemplate(
    input_variables=["question", "candidates"],
    template="""להלן שאלה ורשימת קטעים ממשנה. ענה על השאלה בהתבסס על הקטעים הרלוונטיים.

שאלה: {question}

קטעים:
{candidates}

החזר אובייקט JSON בלבד עם השדות הבאים:
- "answer": תשובה קצרה ותמציתית בעברית
- "used": רשימת מספרי קטעים (אינדקס מ-0) שהשתמשת בהם ישירות לתשובה
- "related": רשימת מספרי קטעים קשורים לנושא אך לא שימשו ישירות (עד 5)

החזר JSON בלבד, ללא טקסט נוסף."""
)

translation_chain = translation_prompt_template | translation_llm | StrOutputParser()
hebrew_llm_chain = hebrew_rerank_prompt_template | generation_llm | StrOutputParser()


def _format_candidates(docs, sources):
    lines = []
    for i, (text, src) in enumerate(zip(docs, sources)):
        name = f"{src['seder']} {src['tractate']} Chapter {src['chapter']}, Mishnah {src['mishnah']}"
        lines.append(f"[{i}] {name}:\n{text}")
    return "\n\n".join(lines)


def _parse_rerank_response(raw, sources):
    """Parse LLM JSON response, returning (answer, used_indices, related_indices)."""
    try:
        clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE)
        data = json.loads(clean)
        answer = data.get("answer", raw)
        used = [i for i in data.get("used", []) if isinstance(i, int) and 0 <= i < len(sources)]
        related = [i for i in data.get("related", []) if isinstance(i, int) and 0 <= i < len(sources) and i not in used]
    except (json.JSONDecodeError, ValueError):
        answer = raw
        used = list(range(min(3, len(sources))))
        related = []
    return answer, used, related


def _build_source(i, sources, use_hebrew_text=True):
    src = sources[i]
    text = re.sub(r'<b>(.*?)</b>', r'**\1**', src['hebrew']) if use_hebrew_text else src.get('english', '')
    return {
        "name": f"{src['seder']} {src['tractate']} Chapter {src['chapter']}, Mishnah {src['mishnah']}",
        "text": text
    }


class SimpleQAChainWithTranslation:
    def __init__(self, translation_chain, retriever, llm_chain):
        self.translation_chain = translation_chain
        self.retriever = retriever
        self.llm_chain = llm_chain

    def __call__(self, inputs):
        hebrew_query = inputs["query"]

        translated_query = self.translation_chain.invoke({"text": hebrew_query})

        english_docs, hebrew_docs, sources = self.retriever(translated_query, k=20)
        hebrew_docs = [remove_vowels_hebrew(doc) for doc in hebrew_docs]

        candidates_text = _format_candidates(hebrew_docs, sources)
        raw = self.llm_chain.invoke({"question": hebrew_query, "candidates": candidates_text})

        answer, used_indices, related_indices = _parse_rerank_response(raw, sources)

        used_sources = [_build_source(i, sources) for i in used_indices]
        related_sources = [_build_source(i, sources) for i in related_indices]

        return answer, used_sources, related_sources


def HebQAChain():
    return SimpleQAChainWithTranslation(translation_chain, simple_retriever, hebrew_llm_chain)
