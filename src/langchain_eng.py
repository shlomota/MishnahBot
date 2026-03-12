import re
import json
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from chroma import simple_retriever

generation_llm = ChatBedrock(
    model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    model_kwargs={"max_tokens": 1500, "temperature": 0.3},
)

english_rerank_prompt_template = PromptTemplate(
    input_variables=["question", "candidates"],
    template="""Given the following question and candidate passages from the Mishnah, answer the question based on the relevant passages.

Question: {question}

Candidates:
{candidates}

Return only a JSON object with:
- "answer": a short and concise answer in English
- "used": list of candidate indices (0-indexed) directly used to answer
- "related": list of candidate indices that are topically related but not directly used (up to 5)

Return only valid JSON, no other text."""
)

english_llm_chain = english_rerank_prompt_template | generation_llm | StrOutputParser()


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


class SimpleQAChain:
    def __init__(self, retriever, llm_chain):
        self.retriever = retriever
        self.llm_chain = llm_chain

    def __call__(self, inputs):
        question = inputs["query"]

        english_docs, hebrew_docs, sources = self.retriever(question, k=20)

        candidates_text = _format_candidates(english_docs, sources)
        raw = self.llm_chain.invoke({"question": question, "candidates": candidates_text})

        answer, used_indices, related_indices = _parse_rerank_response(raw, sources)

        used_sources = []
        for i in used_indices:
            src = sources[i]
            used_sources.append({
                "name": f"{src['seder']} {src['tractate']} Chapter {src['chapter']}, Mishnah {src['mishnah']}",
                "text": english_docs[i]
            })

        related_sources = []
        for i in related_indices:
            src = sources[i]
            related_sources.append({
                "name": f"{src['seder']} {src['tractate']} Chapter {src['chapter']}, Mishnah {src['mishnah']}",
                "text": english_docs[i]
            })

        return answer, used_sources, related_sources


def EngQAChain():
    return SimpleQAChain(simple_retriever, english_llm_chain)
