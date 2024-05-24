import re
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.chat_models import BedrockChat
from langchain.prompts import PromptTemplate
from chroma import simple_retriever

# Initialize AWS Bedrock for Claude Sonnet with specific configurations for generation
generation_llm = BedrockChat(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0"
)

# Define the prompt template for English answers
english_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Answer the following question based on the provided context alone:
    Context: {context}
    Question: {question}
    Answer (short and concise):
    """
)

# Initialize the LLM chain for English answers
english_llm_chain = LLMChain(
    llm=generation_llm,
    prompt=english_prompt_template
)

# Define SimpleQA chain for English
class SimpleQAChain:
    def __init__(self, retriever, llm_chain):
        self.retriever = retriever
        self.llm_chain = llm_chain

    def __call__(self, inputs):
        question = inputs["query"]
        
        english_docs, hebrew_docs, sources = self.retriever(question, k=3)
        context = "\n".join(english_docs)
        
        response = self.llm_chain.run({"context": context, "question": question})
        sources_with_bold = []
        for i, source in enumerate(sources):
            source_text = english_docs[i]
            #source_text = re.sub(r'<b>(.*?)</b>', r'**\1**', source_text)  # Replace <b> with markdown bold
            sources_with_bold.append({
                "name": f"{source['seder']} {source['tractate']} Chapter {source['chapter']}, Mishnah {source['mishnah']}",
                "text": source_text
            })
        
        return response, sources_with_bold

# Function to initialize the English QA Chain
def EngQAChain():
    return SimpleQAChain(simple_retriever, english_llm_chain)

