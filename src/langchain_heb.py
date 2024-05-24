import re
from langchain.chains import LLMChain
from langchain.llms import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain.prompts import PromptTemplate
from chroma import simple_retriever, remove_vowels_hebrew

# Initialize AWS Bedrock for Llama 3 70B Instruct with specific configurations for translation
translation_llm = Bedrock(
    model_id="meta.llama3-70b-instruct-v1:0",
    model_kwargs={
        "temperature": 0.0,  # Set lower temperature for translation
        "max_gen_len": 50  # Limit number of tokens for translation
    }
)

translation_llm = BedrockChat(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={
        "temperature": 0.0,  # Set lower temperature for translation
        "max_tokens": 100
    }
)



# Initialize AWS Bedrock for Claude Sonnet with specific configurations for generation
generation_llm = BedrockChat(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"max_tokens": 300, "temperature": 0.3},
)

# Define the translation prompt template
translation_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""Translate the following Hebrew text to English, take into account that it relates to the mishnah. 
Provide only the resulting English query NO OTHER TEXT.
Input text: {text}
Translation: """
)

# Define the prompt template for Hebrew answers
hebrew_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""ענה על השאלה הבאה בהתבסס על ההקשר המסופק בלבד:
    הקשר: {context}
    שאלה: {question}
    תשובה (קצרה ותמציתית):
    """
)

# Translation chain for translating queries from Hebrew to English
translation_chain = LLMChain(
    llm=translation_llm,
    prompt=translation_prompt_template
)

# Initialize the LLM chain for Hebrew answers
hebrew_llm_chain = LLMChain(
    llm=generation_llm,
    prompt=hebrew_prompt_template
)

# Define SimpleQA chain with translation for Hebrew
class SimpleQAChainWithTranslation:
    def __init__(self, translation_chain, retriever, llm_chain):
        self.translation_chain = translation_chain
        self.retriever = retriever
        self.llm_chain = llm_chain

    def __call__(self, inputs):
        hebrew_query = inputs["query"]
        
        translated_query = self.translation_chain.run({"text": hebrew_query})
        
        english_docs, hebrew_docs, sources = self.retriever(translated_query, k=3)
        hebrew_docs = [remove_vowels_hebrew(doc) for doc in hebrew_docs]

        context = "\n".join(hebrew_docs)
        
        response = self.llm_chain.run({"context": context, "question": hebrew_query})
        sources_with_bold = []
        for source in sources:
            source_text = re.sub(r'<b>(.*?)</b>', r'**\1**', source['hebrew'])  # Replace <b> with markdown bold
            sources_with_bold.append({
                "name": f"{source['seder']} {source['tractate']} Chapter {source['chapter']}, Mishnah {source['mishnah']}",
                "text": source_text
            })
        
        return response, sources_with_bold

# Function to initialize the Hebrew QA Chain
def HebQAChain():
    return SimpleQAChainWithTranslation(translation_chain, simple_retriever, hebrew_llm_chain)

