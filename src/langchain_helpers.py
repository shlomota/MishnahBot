# langchain_helpers.py
from langchain.chains import LLMChain, RetrievalQA
from langchain.llms import Bedrock
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List
import re
from langchain_community.chat_models import BedrockChat

# Initialize AWS Bedrock for Llama 3 70B Instruct with specific configurations for translation
translation_llm = Bedrock(
    model_id="meta.llama3-70b-instruct-v1:0",
    model_kwargs={
        "temperature": 0.0,  # Set lower temperature for translation
        "max_gen_len": 50  # Limit number of tokens for translation
    }
)

# Initialize AWS Bedrock for Claude Sonnet with specific configurations for generation
generation_llm = BedrockChat(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0"
)

# Define the translation prompt template
translation_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""Translate the following Hebrew text to English:
    Input text: {text}
    Translation: 
    """
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

# Initialize ChromaDB
chroma_client = chromadb.Client(Settings(persist_directory="chroma_db", is_persistent=True))
collection = chroma_client.get_collection("mishnah")

# Define the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

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

# Define a simple retriever function for Hebrew texts
def simple_retriever(query: str, k: int = 3) -> List[str]:
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    documents = [meta['hebrew'] for meta in results['metadatas'][0]]  # Access Hebrew texts
    sources = results['metadatas'][0]  # Access the metadata for sources
    return documents, sources

# Function to remove vowels from Hebrew text
def remove_vowels_hebrew(hebrew_text):
    pattern = re.compile(r'[\u0591-\u05C7]')
    hebrew_text_without_vowels = re.sub(pattern, '', hebrew_text)
    return hebrew_text_without_vowels

# Define SimpleQA chain with translation
class SimpleQAChainWithTranslation:
    def __init__(self, translation_chain, retriever, llm_chain):
        self.translation_chain = translation_chain
        self.retriever = retriever
        self.llm_chain = llm_chain

    def __call__(self, inputs):
        hebrew_query = inputs["query"]
        print("#" * 50)
        print(f"Hebrew query: {hebrew_query}")
        
        # Print the translation prompt
        translation_prompt = translation_prompt_template.format(text=hebrew_query)
        print("#" * 50)
        print(f"Translation Prompt: {translation_prompt}")
        
        # Perform the translation using the translation chain with specific configurations
        translated_query = self.translation_chain.run({"text": hebrew_query})
        print("#" * 50)
        print(f"Translated Query: {translated_query}")  # Print the translated query for debugging
        
        retrieved_docs, sources = self.retriever(translated_query)
        retrieved_docs = [remove_vowels_hebrew(doc) for doc in retrieved_docs]

        context = "\n".join(retrieved_docs)
        
        # Print the final prompt for generation
        final_prompt = hebrew_prompt_template.format(context=context, question=hebrew_query)
        print("#" * 50)
        print(f"Final Prompt for Generation:\n {final_prompt}")
        
        response = self.llm_chain.run({"context": context, "question": hebrew_query})
        
        # Collect the source names and texts
        source_names = [f"{source['seder']} {source['tractate']} פרק {source['chapter']}, משנה {source['mishnah']}" for source in sources]
        source_texts = [source['hebrew'] for source in sources]
        
        return {"answer": response, "source_names": source_names, "source_texts": source_texts}

def get_qa_chain():
    translation_chain = LLMChain(
        llm=translation_llm,
        prompt=translation_prompt_template
    )

    hebrew_llm_chain = LLMChain(
        llm=generation_llm,
        prompt=hebrew_prompt_template
    )

    return SimpleQAChainWithTranslation(translation_chain, simple_retriever, hebrew_llm_chain)

