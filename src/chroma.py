import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import re

# Define the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Initialize ChromaDB
chroma_client = chromadb.Client(Settings(persist_directory="chroma_db", is_persistent=True))
collection = chroma_client.get_collection("mishnah")

# Define a simple retriever function for Hebrew and English texts
def simple_retriever(query: str, k: int = 3):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    hebrew_texts = [meta["hebrew"] for meta in results['metadatas'][0]]  # Access Hebrew texts
    english_texts = results['documents'][0]  # Access English texts
    sources = results['metadatas'][0]  # Access the metadata for sources
    return english_texts, hebrew_texts, sources

# Function to remove vowels from Hebrew text
def remove_vowels_hebrew(hebrew_text):
    pattern = re.compile(r'[\u0591-\u05C7]')
    hebrew_text_without_vowels = re.sub(pattern, '', hebrew_text)
    return hebrew_text_without_vowels

