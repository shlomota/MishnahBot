import chromadb
from chromadb.config import Settings

# Initialize ChromaDB client with the path to the persisted database
chroma_client = chromadb.Client(Settings(is_persistent=True, persist_directory="chroma_db"))

# List all collections to verify if the "mishnah" collection exists
collections = chroma_client.list_collections()
print("Collections:", collections)

# Try to get the "mishnah" collection
try:
        collection = chroma_client.get_collection("mishnah")
        print("Successfully loaded the 'mishnah' collection.")
except ValueError as e:
        print(e)
        
import pdb;pdb.set_trace()
