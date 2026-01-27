import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
import qdrant_client
import re

# Load environment variables
load_dotenv()

def clean_text(text: str) -> str:
    """
    Cleans the text by removing excessive whitespace and repetitive headers/footers.
    Modify this function based on specific PDF artifacts.
    """
    # Remove multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def ingest_documents():
    print("🚀 Starting ingestion process...")
    
    # Configuration
    api_key = os.getenv("GOOGLE_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "rag_collection")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env")

    # Setup Gemini Embedding Model
    Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=api_key)
    
    # Initialize Qdrant Client
    print(f"🔌 Connecting to Qdrant at {qdrant_url}...")
    client = qdrant_client.QdrantClient(url=qdrant_url)
    
    # Initialize Vector Store
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load Documents
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"⚠️ '{data_dir}' directory created. Please put your PDFs there.")
        return

    print(f"📂 Loading documents from '{data_dir}'...")
    documents = SimpleDirectoryReader(data_dir).load_data()
    
    if not documents:
        print("⚠️ No documents found. Put PDFs in the 'data' folder.")
        return

    # Clean Documents
    print("🧹 Cleaning documents...")
    for doc in documents:
        doc.text = clean_text(doc.text)

    # Create Index (Chunking happens automatically here, default is 1024)
    # LlamaIndex defaults are usually good, but we can customize transformations if needed.
    print("Current Settings chunk size:", Settings.chunk_size) # Default is 1024

    print("Embeddings and Indexing (this may take a while)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    print(f"✅ Ingestion complete! {len(documents)} documents indexed in '{collection_name}'.")

if __name__ == "__main__":
    ingest_documents()
