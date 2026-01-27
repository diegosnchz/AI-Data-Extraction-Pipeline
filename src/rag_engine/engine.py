import os
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
import qdrant_client

def initialize_settings():
    """Configures LlamaIndex global settings."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    # Configure Gemini Model
    Settings.llm = Gemini(model_name="models/gemini-1.5-flash", api_key=api_key, temperature=0.1)
    Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=api_key)

def get_index():
    """Establishes connection to Qdrant and returns the VectorStoreIndex."""
    initialize_settings()
    
    qdrant_url = os.getenv("QDRANT_URL")
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "rag_collection")
    
    print(f"🔌 Connecting to Qdrant at {qdrant_url}...")
    client = qdrant_client.QdrantClient(url=qdrant_url)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    
    # We assume the index already exists from ingestion
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index

def get_query_engine():
    """Returns a query engine ready for RAG."""
    index = get_index()
    # similarity_top_k=5 is a good default for retrieval
    return index.as_query_engine(similarity_top_k=5)
