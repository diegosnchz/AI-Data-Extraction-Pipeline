import os
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek
import qdrant_client

def initialize_settings():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    # LLM (DeepSeek) para generar la respuesta
    Settings.llm = DeepSeek(model="deepseek-chat", api_key=api_key, temperature=0.1)
    # Embedding (Local) para buscar en el PDF
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def get_query_engine():
    initialize_settings()
    client = qdrant_client.QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))
    vector_store = QdrantVectorStore(client=client, collection_name=os.getenv("QDRANT_COLLECTION_NAME", "rag_collection"))
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index.as_query_engine(similarity_top_k=5)