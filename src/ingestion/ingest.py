import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import qdrant_client

load_dotenv()

def ingest_documents():
    print("🚀 Iniciando ingesta con Embeddings Locales...")
    
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333") 
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "rag_collection")

    # CONFIGURACIÓN LOCAL (Adiós al Error 429)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    print(f"🔌 Conectando a Qdrant en {qdrant_url}...")
    client = qdrant_client.QdrantClient(url=qdrant_url)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if not os.path.exists("data"):
        os.makedirs("data")
        return

    print("📂 Cargando documentos...")
    documents = SimpleDirectoryReader("data").load_data()
    
    if not documents:
        print("⚠️ Carpeta 'data' vacía.")
        return

    print("🧠 Indexando localmente (sin cuotas de Google)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print(f"✅ Éxito! Documentos indexados en '{collection_name}'.")

if __name__ == "__main__":
    ingest_documents()