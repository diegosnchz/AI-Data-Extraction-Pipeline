import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
import qdrant_client

load_dotenv()

def ingest_documents():
    print("🚀 Starting ingestion process...")
    api_key = os.getenv("GOOGLE_API_KEY")
    # Usamos nombre del servicio Docker para la red interna
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333") 
    collection_name = os.getenv("QDRANT_COLLECTION_NAME", "rag_collection")

    if not api_key:
        print("❌ Error: GOOGLE_API_KEY not found in .env")
        return

    try:
        # Embedding-001 es el modelo estable gratuito
        Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=api_key)
    except Exception as e:
        print(f"⚠️ Error setting up embedding model: {e}")
        return
    
    print(f"🔌 Connecting to Qdrant at {qdrant_url}...")
    try:
        client = qdrant_client.QdrantClient(url=qdrant_url)
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        return

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

    print("🧠 Generating embeddings and Indexing (this may take a while)...")
    try:
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        print(f"✅ Ingestion complete! {len(documents)} documents indexed.")
    except Exception as e:
        print(f"❌ Error during indexing: {e}")

if __name__ == "__main__":
    ingest_documents()