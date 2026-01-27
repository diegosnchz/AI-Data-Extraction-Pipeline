# Evaluation-Driven RAG System

This project implements a Retrieval-Augmented Generation (RAG) system with integrated quality evaluation using **LlamaIndex**, **Google Gemini**, **Qdrant**, and **Ragas**.

## 🚀 Quick Start

### 1. Environment Setup
1.  Rename `.env.example` to `.env`.
2.  Add your **Google API Key** to `.env` (`GOOGLE_API_KEY`).
    - Get it from [Google AI Studio](https://aistudio.google.com/).

### 2. Data Ingestion
1.  Place your PDF documents in the `data/` folder.
2.  Install dependencies (if running locally):
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the ingestion script to clean and index your data:
    ```bash
    python src/ingestion/ingest.py
    ```
    *Note: Ensure Qdrant is running. If you don't have a local Qdrant, use Docker.*

### 3. Run with Docker (Recommended)
This will start Qdrant and the Streamlit App together.

1.  Build and start the services:
    ```bash
    docker-compose up --build
    ```
2.  Access the app at: [http://localhost:8501](http://localhost:8501)
3.  Qdrant dashboard: [http://localhost:6333](http://localhost:6333)

### 4. Evaluation
Every time you ask a question, the system uses a "Judge LLM" (Gemini) to evaluate:
- **Faithfulness**: Is the answer derived from the context?
- **Answer Relevancy**: Is the answer relevant to the query?
Metrics are displayed in the "Quality Metrics" expander below the answer.

## 📂 Project Structure
- `src/ingestion`: Scripts to load, clean, and index PDFs.
- `src/rag_engine`: LlamaIndex configuration and query logic.
- `src/evaluation`: Ragas metrics implementation.
- `src/app`: Streamlit frontend.
- `data`: Folder for your PDF documents.
