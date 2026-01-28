# Enterprise RAG System: DeepSeek R1 & Oracle Cloud

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![LlamaIndex](https://img.shields.io/badge/Framework-LlamaIndex-orange?style=flat-square)](https://www.llamaindex.ai/)
[![Qdrant](https://img.shields.io/badge/Vector%20Store-Qdrant-red?style=flat-square&logo=qdrant)](https://qdrant.tech/)
[![Docker](https://img.shields.io/badge/Infrastructure-Docker-blue?style=flat-square&logo=docker)](https://www.docker.com/)

This repository implements a high-performance Retrieval-Augmented Generation (RAG) system designed for precise knowledge extraction from private documents (PDF, TXT) using state-of-the-art language models and a robust microservices architecture.

## Technical Architecture

The system is built on four fundamental pillars to ensure privacy, speed, and cost-efficiency:

- **Reasoning Engine**: Utilizes DeepSeek R1, a language model optimized for complex reasoning tasks and data extraction.
- **Privacy-First Embeddings**: Documents are vectorized locally using the HuggingFace `BGE-Small-EN-v1.5` model, ensuring sensitive data never leaves your infrastructure during indexing.
- **Vector Database**: Employs Qdrant for high-speed storage and semantic search of document fragments.
- **Orchestration**: The entire system is containerized with Docker, facilitating seamless deployment on Oracle Cloud Infrastructure (OCI).

## Technical Stack

| Component | Technology |
| :--- | :--- |
| **RAG Framework** | LlamaIndex |
| **LLM** | DeepSeek R1 (via API) |
| **Embeddings** | BAAI/bge-small-en-v1.5 (Local) |
| **Vector Store** | Qdrant |
| **Interface** | Streamlit |
| **Infrastructure** | Docker & Oracle Cloud (Ubuntu Instance) |

## Installation and Deployment

### 1. Clone and Configure

```bash
git clone https://github.com/diegosnchz/AI-Data-Extraction-Pipeline.git
cd AI-Data-Extraction-Pipeline
```

### 2. Environment Variables

Create a `.env` file in the root directory (do not commit this file to version control) with the following credentials:

```env
DEEPSEEK_API_KEY=your_key_here
QDRANT_URL=http://qdrant:6333
```

### 3. Build and Run

```bash
docker compose up -d --build
```

### 4. Document Ingestion

Place your PDF/TXT files in the `/data` folder and run the indexing pipeline:

```bash
docker compose exec rag-app python src/ingestion/ingest.py
```

## Project Structure

- `src/app/`: User interface built with Streamlit.
- `src/ingestion/`: Document processing and cleaning scripts.
- `src/rag_engine/`: Core logic for DeepSeek and Qdrant integration.
- `data/`: Directory for source documents.
- `qdrant_data/`: Local persistence for the vector database.

## Security

This repository includes a pre-configured `.gitignore` to protect sensitive information, including API keys, virtual environments, and local databases.

---
**Developed by Diego Sánchez** - Exploring the frontier of AI and Cloud infrastructure.