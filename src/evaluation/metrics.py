import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def evaluate_response(query: str, response_text: str, contexts: list[str]):
    """
    Evaluates the RAG response using Ragas metrics: Faithfulness and Answer Relevancy.
    Uses Gemini as the Judge LLM.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Configure Gemini for Ragas (via LangChain wrapper)
    # Ragas uses LangChain LLMs for evaluation
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0
    )
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    # Prepare dataset for Ragas (single row)
    data = {
        "question": [query],
        "answer": [response_text],
        "contexts": [contexts],
        # "ground_truth": [ground_truth] # Optional, not available in live chat
    }
    dataset = Dataset.from_dict(data)

    # Configure metrics with the LLM
    # Note: Ragas v0.1+ allows passing llm to evaluate() or metrics
    
    # Run evaluation
    print("⚖️ Running Ragas evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings
    )

    return results
