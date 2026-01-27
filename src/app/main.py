import streamlit as st
import nest_asyncio
import asyncio
import os
import sys

# Add project root to path for imports to work
sys.path.append(os.getcwd())

from src.rag_engine.engine import get_query_engine
from src.evaluation.metrics import evaluate_response

# Apply nest_asyncio because LlamaIndex uses asyncio and Streamlit runs in an event loop
nest_asyncio.apply()

st.set_page_config(page_title="Evaluation-Driven RAG", layout="wide")

st.title("🔎 Evaluation-Driven RAG Chat")
st.markdown("""
This system retrieves information from your documents and **evaluates** the quality of its own answers using a Judge LLM.
""")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_engine" not in st.session_state:
    try:
        with st.spinner("⏳ Initializing RAG Engine..."):
            st.session_state.query_engine = get_query_engine()
        st.success("Engine ready!")
    except Exception as e:
        st.error(f"Error initializing engine: {e}")
        st.info("Make sure you have ingested data and set your GOOGLE_API_KEY.")
        st.stop()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metrics" in message:
            with st.expander("📊 Quality Metrics"):
                cols = st.columns(2)
                cols[0].metric("Faithfulness", f"{message['metrics']['faithfulness']:.2f}")
                cols[1].metric("Answer Relevancy", f"{message['metrics']['answer_relevancy']:.2f}")

# User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking & Evaluating..."):
            try:
                # Query RAG
                response = st.session_state.query_engine.query(prompt)
                response_text = response.response
                
                # Extract contexts
                source_nodes = response.source_nodes
                contexts = [node.node.get_content() for node in source_nodes]
                
                # Evaluate
                eval_results = evaluate_response(prompt, response_text, contexts)
                
                # Display result
                message_placeholder.markdown(response_text)
                
                # Display metrics
                faithful_score = eval_results["faithfulness"]
                relevancy_score = eval_results["answer_relevancy"]
                
                with st.expander("📊 Quality Metrics", expanded=True):
                    cols = st.columns(2)
                    cols[0].metric("Faithfulness", f"{faithful_score:.2f}", help="Is the answer derived from the context?")
                    cols[1].metric("Answer Relevancy", f"{relevancy_score:.2f}", help="Is the answer relevant to the query?")
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "metrics": {
                        "faithfulness": faithful_score,
                        "answer_relevancy": relevancy_score
                    }
                })

            except Exception as e:
                st.error(f"An error occurred: {e}")

# Sidebar
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        st.warning("For this demo, please place PDFs in the 'data' folder and restart the ingestion script.")
        # Logic to save and ingest could be added here, but staying simple for boilerplate
