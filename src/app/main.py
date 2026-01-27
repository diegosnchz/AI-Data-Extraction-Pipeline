import streamlit as st
import sys
import os

# Añadir directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rag_engine.engine import get_query_engine
from src.evaluation.metrics import evaluate_response

st.set_page_config(page_title="Oracle RAG - Gemini", layout="wide")

st.title("🤖 Enterprise RAG System")
st.subheader("Powered by Oracle Cloud & Google Gemini")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_engine" not in st.session_state:
    try:
        st.session_state.query_engine = get_query_engine()
        st.success("Motor RAG conectado a Qdrant y Gemini.")
    except Exception as e:
        st.error(f"Error conectando motor: {e}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metrics" in message:
            with st.expander("📊 Métricas de Calidad"):
                col1, col2 = st.columns(2)
                col1.metric("Fidelidad", message["metrics"]["faithfulness"])
                col2.metric("Relevancia", message["metrics"]["answer_relevancy"])

if prompt := st.chat_input("Consulta a tus documentos..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            try:
                response = st.session_state.query_engine.query(prompt)
                metrics = evaluate_response(prompt, response)
                
                st.markdown(response.response)
                with st.expander("📊 Métricas de Calidad"):
                    col1, col2 = st.columns(2)
                    col1.metric("Fidelidad", metrics["faithfulness"])
                    col2.metric("Relevancia", metrics["answer_relevancy"])
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response.response,
                    "metrics": metrics
                })
            except Exception as e:
                st.error(f"Ocurrió un error: {e}")