import streamlit as st
import sys
import os
import hmac

# Añadir directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.rag_engine.engine import get_query_engine
from src.evaluation.metrics import evaluate_response

st.set_page_config(page_title="DeepSeek R1 & Oracle Cloud", layout="wide")

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets.get("APP_PASSWORD", os.getenv("APP_PASSWORD", ""))):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password
        else:
            st.session_state["password_correct"] = False

    # Password validation logic
    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    
    if "password_correct" in st.session_state:
        st.error("Password incorrect")
        
    return False

if not check_password():
    st.stop()

st.title("Enterprise RAG System")
st.subheader("Powered by Oracle Cloud & DeepSeek R1")

with st.sidebar:
    st.header("Estado del Sistema")
    st.success("Embeddings: BGE-Small (Local)")
    if "query_engine" in st.session_state:
        st.success("Qdrant: Conectado")
    else:
        st.warning("Qdrant: Desconectado")

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
        
        # Mostrar fuentes si existen
        if "sources" in message:
            with st.expander("Fuentes Consultadas"):
                for source in message["sources"]:
                    st.markdown(f"**{source['file']}** (Relevance: {source['score']:.2f})")
                    st.caption(f"...{source['text']}...")

        if "metrics" in message:
            with st.expander("Metricas de Calidad"):
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
                
                # Procesar fuentes
                sources = []
                if hasattr(response, 'source_nodes'):
                    for node in response.source_nodes:
                        sources.append({
                            "file": node.metadata.get("file_name", "Unknown Document"),
                            "score": node.score or 0.0,
                            "text": node.get_text()[:200]
                        })

                st.markdown(response.response)
                
                # Mostrar fuentes
                if sources:
                    with st.expander("Fuentes Consultadas"):
                        for source in sources:
                            st.markdown(f"**{source['file']}** (Relevance: {source['score']:.2f})")
                            st.caption(f"...{source['text']}...")

                with st.expander("Metricas de Calidad"):
                    col1, col2 = st.columns(2)
                    col1.metric("Fidelidad", metrics["faithfulness"])
                    col2.metric("Relevancia", metrics["answer_relevancy"])
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response.response,
                    "metrics": metrics,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"Ocurrio un error: {e}")