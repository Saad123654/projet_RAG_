import streamlit as st
import requests
from rag_utils import build_rag_chain
import time

st.set_page_config(page_title="Assistant RAG", layout="wide")
st.title("🤖 Assistant RAG")
st.markdown("Posez votre question sur la recette...")

# =========================
# Initialisation avec logs
# =========================
if "rag_ready" not in st.session_state:
    with st.spinner("Initialisation de l’assistant..."):
        with st.status("🔄 Chargement du document PDF...", expanded=True) as status:
            st.write("Chargement du PDF...")
            rag_chain, retriever = build_rag_chain()
            st.session_state["rag_chain"] = rag_chain
            st.session_state["retriever"] = retriever
            st.session_state["chat_history"] = []
            st.session_state["messages"] = []
            status.update(label="✅ Document chargé et assistant prêt.", state="complete", expanded=False)
        st.session_state["rag_ready"] = True

# Affichage de l'historique du chat
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrée utilisateur
question = st.chat_input("Que voulez-vous savoir ?")

# Gestion de la question
if question:
    st.chat_message("user").markdown(question)
    st.session_state["messages"].append({"role": "user", "content": question})

    with st.spinner("Réflexion en cours..."):
        response = requests.post("http://localhost:8000/ask", json={"question": question})
        answer = response.json()["response"]

    st.chat_message("assistant").markdown(answer)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
