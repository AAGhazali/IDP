# streamlit_rag_chatbot.py

import os
from io import BytesIO
from dotenv import load_dotenv

import streamlit as st

import openai
import numpy as np
import faiss
from PyPDF2 import PdfReader

# ---------------------------------------
# 1️⃣ Chargement des variables d’environnement
# ---------------------------------------
load_dotenv()

# Renommage éventuel pour compatibilité
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT_OPENAI")
api_key  = os.getenv("AZURE_OPENAI_KEY")
embed_model = os.getenv("AZURE_OPENAI_EMBED_MODEL", "text-embedding-3-large")
chat_model  = os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o-mini")

if not endpoint or not api_key:
    st.error(
        "⚠️ Veuillez définir dans votre `.env` :\n"
        "- AZURE_OPENAI_ENDPOINT (ou AZURE_OPENAI_ENDPOINT_OPENAI)\n"
        "- AZURE_OPENAI_KEY\n"
        "- AZURE_OPENAI_EMBED_MODEL\n"
        "- AZURE_OPENAI_CHAT_MODEL"
    )
    st.stop()

# ---------------------------------------
# 2️⃣ Configuration du SDK OpenAI pour Azure
# ---------------------------------------
openai.api_type    = "azure"
openai.api_base    = endpoint
openai.api_key     = api_key
openai.api_version = "2024-12-01-preview"

# ---------------------------------------
# 3️⃣ Initialisation de l’historique de la conversation
# ---------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "Vous êtes un analyste de données capable d'analyser les documents et qui répond aux questions "
                "en vous basant sur un document fourni."
            )
        }
    ]

# ---------------------------------------
# 4️⃣ Configuration de la page Streamlit
# ---------------------------------------
st.set_page_config(page_title="Chatbot RAG Azure OpenAI", layout="wide")
st.title("📄🔎 Chatbot RAG avec Azure OpenAI")

# ---------------------------------------
# 5️⃣ Téléversement du document PDF
# ---------------------------------------
uploaded_file = st.file_uploader(
    "1. Téléversez un document PDF",
    type=["pdf"],
    help="Le PDF sera découpé, embeddi et indexé pour le chat RAG."
)

if uploaded_file:
    # Lecture et extraction du texte
    pdf_bytes = uploaded_file.read()
    reader    = PdfReader(BytesIO(pdf_bytes))
    pages     = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n\n".join(pages).strip()
    word_count = len(full_text.split())
    st.success(f"Document chargé — {word_count} mots extraits.")

    # Découpage en chunks (~1000 caractères)
    CHUNK_SIZE = 1000
    chunks = [
        full_text[i : i + CHUNK_SIZE]
        for i in range(0, len(full_text), CHUNK_SIZE)
        if full_text[i : i + CHUNK_SIZE].strip()
    ]
    st.info(f"{len(chunks)} chunks créés pour l’index RAG.")

    # ---------------------------------------
    # 6️⃣ Calcul des embeddings & index FAISS
    # ---------------------------------------
    progress = st.progress(0)
    embeddings = []
    for idx, chunk in enumerate(chunks, start=1):
        resp = openai.embeddings.create(
            model=embed_model,
            input=chunk
        )
        embeddings.append(resp.data[0].embedding)
        progress.progress(idx / len(chunks))
    progress.empty()

    emb_dim = len(embeddings[0])
    index   = faiss.IndexFlatL2(emb_dim)
    index.add(np.array(embeddings, dtype="float32"))

    # ---------------------------------------
    # 7️⃣ Interface de chat
    # ---------------------------------------
    st.subheader("2. Posez votre question")
    question = st.text_input("Votre question ici…")

    if st.button("Envoyer"):
        # Embedding de la question
        q_resp = openai.embeddings.create(
            model=embed_model,
            input=question
        )
        q_emb = q_resp.data[0].embedding

        # Recherche des 5 chunks les plus similaires
        D, I = index.search(np.array([q_emb], dtype="float32"), k=5)
        context = "\n\n---\n\n".join(chunks[i] for i in I[0])

        # Construction du prompt RAG
        rag_content = (
            f"Voici des passages extraits du document :\n\n{context}\n\n"
            f"Question : {question}\n\nRéponse :"
        )
        st.session_state.messages.append({"role": "user", "content": rag_content})

        # — Correction ici : utilisation de `model=` au lieu de `deployment_id=` —
        chat_resp = openai.chat.completions.create(
            model=chat_model,
            messages=st.session_state.messages
        )
        answer = chat_resp.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # ---------------------------------------
    # 8️⃣ Affichage de la conversation
    # ---------------------------------------
    st.subheader("💬 Conversation")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])
