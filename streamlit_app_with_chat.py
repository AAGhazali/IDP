# streamlit_app_with_chat.py

import os
import json
from io import BytesIO
from dotenv import load_dotenv

import streamlit as st
import openai
import numpy as np
import faiss
from PyPDF2 import PdfReader
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.storage.blob import BlobServiceClient

class DocumentRAGApp:
    def __init__(self):
        load_dotenv()

        # Azure Document Intelligence
        di_endpoint = os.getenv("AZURE_DI_ENDPOINT")
        di_key      = os.getenv("AZURE_DI_KEY")
        if not di_endpoint or not di_key:
            st.error("⚠️ Définissez AZURE_DI_ENDPOINT et AZURE_DI_KEY dans votre .env")
            st.stop()
        self.di_client = DocumentIntelligenceClient(
            di_endpoint, AzureKeyCredential(di_key)
        )

        # Azure Storage (optionnel, pour upload du JSON)
        self.conn_str = os.getenv("AZURE_STORAGE_CONN_STR")

        # Azure OpenAI / RAG
        ao_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT_OPENAI")
        ao_key      = os.getenv("AZURE_OPENAI_KEY")
        self.embed_model = os.getenv("AZURE_OPENAI_EMBED_MODEL", "text-embedding-3-large")
        self.chat_model  = os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o-mini")
        if not ao_endpoint or not ao_key:
            st.error("⚠️ Définissez AZURE_OPENAI_ENDPOINT et AZURE_OPENAI_KEY dans votre .env")
            st.stop()
        openai.api_type    = "azure"
        openai.api_base    = ao_endpoint
        openai.api_key     = ao_key
        openai.api_version = "2024-12-01-preview"

    def analyze_document(self, file_bytes, model_id):
        """Envoie le binaire PDF à Azure IDP et retourne un dict JSON."""
        poller = self.di_client.begin_analyze_document(
            model_id,
            body=file_bytes,
            content_type="application/pdf"
        )
        result = poller.result()
        return result.as_dict()

    def save_result(self, result_json, filename, destination, container):
        data = json.dumps(result_json, indent=2, ensure_ascii=False)
        if destination == "local":
            os.makedirs("result", exist_ok=True)
            path = os.path.join("result", filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
            st.info(f"✅ JSON sauvegardé localement : `{path}`")
        else:
            if not self.conn_str:
                st.error("❌ AZURE_STORAGE_CONN_STR non défini.")
                return
            blob_svc = BlobServiceClient.from_connection_string(self.conn_str)
            if container not in [c.name for c in blob_svc.list_containers()]:
                blob_svc.create_container(container)
            blob = blob_svc.get_blob_client(container, filename)
            blob.upload_blob(data, overwrite=True)
            st.info(f"✅ JSON uploadé dans Azure Blob : {blob.url}")

    def build_rag_index(self, full_text):
        CHUNK_SIZE = 1000
        chunks = [
            full_text[i : i + CHUNK_SIZE]
            for i in range(0, len(full_text), CHUNK_SIZE)
            if full_text[i : i + CHUNK_SIZE].strip()
        ]
        progress = st.progress(0)
        embeddings = []
        for idx, chunk in enumerate(chunks, start=1):
            resp = openai.embeddings.create(model=self.embed_model, input=chunk)
            embeddings.append(resp.data[0].embedding)
            progress.progress(idx / len(chunks))
        progress.empty()

        emb_dim = len(embeddings[0])
        index   = faiss.IndexFlatL2(emb_dim)
        index.add(np.array(embeddings, dtype="float32"))

        st.session_state.chunks = chunks
        st.session_state.index  = index

    def chat_rag(self, user_question):
        q_resp = openai.embeddings.create(model=self.embed_model, input=user_question)
        q_emb  = q_resp.data[0].embedding
        D, I   = st.session_state.index.search(np.array([q_emb], dtype="float32"), k=5)
        context = "\n\n---\n\n".join(st.session_state.chunks[i] for i in I[0])

        prompt = (
            f"Extraits du document :\n\n{context}\n\n"
            f"Question : {user_question}\nRéponse :"
        )
        st.session_state.messages.append({"role": "user", "content": prompt})

        chat_resp = openai.chat.completions.create(
            model=self.chat_model,
            messages=st.session_state.messages
        )
        answer = chat_resp.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": answer})

    def run(self):
        st.set_page_config(page_title="IDP + Chat RAG", layout="wide")
        st.title("IDP App – Analyse Documentaire & Chat RAG")

        # Initialisation du toggle chat
        if "show_chat" not in st.session_state:
            st.session_state.show_chat = False

        # 1️⃣ Téléversement du document
        uploaded = st.file_uploader("1. Téléversez votre document", type=["pdf"])

        if uploaded:
            # 2️⃣ Choix du modèle pré-entraîné
            models = [
                "prebuilt-read", "prebuilt-layout", "prebuilt-receipt",
                "prebuilt-invoice", "prebuilt-idDocument",
                "prebuilt-businessCard", "prebuilt-document"
            ]
            model_id = st.selectbox(
                "2. Choisissez le modèle Azure Document Intelligence",
                models,
                index=0
            )

            # 3️⃣ Destination du JSON
            dest = st.selectbox("3. Destination du JSON", ["local", "blob"])
            container = None
            if dest == "blob":
                container = st.text_input("Nom du container Azure Blob", value="result")

            # 4️⃣ Lancer l’analyse
            if st.button("4. Lancer l’analyse"):
                file_bytes = uploaded.read()
                with st.spinner("Analyse en cours…"):
                    result_json = self.analyze_document(file_bytes, model_id)
                st.success("✅ Analyse terminée !")
                st.json(result_json)

                filename = f"{uploaded.name}.json"
                self.save_result(result_json, filename, dest, container)

                # Préparation du RAG index
                reader    = PdfReader(BytesIO(file_bytes))
                full_text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
                self.build_rag_index(full_text)

                # Initialise l’historique du chat
                st.session_state.messages = [
                    {
                        "role": "system",
                        "content": (
                            "Vous êtes un assistant qui répond aux questions "
                            "en vous basant sur le document fourni."
                        )
                    }
                ]

        # 5️⃣ Toggle chat if index ready
        if "index" in st.session_state:
            st.sidebar.markdown("### Actions")
            st.session_state.show_chat = st.sidebar.checkbox(
                "💬 Discuter avec ce document",
                value=st.session_state.show_chat
            )

        # 6️⃣ Chat RAG
        if st.session_state.get("show_chat", False) and "index" in st.session_state:
            st.markdown("---")
            st.subheader("Chat RAG")
            user_q = st.text_input("Posez votre question…", key="chat_input")
            if st.button("Envoyer", key="chat_send"):
                self.chat_rag(user_q)
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.chat_message("user").write(msg["content"])
                else:
                    st.chat_message("assistant").write(msg["content"])

if __name__ == "__main__":
    DocumentRAGApp().run()
