# streamlit_app.py

import os
import json
from dotenv import load_dotenv

import streamlit as st
import openai
import numpy as np
import faiss
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import (
    DocumentIntelligenceClient,
    DocumentIntelligenceAdministrationClient,
)

class ChatRAGApp:
    def __init__(self):
        load_dotenv()

        # ─── Azure Document Intelligence ────────────────────────────────
        di_ep  = os.getenv("AZURE_DI_ENDPOINT")
        di_key = os.getenv("AZURE_DI_KEY")
        if not di_ep or not di_key:
            st.error("Définissez AZURE_DI_ENDPOINT et AZURE_DI_KEY dans `.env`.")
            st.stop()
        self.di_client = DocumentIntelligenceClient(di_ep, AzureKeyCredential(di_key))
        admin_client   = DocumentIntelligenceAdministrationClient(
            di_ep, AzureKeyCredential(di_key)
        )
        # liste des modèles disponibles
        self.models    = [m.model_id for m in admin_client.list_models()]

        # ─── Azure OpenAI pour RAG ───────────────────────────────────────
        ao_ep  = os.getenv("AZURE_OPENAI_ENDPOINT")
        ao_key = os.getenv("AZURE_OPENAI_KEY")
        if not ao_ep or not ao_key:
            st.error("Définissez AZURE_OPENAI_ENDPOINT et AZURE_OPENAI_KEY dans `.env`.")
            st.stop()
        openai.api_type    = "azure"
        openai.api_base    = ao_ep
        openai.api_key     = ao_key
        openai.api_version = "2024-12-01-preview"

        # modèles d'Embedding et de Chat
        self.embed_model = os.getenv("AZURE_OPENAI_EMBED_MODEL", "text-embedding-3-large")
        self.chat_model  = os.getenv("AZURE_OPENAI_CHAT_MODEL",  "gpt-4o-mini")

    def analyze_document(self, body: bytes, model_id: str, content_type: str):
        """Envoie à Azure DI et retourne un dict."""
        poller = self.di_client.begin_analyze_document(
            model_id, body=body, content_type=content_type
        )
        return poller.result().as_dict()

    def prepare_rag(self, text: str):
        """Coupe le texte en chunks, calcule les embeddings, construit un index FAISS."""
        chunks = [
            text[i : i + 1000]
            for i in range(0, len(text), 1000)
            if text[i : i + 1000].strip()
        ]
        embs = []
        for c in chunks:
            r = openai.embeddings.create(model=self.embed_model, input=c)
            embs.append(r.data[0].embedding)
        idx = faiss.IndexFlatL2(len(embs[0]))
        idx.add(np.array(embs, dtype="float32"))

        # on stocke dans la session
        st.session_state.index   = idx
        st.session_state.chunks  = chunks
        st.session_state.history = []

    def chat(self, question: str):
        """Pose la question, récupère l'embedding, recherche le contexte, appelle le chat."""
        # embedding de la question
        r = openai.embeddings.create(model=self.embed_model, input=question)
        q_emb = r.data[0].embedding

        # on recherche les 5 passages les plus proches
        _, I = st.session_state.index.search(
            np.array([q_emb], dtype="float32"), k=5
        )
        context = "\n---\n".join(st.session_state.chunks[i] for i in I[0])

        prompt = f"Contexte :\n{context}\n\nQuestion : {question}\nRéponse :"
        resp = openai.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system",    "content": "Vous êtes un assistant fidèle au document."},
                {"role": "user",      "content": prompt}
            ],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content

        # on historise
        st.session_state.history.append(("user", question))
        st.session_state.history.append(("assistant", answer))

    def run(self):
        # ─── Doit être la **première** commande Streamlit ────────────────
        st.set_page_config(layout="wide", page_title="IDP + Chat RAG")

        st.title("IDP App – Analyse & Chat RAG")
        col1, col2 = st.columns([1, 2])

        # ─── COLONNE GAUCHE : upload + analyse ─────────────────────────
        with col1:
            uploaded = st.file_uploader(
                "Téléversez PDF/PNG/JPG/DOCX",
                type=["pdf","png","jpg","jpeg","docx"],
            )
            if uploaded:
                data = uploaded.read()
                ext = uploaded.name.lower().split(".")[-1]
                ctype = "application/pdf" if ext=="pdf" else f"image/{ext}"

                model_id = st.selectbox("Modèle IDP", self.models)
                dest     = st.selectbox("Destination JSON", ["local","azure-blob"])
                if st.button("Lancer l'analyse"):
                    jr = self.analyze_document(data, model_id, ctype)
                    st.session_state.jr = jr
                    st.success("✅ Analyse terminée")

                    # extraction brute du texte
                    text = jr.get("content","") or ""
                    if not text:
                        docs = jr.get("documents",[])
                        if docs:
                            parts = [f.get("content","") for f in docs[0].get("fields",{}).values()]
                            text = "\n".join(parts)
                    if not text:
                        pages = jr.get("pages",[])
                        lines = [l.get("content","") for p in pages for l in p.get("lines",[])]
                        text = "\n".join(lines)

                    if text.strip():
                        self.prepare_rag(text)
                    else:
                        st.warning("⚠️ Pas de texte exploitable pour le chat.")

                    # sauvegarde du JSON
                    os.makedirs("result", exist_ok=True)
                    fn = f"result/{uploaded.name}.json"
                    with open(fn,"w",encoding="utf8") as f:
                        json.dump(jr, f, ensure_ascii=False, indent=2)
                    if dest=="local":
                        st.info(f"📂 JSON sauvegardé localement : `{fn}`")
                    else:
                        st.info("☁️ (upload vers blob non implémenté)")

        # ─── COLONNE DROITE : affichage + chat RAG ───────────────────────
        with col2:
            # 1) JSON brut + résultats structurés
            if "jr" in st.session_state:
                jr = st.session_state.jr
                with st.expander("Voir JSON brut"):
                    st.json(jr)
                with st.expander("Résultats structurés", expanded=True):
                    docs = jr.get("documents", [])
                    if not docs:
                        st.info("Aucun champ structuré détecté.")
                    else:
                        for k,v in docs[0].get("fields",{}).items():
                            val = v.get("valueString","")
                            cf  = v.get("confidence",0)*100
                            st.markdown(f"- **{k}** : `{val}` _(conf. {cf:.1f}% )_")

            # 2) Chat RAG si l'index existe
            if st.session_state.get("index"):
                st.subheader("💬 Chat RAG")

                # on affiche TOUT l'historique
                for role, msg in st.session_state.history:
                    with st.chat_message(role):
                        st.markdown(msg)

                # un UNIQUE champ de chat : on envoie avec Entrée ou bouton
                user_q = st.chat_input("Posez votre question…")
                if user_q:
                    self.chat(user_q)
                    # **PAS** de experimental_rerun, le chat_input déclenche lui-même le rerun
        

if __name__ == "__main__":
    ChatRAGApp().run()
