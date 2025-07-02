import os
import json
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import HttpResponseError
from azure.ai.documentintelligence import DocumentIntelligenceClient
from io import BytesIO
from PIL import Image

# Chargement des variables d'environnement
load_dotenv()
import os
print("CONN STRING CHARGÉE :", os.getenv("AZURE_STORAGE_CONN_STR"))
endpoint = os.getenv("AZURE_DI_ENDPOINT")
key      = os.getenv("AZURE_DI_KEY")
conn_str = os.getenv("AZURE_STORAGE_CONN_STR")

if not endpoint or not key:
    st.error("Veuillez configurer AZURE_DI_ENDPOINT et AZURE_DI_KEY dans votre .env")
    st.stop()

# Instanciation Azure
credential   = AzureKeyCredential(key)
di_client    = DocumentIntelligenceClient(endpoint, credential)
blob_service = None
if conn_str:
    blob_service = BlobServiceClient.from_connection_string(conn_str)

# UI
st.title("IDP App - Analyse Documentaire")

# 1. Téléversement du document
uploaded_file = st.file_uploader("Téléversez votre document", type=["pdf","jpg","jpeg","png"])

# 2. Sélection du modèle
model_choice = st.selectbox(
    "Sélectionnez le modèle pré-entraîné", [
        "prebuilt-read",
        "prebuilt-layout",
        "prebuilt-receipt",
        "prebuilt-invoice",
        "prebuilt-idDocument",
        "prebuilt-businessCard",
        "prebuilt-document"
    ]
)

# 3. Destination de sauvegarde
dest = st.radio("Destination du résultat", ["Local", "Azure Blob"])
if dest == "Azure Blob":
    container_name = st.text_input("Nom du container Azure", value="result")

# 4. Bouton d'analyse
if st.button("Lancer l'analyse"):
    if not uploaded_file:
        st.warning("Veuillez téléverser un fichier avant de lancer l'analyse.")
    else:
        # Lecture du contenu
        file_bytes = uploaded_file.read()
        size_mb = len(file_bytes) / (1024 * 1024)
        st.write(f"Taille du fichier : {size_mb:.2f} Mo")

        # Compression si nécessaire (images > 4 Mo)
        if size_mb > 4 and uploaded_file.type.startswith("image/"):
            st.info("Compression de l'image en cours...")
            img = Image.open(BytesIO(file_bytes))
            max_width = 2000
            if img.width > max_width:
                ratio = max_width / img.width
                img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=75)
            file_bytes = buf.getvalue()
            st.write(f"Taille après compression : {len(file_bytes)/(1024*1024):.2f} Mo")

        try:
            # Appel Azure Document Intelligence
            poller = di_client.begin_analyze_document(
                model_choice,
                file_bytes,
                content_type=uploaded_file.type
            )
            with st.spinner("Analyse en cours..."):
                result = poller.result()

            # Conversion en dict
            output = result.to_dict() if hasattr(result, 'to_dict') else result.as_dict()

            # 5. Sauvegarde du résultat
            timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            filename = f"result_{timestamp}.json"

            if dest == "Local":
                os.makedirs('result', exist_ok=True)
                local_path = os.path.join('result', filename)
                with open(local_path, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                st.success(f"Résultat sauvegardé localement : {local_path}")

            else:
                if not blob_service or not container_name:
                    st.error("Configuration Azure Blob manquante ou nom de container vide.")
                else:
                    container_client = blob_service.get_container_client(container_name)
                    try:
                        container_client.create_container()
                    except Exception:
                        pass
                    blob_client = container_client.get_blob_client(filename)
                    blob_client.upload_blob(json.dumps(output, ensure_ascii=False), overwrite=True)
                    st.success(f"Résultat sauvegardé sur Blob : {blob_client.url}")

            # Affichage du JSON
            st.header("Résultat JSON")
            st.json(output)

        except HttpResponseError as e:
            st.error(f"Erreur Azure Document Intelligence : {e.message}")
        except Exception as ex:
            st.error(f"Erreur inattendue : {ex}")
