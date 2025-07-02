# test_connection.py

from dotenv import load_dotenv
import os
from azure.core.credentials import AzureKeyCredential

# Importez correctement la classe AnalyzeDocumentRequest
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.ai.documentintelligence import (
    DocumentIntelligenceClient,
    DocumentIntelligenceAdministrationClient
)

# 1️⃣ Charge le .env (AZURE_DI_ENDPOINT et AZURE_DI_KEY)
load_dotenv()

endpoint = os.getenv("AZURE_DI_ENDPOINT")
key      = os.getenv("AZURE_DI_KEY")

if not endpoint or not key:
    raise EnvironmentError(
        "⚠️  Les variables AZURE_DI_ENDPOINT et/ou AZURE_DI_KEY ne sont pas définies."
    )

# 2️⃣ Crée le credential Azure
credential = AzureKeyCredential(key)

# ——— Test d’analyse pré-entraînée ———
di_client = DocumentIntelligenceClient(endpoint, credential)

sample_url = (
    "https://raw.githubusercontent.com/Azure/azure-sdk-for-python/"
    "main/sdk/documentintelligence/azure-ai-documentintelligence/"
    "samples/sample_forms/forms/receipt_ocr.pdf"
)

# Vous pouvez aussi simplement passer l'URL en 2ᵉ argument, 
# mais voici la forme avec AnalyzeDocumentRequest :
poller = di_client.begin_analyze_document(
    "prebuilt-receipt",
    AnalyzeDocumentRequest(url_source=sample_url)
)
result = poller.result()
print(f"✅ Analyse réussie : {len(result.pages)} page(s) extraites.")

# ——— Test d’administration (liste des modèles) ———
admin_client = DocumentIntelligenceAdministrationClient(endpoint, credential)

try:
    custom_models = [m.model_id for m in admin_client.list_models()]
    print("📦 Modèles personnalisés :", custom_models or "— aucun modèle custom —")
except Exception as e:
    # Si vous aviez un 500 Internal Server Error, 
    # ce bloc vous permettra d’en voir la raison exacte
    print("⚠️ Erreur lors de list_models():", type(e).__name__, e)
