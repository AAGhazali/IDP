# test_fr.py
from dotenv import load_dotenv
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

load_dotenv()
endpoint = os.getenv("AZURE_DI_ENDPOINT")
key      = os.getenv("AZURE_DI_KEY")
client   = DocumentAnalysisClient(endpoint, AzureKeyCredential(key))

# Exemple d’analyse “prébuilt-receipt”
poller = client.begin_analyze_document("prebuilt-receipt",
                                       document_url="https://raw.githubusercontent.com/Azure/azure-sdk-for-python/main/sdk/formrecognizer/azure-ai-formrecognizer/tests/sample_forms/forms/receipt_ocr.pdf")
result = poller.result()
print("Pages extraites :", len(result.pages))
print("Total détecté :", result.documents[0].fields.get("Total").value)
