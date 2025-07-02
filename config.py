import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

endpoint = os.getenv("AZURE_DI_ENDPOINT")
key      = os.getenv("AZURE_DI_KEY")
credential = AzureKeyCredential(key)

client = DocumentIntelligenceClient(endpoint, credential)
