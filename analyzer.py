import os
import json
import argparse
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.formrecognizer import DocumentAnalysisClient

# Chargement des variables d'environnement
load_dotenv()
endpoint = os.getenv("AZURE_DI_ENDPOINT")
key      = os.getenv("AZURE_DI_KEY")
if not endpoint or not key:
    raise EnvironmentError("Veuillez définir AZURE_DI_ENDPOINT et AZURE_DI_KEY dans votre .env")

# Clients Azure
credential   = AzureKeyCredential(key)
di_client    = DocumentIntelligenceClient(endpoint, credential)
dan_client   = DocumentAnalysisClient(endpoint, credential)


def _guess_content_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "application/pdf"
    if ext == ".png":
        return "image/png"
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    return "application/octet-stream"


def _result_to_dict(result) -> dict:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if hasattr(result, "as_dict"):
        return result.as_dict()
    raise AttributeError(f"Impossible de convertir le résultat en dict, méthodes manquantes sur {type(result)}")


def analyze_prebuilt_local(model_id: str, file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier spécifié est introuvable : {file_path}")

    content_type = _guess_content_type(file_path)

    with open(file_path, 'rb') as stream:
        try:
            poller = di_client.begin_analyze_document(
                model_id,
                stream,
                content_type=content_type
            )
            result = poller.result()
            return _result_to_dict(result)
        except HttpResponseError as e:
            print(f"⚠️ Nouveau SDK échoué : {e.message}. Tentative fallback...")
            stream.seek(0)
            poller2 = dan_client.begin_analyze_document(
                model_id,
                stream
            )
            result2 = poller2.result()
            return _result_to_dict(result2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyse locale avec Azure Document Intelligence")
    parser.add_argument('--model', required=True, help='ID du modèle prébuilt')
    parser.add_argument('file', help='Chemin vers le fichier à analyser')
    args = parser.parse_args()

    output = analyze_prebuilt_local(args.model, args.file)
    print(json.dumps(output, indent=2, ensure_ascii=False))
