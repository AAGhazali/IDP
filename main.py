# main.py
import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient

from uploader import upload_documents
from analyzer import analyze_prebuilt_local
from trainer import train_custom_model, list_models, delete_model


def save_result_local(data: dict, output_dir: str = 'result') -> str:
    """
    Sauvegarde le résultat JSON dans un fichier local sous le répertoire output_dir.
    Retourne le chemin du fichier créé.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    filename = f'result_{timestamp}.json'
    path = os.path.join(output_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def save_result_blob(data: dict, container_name: str, blob_name: str) -> str:
    """
    Sauvegarde le résultat JSON dans un blob Azure.
    Retourne l'URL du blob.
    """
    conn_str = os.getenv('AZURE_STORAGE_CONN_STR')
    if not conn_str:
        raise EnvironmentError('AZURE_STORAGE_CONN_STR non défini dans .env')
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    container = blob_service.get_container_client(container_name)
    # Crée le conteneur si nécessaire
    try:
        container.create_container()
    except Exception:
        pass
    blob_client = container.get_blob_client(blob_name)
    blob_client.upload_blob(json.dumps(data, ensure_ascii=False), overwrite=True)
    return blob_client.url


def main():
    load_dotenv()
    endpoint = os.getenv('AZURE_DI_ENDPOINT')
    key = os.getenv('AZURE_DI_KEY')
    if not endpoint or not key:
        raise EnvironmentError('Veuillez définir AZURE_DI_ENDPOINT et AZURE_DI_KEY dans votre .env')

    parser = argparse.ArgumentParser(description='Application IDP')
    sub = parser.add_subparsers(dest='cmd')

    # upload
    upload_parser = sub.add_parser('upload', help='Téléverse un ou plusieurs fichiers sur Blob Storage')
    upload_parser.add_argument('--container', required=True, help='Nom du conteneur Blob Storage')
    upload_parser.add_argument('files', nargs='+', help='Chemins locaux des fichiers à téléverser')

    # analyze
    analyze_parser = sub.add_parser('analyze', help='Analyse un ou plusieurs fichiers et sauvegarde le résultat')
    analyze_parser.add_argument('--model', required=True, help='ID du modèle prébuilt')
    analyze_parser.add_argument('--save-to', choices=['local','blob'], default='local', help="Destination du résultat")
    analyze_parser.add_argument('--container', help='Nom du container pour blob (si save-to=blob)')
    analyze_parser.add_argument('files', nargs='+', help='Chemins locaux des fichiers à analyser')

    # train
    train_parser = sub.add_parser('train', help='Entraîne un modèle personnalisé sur un conteneur Blob SAS')
    train_parser.add_argument('--sas-url', required=True, help="URL SAS du conteneur d'entraînement")
    train_parser.add_argument('--name', required=True, help='Nom descriptif du modèle personnalisé')

    # list-models
    sub.add_parser('list-models', help='Liste les modèles personnalisés existants')
    # delete-model
    delete_parser = sub.add_parser('delete-model', help='Supprime un modèle personnalisé')
    delete_parser.add_argument('model_id', help='ID du modèle à supprimer')

    args = parser.parse_args()

    if args.cmd == 'analyze':
        for file_path in args.files:
            print(f"Analyse de {file_path}...")
            result = analyze_prebuilt_local(args.model, file_path)
            if args.save_to == 'local':
                path = save_result_local(result)
                print(f"Résultat sauvegardé localement : {path}")
            else:
                if not args.container:
                    raise ValueError('Vous devez préciser --container quand save-to=blob')
                timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
                blob_name = f'result_{timestamp}.json'
                url = save_result_blob(result, args.container, blob_name)
                print(f"Résultat uploadé sur blob : {url}")

    elif args.cmd == 'upload':
        urls = upload_documents(args.container, args.files)
        print(json.dumps(urls, indent=2, ensure_ascii=False))

    elif args.cmd == 'train':
        model_id = train_custom_model(args.sas_url, args.name)
        print(f'Nouveau modèle créé : {model_id}')

    elif args.cmd == 'list-models':
        models = list_models()
        print(json.dumps(models, indent=2, ensure_ascii=False))

    elif args.cmd == 'delete-model':
        delete_model(args.model_id)
        print(f'Modèle supprimé : {args.model_id}')

    else:
        parser.print_help()

if __name__ == '__main__':
    main()