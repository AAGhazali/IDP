from azure.storage.blob import BlobServiceClient
import os

blob_service = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONN_STR"))

def upload_documents(container_name: str, files: list[str]) -> list[str]:
    container = blob_service.get_container_client(container_name)
    sas_urls = []
    for path in files:
        name = os.path.basename(path)
        blob = container.get_blob_client(name)
        with open(path, "rb") as data:
            blob.upload_blob(data, overwrite=True)
        sas = blob.generate_shared_access_signature(
            permission="r", expiry=datetime.utcnow() + timedelta(hours=1)
        )
        sas_urls.append(f"{blob.url}?{sas}")
    return sas_urls
