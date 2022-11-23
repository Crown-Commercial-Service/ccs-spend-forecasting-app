from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

secret = dbutils.secrets.get(scope = "AzP-UKS-Spend-Forecasting-Development-scope", key = "application_password")
client_id = dbutils.secrets.get(scope = "AzP-UKS-Spend-Forecasting-Development-scope", key = "client_id")
tenant_id  = dbutils.secrets.get(scope = "AzP-UKS-Spend-Forecasting-Development-scope", key = "tenant_id")

storage_account = 'developmentstorageccs'
container_name = 'azp-uks-spend-forecasting-development-transformed'

account_url = "https://developmentstorageccs.blob.core.windows.net"
default_credential = DefaultAzureCredential()
principal_credential = ClientSecretCredential(tenant_id=tenant_id, client_id=client_id, client_secret=secret)

# Create the BlobServiceClient object
blob_service_client = BlobServiceClient(account_url, credential=principal_credential)
container_client = blob_service_client.get_container_client('azp-uks-spend-forecasting-development-transformed')