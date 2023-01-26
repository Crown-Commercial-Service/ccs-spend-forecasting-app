import os
import warnings
from typing import Optional
from configparser import ConfigParser
from datetime import date
from io import BytesIO

from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.storage.blob import BlobServiceClient
from pyspark.sql import SparkSession, DataFrame
import pandas as pd


def is_running_in_databricks() -> bool:
    """Detect whether the current enviroment is in databricks

    Returns:
        bool: A boolean value. True means that the current enviroment is in databricks
    """
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def load_config() -> ConfigParser:
    """Loads the default config file from local file"config.ini"

    Returns:
        ConfigParser: An object that represent the config stored.
    """
    config = ConfigParser()
    config.read("config.ini")
    return config


def get_default_blob_account_name() -> str:
    config = load_config()
    account_name = config.get("blob", "account_name", fallback="developmentstorageccs")

    return account_name


def get_default_blob_container_name() -> str:
    config = load_config()
    container_name = config.get(
        "blob",
        "container_name",
        fallback="azp-uks-spend-forecasting-development-transformed",
    )

    return container_name


def get_spark_session() -> SparkSession:
    spark_found = SparkSession.getActiveSession()
    if spark_found:
        return spark_found
    else:
        return create_spark_session_for_local()


def get_dbutils():
    """Retreive the dbutils object. Provide access to dbutils.secret which is otherwise only available in notebook.

    Returns:
        The pre-defined dbutils object of Databricks
    """
    if not is_running_in_databricks():
        raise RuntimeError(
            "not running in databricks. dbutils is not available in local enviroment"
        )

    from pyspark.dbutils import DBUtils  # pylint: disable=no-name-in-module

    spark = get_spark_session()
    dbutils = DBUtils(spark)
    return dbutils


def load_secret(key: str, scope: Optional[str] = None) -> str:
    """Load a secret from databricks secret

    Args:
        key (str): Key of the said secret
        scope (str, optional): A string that represent the secret scope. If not given, will try to use a predefined scope.

    Returns:
        str: A string value that represent the secret
    """
    if not scope:
        scope = "AzP-UKS-Spend-Forecasting-Development-scope"

    dbutils = get_dbutils()

    return dbutils.secrets.get(scope=scope, key=key)


def get_azure_credential(scope: Optional[str] = None):
    """Return a azure credentials object. Used for accessing blob storage with Azure blob client.
        If run in Databricks, it will try to retrieve the necessary information from Databricks secrets.
        Otherwise, it will try to use the DefaultAzureCredential() method to authenticate.

    Args:
        scope (Optional[str], optional): A string that represent the secret scope. If not given, will try to use a predefined scope.

    Returns:
        An Azure credential object to authenticate blob storage access.
    """
    if is_running_in_databricks():
        secret = load_secret(key="application_password", scope=scope)
        client_id = load_secret(key="client_id", scope=scope)
        tenant_id = load_secret(key="tenant_id", scope=scope)
        return ClientSecretCredential(
            tenant_id=tenant_id, client_id=client_id, client_secret=secret
        )
    else:
        return DefaultAzureCredential()


def get_blob_service_client(storage_account_name: Optional[str] = None):
    """Return a blob storage client object for given storage account name

    Args:
        storage_account_name (str): Storage account name. If not given, will try to use a predefined account name.

    Returns:
        A blob storage client object
    """
    if not storage_account_name:
        storage_account_name = get_default_blob_account_name()
    account_url = f"https://{storage_account_name}.blob.core.windows.net"
    return BlobServiceClient(account_url, get_azure_credential())


def get_blob_container_client(
    container_name: Optional[str] = None, storage_account_name: Optional[str] = None
):
    """Return a blob container client object for given storage account name

    Args:
        container_name (str): Storage container name. If not given, will try to use a predefined container name.
        storage_account_name (str): Storage account name. If not given, will try to use a predefined account name.

    Returns:
        A blob container client object
    """
    if not container_name:
        container_name = get_default_blob_container_name()

    service_client = get_blob_service_client(storage_account_name)
    return service_client.get_container_client(container_name)


def create_spark_session_for_local() -> SparkSession:
    """Create a spark session for local usage. Will try to retrieve jar packages of Azure Blob / MS SQL Server driver

    Returns:
        SparkSession: A spark session with ability to connect to Azure Blob Storage or MS SQL Server
    """
    jar_packages = [
        "org.apache.hadoop:hadoop-azure:3.3.4",
        "com.microsoft.sqlserver:mssql-jdbc:11.2.1.jre8",
        "com.microsoft.azure:spark-mssql-connector_2.12:1.2.0",
    ]

    return (
        SparkSession.builder.config(
            "spark.jars.packages",
            ",".join(jar_packages),
        )
        .config("spark.sql.debug.maxToStringFields", "10000")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
        .config("spark.sql.execution.arrow.pyspark.selfDestruct.enabled", "true")
        .config("spark.driver.maxResultSize", "2048M")
        .config("spark.sql.shuffle.partitions", "7")
        .config("spark.driver.memory", "12g")
        .master("local[*]")
        .appName("TestSparkSession")
        .getOrCreate()
    )


def connect_spark_to_blob_storage(storage_account_name: Optional[str] = None):
    """Connect the current spark session to Azure blob storage.
    If called in Databricks, it will try to get the necessary authentication from Databricks secrets.
    If called in a local environment, it will try to get necessary authentication from the local file 'config.ini'

    Args:
        storage_account_name (str): The blob storage account name to connect to.

    """
    if not storage_account_name:
        storage_account_name = get_default_blob_account_name()

    if is_running_in_databricks():
        return connect_databricks_spark_to_blob_storage(storage_account_name)
    else:
        return connect_local_spark_to_blob_storage(storage_account_name)


def connect_databricks_spark_to_blob_storage(storage_account_name: str):
    spark = get_spark_session()

    if not storage_account_name:
        storage_account_name = get_default_blob_account_name()

    secret = load_secret(key="application_password")
    client_id = load_secret(key="client_id")
    tenant_id = load_secret(key="tenant_id")

    spark.conf.set(
        f"fs.azure.account.auth.type.{storage_account_name}.dfs.core.windows.net",
        "OAuth",
    )
    spark.conf.set(
        f"fs.azure.account.oauth.provider.type.{storage_account_name}.dfs.core.windows.net",
        "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
    )
    spark.conf.set(
        f"fs.azure.account.oauth2.client.id.{storage_account_name}.dfs.core.windows.net",
        client_id,
    )
    spark.conf.set(
        f"fs.azure.account.oauth2.client.secret.{storage_account_name}.dfs.core.windows.net",
        secret,
    )
    spark.conf.set(
        f"fs.azure.account.oauth2.client.endpoint.{storage_account_name}.dfs.core.windows.net",
        f"https://login.microsoftonline.com/{tenant_id}/oauth2/token",
    )

    return spark


def connect_local_spark_to_blob_storage(storage_account_name):
    spark = get_spark_session()
    config = load_config()

    if not storage_account_name:
        storage_account_name = config.get("blob", "account_name")
    blob_access_key = config.get("blob", "access_key")

    spark.conf.set(
        f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net",
        blob_access_key,
    )

    return spark


def get_latest_blob_path_for_table(
    table_name: str, container_name: Optional[str] = None
) -> str:
    """Retrieve the Azure blob uri path for the latest version of given table name.
    In a Databricks enviroment, it will try to use the abfss:// protocol.
    Otherwise, it will try to use the wasbs:// protocol.

    Args:
        table_name (str): The table name of the file to retrieve. It is assumed that the given blob storage should contain at least one copy of parquet file with this name.
        container_name (str): The blob container name to retrieve file from.

    Returns:
        A string that represent the uri of a blob file. Can be read by a pyspark session.
    """
    if not container_name:
        container_name = get_default_blob_container_name()
    container_client = get_blob_container_client(container_name=container_name)

    candidates = [
        blob
        for blob in container_client.list_blobs()
        if blob.name and blob.name.endswith(f"{table_name}.parquet")
    ]
    if not candidates:
        raise ValueError(f"Could not find a blob file of the table {table_name}")

    candidates.sort(key=lambda blob: blob.creation_time, reverse=True)

    blob_file_name = candidates[0].name
    container_name = container_client.container_name
    account_name = container_client.account_name

    if is_running_in_databricks():
        return f"abfss://{container_name}@{account_name}.dfs.core.windows.net/{blob_file_name}"
    else:
        return f"wasbs://{container_name}@{account_name}.blob.core.windows.net/{blob_file_name}"


def load_latest_blob_to_pyspark(
    table_name: str, blob_container_name: Optional[str] = None
) -> DataFrame:
    """Retrieve the latest version of given table name from blob storage, and load it as a spark Dataframe

    Args:
        table_name (str): The table name of the file to retrieve e.g. "Customers" It is assumed that the given blob storage should contain at least one copy of parquet file with this name.
        blob_container_name (str): The blob container name to retrieve file from.

    Returns:
        DataFrame: A spark Dataframe of the given table name
    """
    spark = get_spark_session()
    if not blob_container_name:
        blob_container_name = get_default_blob_container_name()

    parquet_path = get_latest_blob_path_for_table(
        table_name, container_name=blob_container_name
    )
    if not parquet_path:
        raise ValueError(f"Could not find a blob file of the table {table_name}")
    return spark.read.parquet(parquet_path)


def get_blob_storage_root_path(
    blob_container_name: Optional[str] = None,
    blob_account_name: Optional[str] = None,
):
    """Return a blob storage root path with given details.

    Args:
        blob_container_name (Optional[str], optional): _description_. Defaults to None.
        blob_account_name (Optional[str], optional): _description_. Defaults to None.

    Returns:
        str: A string that represent the root uri of a blob storage container
    """
    if not blob_account_name:
        blob_account_name = get_default_blob_account_name()

    if not blob_container_name:
        blob_container_name = get_default_blob_container_name()

    if is_running_in_databricks():
        return f"abfss://{blob_container_name}@{blob_account_name}.dfs.core.windows.net"
    else:
        return (
            f"wasbs://{blob_container_name}@{blob_account_name}.blob.core.windows.net"
        )


def make_blob_storage_path(
    table_name: str,
    blob_container_name: Optional[str] = None,
    blob_account_name: Optional[str] = None,
) -> str:
    """Make a blob storage path dated today with given details.
    For example, if today is "2022-11-28" and table_name is "Customers", it will generate a filename like this:
    "Customers/year=2022/month=11/day=28/Customers.parquet"

    Args:
        table_name (str): The name of table.
        blob_container_name (Optional[str], optional): _description_. Defaults to None.
        blob_account_name (Optional[str], optional): _description_. Defaults to None.

    Returns:
        str: A string that represent the uri of a blob file
    """
    today = date.today()

    blob_storage_root_path = get_blob_storage_root_path(
        blob_account_name=blob_account_name, blob_container_name=blob_container_name
    )
    return f"{blob_storage_root_path}/{table_name}/year={today.year}/month={today.month}/day={today.day}/{table_name}.parquet"


def save_dataframe_to_blob(
    df: DataFrame,
    table_name: str,
    blob_storage_path: Optional[str] = None,
    overwrite: bool = True,
):
    """Save a pyspark dataframe to blob storage in parquet format"""
    if not blob_storage_path:
        blob_storage_path = make_blob_storage_path(table_name=table_name)

    if overwrite:
        df.write.parquet(blob_storage_path, mode="overwrite")
    else:
        df.write.parquet(blob_storage_path)


def save_pandas_dataframe_to_blob(
    pandas_df: pd.DataFrame,
    filename: str,
):
    """Save a pandas dataframe to blob storage in csv format"""

    if is_running_in_databricks():
        blob_storage_root_path = get_blob_storage_root_path()
        full_path = f"{blob_storage_root_path}/{filename}"

        dbutils = get_dbutils()
        dbutils.fs.mkdirs(f"dbfs:/FileStore/{os.path.dirname(filename)}")

        pandas_df.to_csv(f"/dbfs/FileStore/{filename}", index=False)

        dbutils.fs.mv(f"dbfs:/FileStore/{filename}", full_path)

    else:
        container_client = get_blob_container_client()
        file_buffer = BytesIO()
        pandas_df.to_csv(file_buffer, index=False)
        file_buffer.seek(0)

        container_client.upload_blob(name=filename, data=file_buffer, overwrite=True)


def load_csv_from_blob_to_pandas(
    table_name: str,
    blob_container_name: Optional[str] = None,
    storage_account_name: Optional[str] = None,
) -> pd.DataFrame:
    """Load a csv file from blob storage to a pandas dataframe.

    Args:
        table_name (str): A string that represent the the folder name on blob storage.
                          It is expected that a blob file with the path "{table_name}/.../{any_file_name}.csv" exists on said blob storage container.
                          If multiple files meet this criteria, will load the one that is created most recently.
        blob_container_name (Optional[str], optional): Azure Blob Container name. If omitted, will try to use the default provided in other method in this module.
        blob_account_name (Optional[str], optional): Azure Blob Account name. If omitted, will try to use the default provided in other method in this module.

    Returns:
        pd.DataFrame: A pandas dataframe containing the data of said csv file.
    """

    container_client = get_blob_container_client(
        container_name=blob_container_name, storage_account_name=storage_account_name
    )

    candidates = [
        blob
        for blob in container_client.list_blobs()
        if blob.name
        and blob.name.startswith(f"{table_name}/")
        and blob.name.endswith(".csv")
    ]
    if not candidates:
        raise ValueError(
            f"Could not find a csv file in folder {table_name}/ of blob storage"
        )

    candidates.sort(key=lambda blob: blob.creation_time, reverse=True)

    blob_file_name = str(candidates[0].name)
    blob_client = container_client.get_blob_client(blob=blob_file_name)
    file_buffer = BytesIO()
    blob_client.download_blob().readinto(file_buffer)
    file_buffer.seek(0)

    return pd.read_csv(file_buffer, index_col=None)


def suppress_unwanted_warnings():
    """Supress spam warning messages generated during model fitting"""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
