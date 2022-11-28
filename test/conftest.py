import pytest
import io
from datetime import date

from pipeline.utils import (
    get_blob_container_client,
    connect_spark_to_blob_storage,
    get_spark_session,
)

collect_ignore_glob = ["sample_test.py"]


@pytest.fixture(scope="session", autouse=True)
def before_all():
    connect_spark_to_blob_storage()


@pytest.fixture(scope="session")
def blob_container_for_testing():
    container_name = "azp-uks-spend-forecasting-development-test"
    container_client = get_blob_container_client(container_name)

    if not container_client.exists():
        container_client.create_container()

    yield container_client

    for blob in container_client.list_blob_names():
        container_client.delete_blobs(blob)


@pytest.fixture
def blob_file_for_testing(blob_container_for_testing):
    table_name = "table_for_testing_1"
    filename = f"test/{table_name}.parquet"
    container_client = blob_container_for_testing

    spark = get_spark_session()
    df = spark.createDataFrame(
        data=[["abc", 123, True], ["def", 456, False]], schema=["col1", "col2", "col3"]
    )

    buffer = io.BytesIO()
    df.toPandas().to_parquet(buffer)

    buffer.seek(0)

    container_client.upload_blob(filename, buffer, overwrite=True)

    yield filename

    container_client.delete_blob(filename)


@pytest.fixture
def dataframe_for_testing_basic():
    spark = get_spark_session()
    df = spark.createDataFrame(
        data=[
            ["apple", 123.456, date(2022, 4, 1)],
            ["banana", 789.012, date(2022, 10, 31)],
        ],
        schema=["StringCol", "FloatCol", "DateCol"],
    )
    yield df
