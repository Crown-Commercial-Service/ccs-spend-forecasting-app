import io
import pytest
from pyspark.sql import SparkSession, DataFrame

from databricks.utils import get_spark_session, get_blob_container_client, load_latest_blob_to_pyspark, connect_spark_to_blob_storage, create_spark_session_for_local

def test_get_spark_session():
    spark = get_spark_session()
    assert isinstance(spark, SparkSession) == True

def test_blob_container_access():
    container_client = get_blob_container_client()
    assert container_client.exists() == True



@pytest.fixture
def blob_file_for_testing():
    filename = 'test/table_for_testing.parquet'
    container_client = get_blob_container_client()
    connect_spark_to_blob_storage()
    
    spark = get_spark_session()
    df = spark.createDataFrame(data=[["abc", 123, True], ["def", 456, False]], schema=['col1', 'col2', 'col3'])

    buffer = io.BytesIO()
    df.toPandas().to_parquet(buffer)

    buffer.seek(0)
    
    container_client.upload_blob(filename, buffer, overwrite=True)

    yield

    container_client.delete_blob(filename)



def test_load_latest_blob_to_pyspark(blob_file_for_testing):
    test_table_name = 'table_for_testing'

    df = load_latest_blob_to_pyspark(test_table_name)

    assert isinstance(df, DataFrame)

    expected_columns = ['col1', 'col2', 'col3']
    
    assert df.columns == expected_columns
    assert df.count() == 2

