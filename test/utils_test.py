from datetime import date
from pyspark.sql import SparkSession, DataFrame

from pipeline.utils import (
    get_spark_session,
    get_blob_container_client,
    load_latest_blob_to_pyspark,
    is_running_in_databricks,
    make_blob_storage_path,
    save_dataframe_to_blob,
)


def test_get_spark_session():
    spark = get_spark_session()
    assert isinstance(spark, SparkSession) == True


def test_blob_container_access():
    container_client = get_blob_container_client()
    assert container_client.exists() == True


def test_load_latest_blob_to_pyspark(blob_file_for_testing, blob_container_for_testing):
    table_name = "table_for_testing_1"
    container_name = blob_container_for_testing.container_name

    df = load_latest_blob_to_pyspark(table_name, container_name)

    assert isinstance(df, DataFrame)

    expected_columns = ["col1", "col2", "col3"]

    assert df.columns == expected_columns
    assert df.count() == 2


def test_make_blob_storage_path():
    blob_account_name = "test_blob_account"
    blob_container_name = "test_blob_container"
    table_name = "test_table_1234"

    today = date.today()
    if is_running_in_databricks():
        expected = f"abfss://{blob_container_name}@{blob_account_name}.dfs.core.windows.net/{table_name}/year={today.year}/month={today.month}/day={today.day}/{table_name}.parquet"
    else:
        expected = f"wasbs://{blob_container_name}@{blob_account_name}.blob.core.windows.net/{table_name}/year={today.year}/month={today.month}/day={today.day}/{table_name}.parquet"

    actual = make_blob_storage_path(
        table_name=table_name,
        blob_container_name=blob_container_name,
        blob_account_name=blob_account_name,
    )

    assert actual == expected


def test_save_dataframe_to_blob(
    dataframe_for_testing_basic, blob_container_for_testing
):
    table_name = "table2_for_testing"
    container_name = blob_container_for_testing.container_name
    df = dataframe_for_testing_basic

    path = make_blob_storage_path(
        table_name=table_name,
        blob_account_name=None,
        blob_container_name=container_name,
    )

    save_dataframe_to_blob(df, table_name=table_name, blob_storage_path=path)

    df_load_from_blob = load_latest_blob_to_pyspark(
        table_name=table_name, blob_container_name=container_name
    )

    expected_columns = ["StringCol", "FloatCol", "DateCol"]
    expected_first_row = ["apple", 123.456, date(2022, 4, 1)]
    expected_second_row = ["banana", 789.012, date(2022, 10, 31)]

    df_converted_to_list = [list(row) for row in df_load_from_blob.collect()]

    assert df_load_from_blob.count() == 2
    assert df_load_from_blob.columns == expected_columns
    assert (expected_first_row in df_converted_to_list) == True
    assert (expected_second_row in df_converted_to_list) == True
