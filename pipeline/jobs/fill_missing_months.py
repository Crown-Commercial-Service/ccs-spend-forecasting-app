from pyspark.sql import DataFrame, functions as F


def add_missing_months(
    df: DataFrame,
    date_column: str = "SpendMonth",
    amount_column: str = "EvidencedSpend",
    columns_to_consider: list[str] = [],
) -> DataFrame:
    """
    Receive a dataframe and return a new dataframe with zero-amount placeholder rows added for every missing months.
    The latest date across all combination is taken as the `latest_month`.
    For every combinations, for the period from the earliest record in until this `latest_month`,
    if any month in between doesn't have any spending, a placeholder record of amount = 0 will be inserted.

    Args:
        df (DataFrame): Input Dataframe
        date_column (str, optional): The column name that represent date. Defaults to "SpendMonth".
        amount_column (str, optional): The column name that represent spend amount. The added placeholder rows will have 0 at this column. Defaults to "EvidencedSpend".
        columns_to_consider (list[str], optional): A list of column names in the input dataframe. For each combination of those columns, placeholder rows will be added to cover any missing month. Defaults to [].

    Returns:
        DataFrame: Dataframe with no missing month
    """

    input_df_columns = df.columns

    latest_month_row = df.select(F.max(date_column)).first()
    if not latest_month_row:
        raise ValueError(f"the column {date_column} is missing from input data")
    latest_month = latest_month_row[0]

    all_months_combinations = (
        df.groupBy(columns_to_consider)
        .agg(
            F.min(date_column).alias("minMonth"), F.lit(latest_month).alias("maxMonth")
        )
        .select(
            *columns_to_consider,
            F.expr("sequence(minMonth, maxMonth, interval 1 month)").alias(
                "possible_months"
            ),
        )
        .withColumn(date_column, F.explode("possible_months"))
        .drop("possible_months")
    )

    output_df = (
        df.join(
            all_months_combinations, on=[*columns_to_consider, date_column], how="full"
        )
        .fillna(value=0, subset=[amount_column])
        .select([column_name for column_name in input_df_columns])
    )

    return output_df


def fill_missing_months_for_transformed_spend(
    input_table_name: str,
    output_table_name: str,
    container_name: str = "",
):
    """Actual function that carry out the transformation step in pipeline

    Args:
        input_table_name (str): The table name of input data. It is assumed that a parquet file with this name should exist in the blob storage.
        output_table_name (str): The table name of output data. A parquet file will be saved to blob storage with this value as the folder name.
        container_name (str, optional): The blob container name to retreieve the input file and to save the output blob. Defaults to "".
    """

    from pipeline.utils import (
        connect_spark_to_blob_storage,
        load_latest_blob_to_pyspark,
        make_blob_storage_path,
        save_dataframe_to_blob,
    )

    connect_spark_to_blob_storage()
    input_df = load_latest_blob_to_pyspark(
        table_name=input_table_name, blob_container_name=container_name
    )
    output_df = add_missing_months(
        df=input_df,
        date_column="SpendMonth",
        amount_column="EvidencedSpend",
        columns_to_consider=["MarketSector", "Category"],
    )

    output_path = make_blob_storage_path(
        table_name=output_table_name, blob_container_name=container_name
    )
    save_dataframe_to_blob(
        df=output_df,
        table_name=output_table_name,
        blob_storage_path=output_path,
        overwrite=True,
    )
