from pyspark.sql import DataFrame, functions as F, SparkSession
import pandas as pd
import datetime
from typing import Optional


def create_mock_forecast(
    input_df: DataFrame,
    months_to_forecast: int,
    columns_to_consider: list[str],
    start_month: Optional[datetime.date] = None,
    date_column: str = "SpendMonth",
    amount_column: str = "EvidencedSpend",
    randomness: Optional[float] = 0.5,
) -> DataFrame:
    """Create mock forecast for given spend data. The forecast amount is generated randomly within a certain range.
    For each combination in the input data, it will detect the latest entry and create mock forecast base on that figure.

    Args:
        input_df (DataFrame): The input spend data
        months_to_forecast (int): The number of months to create forecast for.
        columns_to_consider (list[str]): Column names that represent the categorisation to consider when making forecast. For example, ["Category", "MarketSector"] . It will create a set of forecast for n month for each combination.
        start_month: (datetime.date, optional) The month to start making forecast. If omitted, will default to the next month from today's date.
        date_column (str, optional): The column name that represent date. Defaults to "SpendMonth".
        amount_column (str, optional): The column name that represent spend amount. Defaults to "EvidencedSpend". In the output data, this column name will be replaced by 'ForecastSpend'
        randomness (float, optional): A number between 0 to 1 which control the random range of output. For example, if randomness = 0.2, output amount will be within +-20% range of latest month spending.

    Returns:
        A dataframe of mock forecast data.
    """

    input_df_aggregated = aggregate_spends_by_month(
        input_df=input_df,
        columns_to_consider=columns_to_consider,
        date_column=date_column,
        amount_column=amount_column,
    )

    # get the row of latest month for each combination
    latest_month_rows = input_df_aggregated.groupBy(columns_to_consider).agg(
        F.max(date_column).alias(date_column),
        F.max_by(amount_column, date_column).alias(amount_column),
    )

    if isinstance(start_month, datetime.date):
        # force the day to be 1st of month as we only care about year and month
        start_month = start_month.replace(day=1)
    else:
        # if start_month is not a valid date type, replace it with next month from today's date
        today = datetime.date.today()
        start_month = (
            today.replace(month=today.month + 1, day=1)
            if today.month != 12
            else datetime.date(year=today.year + 1, month=1, day=1)
        )

    # populate the dataframe with one row per month in the forecast period
    df_populated_with_months = (
        latest_month_rows.withColumns(
            {
                "forecastStart": F.lit(start_month),
                "forecastEnd": F.add_months(F.lit(start_month), months_to_forecast - 1),
            }
        )
        .withColumn(  # populate a row for every month within the forecast period
            "forecastPeriod",
            F.expr("sequence(forecastStart, forecastEnd, interval 1 month)"),
        )
        .select(
            *columns_to_consider,
            amount_column,
            F.explode("forecastPeriod").alias("ForecastMonth"),
        )
    )

    # fill in random amounts to ForecastSpend
    output_df = (
        df_populated_with_months.withColumn(
            "ForecastSpend",
            F.expr(
                f"round({amount_column} * (1 + {randomness} * (2 * rand() - 1)), 1)"
            ),
        )
        .drop(amount_column)
        .withColumnRenamed("ForecastMonth", date_column)
    )

    return output_df


def create_mock_forecast_pandas(input_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """A simple wrapper around method #create_mock_forecast to make it works with pandas dataframe"""
    spark = SparkSession.getActiveSession()

    if not spark:
        spark = SparkSession.Builder().getOrCreate()
    input_df_in_pandas = spark.createDataFrame(data=input_df)
    output_df = create_mock_forecast(input_df=input_df_in_pandas, **kwargs)

    return output_df.toPandas()


def create_mock_model(randomness: float):
    """Create a mock model that output forecast with a pre-defined randomness factor.

    Args:
        randomness (float): A number between 0 to 1 which control the random range of output. For example, if randomness = 0.2, output amount will be within +-20% range of latest month spending.
    """

    def model(
        input_df: pd.DataFrame,
        months_to_forecast: int,
        start_month: Optional[datetime.date] = None,
    ):
        return create_mock_forecast_pandas(
            input_df=input_df,
            months_to_forecast=months_to_forecast,
            columns_to_consider=["Category", "MarketSector"],
            start_month=start_month,
            date_column="SpendMonth",
            amount_column="EvidencedSpend",
            randomness=randomness,
        )

    return model


def aggregate_spends_by_month(
    input_df: DataFrame,
    columns_to_consider: list[str],
    date_column: str,
    amount_column: str,
) -> DataFrame:
    """Aggregate spend data by the given categorisation and return the monthly total spends amount for each combination.
    For example, if columns_to_consider is ["Category", "MarketSector"], it will sum up each month's spends for every available combination of Category and MarketSector.

    Args:
        input_df (DataFrame): The input spend data
        columns_to_consider (list[str]): Column names that represent the categorisation to consider. For example, ["Category", "MarketSector"] .
        date_column (str): The column name that represent date.
        amount_column (str): The column name that represent spend amount.

    Returns:
        DataFrame: The aggregated data frame
    """
    return input_df.groupBy(date_column, *columns_to_consider).agg(
        F.sum(amount_column).alias(amount_column)
    )
