from pyspark.sql import DataFrame, functions as F
import datetime


def create_mock_forecast(
    input_df: DataFrame,
    months_to_forecast: int,
    columns_to_consider: list[str],
    date_column: str = "SpendMonth",
    amount_column: str = "EvidencedSpend",
) -> DataFrame:
    """Create mock forecast for given spend data. The amount in output is generated randomly.
    For each combination in the input data, it will detect the date of latest entry and create mock forecast for the n months afterwards.

    Args:
        input_df (DataFrame): The input dataframe
        months_to_forecast (int): The number of months to create forecast for.
        columns_to_consider (list[str]): Column names that represent the categorisation to consider when making forecast. For example, ["Category", "MarketSector"] . It will create a set of forecast for n month for each combination.
        date_column (str, optional): The column name that represent date. Defaults to "SpendMonth". The datatype in this column is expected to be valid date type, not string representation of date.
        amount_column (str, optional): The column name that represent spend amount.

    Returns:
        A dataframe of mock forecast data.
    """

    # get the row of latest month for each combination
    latest_month_rows = input_df.groupBy(columns_to_consider).agg(
        F.max(date_column).alias(date_column),
        F.max_by(amount_column, date_column).alias(amount_column),
    )

    date_column_values = [row[date_column] for row in latest_month_rows.collect()]

    if not all(isinstance(value, datetime.date) for value in date_column_values):
        raise ValueError(
            "Invalid date found in input data. Please check whether date column have the correct type."
        )

    output_df = (
        latest_month_rows.withColumns(  # generate the start and end for the forecast period
            {
                "forecastStart": F.add_months(date_column, 1),
                "forecastEnd": F.add_months(date_column, months_to_forecast),
            }
        )
        .withColumn(  # populate a row for every month within the forecast period
            "forecastPeriod",
            F.expr("sequence(forecastStart, forecastEnd, interval 1 month)"),
        )
        .select(*input_df.columns, F.explode("forecastPeriod").alias("ForecastMonth"))
        .withColumn(  # add random amount as ForecastSpend
            "ForecastSpend", F.round((F.rand() + 0.5) * F.col(amount_column), 1)
        )
        .drop(amount_column, date_column)
        .withColumnRenamed("ForecastMonth", date_column)
    )

    return output_df
