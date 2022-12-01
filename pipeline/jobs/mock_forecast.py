from pyspark.sql import DataFrame, functions as F
import datetime
from typing import Union


def create_mock_forecast(
    input_df: DataFrame,
    months_to_forecast: int,
    columns_to_consider: list[str],
    start_month: Union[datetime.date, None] = None,
    date_column: str = "SpendMonth",
    amount_column: str = "EvidencedSpend",
) -> DataFrame:
    """Create mock forecast for given spend data. The amount in output is generated randomly.
    For each combination in the input data, it will detect the date of latest entry and create mock forecast for the n months afterwards.

    Args:
        input_df (DataFrame): The input dataframe
        months_to_forecast (int): The number of months to create forecast for.
        columns_to_consider (list[str]): Column names that represent the categorisation to consider when making forecast. For example, ["Category", "MarketSector"] . It will create a set of forecast for n month for each combination.
        start_month: The month to start making forecast. If omitted, will default to the next month of today's date.
        date_column (str, optional): The column name that represent date. Defaults to "SpendMonth".
        amount_column (str, optional): The column name that represent spend amount. Defaults to "EvidencedSpend". In the output data, this column name will be replaced by 'ForecastSpend'

    Returns:
        A dataframe of mock forecast data.
    """

    # get the row of latest month for each combination
    latest_month_rows = input_df.groupBy(columns_to_consider).agg(
        F.max(date_column).alias(date_column),
        F.max_by(amount_column, date_column).alias(amount_column),
    )

    if not start_month:
        start_month = input_df.select(F.add_months(F.current_date(), 1)).head(1)[0][0]
    elif isinstance(start_month, datetime.date):
        start_month = start_month.replace(day=1)

    output_df = (
        latest_month_rows.withColumns(  # generate the start and end for the forecast period
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
            F.explode("forecastPeriod").alias("ForecastMonth")
        )
        .withColumn(  # add random amount as ForecastSpend
            "ForecastSpend", F.round((F.rand() + 0.5) * F.col(amount_column), 1)
        )
        .drop(amount_column)
        .withColumnRenamed("ForecastMonth", date_column)
    )

    return output_df
