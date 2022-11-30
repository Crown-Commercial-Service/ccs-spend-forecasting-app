from pyspark.sql import DataFrame, Row, SparkSession, functions as F

import datetime


def mock_forecast_single_combination(
    input_df: DataFrame,
    months_to_forecast: int = 12,
    date_column: str = "SpendMonth",
    amount_column: str = "EvidencedSpend",
) -> DataFrame:

    latest_month_row = input_df.orderBy(F.desc(date_column)).first()

    if not latest_month_row or not isinstance(
        latest_month_row[date_column], datetime.date
    ):
        raise ValueError(
            "The date_column of input dataframe does not have a valid date value"
        )

    df = input_df.sparkSession.createDataFrame(data=[latest_month_row])

    output_df = (
        df.withColumns( # create the forecast months
            {
                "forecastStart": F.add_months(date_column, 1),
                "forecastEnd": F.add_months(date_column, months_to_forecast),
            }
        )
        .withColumn(
            "forecastPeriod",
            F.expr("sequence(forecastStart, forecastEnd, interval 1 month)"),
        )
        .select(*input_df.columns, F.explode("forecastPeriod").alias("ForecastMonth"))
        .withColumn( # add random value as ForecastSpend
            "ForecastSpend", F.round((F.rand() + 0.5) * F.col(amount_column), 1)
        )
        .drop(amount_column, date_column)
        .withColumnRenamed("ForecastMonth", date_column)
    )


    return output_df
