import datetime
import pandas as pd
from pyspark.sql import functions as F
import os
import sys

# add /dbfs/ to path so that import statements works on databricks
if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    sys.path.append("/dbfs/")


from utils import get_logger
from pipeline.jobs.sarima_model import SarimaModel
from pipeline.jobs.arima_model import ArimaModel
from pipeline.jobs.arma_model import ArmaModel
from pipeline.jobs.forecast_model import ForecastModel
from pipeline.jobs.models_comparison import create_models_comparison
from pipeline.utils import (
    connect_spark_to_blob_storage,
    load_latest_blob_to_pyspark,
    save_pandas_dataframe_to_blob,
)

logger = get_logger()


def main():
    input_df = fetch_data_from_blob()
    models = model_choices()
    model_suggestions = compare_models_performance(input_df, models=models)
    forecast_df = run_forecast_with_suggested_models(
        input_df=input_df,
        model_suggestions=model_suggestions,
        models=models,
        months_to_forecast=12,
        start_month=datetime.date(2023, 4, 1),
    )
    output_forecast_to_blob(forecast_df=forecast_df)


def fetch_data_from_blob() -> pd.DataFrame:
    logger.debug("loading spend data from Azure blob storage...")

    connect_spark_to_blob_storage()
    table_name = "SpendDataFilledMissingMonth"
    sdf = load_latest_blob_to_pyspark(table_name)

    # select the combinations to make forecast for.
    category_list = [
        "Workforce Health & Education",
        # "Document Management & Logistics",
        # "Financial Services",
        # "Network Services",
    ]
    market_sector_list = [
        "Health",
        "Education",
        # "Local Community and Housing",
        # "Government Policy",
    ]

    df = sdf.filter(
        (F.col("Category").isin(category_list))
        & (F.col("MarketSector").isin(market_sector_list))
    ).toPandas()

    # sum up the spend data by month, so that for each combination, only one row for one month
    input_df = df.groupby(
        ["SpendMonth", "Category", "MarketSector"], as_index=False
    ).agg({"EvidencedSpend": "sum"})

    return input_df


def model_choices() -> list[ForecastModel]:
    logger.debug("Instantiating models...")

    sarima = SarimaModel(search_hyperparameters=True)
    arima = ArimaModel(search_hyperparameters=True)
    arma = ArmaModel(search_hyperparameters=True)

    return [
        sarima,
        arima,
        arma,
    ]


def compare_models_performance(
    input_df: pd.DataFrame, models: list[ForecastModel]
) -> pd.DataFrame:

    logger.debug("Running test with past spend data to compare model performance...")

    comparison_table = create_models_comparison(
        input_df=input_df, train_ratio=0.9, models=models
    )

    columns_to_extract = ["Category", "MarketSector", "Model Suggested"]
    model_suggestions = comparison_table[columns_to_extract].drop_duplicates()

    logger.debug("Comparison result:")
    logger.debug(model_suggestions)

    return model_suggestions


def run_forecast_with_suggested_models(
    input_df: pd.DataFrame,
    model_suggestions: pd.DataFrame,
    models: list[ForecastModel],
    months_to_forecast: int,
    start_month: datetime.date,
) -> pd.DataFrame:

    logger.debug("Generating forecast with suggested models...")

    model_chooser: dict[str, ForecastModel] = {model.name: model for model in models}
    spend_data_with_model_suggestions = input_df.merge(
        right=model_suggestions, on=["Category", "MarketSector"]
    )

    output_dfs = []
    for model_suggested, spend_data in spend_data_with_model_suggestions.groupby(
        "Model Suggested"
    ):
        model_suggested = str(model_suggested)

        forecast = model_chooser[model_suggested].create_forecast(
            input_df=spend_data,
            months_to_forecast=months_to_forecast,
            start_month=start_month,
        )
        forecast["Model used"] = model_suggested
        output_dfs.append(forecast)

    output_df = pd.concat(output_dfs)
    logger.debug("Finished generating forecast.")

    return output_df


def output_forecast_to_blob(forecast_df: pd.DataFrame):

    logger.debug("Saving forecast to blob storage...")

    start_month = min(forecast_df["SpendMonth"])
    end_month = max(forecast_df["SpendMonth"])
    start_month_in_yyyymmdd = start_month.strftime("%Y%m%d")
    end_month_in_yyymmdd = end_month.strftime("%Y%m%d")

    file_name = f"ForecastOutput/Spend_forecast_for_{start_month_in_yyyymmdd}-{end_month_in_yyymmdd}.csv"

    save_pandas_dataframe_to_blob(pandas_df=forecast_df, filename=file_name)

    logger.debug("Job completed.")


if __name__ == "__main__":
    """Run the pipeline process"""

    main()
