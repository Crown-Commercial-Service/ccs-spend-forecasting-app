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
    """ An example workflow that can be run on local machine or databricks.

    It will execute the following tasks:
    1. Fetch latest spend data from Azure blob storage
    2. Create 3 forecast models (SARIMA/ARIMA/ARMA) and compare their performance for each combination, base on the past spend data
    3. Create actual forecast of a future period using the models which performed better for each combination
    4. Output the forecast to blob storage.
    """
    input_df = fetch_data_from_blob()
    models = model_choices()
    model_suggestions = compare_models_performance(input_df, models=models)

    today = datetime.date.today()
    next_month = today.replace(month = today.month + 1, day = 1) if today.month < 12 else today.replace(year = today.year + 1, month = 1, day = 1)

    forecast_df = run_forecast_with_suggested_models(
        input_df=input_df,
        model_suggestions=model_suggestions,
        models=models,
        months_to_forecast=12,
        start_month=next_month
    )
    output_forecast_to_blob(forecast_df=forecast_df)


def fetch_data_from_blob() -> pd.DataFrame:
    """Fetch spend data from blob storage and prepare the data for feeding into model.

    Returns:
        A pandas dataframe of spend data which is summed up by month. For each combination, there is only one row for one month.
    """
    logger.debug("loading spend data from Azure blob storage...")

    connect_spark_to_blob_storage()
    table_name = "SpendDataFilledMissingMonth"
    sdf = load_latest_blob_to_pyspark(table_name)

    # select the combinations to make forecast for.
    category_list = [
        "Workforce Health & Education",
        "Document Management & Logistics",
        "Financial Services",
        "Network Services",
    ]
    market_sector_list = [
        "Health",
        "Education",
        "Local Community and Housing",
        "Government Policy",
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
    """ Instantiate the forecast models to use.

    Returns: A list of ForecastModel object, each one represent a different type of forecast model.
    """
    
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
    """ Test train each model with 90% of spend data and compare their accuracy with the rest 10% data. 
        Will output the result in a table format.

    Args:
        input_df (pd.DataFrame): A dataframe of prepared spend data.
        models (list[ForecastModel]): A list of ForecastModel instances.

    Returns:
        pd.DataFrame: A table which contains the comparison result for each combination.
    """

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
    """ Generate forecast for a future period with the better performed models for each combination.

    Args:
        input_df (pd.DataFrame): A dataframe of prepared spend data.
        model_suggestions (pd.DataFrame): The output dataframe of method #compare_models_performance, which contains the information of which model performed better for which combination.
        models (list[ForecastModel]): A list of ForecastModel instances.
        months_to_forecast (int): The number of months to make forecast for.
        start_month (datetime.date): The starting month to make forecast.

    Returns:
        pd.DataFrame: A dataframe of forecasted future spending.
    """

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
    """ Save the forecast data to blob storage.
    """

    logger.debug("Saving forecast to blob storage...")

    start_month = min(forecast_df["SpendMonth"])
    end_month = max(forecast_df["SpendMonth"])
    start_month_in_yyyymmdd = start_month.strftime("%Y%m%d")
    end_month_in_yyyymmdd = end_month.strftime("%Y%m%d")
    today_in_yyyymmdd = datetime.date.today().strftime("%Y%m%d")

    file_name = f"ForecastOutput/Spend_forecast_for_{start_month_in_yyyymmdd}-{end_month_in_yyyymmdd}__forecast_made_on_{today_in_yyyymmdd}.csv"

    save_pandas_dataframe_to_blob(pandas_df=forecast_df, filename=file_name)

    logger.debug("Job completed.")


if __name__ == "__main__":

    main()
