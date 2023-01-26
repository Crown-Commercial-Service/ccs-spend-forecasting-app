import os
import sys
import datetime
import itertools

import pandas as pd

# add /dbfs/ to path so that import statements works on databricks
if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    sys.path.append("/dbfs/")


from utils import get_logger
from pipeline.models.sarima_model import SarimaModel
from pipeline.models.arima_model import ArimaModel
from pipeline.models.arma_model import ArmaModel
from pipeline.models.prophet_model import ProphetModel
from pipeline.models.forecast_model import ForecastModel
from pipeline.jobs.models_comparison import create_models_comparison
from pipeline.utils import (
    connect_spark_to_blob_storage,
    load_latest_blob_to_pyspark,
    save_pandas_dataframe_to_blob,
    load_csv_from_blob_to_pandas,
    suppress_unwanted_warnings,
)

logger = get_logger()


def main():
    """An example workflow that can be run on local machine or databricks.

    It will execute the following tasks:
    1. Fetch latest spend data from Azure blob storage
    2. Create 4 forecast models (SARIMA/ARIMA/Prophet) and compare their performance for each combination, base on the past spend data.
    3. Create forecast with each models, and also add a column of "suggested forecast" which is given by the model that performed best in each combination.
    4. Output the forecast to blob storage.
    """
    input_df = fetch_data_from_blob()
    models = model_choices()

    suppress_unwanted_warnings()

    # run each model with spend data to get suggestion for each combination
    model_suggestions = compare_models_performance(input_df, models=models)

    today = datetime.date.today()
    last_month = (
        today.replace(month=today.month - 1, day=1)
        if today.month > 1
        else today.replace(year=today.year - 1, month=12, day=1)
    )

    # Generate forecast.
    # Currently the forecast period is set as from last month, until 24 + 2 months from running time, in order to align with EDA.
    forecast_df = run_forecast_with_all_models(
        input_df=input_df,
        model_suggestions=model_suggestions,
        models=models,
        months_to_forecast=24 + 2,
        start_month=last_month,
    )

    # log the hyperparameter for each model.
    params_cache = {model.name: model._hyperparameters_cache for model in models}
    params_df = pd.DataFrame(data=params_cache)
    logger.debug("Hyperparameters used for each models:\n")
    logger.debug(params_df)
    logger.debug("in JSON form (for copy & paste):")
    logger.debug(params_df.to_json())

    output_forecast_to_blob(forecast_df=forecast_df)


def fetch_data_from_blob() -> pd.DataFrame:
    """Fetch spend data from blob storage and prepare the data for feeding into model.

    Returns:
        A pandas dataframe of spend data which is summed up by month. For each combination, there is only one row for one month.
    """
    logger.debug("loading spend data from Azure blob storage...")

    connect_spark_to_blob_storage()
    spend_data = load_latest_blob_to_pyspark("SpendDataFilledMissingMonth")

    active_combinations = load_csv_from_blob_to_pandas("ActiveCombinations")

    ## For development purpose. Uncomment the following lines to manually choose what combinations to use.
    # category_list = [
    #     "Workforce Health & Education",
    #     "Document Management & Logistics",
    #     "Financial Services",
    #     "Network Services",
    # ]
    # market_sector_list = [
    #     "Health",
    #     "Education",
    #     "Local Community and Housing",
    #     "Government Policy",
    # ]
    # active_combinations = pd.DataFrame(data=itertools.product(category_list, market_sector_list), columns=['Category', 'MarketSector'])

    combinations = spend_data.sparkSession.createDataFrame(active_combinations)
    column_names = [str(column_name) for column_name in active_combinations.columns]
    filtered_spend = (
        spend_data.alias("spend")
        .join(combinations, on=column_names, how="inner")
        .select("spend.*")
    )

    df = filtered_spend.toPandas()

    # sum up the spend data by month, so that for each combination, only one row for one month
    input_df = df.groupby(
        ["SpendMonth", "Category", "MarketSector"], as_index=False
    ).agg({"EvidencedSpend": "sum"})

    return input_df


def model_choices() -> list[ForecastModel]:
    """Instantiate the forecast models to use.

    Returns: A list of ForecastModel object, each one represent a different type of forecast model.
    """

    logger.debug("Instantiating models...")

    sarima = SarimaModel(search_hyperparameters=True)
    arima = ArimaModel(search_hyperparameters=True)
    prophet = ProphetModel()

    return [sarima, arima, prophet]


def compare_models_performance(
    input_df: pd.DataFrame, models: list[ForecastModel]
) -> pd.DataFrame:
    """Test train each model with 90% of spend data and compare their accuracy with the rest 10% data.
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

    columns_to_extract = [
        "Category",
        "MarketSector",
        "MAPE of Suggested Model",
        "Model Suggested",
    ]
    model_suggestions = comparison_table[columns_to_extract].drop_duplicates()

    logger.debug("Comparison result:")
    logger.debug(model_suggestions)

    return model_suggestions


def run_forecast_with_all_models(
    input_df: pd.DataFrame,
    model_suggestions: pd.DataFrame,
    models: list[ForecastModel],
    months_to_forecast: int,
    start_month: datetime.date,
) -> pd.DataFrame:
    """Generate forecast for a future period with all models for each combination.

    Args:
        input_df (pd.DataFrame): A dataframe of prepared spend data.
        model_suggestions (pd.DataFrame): The output dataframe of method #compare_models_performance, which contains the information of which model performed better for which combination.
        models (list[ForecastModel]): A list of ForecastModel instances.
        months_to_forecast (int): The number of months to make forecast for.
        start_month (datetime.date): The starting month to make forecast.

    Returns:
        pd.DataFrame: A dataframe of forecasted future spending.
    """

    logger.debug("Generating forecast with all models...")

    # Generate forecast with all models
    output_dfs = []
    for model in models:
        forecast = model.create_forecast(
            input_df=input_df,
            months_to_forecast=months_to_forecast,
            start_month=start_month,
        )
        forecast = forecast.rename(columns={"ForecastSpend": f"{model.name}_Forecast"})
        output_dfs.append(forecast)

    # Combine the forecast into one dataframe
    output_df = output_dfs[0]
    for other_output in output_dfs[1:]:
        output_df = output_df.merge(
            other_output, how="outer", on=["Category", "MarketSector", "SpendMonth"]
        )
    output_df = output_df.merge(
        right=model_suggestions, on=["Category", "MarketSector"], how="left"
    )

    # Copy the forecasted amount from the suggested model to 'Suggested Forecast' column
    for model_name in pd.unique(output_df["Model Suggested"]):
        output_df.loc[
            (output_df["Model Suggested"]) == model_name, "Suggested Forecast"
        ] = output_df[f"{model_name}_Forecast"]

    logger.debug("Finished generating forecast.")

    return output_df


def output_forecast_to_blob(forecast_df: pd.DataFrame):
    """Save the forecast data to blob storage."""

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
