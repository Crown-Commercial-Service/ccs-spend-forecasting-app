import os
import sys

# add /dbfs/ to path so that import statements works on databricks
if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    sys.path.append("/dbfs/")

from pipeline.utils import (
    connect_spark_to_blob_storage,
    load_latest_blob_to_pyspark,
    load_csv_from_blob_to_pandas,
    save_pandas_dataframe_to_blob,
)
from utils import get_logger


def save_historic_spend_as_csv():
    connect_spark_to_blob_storage()
    spend_df = load_latest_blob_to_pyspark("SpendDataFilledMissingMonth")
    pandas_df = spend_df.toPandas()

    filename = "DataForPowerBI/HistoricSpend_latest.csv"
    save_pandas_dataframe_to_blob(pandas_df, filename)


def save_forecast_as_csv():
    connect_spark_to_blob_storage()
    forecast_df = load_csv_from_blob_to_pandas("ForecastOutput")

    filename = "DataForPowerBI/ForecastSpend_latest.csv"
    save_pandas_dataframe_to_blob(forecast_df, filename)


if __name__ == "__main__":

    logger = get_logger()
    logger.debug(
        "Saving the latest version of historic spend and forecast spend as .csv format..."
    )
    save_historic_spend_as_csv()
    save_forecast_as_csv()
    logger.debug("Job completed.")
