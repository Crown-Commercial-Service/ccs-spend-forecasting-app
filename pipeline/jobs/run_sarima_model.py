from pipeline.jobs.sarima_model import SarimaModel
from pipeline.utils import connect_spark_to_blob_storage, load_latest_blob_to_pyspark

import datetime


def run_sarima_model_with_data():

    connect_spark_to_blob_storage()
    sdf = load_latest_blob_to_pyspark(table_name="SpendDataFilledMissingMonth")

    df = sdf.toPandas()

    input_df = df

    model = SarimaModel()

    forecast = model.create_forecast(
        input_df=input_df, months_to_forecast=18, start_month=datetime.date(2023, 4, 1)
    )
    print(forecast)

    return forecast


run_sarima_model_with_data()
