import pandas as pd
from typing import Callable
from sklearn.metrics import mean_absolute_percentage_error


def create_models_comparison(
    input_df: pd.DataFrame,
    train_size: int,
    prediction_size: int,
    models: dict[str, Callable[[pd.DataFrame, int], pd.DataFrame]],
) -> pd.DataFrame:

    df = (
        input_df.groupby(["Category", "MarketSector", "SpendMonth"])["EvidencedSpend"]
        .sum()
        .reset_index()
        .sort_values(by="SpendMonth")
    )

    train_period = df["SpendMonth"].unique()[0:train_size]
    prediction_period = df["SpendMonth"].unique()[
        train_size : train_size + prediction_size
    ]

    df_for_training = df[df["SpendMonth"].isin(train_period)]
    comparison_table = df[df["SpendMonth"].isin(prediction_period)].reset_index(
        drop=True
    )

    forecast_start_month = prediction_period[0]

    for model_name, model_function in models.items():
        forecast_column_name = f"{model_name} Forecast"

        forecast = model_function(
            input_df=df_for_training,
            months_to_forecast=prediction_size,
            start_month=forecast_start_month,
        ).rename(columns={"ForecastSpend": forecast_column_name})
        comparison_table = comparison_table.merge(
            right=forecast, on=["Category", "MarketSector", "SpendMonth"], how="outer"
        )

        comparison_table[f"{model_name} Error %"] = (
            comparison_table[forecast_column_name] - comparison_table["EvidencedSpend"]
        ).abs() / comparison_table["EvidencedSpend"]

        error_perc_for_model = mean_absolute_percentage_error(
            y_true=comparison_table["EvidencedSpend"],
            y_pred=comparison_table[forecast_column_name],
        )
        comparison_table[f"{model_name} MAE %"] = error_perc_for_model

    print(comparison_table)
    return comparison_table
