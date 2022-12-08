import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_percentage_error,
    r2_score,
)

from pipeline.jobs.forecast_model import ForecastModel


def safe_mean_absolute_percentage_error(y_true: pd.Series, y_pred: pd.Series, **kwargs):
    """A wrapper around the mean_absolute_percentage_error method to avoid error arise from NaN value in input.
    If any of the input has NaN, will simply return NaN instead of throwing error.

    Args:
        y_true (pd.Series):  Ground truth (correct) target values.
        y_pred (pd.Series):  Predicted target values.

    Returns:
        pd.Series or scalar value np.NaN
    """
    if y_true.hasnans or y_pred.hasnans:
        return np.NaN

    return mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred, **kwargs)


def safe_r2_score(y_true: pd.Series, y_pred: pd.Series, **kwargs):
    """A wrapper around the r2_score method to avoid error arise from NaN value in input.
    If any of the input has NaN, will simply return NaN instead of throwing error.

    Args:
        y_true (pd.Series):  Ground truth (correct) target values.
        y_pred (pd.Series):  Predicted target values.

    Returns:
        pd.Series or scalar value np.NaN
    """
    if y_true.hasnans or y_pred.hasnans:
        return np.NaN

    return r2_score(y_true=y_true, y_pred=y_pred, **kwargs)


def create_models_comparison(
    input_df: pd.DataFrame,
    train_ratio: float,
    models: list[ForecastModel],
) -> pd.DataFrame:
    """Feed a portion of data into forecast models, and compare the output with remaining data.
    Returns a table with error rate of each model for each combination.

    Args:
        input_df (pd.DataFrame): Spend data
        train_ratio (float): A value within 0 to 1 which denotes the ratio of data used for training the model.
        models (dict[str, Callable[[pd.DataFrame, int], pd.DataFrame]]): A dictionary with keys being name of models and values being a method of the model.

    Returns:
        pd.DataFrame: A table with the outputs and error rate for each given models.
    """

    columns_to_consider = models[0].columns_to_consider
    amount_column = models[0].amount_column
    date_column = models[0].date_column

    aggregated_spend_by_month = (
        input_df.groupby([*columns_to_consider, date_column])[amount_column]
        .sum()
        .reset_index()
        .sort_values(by=date_column)
    )

    unique_dates = aggregated_spend_by_month[date_column].unique()
    dataset_size = len(unique_dates)

    train_size = int(dataset_size * train_ratio)
    prediction_size = dataset_size - train_size

    train_period = unique_dates[0:train_size]

    prediction_period = unique_dates[train_size:]

    # Take a portion of real data to feed in the model. The rest will be compared with the output of the model.
    df_for_training = aggregated_spend_by_month[
        aggregated_spend_by_month[date_column].isin(train_period)
    ]
    comparison_table = aggregated_spend_by_month[
        aggregated_spend_by_month[date_column].isin(prediction_period)
    ].reset_index(drop=True)

    forecast_start_month = prediction_period[0]

    # Loop through a list of models and populate the table
    for model in models:
        model_name = model.name
        forecast_column_name = f"{model_name} Forecast"

        forecast = model.create_forecast(
            input_df=df_for_training,
            months_to_forecast=prediction_size,
            start_month=forecast_start_month,
        ).rename(columns={"ForecastSpend": forecast_column_name})

        comparison_table = comparison_table.merge(
            right=forecast, on=[*columns_to_consider, date_column], how="inner"
        )

        # Calculate absolute error percentage for each single months
        comparison_table[f"{model_name} Error %"] = (
            comparison_table[forecast_column_name] - comparison_table[amount_column]
        ).abs() / comparison_table[amount_column]

        # Calculate MAPE Mean Absolute Percentage Error
        mape_by_combinations = (
            comparison_table.groupby(columns_to_consider)[
                [amount_column, forecast_column_name]
            ]
            .apply(
                lambda df: safe_mean_absolute_percentage_error(
                    y_true=df[amount_column], y_pred=df[forecast_column_name]
                )
            )
            .to_frame(f"{model_name} MAPE")
        )

        comparison_table = comparison_table.merge(
            right=mape_by_combinations,
            on=columns_to_consider,
            how="left",
        )

    comparison_table.sort_values(by=[*columns_to_consider, date_column], inplace=True)

    return comparison_table
