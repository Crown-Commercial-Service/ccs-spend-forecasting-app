import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import (
    mean_absolute_percentage_error,
)

from pipeline.jobs.forecast_model import ForecastModel
from utils import get_logger

logger = get_logger()


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
    date_column = models[0].date_column

    aggregated_spend = prepare_data(input_df=input_df, models=models)

    unique_dates = aggregated_spend[date_column].unique()
    dataset_size = len(unique_dates)

    train_size = int(dataset_size * train_ratio)
    prediction_size = dataset_size - train_size

    train_period = unique_dates[0:train_size]

    prediction_period = unique_dates[train_size:]

    # Separate the data to two portions, one for training the model and one for comparison.
    # The data for comparison will be named as "comparison_table" from this point
    df_for_training = aggregated_spend[aggregated_spend[date_column].isin(train_period)]
    comparison_table = aggregated_spend[
        aggregated_spend[date_column].isin(prediction_period)
    ].reset_index(drop=True)

    forecast_start_month = prediction_period[0]

    # Loop through a list of models and populate the table with forecast and model accuracy
    for model in models:
        comparison_table = fill_in_model_forecast(
            df_for_training=df_for_training,
            comparison_table=comparison_table,
            model=model,
            months_to_forecast=prediction_size,
            start_month=forecast_start_month,
        )

        comparison_table = fill_in_model_accuracy(
            comparison_table=comparison_table, model=model
        )

    # compare the results and fill in the suggested model for each combination
    comparison_table = fill_in_model_suggestion(
        comparison_table=comparison_table, models=models
    )

    # sort the table by Category, MarketSector, Date before output
    comparison_table.sort_values(by=[*columns_to_consider, date_column], inplace=True)

    return comparison_table


def prepare_data(input_df: pd.DataFrame, models: list[ForecastModel]) -> pd.DataFrame:
    """Aggregate the spend data by month and by the categorisation defined in model (e.g. Category & MarketSector)
    This function is meant to be called from create_models_comparison.
    Returns:
        pd.DataFrame: Aggregated data
    """
    columns_to_consider = models[0].columns_to_consider
    amount_column = models[0].amount_column
    date_column = models[0].date_column

    return (
        input_df.groupby([*columns_to_consider, date_column])[amount_column]
        .sum()
        .reset_index()
        .sort_values(by=date_column)
    )


def fill_in_model_forecast(
    df_for_training: pd.DataFrame,
    comparison_table: pd.DataFrame,
    months_to_forecast: int,
    start_month: datetime.date,
    model: ForecastModel,
) -> pd.DataFrame:
    """Fill in the forecast amount for each model to comparison table
    This function is meant to be called from create_models_comparison.
    """
    model_name = model.name
    columns_to_consider = model.columns_to_consider
    date_column = model.date_column

    try:
        forecast = model.create_forecast(
            input_df=df_for_training,
            months_to_forecast=months_to_forecast,
            start_month=start_month,
        ).rename(columns={"ForecastSpend": column_name_for_forecast(model_name)})
    except Exception as err:
        # If the model raise exception, log the error, fill the forecast column with NaN and return.
        logger.error(f"Exception raised when trying to create forecast {err}")
        logger.error(f"Model name: {model.name}")

        return comparison_table.assign(**{column_name_for_forecast(model_name): np.NaN})

    if model.amount_column in forecast.columns:
        forecast.drop(columns=model.amount_column, inplace=True)

    return comparison_table.merge(
        right=forecast, on=[*columns_to_consider, date_column], how="inner"
    )


def fill_in_model_accuracy(
    comparison_table: pd.DataFrame, model: ForecastModel
) -> pd.DataFrame:
    """Fill in the accuracy for each model to comparison table
    This function is meant to be called from create_models_comparison.
    """

    model_name = model.name
    forecast_column_name = column_name_for_forecast(model_name)
    columns_to_consider = model.columns_to_consider
    amount_column = model.amount_column

    # Calculate absolute error percentage for each single months
    comparison_table[f"{model_name} Error %"] = (
        comparison_table[forecast_column_name] - comparison_table[amount_column]
    ).abs() / comparison_table[amount_column]

    mape_by_combinations = (
        comparison_table.groupby(columns_to_consider)[
            [amount_column, forecast_column_name]
        ]
        .apply(
            lambda df: safe_mean_absolute_percentage_error(
                y_true=df[amount_column], y_pred=df[forecast_column_name]
            )
        )
        .to_frame(column_name_for_MAPE(model_name))
    )

    return comparison_table.merge(
        right=mape_by_combinations,
        on=columns_to_consider,
        how="left",
    )


def fill_in_model_suggestion(
    comparison_table: pd.DataFrame, models: list[ForecastModel]
):
    """Fill in the model suggestion for each combination.
    This function is meant to be called from create_models_comparison.
    """

    metric_column_names = [column_name_for_MAPE(model.name) for model in models]
    renamer = {column_name_for_MAPE(model.name): model.name for model in models}

    return comparison_table.assign(
        **{
            "Model Suggested": comparison_table[metric_column_names].idxmin(axis=1)
            # idxmin fills the model suggestion with column names such as "Model A MAPE", so run a replace to change that back to "Model A"
            .replace(renamer)
        }
    )


""" Below are helper methods to standise column names that are reused in several places"""


def column_name_for_forecast(model_name: str) -> str:
    return f"{model_name} Forecast"


def column_name_for_MAPE(model_name: str) -> str:
    return f"{model_name} MAPE"
