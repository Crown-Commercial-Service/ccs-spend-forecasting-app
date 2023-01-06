import datetime
import pandas as pd
from unittest.mock import MagicMock

from pipeline.jobs.sarima_model import SarimaModel


def test_SARIMA_model_initiation():
    model = SarimaModel()

    actual = isinstance(model, SarimaModel)
    expected = True

    assert expected == actual


def test_SARIMA_model_create_forecast():
    """Test that method #create_forecast will return forecast data in correct format and dates"""
    input_df = pd.DataFrame(
        # fmt: off
            data={
                'SpendMonth': [datetime.date(2022, 1, 1), datetime.date(2022, 2, 1), datetime.date(2022, 3, 1), datetime.date(2022, 4, 1), datetime.date(2022, 5, 1), datetime.date(2022, 6, 1)],
                'Category': ['Test Category'] * 6,
                'MarketSector': ['Health'] * 6,
                'EvidencedSpend': [1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]
            }
        # fmt: on
    )
    months_to_forecast = 12
    start_month = datetime.date(2024, 1, 1)

    model = SarimaModel(search_hyperparameters=False)
    output_forecast = model.create_forecast(
        input_df=input_df,
        months_to_forecast=months_to_forecast,
        start_month=start_month,
    )

    # check output is a dataframe
    actual = isinstance(output_forecast, pd.DataFrame)
    expected = True

    # check that the output got the right columns
    actual = list(output_forecast.columns)
    expected = ["SpendMonth", "Category", "MarketSector", "ForecastSpend"]

    assert expected == actual

    # check that output got 12 rows, same as the `months_to_forecast` argument given
    expected = 12
    actual = len(output_forecast)

    assert expected == actual

    # check the dates in output is from 2024-1-1 to 2024-12-1
    expected = [
        datetime.date(2024, month, 1)
        for month in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    ]
    actual = list(output_forecast[model.date_column])

    assert expected == actual


def test_SARIMA_model_create_forecast_with_multiple_combinations():
    """Test that calling create_forecast with input data of multiple Category/MarketSector combination will generate n months forecast for each combination"""
    input_df = pd.DataFrame(
        # fmt: off
            data={
                # four combinations, each with 12 months data
                'SpendMonth': [datetime.date(2022, month, 1) for month in range(1, 12 + 1)] * 4,
                'Category': ['Test Category A'] * 24 + ['Test Category B'] * 24,
                'MarketSector': ['Health'] * 12 + ['Education'] * 12 + ['Health'] * 12 + ['Education'] * 12,
                'EvidencedSpend': [1000.0, 1200.0, 1400.0, 1200.0, 1000.0, 1200.0] * 8
            }
        # fmt: on
    )
    months_to_forecast = 24
    start_month = datetime.date(2023, 4, 1)

    model = SarimaModel(search_hyperparameters=False)

    output_forecast = model.create_forecast(
        input_df=input_df,
        months_to_forecast=months_to_forecast,
        start_month=start_month,
    )

    # check that the output got the right columns
    actual = list(output_forecast.columns)
    expected = ["SpendMonth", "Category", "MarketSector", "ForecastSpend"]

    assert expected == actual

    # check that output got 4 * 24 rows. 4 combinations * 24 months each
    expected = 4 * 24
    actual = len(output_forecast)

    assert expected == actual

    # check the dates in output are within 2023-4-1 to 2025-3-1
    expected_months_in_2023 = [
        datetime.date(2023, month, 1) for month in (4, 5, 6, 7, 8, 9, 10, 11, 12)
    ]
    expected_months_in_2024 = [
        datetime.date(2024, month, 1)
        for month in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    ]
    expected_months_in_2025 = [datetime.date(2025, month, 1) for month in (1, 2, 3)]
    all_expected_months = (
        expected_months_in_2023 + expected_months_in_2024 + expected_months_in_2025
    )

    for months in all_expected_months:
        # for each month, we should got 4 rows, one for each combination
        forecast_for_month = output_forecast[output_forecast["SpendMonth"] == months]

        expected = 4
        actual = len(forecast_for_month)

        assert expected == actual

        expected = [
            "Test Category A",
            "Test Category A",
            "Test Category B",
            "Test Category B",
        ]
        actual = sorted(forecast_for_month["Category"])

        assert expected == actual

        expected = ["Education", "Education", "Health", "Health"]
        actual = sorted(forecast_for_month["MarketSector"])

        assert expected == actual


def test_SARIMA_model_will_search_for_hyperparameter_when_flag_is_true():
    """
    Test the behaviours when `search_hyperparameters` is set to True.

    To reduce testing time and remove randomness, the method for searching hyperparameter and the method for running the model are substituted by mock.
    Here, we test that when `search_hyperparameters` is True,
    calling #create_forecast will run the search, and it use the params that the search returned to generate forecast
    """
    model = SarimaModel(search_hyperparameters=True)

    mock_hyperparameters = {
        "order": (2, 2, 3),
        "seasonal_order": (0, 1, 2, 12),
    }
    mock_parameter_search = MagicMock(return_value=mock_hyperparameters)
    mock_forecast_creation = MagicMock(
        return_value=pd.DataFrame(
            data={"SpendMonth": datetime.date(2020, 1, 1), "ForecastAmount": 1000.0},
            index=[0],
        )
    )

    # subtitute two methods with mock , one is the search for hyperparameter, and one is the method that actually running the forecast with the hyperparameter
    model.find_best_hyperparameter_for_single_combination = mock_parameter_search
    model.create_forecast_with_given_hyperparameter = mock_forecast_creation

    input_df = pd.DataFrame(
        # fmt: off
            data={
                # four combinations, each with 12 months data
                'SpendMonth': [datetime.date(2022, month, 1) for month in range(1, 12 + 1)] * 4,
                'Category': ['Test Category A'] * 24 + ['Test Category B'] * 24,
                'MarketSector': ['Health'] * 12 + ['Education'] * 12 + ['Health'] * 12 + ['Education'] * 12,
                'EvidencedSpend': [1000.0, 1200.0, 1400.0, 1200.0, 1000.0, 1200.0] * 8
            }
        # fmt : on
    )

    model.create_forecast(
        input_df=input_df, months_to_forecast=12, start_month=datetime.date(2023, 1, 1)
    )

    # assert running create_forecast called the search for hyperparameter 4 times
    expected = True
    actual = mock_parameter_search.called
    assert actual == expected

    expected = 4
    actual = mock_parameter_search.call_count
    assert actual == expected

    # check that it also calls the actual forecast method 4 times, with the params that mock_parameter_search supplied
    expected = True
    actual = mock_forecast_creation.called
    assert actual == expected

    expected = 4
    actual = mock_forecast_creation.call_count
    assert actual == expected

    # check that the hyperparameter that it use is same as the ones provided by mock_parameter_search
    actual_called_with_arguments = mock_forecast_creation.call_args.kwargs

    expected = mock_parameter_search.return_value["order"]
    actual = actual_called_with_arguments["order"]
    assert actual == expected

    expected = mock_parameter_search.return_value["seasonal_order"]
    actual = actual_called_with_arguments["seasonal_order"]

    assert actual == expected
