import datetime
import pandas as pd
from pipeline.jobs.prophet_model import ProphetModel


def test_Prophet_model_initiation():
    model = ProphetModel()

    actual = isinstance(model, ProphetModel)
    expected = True

    assert expected == actual


def test_Profect_model_create_forecast():
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

    model = ProphetModel()
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


def test_prophet_model_create_forecast_with_multiple_combinations():
    """Test that calling create_forecast with input data of multiple Category/MarketSector combination 
    will generate n months forecast for each combination"""
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

    model = ProphetModel()

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
