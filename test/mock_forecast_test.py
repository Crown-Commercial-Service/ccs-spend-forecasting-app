from unittest import TestCase
import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta

from pipeline.jobs.mock_forecast import (
    MockForecastModel,
)


class TestMockForecastModel(TestCase):
    def test_forecast_for_one_combination(self):
        """Reimplement in pandas. Test creating mock forecast for just one combination."""
        input_df = pd.DataFrame(
            # fmt: off
            data=[
                [datetime.date(2022, 4, 1), 'Test Category', 'Health', 1000.0],
                [datetime.date(2022, 4, 1), 'Test Category', 'Health', 1000.0],
                [datetime.date(2022, 4, 1), 'Test Category', 'Health', 1000.0],
                [datetime.date(2022, 4, 1), 'Test Category', 'Health', 1000.0],
            ],
            # fmt: on
            columns=["SpendMonth", "Category", "MarketSector", "EvidencedSpend"]
        )

        expected_forecast_period = [
            datetime.date.today().replace(day=1) + relativedelta(months=m + 1)
            for m in range(12)
        ]

        expected_column_names = [
            "SpendMonth",
            "Category",
            "MarketSector",
            "ForecastSpend",
        ]

        mock_model = MockForecastModel()
        actual = mock_model.create_forecast(
            input_df=input_df,
            months_to_forecast=12,
        )

        assert len(actual) == 12
        assert sorted(actual.columns) == sorted(expected_column_names)

        self.assertListEqual(expected_forecast_period, sorted(actual["SpendMonth"]))

    def test_forecast_for_multiple_combinations(self):
        """Reimplement in pandas. Test creating mock forecast for multiple combinations of category / marketsector."""

        # Arrange
        input_df = pd.DataFrame(
            # fmt: off
            data={
                'SpendMonth': [datetime.date(2021, month, 1) for month in range(1, 12 + 1)] * 2,
                'Category': ['Construction'] * 12 + ['Workplace'] * 12,
                'MarketSector': ['Health'] * 6 + ['Infrastructure'] * 6 + ['Health'] * 6 + ['Education'] * 6,
                'EvidencedSpend': [1000] * 6 + [2000]* 6 + [3000] * 6 + [4000]* 6
            }
            # fmt: on
        )
        start_month = datetime.date(2022, 1, 1)

        expected_column_names = [
            "SpendMonth",
            "Category",
            "MarketSector",
            "ForecastSpend",
        ]
        expected_output_combinations = [
            ("Construction", "Health"),
            ("Construction", "Infrastructure"),
            ("Workplace", "Health"),
            ("Workplace", "Education"),
        ]

        expected_forecast_period = [
            datetime.date(2022, month, 1) for month in range(1, 12 + 1)
        ]

        # Act
        mock_model = MockForecastModel()
        actual = mock_model.create_forecast(
            input_df=input_df, months_to_forecast=12, start_month=start_month
        )

        # Assert
        assert len(actual) == 48  # 4 combinations * 12 months = 48 output rows
        assert sorted(actual.columns) == sorted(expected_column_names)

        for category, market_sector in expected_output_combinations:
            rows_for_this_combination = actual[
                (actual["Category"] == category)
                & (actual["MarketSector"] == market_sector)
            ]
            # each combination should get 12 months prediction
            assert len(rows_for_this_combination) == 12
            assert (
                sorted(rows_for_this_combination["SpendMonth"])
                == expected_forecast_period
            )

    def test_handle_unrelated_columns(self):
        """Test that the function works correctly for input data with unrelated columns"""
        input_df = pd.DataFrame(
            # fmt: off
            data={
                'SpendMonth': [datetime.date(2021, month, 1) for month in range(1, 12 + 1)],
                'Category': ['Construction'] * 12,
                'MarketSector': ['Health'] * 12,
                'EvidencedSpend': [1000] * 12,
                'CustomerInvoiceNo': ['ABC123456'] * 12,
                'Unrelated Columns With Whitespaces': ['foo'] * 12
            }
            # fmt: on
        )
        start_month = datetime.date(2022, 1, 1)

        expected_forecast_period = [
            datetime.date(2022, month, 1) for month in range(1, 12 + 1)
        ]

        expected_column_names = [
            "SpendMonth",
            "Category",
            "MarketSector",
            "ForecastSpend",
        ]

        mock_model = MockForecastModel()
        actual = mock_model.create_forecast(
            input_df=input_df, months_to_forecast=12, start_month=start_month
        )

        assert len(actual) == 12
        assert sorted(actual.columns) == sorted(expected_column_names)
        assert sorted(actual["SpendMonth"]) == expected_forecast_period

    def test_mock_forecast_random_range(self):
        """Test that the mock forecast output value within a given random range."""
        input_df = pd.DataFrame(
            # fmt: off
            data={
                'SpendMonth': [datetime.date(2022, month, 1) for month in range(1, 12 + 1)],
                'Category': ['Workplace'] * 12,
                'MarketSector': ['Health'] * 12,
                'EvidencedSpend': [1000.0, 2000.0, 3000.0, 4000.0] * 3
            }
            # fmt: on
        )

        test_input_randomness = [0.15, 0.2, 0.3, 0.5, 0.6]

        for input_randomness in test_input_randomness:

            # mean of input data is 2500.0, expect all forecast value within this range
            expected_lower_limit = (2500) * (1 - input_randomness)
            expected_upper_limit = (2500) * (1 + input_randomness)

            model = MockForecastModel(randomness=input_randomness)
            actual = model.create_forecast(input_df=input_df, months_to_forecast=36)

            assert (expected_lower_limit <= actual["ForecastSpend"]).all()
            assert (actual["ForecastSpend"] <= expected_upper_limit).all()
