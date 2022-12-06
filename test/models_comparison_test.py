import datetime
import pandas as pd

from base_test import ReusableSparkTestCase

from pipeline.jobs.mock_forecast import (
    create_mock_forecast,
    create_mock_forecast_pandas,
    create_mock_model,
)
from pipeline.jobs.models_comparison import create_models_comparison


class ModelComparisonTest(ReusableSparkTestCase):
    def test_for_one_combination(self):
        """Basic test for comparing models with only one Category/MarketSector combination in input data"""
        input_df = pd.DataFrame(
            # fmt: off
            data={
                'SpendMonth': [datetime.date(2022, 1, 1), datetime.date(2022, 2, 1), datetime.date(2022, 3, 1), datetime.date(2022, 4, 1), datetime.date(2022, 5, 1), datetime.date(2022, 6, 1)],
                'Category': ['Test Category'] * 6,
                'MarketSector': ['Health'] * 6,
                'EvidencedSpend': [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
            }
        )

        models = {
            "Model A": create_mock_model(randomness=0.8),
            "Model B": create_mock_model(randomness=0.05),
        }

        expected_output_format = pd.DataFrame(
            # fmt: off
            data={
                'Category': ['Test Category', 'Test Category'],
                'MarketSector': ['Health', 'Health'],
                'EvidencedSpend': [1000.0, 1000.0],
                'SpendMonth': [datetime.date(2022, 5, 1), datetime.date(2022, 6, 1)],
                ## the below figures are just for example. Not to be campared with actual output as randomness are involved.
                'Model A Forecast': [1300.0, 700.0],
                'Model A Error %': [0.3, 0.3],
                'Model A MAE %': [0.3, 0.3],
                'Model B Forecast': [950.0, 1050],
                'Model B Error %': [0.05, 0.05],
                'Model B MAE %': [0.05, 0.05],
            }
        )

        actual = create_models_comparison(
            input_df=input_df,
            train_ratio=0.67,
            models=models,
        )

        self.assertListEqual(
            sorted(expected_output_format.columns), sorted(actual.columns)
        )

        self.assertListEqual(
            sorted(actual["Category"]), sorted(expected_output_format["Category"])
        )

        self.assertListEqual(
            sorted(actual["MarketSector"]),
            sorted(expected_output_format["MarketSector"]),
        )

        for value in actual["Model A MAE %"]:
            self.assertTrue(0 < value < 0.8)

        for value in actual["Model B MAE %"]:
            self.assertTrue(0 < value < 0.05)

    def test_for_multiple_combinations(self):
        """Basic test for comparing models with multiple Category/MarketSector combination in input data"""
        input_df = pd.DataFrame(
            # fmt: off
            data={
                # In total 48 rows, which are 12 rows * 4 combinations. The spend values are simplified for testing purpose.
                'SpendMonth': [datetime.date(2021, month, 1) for month in range(1, 12 + 1)] * 4,
                'Category': ['Construction'] * 24 + ['Digital Future'] * 24,
                'MarketSector': ['Health'] * 12 + ['Education']  * 12 + ['Health'] * 12 + ['Education']  * 12,
                'EvidencedSpend': [1000] * 12 + [2000] * 12 + [3000] * 12 + [4000] * 12
            }
        )

        train_size = 8
        prediction_size = 4
        models = {
            "Model A": create_mock_model(randomness=0.8),
            "Model B": create_mock_model(randomness=0.05),
        }

        expected_columns = [
            "Category",
            "MarketSector",
            "EvidencedSpend",
            "SpendMonth",
            "Model A Forecast",
            "Model A Error %",
            "Model A MAE %",
            "Model B Forecast",
            "Model B Error %",
            "Model B MAE %",
        ]

        actual = create_models_comparison(
            input_df=input_df,
            train_ratio=0.9,
            models=models,
        )

        self.assertListEqual(sorted(expected_columns), sorted(actual.columns))

        for value in actual["Model A MAE %"]:
            self.assertTrue(0 < value < 0.8)

        for value in actual["Model B MAE %"]:
            self.assertTrue(0 < value < 0.05)
