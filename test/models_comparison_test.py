import datetime
import pandas as pd
import numpy as np
from random import random
from unittest import TestCase
from unittest.mock import Mock

from pipeline.jobs.mock_forecast import create_mock_model
from pipeline.jobs.models_comparison import (
    create_models_comparison,
    fill_in_model_suggestion,
)


class ModelComparisonTest(TestCase):
    def test_for_one_combination(self):
        """Basic test for comparing models with only one Category/MarketSector combination in input data"""
        input_df = pd.DataFrame(
            # fmt: off
            data={
                'SpendMonth': [datetime.date(2022, 1, 1), datetime.date(2022, 2, 1), datetime.date(2022, 3, 1), datetime.date(2022, 4, 1), datetime.date(2022, 5, 1), datetime.date(2022, 6, 1)],
                'Category': ['Test Category'] * 6,
                'MarketSector': ['Health'] * 6,
                'EvidencedSpend': [1000.0, 1200.0, 800.0, 1000.0, 1200.0, 800.0]
            }
        )

        models = [
            create_mock_model(name="Model A", randomness=0.8),
            create_mock_model(name="Model B", randomness=0.05),
        ]

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
                'Model A MAPE': [0.3, 0.3],
                'Model B Forecast': [950.0, 1050],
                'Model B Error %': [0.05, 0.05],
                'Model B MAPE': [0.05, 0.05],
                'Model Suggested': ['Model B', 'Model B']
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

        # since we only have one combination, all MAPE for the same model should be the same
        assert len(set(actual["Model A MAPE"])) == 1
        assert len(set(actual["Model B MAPE"])) == 1

    def test_for_multiple_combinations(self):
        """Basic test for comparing models with multiple Category/MarketSector combination in input data"""
        input_df = pd.DataFrame(
            # fmt: off
            data={
                # In total 48 rows, which are 12 rows * 4 combinations. The spend values are simplified for testing purpose.
                'SpendMonth': [datetime.date(2021, month, 1) for month in range(1, 12 + 1)] * 4,
                'Category': ['Construction'] * 24 + ['Digital Future'] * 24,
                'MarketSector': ['Health'] * 12 + ['Education']  * 12 + ['Health'] * 12 + ['Education']  * 12,
                'EvidencedSpend': [1000 + 100 * random() for _ in range(12)] + [2000 + 100 * random() for _ in range(12)] + [3000 + 100 * random() for _ in range(12)] + [4000 + 100 * random() for _ in range(12)]
            }
        )

        models = [
            create_mock_model(name="Model A", randomness=0.8),
            create_mock_model(name="Model B", randomness=0.05),
        ]

        expected_columns = [
            "Category",
            "MarketSector",
            "EvidencedSpend",
            "SpendMonth",
            "Model A Forecast",
            "Model A Error %",
            "Model A MAPE",
            "Model B Forecast",
            "Model B Error %",
            "Model B MAPE",
            "Model Suggested",
        ]

        actual = create_models_comparison(
            input_df=input_df,
            train_ratio=0.9,
            models=models,
        )

        self.assertListEqual(sorted(expected_columns), sorted(actual.columns))

        # for each combination of Category/MarketSector, assert the MAPE is the same
        # this is to check that we are calculating a separate score for each combination
        for category in input_df["Category"].unique():
            for market_sector in input_df["MarketSector"].unique():
                output_for_current_combination = actual[
                    (actual["Category"] == category)
                    & (actual["MarketSector"] == market_sector)
                ]
                assert len(set(output_for_current_combination["Model A MAPE"])) == 1
                assert len(set(output_for_current_combination["Model B MAPE"])) == 1

    def test_for_handling_na_data(self):
        """Test for handling input data with months of no spending, N/A or zero values"""
        input_df = pd.DataFrame(
            # fmt: off
            data={
                # No spending in November
                'SpendMonth': [datetime.date(2021, month, 1) for month in range(1, 10 + 1)] + [datetime.date(2021, 12, 1)],
                'Category': ['Test Category'] * 11,
                'MarketSector': ['Health'] * 11,
                # Some months got 0 spending or NaN spending
                'EvidencedSpend': [1000 + 100 * random() for _ in range(4)] + [np.NaN] + [0] + [1000 + 100 * random() for _ in range(4)] + [0]
            }
        )

        models = [
            create_mock_model(name="Model A", randomness=0.8),
            create_mock_model(name="Model B", randomness=0.05),
        ]

        actual = create_models_comparison(
            input_df=input_df,
            train_ratio=0.80,
            models=models,
        )

        # 11 * 0.8 = 8.8, so it is 8 months for training and 3 months for testing
        # actual data for Dec got zero, so only Sept and Oct is used for comparison
        assert list(actual["SpendMonth"]) == [
            datetime.date(2021, 9, 1),
            datetime.date(2021, 10, 1),
        ]

        assert all(actual["Model A MAPE"].notna())

    def test_for_model_suggestion(self):
        """Test that the comparison table provide a model suggestion for each combination base on measuring metric"""

        input_df = pd.DataFrame(
            # fmt: off
            data={
                'SpendMonth': [datetime.date(2021, month, 1) for month in range(1, 12 + 1)] * 4,
                'Category': ['Construction'] * 24 + ['Digital Future'] * 24,
                'MarketSector': ['Health'] * 12 + ['Education']  * 12 + ['Health'] * 12 + ['Education']  * 12,
                'EvidencedSpend': [1000 + 100 * random() for _ in range(12)] + [2000 + 100 * random() for _ in range(12)] + [3000 + 100 * random() for _ in range(12)] + [4000 + 100 * random() for _ in range(12)]
            }
            # fmt: on
        )

        models = [
            create_mock_model(name="Model A", randomness=0.1),
            create_mock_model(name="Model B", randomness=0.1),
        ]

        actual = create_models_comparison(
            input_df=input_df,
            train_ratio=0.9,
            models=models,
        )

        # fmt: off
        for category in ['Construction', 'Digital Future']:
            for market_sector in ['Health', 'Education']:
                rows_for_current_combination = actual[(actual["Category"] == category) & (actual["MarketSector"] == market_sector)]

                # check that for each combination, every row should suggest the same model
                assert len(set(rows_for_current_combination["Model Suggested"])) == 1

                # check that the suggestion agrees with the MAPE score. The model with lower MAPE should be chosen.
                first_row = rows_for_current_combination.head(1)

                if (first_row["Model A MAPE"] <= first_row["Model B MAPE"]).bool():
                    assert all(rows_for_current_combination["Model Suggested"] == "Model A")
                else:
                    assert all(rows_for_current_combination["Model Suggested"] == "Model B")
        # fmt: on

    def test_for_more_than_two_models(self):
        """Test that the function can compare more than two models"""

        input_df = pd.DataFrame(
            # fmt: off
            data={
                'SpendMonth': [datetime.date(2021, month, 1) for month in range(1, 12 + 1)] * 4,
                'Category': ['Construction'] * 24 + ['Digital Future'] * 24,
                'MarketSector': ['Health'] * 12 + ['Education']  * 12 + ['Health'] * 12 + ['Education']  * 12,
                'EvidencedSpend': [1000 + 100 * random() for _ in range(12)] + [2000 + 100 * random() for _ in range(12)] + [3000 + 100 * random() for _ in range(12)] + [4000 + 100 * random() for _ in range(12)]
            }
            # fmt: on
        )

        models = [
            create_mock_model(name="Model A", randomness=0.2),
            create_mock_model(name="Model B", randomness=0.2),
            create_mock_model(name="Model C", randomness=0.2),
        ]

        expected_column_names = [
            "Model A Forecast",
            "Model B Forecast",
            "Model C Forecast",
            "Model A MAPE",
            "Model B MAPE",
            "Model C MAPE",
        ]

        actual = create_models_comparison(
            input_df=input_df,
            train_ratio=0.8,
            models=models,
        )

        for column_name in expected_column_names:
            self.assertIn(column_name, list(actual.columns))

    def test_fill_in_model_suggestion(self):
        """Basic test for comparing models with only one Category/MarketSector combination in input data"""
        comparison_table = pd.DataFrame(
            # fmt: off
            data={
                'SpendMonth': [datetime.date(2022, 1, 1), datetime.date(2022, 2, 1), datetime.date(2022, 3, 1)],
                'Category': ['Test Category'] * 3,
                'MarketSector': ['Health'] * 3,
                'EvidencedSpend': [1000.0, 1200.0, 800.0],
                'Model A MAPE': [0.3, 0.4, 0.5],
                'Model B MAPE': [0.4, 0.4, 0.4]
            }
            # fmt: on
        )
        models = [
            create_mock_model(name="Model A", randomness=0.8),
            create_mock_model(name="Model B", randomness=0.05),
        ]

        output_df = fill_in_model_suggestion(
            comparison_table=comparison_table, models=models
        )

        assert "Model Suggested" in output_df.columns
        assert list(output_df["Model Suggested"]) == [
            "Model A",
            "Model A",
            "Model B",
        ]

    def test_handle_exception_from_model_class(self):
        """Test for handling exception raised by model when generating forecast"""
        input_df = pd.DataFrame(
            # fmt: off
            data={
                'SpendMonth': [datetime.date(2022, month, 1) for month in range(1, 11)],
                'Category': ['Test Category'] * 10,
                'MarketSector': ['Health'] * 10,
                'EvidencedSpend': [1000.0, 1200.0, 800.0, 1000.0, 1200.0, 800.0, 1000.0, 1200.0, 800.0, 1000.0]
            }
        )

        models = [
            create_mock_model(name="Model A", randomness=0.8),
            create_mock_model(name="Model B", randomness=0.05),
        ]

        # substitute model A's create forecast for a mock which raise error
        mock_method = Mock(side_effect=ValueError("Mock error for test"))
        models[0].create_forecast = mock_method

        output = create_models_comparison(
            input_df=input_df, train_ratio=0.9, models=models
        )

        # Assert Model A's forecast to be with filled with N/A
        expected = True
        actual = output["Model A Forecast"].isnull().all()

        assert expected == actual
