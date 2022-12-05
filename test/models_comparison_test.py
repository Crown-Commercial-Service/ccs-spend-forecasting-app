import datetime
import pandas as pd

from base_test import ReusableSparkTestCase

from pipeline.jobs.mock_forecast import create_mock_forecast, create_mock_forecast_pandas, create_mock_model
from pipeline.jobs.models_comparison import create_models_comparison


class ModelComparisonTest(ReusableSparkTestCase):
    def test_for_one_combination(self):
        """ Basic test to generate model comparison for only one Category/MarketSector combination"""
        input_df = pd.DataFrame(
            # fmt: off
            data={
                'SpendMonth': [datetime.date(2022, 1, 1), datetime.date(2022, 2, 1), datetime.date(2022, 3, 1), datetime.date(2022, 4, 1), datetime.date(2022, 5, 1), datetime.date(2022, 6, 1)],
                'Category': ['Test Category'] * 6,
                'MarketSector': ['Health'] * 6,
                'EvidencedSpend': [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
            }
        )

        train_size = 4
        models = {
            'Model A': create_mock_model(randomness=0.8), 
            'Model B': create_mock_model(randomness=0.05)
        }


        expected_output_format = pd.DataFrame(
            # fmt: off
            data={
                'Category': ['Test Category'],
                'MarketSector': ['Health'],
                'Model A Forecast': [1300.0],
                'Model B Forecast': [950.0],
                'Model A Accuracy': [0.3],
                'Model B Accuracy': [0.05],
                'Model Chosen': ['Model B'],
            }
        )

        actual = create_models_comparison(
            input_df=input_df,
            train_size=train_size,
            models=models
        )

        assert list(expected_output_format.columns) == list(actual.columns)

        assert list(actual['Category']) == ['Test Category']
        assert list(actual['MarketSector']) == ['Health']
        
        for value in actual['Model A Accuracy']:
            self.assertTrue(value > 0.2)

        for value in actual['Model B Accuracy']:
            self.assertTrue(value > 0.95)


            