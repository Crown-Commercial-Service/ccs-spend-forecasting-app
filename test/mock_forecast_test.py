from base_test import ReusableSparkTestCase
from pyspark.sql import Row, functions as F
import datetime
from dateutil.relativedelta import relativedelta

from pipeline.jobs.mock_forecast import mock_forecast_single_combination


class MockForecastTest(ReusableSparkTestCase):
    def test_forecast_for_one_combination(self):
        input_df = self.spark.createDataFrame(
            # fmt: off
            data=[
                Row(SpendMonth=datetime.date(2022, 1, 1), Category='Test Category', MarketSector='Health', EvidencedSpend=1234.0),
                Row(SpendMonth=datetime.date(2022, 2, 1), Category='Test Category', MarketSector='Health', EvidencedSpend=2234.0),
                Row(SpendMonth=datetime.date(2022, 3, 1), Category='Test Category', MarketSector='Health', EvidencedSpend=3456.12),
                Row(SpendMonth=datetime.date(2022, 4, 1), Category='Test Category', MarketSector='Health', EvidencedSpend=876.5),
            ]
            # fmt: on
        )

        expected_forecast_period = [datetime.date(2022, 4, 1) + relativedelta(months=m) for m in range(1, 12 + 1)]

        expected_column_names = ['SpendMonth', 'Category', 'MarketSector', 'ForecastSpend']

        actual = mock_forecast_single_combination(
            input_df=input_df,
            months_to_forecast=12,
            date_column="SpendMonth",
            amount_column="EvidencedSpend",
        )

        assert actual.count() == 12
        assert sorted(actual.columns) == sorted(expected_column_names)

        dates_in_output_df = [row['SpendMonth'] for row in actual.collect()]
        for date in expected_forecast_period:
            assert date in dates_in_output_df


    def test_forecast_for_multiple_combinations(self):
        input_df = self.spark.createDataFrame(
            # fmt: off
            data=[
                Row(SpendMonth=datetime.date(2022, 1, 1), Category='Construction', MarketSector='Health', EvidencedSpend=1234.0),
                Row(SpendMonth=datetime.date(2022, 2, 1), Category='Construction', MarketSector='Infrastructure', EvidencedSpend=2234.0),
                Row(SpendMonth=datetime.date(2022, 3, 1), Category='Workplace', MarketSector='Health', EvidencedSpend=3456.12),
                Row(SpendMonth=datetime.date(2022, 4, 1), Category='Workplace', MarketSector='Infrastructure', EvidencedSpend=876.5),
            ]
            # fmt: on
        )

        expected_column_names = ['SpendMonth', 'Category', 'MarketSector', 'ForecastSpend']
        expected_output_combinations = [
            {'Category':'Construction', 'MarketSector' : 'Health', 'start_month': datetime.date(2022, 2, 1)},
            {'Category':'Construction', 'MarketSector' : 'Infrastructure', 'start_month': datetime.date(2022, 3, 1)},
            {'Category':'Workplace', 'MarketSector' : 'Health', 'start_month': datetime.date(2022, 4, 1)},
            {'Category':'Workplace', 'MarketSector' : 'Infrastructure', 'start_month': datetime.date(2022, 5, 1)}
        ]

        actual = mock_forecast_single_combination(
            input_df=input_df,
            months_to_forecast=12,
            date_column="SpendMonth",
            amount_column="EvidencedSpend",
        )

        assert actual.count() == 48  # 4 combinations * 12 months

        for combination in expected_output_combinations:
            forecasts_of_this_combination = actual.filter( (F.col('Category') == combination['Category']) &  (F.col('MarketSector') == combination['MarketSector']) )
            assert forecasts_of_this_combination.count() == 12

            expected_forecast_period = [combination['start_month'] + relativedelta(months=m) for m in range(12)]
            
            for date in expected_forecast_period:
                assert forecasts_of_this_combination.filter( F.col('SpendMonth') == date).count() == 1


  