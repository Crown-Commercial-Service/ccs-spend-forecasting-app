from base_test import ReusableSparkTestCase
from pyspark.sql import Row, functions as F
import datetime
from dateutil.relativedelta import relativedelta

from pipeline.jobs.mock_forecast import create_mock_forecast


class MockForecastTest(ReusableSparkTestCase):
    def test_forecast_for_one_combination(self):
        """Test creating mock forecast for just one combination."""
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

        actual = create_mock_forecast(
            input_df=input_df,
            months_to_forecast=12,
            columns_to_consider=["Category", "MarketSector"],
            date_column="SpendMonth",
            amount_column="EvidencedSpend",
        )

        assert actual.count() == 12
        assert sorted(actual.columns) == sorted(expected_column_names)

        dates_in_output_df = [row["SpendMonth"] for row in actual.collect()]
        for date in expected_forecast_period:
            assert date in dates_in_output_df

    def test_forecast_for_multiple_combinations(self):
        """Test creating mock forecast for multiple combinations of category / marketsector."""
        input_df = self.spark.createDataFrame(
            # fmt: off
            data=[
                Row(SpendMonth=datetime.date(2022, 1, 1), Category='Construction', MarketSector='Health', EvidencedSpend=1234.0),
                Row(SpendMonth=datetime.date(2022, 2, 1), Category='Construction', MarketSector='Health', EvidencedSpend=1234.0),
                Row(SpendMonth=datetime.date(2022, 3, 1), Category='Construction', MarketSector='Infrastructure', EvidencedSpend=2234.0),
                Row(SpendMonth=datetime.date(2022, 4, 1), Category='Construction', MarketSector='Infrastructure', EvidencedSpend=2234.0),
                Row(SpendMonth=datetime.date(2022, 5, 1), Category='Workplace', MarketSector='Health', EvidencedSpend=3456.12),
                Row(SpendMonth=datetime.date(2022, 6, 1), Category='Workplace', MarketSector='Health', EvidencedSpend=3456.12),
                Row(SpendMonth=datetime.date(2022, 7, 1), Category='Workplace', MarketSector='Education', EvidencedSpend=876.5),
                Row(SpendMonth=datetime.date(2022, 8, 1), Category='Workplace', MarketSector='Education', EvidencedSpend=876.5),
            ]
            # fmt: on
        )

        expected_column_names = [
            "SpendMonth",
            "Category",
            "MarketSector",
            "ForecastSpend",
        ]
        expected_output_combinations = [
            {
                "Category": "Construction",
                "MarketSector": "Health",
            },
            {
                "Category": "Construction",
                "MarketSector": "Infrastructure",
            },
            {
                "Category": "Workplace",
                "MarketSector": "Health",
            },
            {
                "Category": "Workplace",
                "MarketSector": "Education",
            },
        ]
        expected_forecast_period = [
            datetime.date.today().replace(day=1) + relativedelta(months=m + 1)
            for m in range(12)
        ]

        actual = create_mock_forecast(
            input_df=input_df,
            months_to_forecast=12,
            columns_to_consider=["Category", "MarketSector"],
            date_column="SpendMonth",
            amount_column="EvidencedSpend",
        )

        assert actual.count() == 48  # 4 combinations * 12 months = 48 output rows
        assert sorted(actual.columns) == sorted(expected_column_names)

        for combination in expected_output_combinations:
            forecasts_of_this_combination = actual.filter(
                (F.col("Category") == combination["Category"])
                & (F.col("MarketSector") == combination["MarketSector"])
            )
            assert forecasts_of_this_combination.count() == 12
            for date in expected_forecast_period:
                assert (
                    forecasts_of_this_combination.filter(
                        F.col("SpendMonth") == date
                    ).count()
                    == 1
                )

    def test_forecast_for_given_start_month(self):

        input_df = self.spark.createDataFrame(
            # fmt: off
            data=[
                Row(SpendMonth=datetime.date(2022, 1, 1), Category='Workplace', MarketSector='Health', EvidencedSpend=1234.0),
            ]
            # fmt: on
        )
        start_month = datetime.date(2024, 10, 1)
        expected_forecast_period = [
            datetime.date(2024, m, 1) for m in range(10, 12 + 1)
        ] + [datetime.date(2025, m, 1) for m in range(1, 10)]

        actual = create_mock_forecast(
            input_df=input_df,
            months_to_forecast=12,
            columns_to_consider=["Category", "MarketSector"],
            start_month=start_month,
        )

        assert actual.count() == 12

        dates_in_output_df = [row["SpendMonth"] for row in actual.collect()]
        assert sorted(dates_in_output_df) == expected_forecast_period

    def test_forecast_period_of_non_12_months(self):
        input_df = self.spark.createDataFrame(
            # fmt: off
            data=[
                Row(SpendMonth=datetime.date(2022, 1, 1), Category='Workplace', MarketSector='Health', EvidencedSpend=1234.0),
                Row(SpendMonth=datetime.date(2022, 2, 1), Category='Workplace', MarketSector='Health', EvidencedSpend=0.0),
                Row(SpendMonth=datetime.date(2022, 3, 1), Category='Workplace', MarketSector='Education', EvidencedSpend=0.0),
                Row(SpendMonth=datetime.date(2022, 4, 1), Category='Network Service', MarketSector='Education', EvidencedSpend=0.0),
            ]
            # fmt: on
        )
        actual = create_mock_forecast(
            input_df=input_df,
            months_to_forecast=36,
            columns_to_consider=["Category", "MarketSector"],
        )

        assert (
            actual.count() == 36 * 3
        )  # Workplace & Health got two rows, so it is 3 combinations * 36 months

    def test_handle_unrelated_columns(self):
        """Test that the function works correctly for input data with unrelated columns"""
        input_df = self.spark.createDataFrame(
            # fmt: off
            data=[
                Row(CustomerURN=12345, FinancialYear='FY2021/22', SpendMonth=datetime.date(2022, 1, 1), Category='Test Category', MarketSector='Health', EvidencedSpend=1234.0),
                Row(CustomerURN=67890, FinancialYear='FY2021/22', SpendMonth=datetime.date(2022, 2, 1), Category='Test Category', MarketSector='Health', EvidencedSpend=2234.0),
                Row(CustomerURN=13579, FinancialYear='FY2021/22', SpendMonth=datetime.date(2022, 3, 1), Category='Test Category', MarketSector='Health', EvidencedSpend=3456.12),
                Row(CustomerURN=24680, FinancialYear='FY2022/21', SpendMonth=datetime.date(2022, 4, 1), Category='Test Category', MarketSector='Health', EvidencedSpend=876.5),
            ]
            # fmt: on
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

        actual = create_mock_forecast(
            input_df=input_df,
            months_to_forecast=12,
            columns_to_consider=["Category", "MarketSector"],
            date_column="SpendMonth",
            amount_column="EvidencedSpend",
        )

        assert actual.count() == 12
        assert sorted(actual.columns) == sorted(expected_column_names)

        dates_in_output_df = [row["SpendMonth"] for row in actual.collect()]
        for date in expected_forecast_period:
            assert date in dates_in_output_df
