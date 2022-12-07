from unittest import TestCase
import datetime
import pandas as pd
from dateutil.relativedelta import relativedelta

from pipeline.jobs.mock_forecast import (
    MockForecastModel,
)


# class MockForecastTest(ReusableSparkTestCase):
#     def test_forecast_for_one_combination(self):
#         """Test creating mock forecast for just one combination."""
#         input_df = self.spark.createDataFrame(
#             # fmt: off
#             data=[
#                 Row(SpendMonth=datetime.date(2022, 4, 1), Category='Test Category', MarketSector='Health', EvidencedSpend=1000.0),
#                 Row(SpendMonth=datetime.date(2022, 4, 1), Category='Test Category', MarketSector='Health', EvidencedSpend=1000.0),
#                 Row(SpendMonth=datetime.date(2022, 4, 1), Category='Test Category', MarketSector='Health', EvidencedSpend=1000.0),
#                 Row(SpendMonth=datetime.date(2022, 4, 1), Category='Test Category', MarketSector='Health', EvidencedSpend=1000.0),
#             ]
#             # fmt: on
#         )

#         expected_forecast_period = [
#             datetime.date.today().replace(day=1) + relativedelta(months=m + 1)
#             for m in range(12)
#         ]

#         expected_column_names = [
#             "SpendMonth",
#             "Category",
#             "MarketSector",
#             "ForecastSpend",
#         ]

#         actual = create_mock_forecast(
#             input_df=input_df,
#             months_to_forecast=12,
#             columns_to_consider=["Category", "MarketSector"],
#             date_column="SpendMonth",
#             amount_column="EvidencedSpend",
#         )

#         assert actual.count() == 12
#         assert sorted(actual.columns) == sorted(expected_column_names)

#         dates_in_output_df = [row["SpendMonth"] for row in actual.collect()]
#         for date in expected_forecast_period:
#             assert date in dates_in_output_df

#     def test_forecast_for_multiple_combinations(self):
#         """Test creating mock forecast for multiple combinations of category / marketsector."""
#         input_df = self.spark.createDataFrame(
#             # fmt: off
#             data=[
#                 Row(SpendMonth=datetime.date(2022, 1, 1), Category='Construction', MarketSector='Health', EvidencedSpend=1234.0),
#                 Row(SpendMonth=datetime.date(2022, 2, 1), Category='Construction', MarketSector='Health', EvidencedSpend=1234.0),
#                 Row(SpendMonth=datetime.date(2022, 3, 1), Category='Construction', MarketSector='Infrastructure', EvidencedSpend=2234.0),
#                 Row(SpendMonth=datetime.date(2022, 4, 1), Category='Construction', MarketSector='Infrastructure', EvidencedSpend=2234.0),
#                 Row(SpendMonth=datetime.date(2022, 5, 1), Category='Workplace', MarketSector='Health', EvidencedSpend=3456.12),
#                 Row(SpendMonth=datetime.date(2022, 6, 1), Category='Workplace', MarketSector='Health', EvidencedSpend=3456.12),
#                 Row(SpendMonth=datetime.date(2022, 7, 1), Category='Workplace', MarketSector='Education', EvidencedSpend=876.5),
#                 Row(SpendMonth=datetime.date(2022, 8, 1), Category='Workplace', MarketSector='Education', EvidencedSpend=876.5),
#             ]
#             # fmt: on
#         )

#         expected_column_names = [
#             "SpendMonth",
#             "Category",
#             "MarketSector",
#             "ForecastSpend",
#         ]
#         expected_output_combinations = [
#             {
#                 "Category": "Construction",
#                 "MarketSector": "Health",
#             },
#             {
#                 "Category": "Construction",
#                 "MarketSector": "Infrastructure",
#             },
#             {
#                 "Category": "Workplace",
#                 "MarketSector": "Health",
#             },
#             {
#                 "Category": "Workplace",
#                 "MarketSector": "Education",
#             },
#         ]
#         expected_forecast_period = [
#             datetime.date.today().replace(day=1) + relativedelta(months=m + 1)
#             for m in range(12)
#         ]

#         actual = create_mock_forecast(
#             input_df=input_df,
#             months_to_forecast=12,
#             columns_to_consider=["Category", "MarketSector"],
#             date_column="SpendMonth",
#             amount_column="EvidencedSpend",
#         )

#         assert actual.count() == 48  # 4 combinations * 12 months = 48 output rows
#         assert sorted(actual.columns) == sorted(expected_column_names)

#         for combination in expected_output_combinations:
#             forecasts_of_this_combination = actual.filter(
#                 (F.col("Category") == combination["Category"])
#                 & (F.col("MarketSector") == combination["MarketSector"])
#             )
#             assert forecasts_of_this_combination.count() == 12
#             for date in expected_forecast_period:
#                 assert (
#                     forecasts_of_this_combination.filter(
#                         F.col("SpendMonth") == date
#                     ).count()
#                     == 1
#                 )

#     def test_forecast_for_given_start_month(self):

#         input_df = self.spark.createDataFrame(
#             # fmt: off
#             data=[
#                 Row(SpendMonth=datetime.date(2022, 1, 1), Category='Workplace', MarketSector='Health', EvidencedSpend=1234.0),
#             ]
#             # fmt: on
#         )
#         start_month = datetime.date(2024, 10, 1)
#         expected_forecast_period = [
#             datetime.date(2024, m, 1) for m in range(10, 12 + 1)
#         ] + [datetime.date(2025, m, 1) for m in range(1, 10)]

#         actual = create_mock_forecast(
#             input_df=input_df,
#             months_to_forecast=12,
#             columns_to_consider=["Category", "MarketSector"],
#             start_month=start_month,
#         )

#         assert actual.count() == 12

#         dates_in_output_df = [row["SpendMonth"] for row in actual.collect()]
#         assert sorted(dates_in_output_df) == expected_forecast_period

#     def test_forecast_period_of_non_12_months(self):
#         input_df = self.spark.createDataFrame(
#             # fmt: off
#             data=[
#                 Row(SpendMonth=datetime.date(2022, 1, 1), Category='Workplace', MarketSector='Health', EvidencedSpend=1234.0),
#                 Row(SpendMonth=datetime.date(2022, 2, 1), Category='Workplace', MarketSector='Health', EvidencedSpend=0.0),
#                 Row(SpendMonth=datetime.date(2022, 3, 1), Category='Workplace', MarketSector='Education', EvidencedSpend=0.0),
#                 Row(SpendMonth=datetime.date(2022, 4, 1), Category='Network Service', MarketSector='Education', EvidencedSpend=0.0),
#             ]
#             # fmt: on
#         )
#         actual = create_mock_forecast(
#             input_df=input_df,
#             months_to_forecast=36,
#             columns_to_consider=["Category", "MarketSector"],
#         )

#         assert (
#             actual.count() == 36 * 3
#         )  # Workplace & Health got two rows, so it is 3 combinations * 36 months

#     def test_handle_unrelated_columns(self):
#         """Test that the function works correctly for input data with unrelated columns"""
#         input_df = self.spark.createDataFrame(
#             # fmt: off
#             data=[
#                 Row(CustomerURN=12345, FinancialYear='FY2021/22', SpendMonth=datetime.date(2022, 1, 1), Category='Test Category', MarketSector='Health', EvidencedSpend=1234.0),
#                 Row(CustomerURN=67890, FinancialYear='FY2021/22', SpendMonth=datetime.date(2022, 2, 1), Category='Test Category', MarketSector='Health', EvidencedSpend=2234.0),
#                 Row(CustomerURN=13579, FinancialYear='FY2021/22', SpendMonth=datetime.date(2022, 3, 1), Category='Test Category', MarketSector='Health', EvidencedSpend=3456.12),
#                 Row(CustomerURN=24680, FinancialYear='FY2022/21', SpendMonth=datetime.date(2022, 4, 1), Category='Test Category', MarketSector='Health', EvidencedSpend=876.5),
#             ]
#             # fmt: on
#         )

#         expected_forecast_period = [
#             datetime.date.today().replace(day=1) + relativedelta(months=m + 1)
#             for m in range(12)
#         ]
#         expected_column_names = [
#             "SpendMonth",
#             "Category",
#             "MarketSector",
#             "ForecastSpend",
#         ]

#         actual = create_mock_forecast(
#             input_df=input_df,
#             months_to_forecast=12,
#             columns_to_consider=["Category", "MarketSector"],
#             date_column="SpendMonth",
#             amount_column="EvidencedSpend",
#         )

#         assert actual.count() == 12
#         assert sorted(actual.columns) == sorted(expected_column_names)

#         dates_in_output_df = [row["SpendMonth"] for row in actual.collect()]
#         for date in expected_forecast_period:
#             assert date in dates_in_output_df

#     def test_mock_forecast_random_range(self):
#         """Test that the mock forecast output value within a given random range."""
#         input_df = self.spark.createDataFrame(
#             # fmt: off
#             data=[
#                 Row(SpendMonth=datetime.date(2022, 4, 1), Category='Category A', MarketSector='Health', EvidencedSpend=1000.0),
#                 Row(SpendMonth=datetime.date(2022, 4, 1), Category='Category A', MarketSector='Health', EvidencedSpend=3000.0),
#             ]
#             # fmt: on
#         )

#         test_inputs = [0.15, 0.2, 0.3, 0.5, 0.6]

#         for input_randomness in test_inputs:

#             expected_lower_limit = (1000 + 3000) * (1 - input_randomness)
#             expected_upper_limit = (1000 + 3000) * (1 + input_randomness)

#             actual = create_mock_forecast(
#                 input_df=input_df,
#                 months_to_forecast=36,
#                 columns_to_consider=["Category", "MarketSector"],
#                 date_column="SpendMonth",
#                 amount_column="EvidencedSpend",
#                 randomness=input_randomness,
#             )

#             for row in actual.collect():
#                 assert row["ForecastSpend"] <= expected_upper_limit
#                 assert row["ForecastSpend"] >= expected_lower_limit


# class AggregateSpendsByMonth(ReusableSparkTestCase):
#     def test_aggregate_spends(self):
#         """Test creating mock forecast for just one combination."""
#         input_df = self.spark.createDataFrame(
#             # fmt: off
#             data=[
#                 Row(CustomerURN="abcdef", InvoiceNumber='dw9jj-adkak', SpendMonth=datetime.date(2022, 1, 1), Category='Network Service', MarketSector='Health', EvidencedSpend=100.0),
#                 Row(CustomerURN="d09ais", InvoiceNumber='o9jas-10k93', SpendMonth=datetime.date(2022, 1, 1), Category='Network Service', MarketSector='Health', EvidencedSpend=200.0),
#                 Row(CustomerURN="r2jjem", InvoiceNumber='dw9jj-i0da0', SpendMonth=datetime.date(2022, 2, 1), Category='Network Service', MarketSector='Health', EvidencedSpend=300.0),
#                 Row(CustomerURN="29ejin", InvoiceNumber='929ej-0di0d', SpendMonth=datetime.date(2022, 2, 1), Category='Network Service', MarketSector='Health', EvidencedSpend=400.0),
#                 Row(CustomerURN="ejqidj", InvoiceNumber='0ix0s-0akk0', SpendMonth=datetime.date(2022, 2, 1), Category='Office Supplies', MarketSector='Health', EvidencedSpend=500.0),
#                 Row(CustomerURN="0xajos", InvoiceNumber='xak09-xa0j0', SpendMonth=datetime.date(2022, 2, 1), Category='Office Supplies', MarketSector='Health', EvidencedSpend=600.0),
#                 Row(CustomerURN="1au9ji", InvoiceNumber='ax0ix-0ka0x', SpendMonth=datetime.date(2022, 3, 1), Category='Office Supplies', MarketSector='Health', EvidencedSpend=700.0),
#                 Row(CustomerURN="i0xajk", InvoiceNumber='0xka0-ax0i0', SpendMonth=datetime.date(2022, 1, 1), Category='Office Supplies', MarketSector='Education', EvidencedSpend=800.0),
#                 Row(CustomerURN="cmlam0", InvoiceNumber='-axk--0aas0', SpendMonth=datetime.date(2022, 1, 1), Category='Office Supplies', MarketSector='Education', EvidencedSpend=900.0),
#             ]
#             # fmt: on
#         )

#         expected = self.spark.createDataFrame(
#             # fmt: off
#             data=[
#                 Row(SpendMonth=datetime.date(2022, 1, 1), Category='Network Service', MarketSector='Health', EvidencedSpend=300.0),
#                 Row(SpendMonth=datetime.date(2022, 2, 1), Category='Network Service', MarketSector='Health', EvidencedSpend=700.0),
#                 Row(SpendMonth=datetime.date(2022, 2, 1), Category='Office Supplies', MarketSector='Health', EvidencedSpend=1100.0),
#                 Row(SpendMonth=datetime.date(2022, 3, 1), Category='Office Supplies', MarketSector='Health', EvidencedSpend=700.0),
#                 Row(SpendMonth=datetime.date(2022, 1, 1), Category='Office Supplies', MarketSector='Education', EvidencedSpend=1700.0),
#             ]
#             # fmt: on
#         )

#         actual = aggregate_spends_by_month(
#             input_df=input_df,
#             columns_to_consider=["Category", "MarketSector"],
#             date_column="SpendMonth",
#             amount_column="EvidencedSpend",
#         )

#         diff = actual.exceptAll(expected)

#         assert diff.count() == 0


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
