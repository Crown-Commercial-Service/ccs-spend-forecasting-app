from base_test import ReusableSparkTestCase
import datetime
from pyspark.sql import Row

from pipeline.jobs.fill_missing_months import add_missing_months


class FillMissingMonths(ReusableSparkTestCase):
    def make_sample_test_data(self):
        spark = self.spark

        # fmt: off
        df = spark.createDataFrame(data=[
            Row(SpendMonth=datetime.date(2013, 4, 1), MarketSector="Culture, Media and Sport", Pillar='Corporate Solutions', Category='Document Management & Logistics', SubCategory='MFD & Record Management', EvidencedSpend=1234.56),
            Row(SpendMonth=datetime.date(2013, 5, 1), MarketSector="Culture, Media and Sport", Pillar='Corporate Solutions', Category='Document Management & Logistics', SubCategory='MFD & Record Management', EvidencedSpend=1111.22),
            Row(SpendMonth=datetime.date(2013, 11, 1), MarketSector="Culture, Media and Sport", Pillar='Corporate Solutions', Category='Document Management & Logistics', SubCategory='MFD & Record Management', EvidencedSpend=1234.56),
            Row(SpendMonth=datetime.date(2013, 3, 1), MarketSector="Health", Pillar='People', Category='Professional Services', SubCategory='Consultancy', EvidencedSpend=1234.56),
            Row(SpendMonth=datetime.date(2013, 4, 1), MarketSector="Health", Pillar='People', Category='Professional Services', SubCategory='Consultancy', EvidencedSpend=1111.22),
            Row(SpendMonth=datetime.date(2013, 11, 1), MarketSector="Health", Pillar='People', Category='Professional Services', SubCategory='Consultancy', EvidencedSpend=1234.56),
            Row(SpendMonth=datetime.date(2013, 7, 1), MarketSector="Health", Pillar='People', Category='Professional Services', SubCategory='Consultancy', EvidencedSpend=1111.22),
            Row(SpendMonth=datetime.date(2013, 12, 1), MarketSector="Health", Pillar='People', Category='Professional Services', SubCategory='Legal Services', EvidencedSpend=1234.56),
            Row(SpendMonth=datetime.date(2014, 2, 1), MarketSector="Health", Pillar='People', Category='Professional Services', SubCategory='Legal Services', EvidencedSpend=1111.22),
            Row(SpendMonth=datetime.date(2014, 4, 1), MarketSector="Health", Pillar='People', Category='Professional Services', SubCategory='Legal Services', EvidencedSpend=1234.56),
        ])
        # fmt: on
        return df

    def test_add_missing_months_basic(self):
        """Basic test case, not considering MarketSector or Category. Just fill in missing months with zero rows"""

        input_df = self.make_sample_test_data()

        expected = self.spark.createDataFrame(
            # fmt: off
            data=[
                # expected to output a dataframe that fill missing months with zero record.
                Row( SpendMonth=datetime.date(2013, 4, 1), MarketSector="Culture, Media and Sport", Pillar="Corporate Solutions", Category="Document Management & Logistics", SubCategory="MFD & Record Management", EvidencedSpend=1234.56,),
                Row( SpendMonth=datetime.date(2013, 5, 1), MarketSector="Culture, Media and Sport", Pillar="Corporate Solutions", Category="Document Management & Logistics", SubCategory="MFD & Record Management", EvidencedSpend=1111.22,),
                Row( SpendMonth=datetime.date(2013, 6, 1), MarketSector=None, Pillar=None, Category=None, SubCategory=None, EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 8, 1), MarketSector=None, Pillar=None, Category=None, SubCategory=None, EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 9, 1), MarketSector=None, Pillar=None, Category=None, SubCategory=None, EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 10, 1), MarketSector=None, Pillar=None, Category=None, SubCategory=None, EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 11, 1), MarketSector="Culture, Media and Sport", Pillar="Corporate Solutions", Category="Document Management & Logistics", SubCategory="MFD & Record Management", EvidencedSpend=1234.56,),
                Row( SpendMonth=datetime.date(2013, 3, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Consultancy", EvidencedSpend=1234.56,),
                Row( SpendMonth=datetime.date(2013, 4, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Consultancy", EvidencedSpend=1111.22,),
                Row( SpendMonth=datetime.date(2013, 11, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Consultancy", EvidencedSpend=1234.56,),
                Row( SpendMonth=datetime.date(2013, 7, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Consultancy", EvidencedSpend=1111.22,),
                Row( SpendMonth=datetime.date(2013, 12, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Legal Services", EvidencedSpend=1234.56,),
                Row( SpendMonth=datetime.date(2014, 1, 1), MarketSector=None, Pillar=None, Category=None, SubCategory=None, EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2014, 2, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Legal Services", EvidencedSpend=1111.22,),
                Row( SpendMonth=datetime.date(2014, 3, 1), MarketSector=None, Pillar=None, Category=None, SubCategory=None, EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2014, 4, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Legal Services", EvidencedSpend=1234.56,),
            ]
            # fmt: on
        )

        actual = add_missing_months(
            input_df,
            date_column="SpendMonth",
            amount_column="EvidencedSpend",
            columns_to_consider=[],
        )

        assert actual.count() == expected.count()

        diff = actual.exceptAll(expected)
        assert diff.isEmpty() == True

    def test_not_mutating_input_df(self):
        """Test that the function doesn't mutate the input dataframe"""
        input_df = self.make_sample_test_data()

        columns_to_consider = ["MarketSector", "Category"]

        clone_df = self.spark.createDataFrame(
            data=input_df.toPandas(), schema=input_df.schema
        )

        add_missing_months(
            input_df,
            date_column="SpendMonth",
            amount_column="EvidencedSpend",
            columns_to_consider=columns_to_consider,
        )

        assert input_df.count() == clone_df.count()

        diff = input_df.exceptAll(clone_df)
        assert diff.isEmpty() == True

    def test_columns_order_remain_the_same(self):
        """Test that the function doesn't change column order"""

        input_df = self.make_sample_test_data()
        expected_columns = input_df.columns

        columns_to_consider = ["MarketSector", "Category"]

        actual = add_missing_months(
            input_df,
            date_column="SpendMonth",
            amount_column="EvidencedSpend",
            columns_to_consider=columns_to_consider,
        )

        assert actual.columns == expected_columns

    def test_add_missing_months_with_marketsector_and_category(self):
        """Advanced test case, considering MarketSector and Category"""
        input_df = self.make_sample_test_data()
        columns_to_consider = ["MarketSector", "Category"]

        expected = self.spark.createDataFrame(
            # fmt: off
            data=[
                Row( SpendMonth=datetime.date(2013, 4, 1), MarketSector="Culture, Media and Sport", Pillar="Corporate Solutions", Category="Document Management & Logistics", SubCategory="MFD & Record Management", EvidencedSpend=1234.56,),
                Row( SpendMonth=datetime.date(2013, 5, 1), MarketSector="Culture, Media and Sport", Pillar="Corporate Solutions", Category="Document Management & Logistics", SubCategory="MFD & Record Management", EvidencedSpend=1111.22,),
                Row( SpendMonth=datetime.date(2013, 6, 1), MarketSector="Culture, Media and Sport", Pillar=None, Category="Document Management & Logistics", SubCategory=None, EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 7, 1), MarketSector="Culture, Media and Sport", Pillar=None, Category="Document Management & Logistics", SubCategory=None, EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 8, 1), MarketSector="Culture, Media and Sport", Pillar=None, Category="Document Management & Logistics", SubCategory=None, EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 9, 1), MarketSector="Culture, Media and Sport", Pillar=None, Category="Document Management & Logistics", SubCategory=None, EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 10, 1), MarketSector="Culture, Media and Sport", Pillar=None, Category="Document Management & Logistics", SubCategory=None, EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 11, 1), MarketSector="Culture, Media and Sport", Pillar="Corporate Solutions", Category="Document Management & Logistics", SubCategory="MFD & Record Management", EvidencedSpend=1234.56,),
                Row( SpendMonth=datetime.date(2013, 3, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Consultancy", EvidencedSpend=1234.56,),
                Row( SpendMonth=datetime.date(2013, 4, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Consultancy", EvidencedSpend=1111.22,),
                Row( SpendMonth=datetime.date(2013, 5, 1), MarketSector="Health", Pillar=None, Category="Professional Services", SubCategory=None, EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 6, 1), MarketSector="Health", Pillar=None, Category="Professional Services", SubCategory=None, EvidencedSpend=0.0,),
                # Function should not depends on order of input rows. A row of 2013 07 already exist at the bottom,  so it shouldn't fill a 2013 07 here.
                Row( SpendMonth=datetime.date(2013, 8, 1), MarketSector="Health", Pillar=None, Category="Professional Services", SubCategory=None, EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 9, 1), MarketSector="Health", Pillar=None, Category="Professional Services", SubCategory=None, EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 10, 1), MarketSector="Health", Pillar=None, Category="Professional Services", SubCategory=None, EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 11, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Consultancy", EvidencedSpend=1234.56,),
                Row( SpendMonth=datetime.date(2013, 7, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Consultancy", EvidencedSpend=1111.22,),
                Row( SpendMonth=datetime.date(2013, 12, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Legal Services", EvidencedSpend=1234.56,),
                Row( SpendMonth=datetime.date(2014, 1, 1), MarketSector="Health", Pillar=None, Category="Professional Services", SubCategory=None, EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2014, 2, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Legal Services", EvidencedSpend=1111.22,),
                Row( SpendMonth=datetime.date(2014, 3, 1), MarketSector="Health", Pillar=None, Category="Professional Services", SubCategory=None, EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2014, 4, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Legal Services", EvidencedSpend=1234.56,),
            ]
            # fmt: on
        )

        actual = add_missing_months(
            input_df,
            date_column="SpendMonth",
            amount_column="EvidencedSpend",
            columns_to_consider=columns_to_consider,
        )

        assert actual.count() == expected.count()

        diff = actual.exceptAll(expected)
        assert diff.isEmpty() == True

    def test_add_missing_months_with_marketsector_and_Subcategory(self):
        """Advanced test case, considering MarketSector and SubCategory"""

        input_df = self.make_sample_test_data()
        columns_to_consider = ["MarketSector", "SubCategory"]

        expected = self.spark.createDataFrame(
            # fmt: off
            data=[
                Row( SpendMonth=datetime.date(2013, 4, 1), MarketSector="Culture, Media and Sport", Pillar="Corporate Solutions", Category="Document Management & Logistics", SubCategory="MFD & Record Management", EvidencedSpend=1234.56,),
                Row( SpendMonth=datetime.date(2013, 5, 1), MarketSector="Culture, Media and Sport", Pillar="Corporate Solutions", Category="Document Management & Logistics", SubCategory="MFD & Record Management", EvidencedSpend=1111.22,),
                Row( SpendMonth=datetime.date(2013, 6, 1), MarketSector="Culture, Media and Sport", Pillar=None, Category=None, SubCategory="MFD & Record Management", EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 7, 1), MarketSector="Culture, Media and Sport", Pillar=None, Category=None, SubCategory="MFD & Record Management", EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 8, 1), MarketSector="Culture, Media and Sport", Pillar=None, Category=None, SubCategory="MFD & Record Management", EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 9, 1), MarketSector="Culture, Media and Sport", Pillar=None, Category=None, SubCategory="MFD & Record Management", EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 10, 1), MarketSector="Culture, Media and Sport", Pillar=None, Category=None, SubCategory="MFD & Record Management", EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 11, 1), MarketSector="Culture, Media and Sport", Pillar="Corporate Solutions", Category="Document Management & Logistics", SubCategory="MFD & Record Management", EvidencedSpend=1234.56,),
                Row( SpendMonth=datetime.date(2013, 3, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Consultancy", EvidencedSpend=1234.56,),
                Row( SpendMonth=datetime.date(2013, 4, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Consultancy", EvidencedSpend=1111.22,),
                Row( SpendMonth=datetime.date(2013, 5, 1), MarketSector="Health", Pillar=None, Category=None, SubCategory="Consultancy", EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 6, 1), MarketSector="Health", Pillar=None, Category=None, SubCategory="Consultancy", EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 8, 1), MarketSector="Health", Pillar=None, Category=None, SubCategory="Consultancy", EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 9, 1), MarketSector="Health", Pillar=None, Category=None, SubCategory="Consultancy", EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 10, 1), MarketSector="Health", Pillar=None, Category=None, SubCategory="Consultancy", EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2013, 11, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Consultancy", EvidencedSpend=1234.56,),
                Row( SpendMonth=datetime.date(2013, 7, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Consultancy", EvidencedSpend=1111.22,),
                Row( SpendMonth=datetime.date(2013, 12, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Legal Services", EvidencedSpend=1234.56,),
                Row( SpendMonth=datetime.date(2014, 1, 1), MarketSector="Health", Pillar=None, Category=None, SubCategory="Legal Services", EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2014, 2, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Legal Services", EvidencedSpend=1111.22,),
                Row( SpendMonth=datetime.date(2014, 3, 1), MarketSector="Health", Pillar=None, Category=None, SubCategory="Legal Services", EvidencedSpend=0.0,),
                Row( SpendMonth=datetime.date(2014, 4, 1), MarketSector="Health", Pillar="People", Category="Professional Services", SubCategory="Legal Services", EvidencedSpend=1234.56,),
            ]
            # fmt: on
        )

        actual = add_missing_months(
            input_df,
            date_column="SpendMonth",
            amount_column="EvidencedSpend",
            columns_to_consider=columns_to_consider,
        )

        assert actual.count() == expected.count()

        diff = actual.exceptAll(expected)
        assert diff.isEmpty() == True
