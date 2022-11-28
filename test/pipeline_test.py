import pytest
import datetime
from pyspark.sql import Row
from pipeline.utils import (
    get_spark_session,
    connect_spark_to_blob_storage,
    save_dataframe_to_blob,
    load_latest_blob_to_pyspark,
    make_blob_storage_path,
)
from pipeline.jobs.fill_missing_months import (
    fill_blanks_with_zeros,
    fill_missing_months_for_transformed_spend,
)


@pytest.fixture()
def dataframe_to_test_fill_blanks_with_zeros():
    spark = get_spark_session()

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
    yield df


def test_fill_blanks_with_zeros_basic(dataframe_to_test_fill_blanks_with_zeros):
    """Basic test case, not considering MarketSector or Category"""
    spark = get_spark_session()

    input_df = dataframe_to_test_fill_blanks_with_zeros

    expected = spark.createDataFrame(
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

    actual = fill_blanks_with_zeros(
        input_df,
        date_column="SpendMonth",
        amount_column="EvidencedSpend",
        columns_to_consider=[],
    )

    assert actual.count() == expected.count()

    diff = actual.exceptAll(expected)
    assert diff.isEmpty() == True


def test_fill_blanks_with_zeros_not_mutating_input_df(
    dataframe_to_test_fill_blanks_with_zeros,
):
    """Test that the function doesn't mutate the input dataframe"""
    spark = get_spark_session()

    input_df = dataframe_to_test_fill_blanks_with_zeros
    columns_to_consider = ["MarketSector", "Category"]

    clone_df = spark.createDataFrame(data=input_df.toPandas(), schema=input_df.schema)

    fill_blanks_with_zeros(
        input_df,
        date_column="SpendMonth",
        amount_column="EvidencedSpend",
        columns_to_consider=columns_to_consider,
    )

    assert input_df.count() == clone_df.count()

    diff = input_df.exceptAll(clone_df)
    assert diff.isEmpty() == True


def test_fill_blanks_with_zeros_output_columns_order_remain_the_same(
    dataframe_to_test_fill_blanks_with_zeros,
):
    """Test that the function doesn't change column order"""

    input_df = dataframe_to_test_fill_blanks_with_zeros
    expected_columns = input_df.columns

    columns_to_consider = ["MarketSector", "Category"]

    actual = fill_blanks_with_zeros(
        input_df,
        date_column="SpendMonth",
        amount_column="EvidencedSpend",
        columns_to_consider=columns_to_consider,
    )

    assert actual.columns == expected_columns


def test_fill_blanks_with_zeros_with_marketsector_and_category(
    dataframe_to_test_fill_blanks_with_zeros,
):
    """Advanced test case, considering MarketSector and Category"""
    spark = get_spark_session()

    input_df = dataframe_to_test_fill_blanks_with_zeros
    columns_to_consider = ["MarketSector", "Category"]

    expected = spark.createDataFrame(
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

    actual = fill_blanks_with_zeros(
        input_df,
        date_column="SpendMonth",
        amount_column="EvidencedSpend",
        columns_to_consider=columns_to_consider,
    )

    assert actual.count() == expected.count()

    diff = actual.exceptAll(expected)
    assert diff.isEmpty() == True


def test_fill_blanks_with_zeros_with_marketsector_and_Subcategory(
    dataframe_to_test_fill_blanks_with_zeros,
):
    """Advanced test case, considering MarketSector and SubCategory"""
    spark = get_spark_session()

    input_df = dataframe_to_test_fill_blanks_with_zeros
    columns_to_consider = ["MarketSector", "SubCategory"]

    expected = spark.createDataFrame(
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

    actual = fill_blanks_with_zeros(
        input_df,
        date_column="SpendMonth",
        amount_column="EvidencedSpend",
        columns_to_consider=columns_to_consider,
    )

    assert actual.count() == expected.count()

    diff = actual.exceptAll(expected)
    assert diff.isEmpty() == True


def test_fill_missing_months_for_transformed_spend(
    dataframe_to_test_fill_blanks_with_zeros, blob_container_for_testing
):
    ## arrange
    spark = get_spark_session()
    connect_spark_to_blob_storage()

    container_client = blob_container_for_testing
    input_df = dataframe_to_test_fill_blanks_with_zeros

    input_table_name = "test_input_table"
    output_table_name = "test_output_table"
    container_name = container_client.container_name

    input_blob_path = make_blob_storage_path(
        table_name=input_table_name, blob_container_name=container_name
    )
    save_dataframe_to_blob(
        df=input_df, table_name=input_table_name, blob_storage_path=input_blob_path
    )

    ## act
    fill_missing_months_for_transformed_spend(
        input_table_name=input_table_name,
        output_table_name=output_table_name,
        container_name=container_name,
    )

    ## assert
    actual = load_latest_blob_to_pyspark(
        table_name=output_table_name, blob_container_name=container_name
    )

    expected = spark.createDataFrame(
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

    assert actual.count() == expected.count()

    diff = actual.exceptAll(expected)
    assert diff.isEmpty() == True
