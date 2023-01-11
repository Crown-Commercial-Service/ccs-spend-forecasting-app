from pyspark.sql import Row

from pipeline.jobs.get_active_combinations import get_active_combinations
from pipeline.utils import get_spark_session


def make_sample_test_data():
    """Generate sample data this test"""
    spark = get_spark_session()

    # fmt: off
    customers = spark.createDataFrame(data=[
        Row(SalesforceID='123', CustomerKey='ABC', CustomerName='Customer A', MarketSector = "Culture, Media and Sport"),
        Row(SalesforceID='234', CustomerKey='DEF', CustomerName='Customer B', MarketSector = "Education"),
        Row(SalesforceID='345', CustomerKey='XYZ', CustomerName='Customer C', MarketSector = "Health"),
        Row(SalesforceID='456', CustomerKey='PQJ', CustomerName='Customer D', MarketSector = "Government Policy"),
    ])

    framework_category_pillar = spark.createDataFrame(
        data=[
            # below categories got status 'Live' or 'Expired - Data Still Received'. They should appear in result.
            Row(Category='Digital Future', CategoryKey=111, FrameworkNumber='RM1234', FrameworkName='', Status='Live'),
            Row(Category='Network Service', CategoryKey=222, FrameworkNumber='RM3456', FrameworkName='', Status='Expired - Data Still Received'),

            # below categories should not appear in result
            Row(Category='Construction', CategoryKey=333, FrameworkNumber='RM5678', FrameworkName='', Status='Expired'),
            Row(Category='Digital Future', CategoryKey=111, FrameworkNumber='Z123456', FrameworkName='', Status='Some Other Status'),
            Row(Category='Some random category', CategoryKey=444, FrameworkNumber='DFGH', FrameworkName='', Status=None),
        ]
    )

    spend_aggregated = spark.createDataFrame(data=[
        # Customer A : "Culture, Media and Sport". Should match with "Digital Future" only, as RM1234 is live but RM5678 is expired.
        Row(CustomerURN='ABC', CustomerName='Customer A', Category='Digital Future', FrameworkNumber='RM1234', EvidencedSpend=3000.0),
        Row(CustomerURN='ABC', CustomerName='Customer A', Category='Construction', FrameworkNumber='RM5678', EvidencedSpend=3000.0),
        
        # Customer B : "Education". Should match with both "Digital Future" and "Network Service"
        Row(CustomerURN='DEF', CustomerName='Customer B', Category='Digital Future', FrameworkNumber='RM1234', EvidencedSpend=2000.0),
        Row(CustomerURN='DEF', CustomerName='Customer B', Category='Network Service', FrameworkNumber='RM3456', EvidencedSpend=2000.0),

        # Customer C : "Health". Should only match with "Network Service".
        Row(CustomerURN='XYZ', CustomerName='Customer C', Category='Network Service', FrameworkNumber='RM3456', EvidencedSpend=1500.0),
        Row(CustomerURN='XYZ', CustomerName='Customer C', Category='Construction', FrameworkNumber='RM5678', EvidencedSpend=1500.0),
        Row(CustomerURN='XYZ', CustomerName='Customer C', Category='Digital Future', FrameworkNumber='Z123456', EvidencedSpend=1500.0),

        # below 2 spends should not contribute to result, as frameworks are not 'Live' or 'Expired - Data Still Received'
        Row(CustomerURN='PQJ', CustomerName='Customer D', Category='Digital Future', FrameworkNumber='Z123456', EvidencedSpend=1000.0),
        Row(CustomerURN=None, CustomerName=None, Category='Construction', FrameworkNumber='DFGH', EvidencedSpend=1000.0),

        # below spend should give a "('Digital Future', 'Unassigned')" combination, as currently we treat Null customer as MarketSector: Unassigned
        Row(CustomerURN=None, CustomerName=None, Category='Digital Future', FrameworkNumber='RM1234', EvidencedSpend=1000.0),
    ])

    # fmt: on
    return (customers, framework_category_pillar, spend_aggregated)


def test_get_active_combinations():
    """Test that the method #get_active_combinations correctly filter out the Category/MarketSector combination that is considered active."""

    customers, framework_category_pillar, spend_aggregated = make_sample_test_data()

    expected = [
        ("Digital Future", "Culture, Media and Sport"),
        ("Digital Future", "Education"),
        ("Network Service", "Education"),
        ("Network Service", "Health"),
        ("Digital Future", "Unassigned"),
    ]

    actual = get_active_combinations(
        customers=customers,
        framework_category_pillar=framework_category_pillar,
        spend_aggregated=spend_aggregated,
    )

    assert actual == expected
