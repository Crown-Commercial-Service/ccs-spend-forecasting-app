from pyspark.sql import DataFrame, functions as F
import pandas as pd

from pipeline.utils import connect_spark_to_blob_storage, load_latest_blob_to_pyspark


def filter_active_combinations(
    customers: DataFrame,
    framework_category_pillar: DataFrame,
    spend_aggregated: DataFrame,
) -> list[tuple[str, str]]:

    active_frameworks = framework_category_pillar.filter(F.col('Status').isin("Live", "Expired - Data Still Received"))
    active_framework_numbers = [row['FrameworkNumber'] for row in active_frameworks.collect()]

    spends_of_active_frameworks = spend_aggregated.filter(spend_aggregated.FrameworkNumber.isin(active_framework_numbers))

    joined_spend = spends_of_active_frameworks.join(
        other=customers, 
        on=spends_of_active_frameworks.CustomerURN == customers.CustomerKey, 
        how='left')

    joined_spend = joined_spend.fillna({'MarketSector': 'Unassigned'})

    active_combinations = joined_spend.select('Category', 'MarketSector').drop_duplicates()

    output = [tuple(row) for row in active_combinations.collect()]

    return output


def main():
    connect_spark_to_blob_storage()

    customers = load_latest_blob_to_pyspark(table_name='Customers')
    framework_category_pillar = load_latest_blob_to_pyspark(table_name='FrameworkCategoryPillar')
    spend_aggregated = load_latest_blob_to_pyspark(table_name='SpendAggregated')

    active_combinations = filter_active_combinations(
        customers=customers,
        framework_category_pillar=framework_category_pillar,
        spend_aggregated=spend_aggregated
    )

    df = pd.DataFrame(data=active_combinations, columns=['Category', 'MarketSector'])
    