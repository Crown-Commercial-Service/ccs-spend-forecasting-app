from pyspark.sql import DataFrame, functions as F
import pandas as pd
import datetime
import os
import sys

# add /dbfs/ to path so that import statements works on databricks
if "DATABRICKS_RUNTIME_VERSION" in os.environ:
    sys.path.append("/dbfs/")


from utils import get_logger
from pipeline.utils import (
    connect_spark_to_blob_storage,
    load_latest_blob_to_pyspark,
    save_pandas_dataframe_to_blob,
)


def get_active_combinations(
    customers: DataFrame,
    framework_category_pillar: DataFrame,
    spend_aggregated: DataFrame,
) -> list[tuple[str, str]]:
    """From the given data, make a list of active Category / MarketSector combinations.
    The current way of selecting active combinations:
    For any framework that has status: 'Live' or 'Expired - Data Still Received',
    find all spend record of such framework, and convert that to a unique list of Category / MarketSector.
    The current implementation return the result in desc order of total spend amount.

    Args:
        customers (DataFrame): Spark dataframe of dbo.Customers table
        framework_category_pillar (DataFrame): Spark dataframe of dbo.FrameworkCategoryPillar table
        spend_aggregated (DataFrame): Spark dataframe of dbo.SpendAggregated table

    Returns:
        list[tuple[str, str]]: A list of string pairs, each pair representating a Category / MarketSector combination.
    """

    active_frameworks = framework_category_pillar.filter(
        F.col("Status").isin("Live", "Expired - Data Still Received")
    )
    active_framework_numbers = [
        row["FrameworkNumber"] for row in active_frameworks.collect()
    ]

    spends_of_active_frameworks = spend_aggregated.filter(
        spend_aggregated.FrameworkNumber.isin(active_framework_numbers)
    )

    joined_with_customer_table = spends_of_active_frameworks.join(
        other=customers,
        on=spends_of_active_frameworks.CustomerURN == customers.CustomerKey,
        how="left",
    ).fillna({"MarketSector": "Unassigned"})

    sorted_by_spend_amount = (
        joined_with_customer_table.groupby("Category", "MarketSector")
        .agg(F.sum("EvidencedSpend"))
        .sort(F.desc("sum(EvidencedSpend)"))
    )
    active_combinations = sorted_by_spend_amount.select(
        "Category", "MarketSector"
    ).drop_duplicates()

    output = [tuple(row) for row in active_combinations.collect()]

    return output


def main():
    """The main method to be runned as a job. Will carry out below tasks:
    1. Fetch relavant data tables from blob storage.
    2. Filter out a list of active Category/Market Sector combination base on input data.
    3. Save the result to blob storage.
    """
    logger = get_logger()
    logger.debug("Started Job: Get active combinations")

    logger.debug("Connecting to blob storage...")
    connect_spark_to_blob_storage()

    logger.debug("Loading data tables from blob storage")
    customers = load_latest_blob_to_pyspark(table_name="Customers")
    framework_category_pillar = load_latest_blob_to_pyspark(
        table_name="FrameworkCategoryPillar"
    )
    spend_aggregated = load_latest_blob_to_pyspark(table_name="SpendAggregated")

    logger.debug("Filtering active Category/MarketSector combinations based on data")
    active_combinations = get_active_combinations(
        customers=customers,
        framework_category_pillar=framework_category_pillar,
        spend_aggregated=spend_aggregated,
    )

    df = pd.DataFrame(data=active_combinations, columns=["Category", "MarketSector"])
    today = datetime.date.today().strftime("%Y%m%d")
    filename = f"ActiveCombinations/ActiveCombinations_{today}.csv"

    logger.debug("Saving result to blob storage...")
    save_pandas_dataframe_to_blob(pandas_df=df, filename=filename)

    logger.debug(f"Saved to file: {filename}")
    logger.debug("Job completed")


if __name__ == "__main__":
    """Run the pipeline process"""

    main()
