import os
import sys


if __name__ == "__main__":
    """Run the pipeline process"""

    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        sys.path.append(
            "/dbfs/"
        )  # add /dbfs/ to path so that import statements works on databricks

    from pipeline.jobs.fill_missing_months import (
        fill_missing_months_for_transformed_spend,
    )

    fill_missing_months_for_transformed_spend(
        input_table_name="TransformedSpendData",
        output_table_name="SpendDataFilledMissingMonth",
        container_name="azp-uks-spend-forecasting-development-transformed",
    )
