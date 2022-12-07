import pandas as pd
import numpy as np
import datetime
from typing import Optional

from pipeline.jobs.forecast_model import ForecastModel


class MockForecastModel(ForecastModel):
    """A mock forecast model that generate mock forecast data for given spend data. The forecast amount is generated randomly within a certain range.
    For each combination in the input data, it will detect the mean spend value and create mock forecast base on that figure.

    """

    def __init__(
        self,
        name: str = "Mock Model A",
        columns_to_consider: list[str] = ["Category", "MarketSector"],
        date_column: str = "SpendMonth",
        amount_column: str = "EvidencedSpend",
        randomness: float = 0.5,
    ):
        """_summary_

        Args:
            name (str, optional): The name of this forecast model. Defaults to 'Mock Model A'.
            columns_to_consider (list[str], optional): Column names that represent the categorisation to consider when making forecast. Defaults to ['Category', 'MarketSector'].
            date_column (str, optional): The column name that represent date in input data. Defaults to "SpendMonth". Note: It is assumed that this field in the input data should be ub valid date type, NOT string representation of date (e.g. "20220101")
            amount_column (str, optional):The column name that represent spend amount in input data. Defaults to "EvidencedSpend".
            randomness (float, optional): A number between 0 to 1 which control the random range of output. For example, if randomness = 0.2, output amount will be within +-20% range of latest month spending.
        """
        super().__init__(
            name=name,
            columns_to_consider=columns_to_consider,
            date_column=date_column,
            amount_column=amount_column,
        )
        self.randomness = randomness

    def create_forecast(
        self,
        input_df: pd.DataFrame,
        months_to_forecast: int,
        start_month: Optional[datetime.date] = None,
    ) -> pd.DataFrame:
        """Create mock forecast for given spend data. The forecast amount is generated randomly within a certain range.
        For each combination in the input data, it will detect the mean value and create mock forecast amount base on that value.

        Args:
            input_df (DataFrame): The input spend data
            months_to_forecast (int): The number of months to create forecast for.
            start_month: (datetime.date, optional) The month to start making forecast. If omitted, will default to the next month from today's date.

        Returns:
            A pandas dataframe of mock forecast data.
        """
        df_aggregated_spend_by_month = (
            input_df.groupby([*self.columns_to_consider, self.date_column])[
                self.amount_column
            ]
            .sum()
            .reset_index()
            .sort_values(by=self.date_column)
        )

        if not isinstance(start_month, datetime.date):
            # if start_month is not a valid date type, replace it with next month from today's date
            today = datetime.date.today()
            start_month = (
                today.replace(month=today.month + 1, day=1)
                if today.month != 12
                else datetime.date(year=today.year + 1, month=1, day=1)
            )

        mean_values = (
            df_aggregated_spend_by_month.groupby(self.columns_to_consider)[
                self.amount_column
            ]
            .mean()
            .reset_index()
        )

        date_range = pd.DataFrame(
            data=pd.date_range(
                start=start_month, periods=months_to_forecast, freq="MS"
            ).date,
            columns=[self.date_column],
        )

        output_df = mean_values.merge(date_range, how="cross")

        output_df["ForecastSpend"] = output_df[self.amount_column] * (
            1 + self.randomness * (np.random.rand(len(output_df)) * 2 - 1)
        )

        output_df.drop(columns=self.amount_column, inplace=True)

        return output_df


def create_mock_model(randomness: float, name: str = "Mock Model A") -> ForecastModel:
    """Create a mock model that output forecast with a pre-defined randomness factor.

    Args:
        randomness (float): A number between 0 to 1 which control the random range of output. For example, if randomness = 0.2, output amount will be within +-20% range of latest month spending.
        name (str): The name of the mock model.
    """

    return MockForecastModel(name=name, randomness=randomness)
