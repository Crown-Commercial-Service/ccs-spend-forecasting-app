import pandas as pd
import datetime
from typing import Optional
from abc import ABC


class ForecastModel(ABC):
    """
    An abstract class of a spend prediction model.
    The relevant column names of input data are to be registered at initialisation, to avoid hardcoding field names in machine learning codes.
    """

    def __init__(
        self,
        name: str,
        columns_to_consider: list[str] = ["Category", "MarketSector"],
        date_column: str = "SpendMonth",
        amount_column: str = "EvidencedSpend",
    ):
        """
        Args:
            name (str): The name of this forecast model. e.g. ARMA(1,4)
            columns_to_consider (list[str], optional): Column names that represent the categorisation to consider when making forecast. Defaults to ['Category', 'MarketSector'].
            date_column (str, optional): The column name that represent date in input data. Defaults to "SpendMonth". Note: It is assumed that this field in the input data should be ub valid date type, NOT string representation of date (e.g. "20220101")
            amount_column (str, optional):The column name that represent spend amount in input data. Defaults to "EvidencedSpend".

        """
        self.name = name
        self.columns_to_consider = columns_to_consider
        self.date_column = date_column
        self.amount_column = amount_column

    def create_forecast(
        self,
        input_df: pd.DataFrame,
        months_to_forecast: int,
        start_month: Optional[datetime.date] = None,
    ) -> pd.DataFrame:
        """An abstract interface of creating a forecast. To be implemented by each model

        Args:
            input_df (pd.DataFrame): The input spend data
            months_to_forecast (int): The number of months to create forecast for.
            start_month: (datetime.date) The month to start making forecast.

        Returns:
            pd.DataFrame: A pandas dataframe that contains spend forecast data.
        """
        raise NotImplementedError
