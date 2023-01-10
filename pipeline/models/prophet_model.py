from typing import Optional
from itertools import product
import datetime
import logging

import pandas as pd
from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_percentage_error,
)

from pipeline.models.forecast_model import ForecastModel
from utils import get_logger



class ProphetModel(ForecastModel):
    def __init__(
        self,
        name: str = "Prophet",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.logger = get_logger()
        self.suppress_prophet_info_logging()

        # a local dictionary that store the best hyperparameter found for each combination.
        self._hyperparameters_cache = {}

    def rename_columns_for_input(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """ Rename dataframe column for input to prophet model.

        Args:
            input_df (pd.DataFrame): Spend data

        Returns:
            pd.DataFrame: Spend data with column name matching prophet's required input format
        """
        return input_df.rename(columns = {
            self.amount_column: 'y',
            self.date_column: 'ds'
        })

    def rename_columns_for_output(self, output_df: pd.DataFrame) -> pd.DataFrame:
        """ Rename dataframe column of the output from prophet model.

        Args:
            output_df (pd.DataFrame): Forecast from prophet

        Returns:
            pd.DataFrame: Spend data with column name 'y', 'ds', 'yhat' renamed to meaningful names to user
        """
        return output_df.rename(columns = {
            'y': self.amount_column,
            'ds': self.date_column,
            'yhat': 'ForecastSpend'
        })


    def suppress_prophet_info_logging(self):
        """ 
        Suppress the spam logs of 'INFO: cmdstanpy:start chain 1' from prophet
        """
        stan_logger = logging.getLogger('cmdstanpy')
        stan_logger.addHandler(logging.NullHandler())
        stan_logger.propagate = False
        stan_logger.setLevel(logging.WARNING)

    def create_forecast(
        self,
        input_df: pd.DataFrame,
        months_to_forecast: int,
        start_month: Optional[datetime.date] = None,
    ) -> pd.DataFrame:
        """Create forecast for the given spend data.
        This method will automatically split data into multiple Category/MarketSector combinations,
        fit a different model for every combination,
        and finally combine all the forecast output in one dataframe.

        Args:
            input_df (pd.DataFrame): Spend data
            months_to_forecast (int): The number of months to forecast for.
            start_month (datetime.date): The first month to forecast for. Should be in Python native date type. 
                                         If omitted, will use the next month of today's date.

        Returns:
            pd.DataFrame: A dataframe with forecast spend amounts.
        """

        if not isinstance(start_month, datetime.date):
            # if start_month is not a valid date type, replace it with next month from today's date
            today = datetime.date.today()
            start_month = (
                today.replace(month=today.month + 1, day=1)
                if today.month != 12
                else datetime.date(year=today.year + 1, month=1, day=1)
            )

        # prepare data by summing up each month, then rename columns to fulfill Prophet requirement
        prepared_data = self.prepare_input_data(input_df=input_df)
        prepared_data = self.rename_columns_for_input(prepared_data)
        

        # create a dataframe to collect output values
        future = pd.DataFrame({'ds': pd.date_range(start=start_month, periods=months_to_forecast, freq="MS")})

        output_df_list = []

        for combination, category_sector_spend in prepared_data.groupby(
            self.columns_to_consider, as_index=False
        ):
            # if already searched for changepoint_prior_scale and seasonality_prior_scale for this combination, reuse the best values. 
            # otherwise, run a search for the best prior scale values
            if combination in self._hyperparameters_cache:
                changepoint_prior_scale, seasonality_prior_scale = self._hyperparameters_cache[combination]
            else:
                self.logger.debug(f"{self.name}: Start searching for best prior scales for {combination}...")
                changepoint_prior_scale, seasonality_prior_scale = self.find_best_prior_scales(category_sector_spend=category_sector_spend)
                self._hyperparameters_cache[combination] = (changepoint_prior_scale, seasonality_prior_scale)


            forecast_df = self.create_forecast_with_given_prior_scales(
                category_sector_spend=category_sector_spend,
                future=future,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
            )

            output_df_list.append(forecast_df)

        output_df = pd.concat(output_df_list)

        return output_df

    def create_forecast_with_given_prior_scales(
        self,
        category_sector_spend: pd.DataFrame,
        future: pd.DataFrame,
        changepoint_prior_scale: float,
        seasonality_prior_scale: float,
    ) -> pd.DataFrame:
        """Create forecast with a given changepoint prior scale and seasonality prior scale

        Args:
            category_sector_spend (pd.DataFrame): Spend data under one particular combination. Assumed to have the columns 'y' and 'ds' that is required by prophet
            future (pd.DataFrame): A dataframe which specify the forecast period. Assume to have a column named 'ds' which contains dates.
            changepoint_prior_scale (float): changepoint prior scale for Prophet model
            seasonality_prior_scale (float): seasonality prior scale for Prophet model

        Returns:
            pd.DataFrame: A dataframe that contains the forecast for one particular combination.
        """

        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
        )
        model.add_country_holidays(country_name="UK")
        model.fit(category_sector_spend)

        raw_forecast = model.predict(future)

        # extract relevant columns from the output forecast, then rename the column names
        forecast = raw_forecast[["ds", "yhat"]]
        forecast = self.rename_columns_for_output(forecast)
        
        # add back the category/marketsector columns and rearrange columns order
        for column_name in [self.columns_to_consider]:
            forecast[column_name] = category_sector_spend.iloc[0][column_name]
        forecast = forecast[
            [self.date_column, *self.columns_to_consider, "ForecastSpend"]
        ]

        # as Prophet use dates as timestamp format, convert that to date format.
        forecast[self.date_column] = forecast[self.date_column].transform(
            lambda datetime: datetime.date()
        )

        return forecast

    def find_best_prior_scales(self, category_sector_spend: pd.DataFrame) -> tuple[float, float]:
        """Find a best set of changepoint/seasonality prior scales for one single Category/MarketSector combination,
           by testing with past data and comparing the Mean Absolute Percentage Error (MAPE).

        Args:
            category_sector_spend (pd.DataFrame): Spend data of one single Category and MarketSector combination.

        Returns:
            tuple[float, float]: A tuple of the best values for changepoint_prior_scale and seasonality_prior_scale
        """

        # get the current Category/MarketSector combination as a tuple
        current_combination = tuple(category_sector_spend.iloc[0][self.columns_to_consider])

        self.logger.debug(f"Finding best prior scales for combination: {current_combination}")
        changepoint_prior_scale_values = [0.001, 0.01, 0.1, 0.5]
        seasonality_prior_scale_values = [0.01, 0.1, 1.0, 10.0]

        train_size = int(len(category_sector_spend)* 0.9) 

        train_data = category_sector_spend.iloc[:train_size]
        test_data = category_sector_spend.iloc[train_size:]

        all_mape = []
        
        # loop through 4x4 pairs of prior_scale_values and record the MAPEs each the trial
        search_range = product(changepoint_prior_scale_values, seasonality_prior_scale_values)
        for changepoint_prior_scale, seasonality_prior_scale in search_range:
            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
            )
            model.add_country_holidays(country_name="UK")
            model.fit(train_data)

            forecast = model.predict(test_data)
            mape = mean_absolute_percentage_error(y_true=test_data['y'], y_pred=forecast['yhat'])
            all_mape.append([changepoint_prior_scale, seasonality_prior_scale, mape])

        all_mape = pd.DataFrame(data=all_mape, columns = ['changepoint_prior_scale', 'seasonality_prior_scale', 'mape']).sort_values('mape', ascending=True)
        best_params = tuple(all_mape.iloc[0][['changepoint_prior_scale', 'seasonality_prior_scale']])

        self.logger.debug(f"MAPE scores for {current_combination}:")
        self.logger.debug(all_mape)
        self.logger.debug(f"Best params were: {best_params}")

        return best_params


