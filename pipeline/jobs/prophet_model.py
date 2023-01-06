from typing import Optional
from itertools import product
import datetime

import pandas as pd
from prophet import Prophet
from sklearn.metrics import (
    mean_absolute_percentage_error,
)

from pipeline.jobs.forecast_model import ForecastModel
from utils import get_logger

logger = get_logger()


class ProphetModel(ForecastModel):
    def __init__(
        self,
        name: str = "Prophet",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        # a local dictionary that store the best hyperparameter found for each combination.
        # key to be the combination and value to be the hyperparameters
        # for now, this cache only live in the model instance, and will not be stored in anywhere.
        self._hyperparameters_cache = {}

    def rename_columns_for_input(self, input_df: pd.DataFrame) -> pd.DataFrame:
        return input_df.rename(columns = {
            self.amount_column: 'y',
            self.date_column: 'ds',
            'ForecastSpend': 'yhat'
        })

    def rename_columns_for_output(self, output_df: pd.DataFrame) -> pd.DataFrame:
        return output_df.rename(columns = {
            'y': self.amount_column,
            'ds': self.date_column,
            'yhat': 'ForecastSpend'
        })


    def create_forecast(
        self,
        input_df: pd.DataFrame,
        months_to_forecast: int,
        start_month: Optional[datetime.date] = None,
    ) -> pd.DataFrame:

        if not isinstance(start_month, datetime.date):
            # if start_month is not a valid date type, replace it with next month from today's date
            today = datetime.date.today()
            start_month = (
                today.replace(month=today.month + 1, day=1)
                if today.month != 12
                else datetime.date(year=today.year + 1, month=1, day=1)
            )

        # prepare data by summing up each month, remove irrelavent columns and rename columns to meet Prophet interface
        prepared_data = self.prepare_input_data(input_df=input_df)

        # create dataframe to collect output values
        future = pd.DataFrame({'ds': pd.date_range(start=start_month, periods=months_to_forecast, freq="MS")})

        output_df_list = []

        for combination, category_sector_spend in prepared_data.groupby(
            self.columns_to_consider, as_index=False
        ):
            if combination in self._hyperparameters_cache:
                changepoint_prior_scale, seasonality_prior_scale = self._hyperparameters_cache[combination]
            else:
                logger.debug(f"{self.name}: Start searching for best prior scale for {combination}...")
                changepoint_prior_scale, seasonality_prior_scale = self.find_best_prior_scales(input_df=category_sector_spend)
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

        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
        )
        model.add_country_holidays(country_name="UK")
        model.fit(category_sector_spend)

        raw_forecast = model.predict(future)

        # extract relevant columns from model output, then rename the column names
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

    def find_best_prior_scales(self, input_df: pd.DataFrame) -> tuple[float, float]:
        current_combination = tuple(input_df.iloc[0][self.columns_to_consider])
        logger.debug(f"Finding best prior scales for combination: {current_combination}")
        changepoint_prior_scale_values = [0.001, 0.01, 0.1, 0.5]
        seasonality_prior_scale_values = [0.01, 0.1, 1.0, 10.0]

        train_size = int(len(input_df)* 0.9) 

        train_data = input_df.iloc[:train_size]
        test_data = input_df.iloc[train_size:]

        all_mape = []
        
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

        logger.debug(f"MAPE scores for {current_combination}:")
        logger.debug(all_mape)
        logger.debug(f"Best params were: {best_params}")

        return best_params



    def prepare_input_data(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Sum up the spend data by month, so that for each combination, there is only one row for one month.
        Also strips away any irrelavant columns from input data

        Args:
            input_df (pd.DataFrame): Input spend data

        Returns:
            pd.DataFrame: Prepared data
        """

        aggregated_spend =  input_df.groupby(
            [self.date_column, *self.columns_to_consider], as_index=False
        ).agg({self.amount_column: "sum"})
        return self.rename_columns_for_input(input_df=aggregated_spend)
