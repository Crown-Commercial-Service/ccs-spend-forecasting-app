from typing import Optional, Union
from itertools import product
import datetime
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from statsmodels.tsa.statespace.sarimax import SARIMAX

from pipeline.jobs.model_utils import (
    find_integration_order,
    find_seasonal_integration_order,
    ljung_box_residual_test,
)
from pipeline.jobs.forecast_model import ForecastModel
from utils import get_logger

logger = get_logger()


class SarimaModel(ForecastModel):
    """Forecast the data using SARIMA(p,d,q)(P,D,Q)m model."""

    def __init__(self, name: str = "SARIMA", **kwargs):
        super().__init__(name=name, **kwargs)

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
            start_month (datetime.date): The first month to forecast for. Should be in Python native date type. If omitted, will use the next month of today's date.

        Returns:
            pd.DataFrame: A dataframe with forecast spend amounts.
        """

        # find_hyperparameters: set this to True to search for best hyperparameter, False to just use a set of hardcoded numbers
        # This process take around 10 minutes for each Category/MarketSector combination.
        # During debug, change this to false to enable faster feedback loop.
        # To be removed later when we got a better way to store the parameters for each combination.
        # find_hyperparameters = True
        find_hyperparameters = False

        output_df_list = []
        if not isinstance(start_month, datetime.date):
            # if start_month is not a valid date type, replace it with next month from today's date
            today = datetime.date.today()
            start_month = (
                today.replace(month=today.month + 1, day=1)
                if today.month != 12
                else datetime.date(year=today.year + 1, month=1, day=1)
            )

        for combination, category_sector_spend in input_df.groupby(
            self.columns_to_consider, as_index=False
        ):
            # make sure the data are sorted by date in asc order. drop the index
            category_sector_spend = category_sector_spend.sort_values(
                self.date_column, ascending=True
            ).reset_index(drop=True)

            if find_hyperparameters:
                logger.debug(
                    f"Start searching for best hyperparameters for {combination}..."
                )
                params = self.find_best_hyperparameter_for_single_combination(
                    category_sector_spend
                )
            else:
                # If find_hyperparameters is false, will use a hardcoded set of params.
                # Below is the best params found for Category="Workforce Health & Education" and MarketSector="Health". Probably won't work well for other combinations.
                params = {
                    "order": (2, 2, 3),
                    "seasonal_order": (0, 1, 2, 12),
                }

            logger.debug(f"Generating forecast for {combination}...")

            # create the forecast of required time period.
            forecast_df = self.create_forecast_with_given_hyperparameter(
                category_sector_spend=category_sector_spend,
                order=params["order"],
                seasonal_order=params["seasonal_order"],
                months_to_forecast=months_to_forecast,
                start_month=start_month,
            )
            output_df_list.append(forecast_df)

        output_df = pd.concat(output_df_list)
        return output_df

    def sarima_aic_scores(
        self,
        timeseries: Union[pd.Series, np.ndarray, list],
        pqPQ_combinations: list,
        d: int,
        D: int,
        s: int,
    ) -> pd.DataFrame:
        """
        Args:
            timeseries: Timeseries data
            pqPQ_combinations: List of all combination of p, q, P and Q that you want to test
            d: Integration order for the series
            D: Integration order for seasonality
            s: Seasonality

        Returns:
            DataFrame containing parameters and its respective aic score, sorted by aic score ascending.

        """
        aic_scores = []
        for p, q, P, Q in pqPQ_combinations:
            try:
                model = SARIMAX(
                    timeseries,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    simple_differencing=False,
                ).fit(disp=False)
                aic = model.aic
                aic_scores.append([p, q, d, P, Q, D, s, aic])
            except Exception as e:
                logger.error(f"Error while calculating SARIMA aic scores: {e}")

        df = (
            pd.DataFrame(
                data=aic_scores, columns=["p", "q", "d", "P", "Q", "D", "s", "aic"]
            )
            .sort_values(by="aic", ascending=True)
            .reset_index(drop=True)
        )

        return df

    def create_forecast_with_given_hyperparameter(
        self,
        category_sector_spend: pd.DataFrame,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int],
        start_month: datetime.date,
        months_to_forecast: int,
    ) -> pd.DataFrame:
        """Create SARIMA model forecast with a given set of hyper parameters for one particular combination.

        Args:
            category_sector_spend (pd.DataFrame): Spend data under one particular combination. Assumed to be sorted by date in asc order, with only one row per month.
            order: parameter (p,d,q) for SARIMA model.
            seasonal_order: parameter (P,D,Q,M) for SARIMA model.
            start_month (datetime.date): The first month to make forecast for.
            months_to_forecast (int): The number of months to make forecast for.

        Returns:
            pd.Dataframe: A dataframe that contains the forecast for one particular combination.
        """
        spend = category_sector_spend[self.amount_column]

        # determine the time period to forecast for.
        input_data_start_date = category_sector_spend[self.date_column].iloc[0]
        forecast_end_date = start_month + relativedelta(months=months_to_forecast - 1)
        date_delta = relativedelta(forecast_end_date, input_data_start_date)
        total_prediction_size = date_delta.months + date_delta.years * 12

        # feed the spend data and hyperparameters to model.
        sarima_model = SARIMAX(
            spend,
            order=order,
            seasonal_order=seasonal_order,
            simple_differencing=False,
        )

        # fit the model and get prediction.
        sarima_model_fit = sarima_model.fit(disp=False)
        sarima_pred = sarima_model_fit.get_prediction(
            start=0, end=total_prediction_size
        ).predicted_mean

        # As prediction came out as a non-label list of values, match date labels to the output of model
        forecast_with_dates = sarima_pred.to_frame(name="ForecastSpend")
        forecast_with_dates[self.date_column] = pd.date_range(
            start=input_data_start_date, end=forecast_end_date, freq="1MS"
        ).date

        # join the forecasts with input data
        combined_df = category_sector_spend.merge(
            forecast_with_dates, how="outer", on=self.date_column
        )
        combined_df[self.columns_to_consider] = combined_df[
            self.columns_to_consider
        ].fillna(method="ffill")

        # Select only the relevent about and return
        # For example, if start_month = 2023 01 01 and months_to_forecast = 12, the model will produce prediction/forecast for all period up to 2023 12 01.
        # Then we select only the period within 2023 01 01 to 2023 12 01 as the output.
        output_df = combined_df[
            (combined_df[self.date_column] >= start_month)
            & (combined_df[self.date_column] <= forecast_end_date)
        ]
        return output_df

    def find_best_hyperparameter_for_single_combination(
        self, category_sector_spend: pd.DataFrame
    ) -> dict:

        spend = category_sector_spend[self.amount_column]
        s = 12  # s is same as m
        D = find_seasonal_integration_order(spend, s=s)
        ps = range(0, 4, 1)
        qs = range(0, 4, 1)
        Ps = range(0, 4, 1)
        Qs = range(0, 4, 1)
        pqPQ_combinations = list(product(ps, qs, Ps, Qs))
        d = find_integration_order(spend)

        aic_scores = self.sarima_aic_scores(spend, pqPQ_combinations, d, D, s)
        logger.debug(f"AIC scores are:\n{aic_scores.head(len(pqPQ_combinations))}")
        lowest_aic_score = aic_scores.iloc[0]
        best_p = int(lowest_aic_score["p"])
        best_q = int(lowest_aic_score["q"])
        best_P = int(lowest_aic_score["P"])
        best_Q = int(lowest_aic_score["Q"])
        logger.debug(
            f"Best p is: {best_p}\nBest q is {best_q}\nBest P is: {best_P}\nBest Q is: {best_Q}"
        )

        # Analysis residuals and store to logs
        sarima_model = SARIMAX(
            spend,
            order=(best_p, d, best_q),
            seasonal_order=(best_P, D, best_Q, s),
            simple_differencing=False,
        )
        sarima_model_fit = sarima_model.fit(disp=False)
        logger.debug(f"Model fit summary:\n{sarima_model_fit.summary()}")
        try:
            sarima_model_fit.plot_diagnostics(figsize=(10, 8))
        except Exception as e:
            logger.error(f"Exception occurred due to {e}")

        logger.debug("Performing Ljung-Box test on for the residuals, on 10 lags")
        residuals = sarima_model_fit.resid
        is_residual_white_noise = ljung_box_residual_test(residuals)
        logger.debug(
            "Is SARIMA({best_p},{d},{best_q})({best_P},{D},{best_Q}){s} residual just random error?",
            is_residual_white_noise,
        )

        return {"order": (best_p, d, best_q), "seasonal_order": (best_P, D, best_Q, s)}
