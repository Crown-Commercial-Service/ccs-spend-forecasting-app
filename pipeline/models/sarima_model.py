from typing import Optional, Union, Iterable
from itertools import product
from dateutil.relativedelta import relativedelta
import datetime

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pipeline.models.model_utils import (
    find_integration_order,
    find_seasonal_integration_order,
)
from pipeline.models.forecast_model import ForecastModel
from utils import get_logger

logger = get_logger()


class SarimaModel(ForecastModel):
    """Forecast the data using SARIMA(p,d,q)(P,D,Q)m model.

    Args:
        name (str):
            Name of the model instance. Default to be "SARIMA"

        search_hyperparameters (bool):
            If True, will run a search for hyperparameter for each combination.
            Note: This could potentially take a very long time to run (>= 10min per combination.)
            If False, will use a pre-defined set of hyperparameter, which largely reduce computation time but may lead to much less accurate forecast.
            The same name argument of method #create_forecast will take precedence over this value.

        default_hyperparameters (dict): A dict that contains the necessary hyperparameters for the model.
            For SARIMA, it requires input with the format {"order": (p, d, q), "seasonal_order": (P, D, Q, m)}
            If `search_hyperparameters` is False and this value is given, will use this instead of the pre-defined ones.
    """

    def __init__(
        self,
        name: str = "SARIMA",
        search_hyperparameters: bool = False,
        default_hyperparameters: Optional[dict] = None,
        **kwargs,
    ):

        super().__init__(name=name, **kwargs)
        self._search_hyperparameters = search_hyperparameters

        if default_hyperparameters:
            self._default_hyperparameters = default_hyperparameters
        else:
            # This is found to be a good param set for Category="Workforce Health & Education" and MarketSector="Health".
            # May not work well for other combinations.
            self._default_hyperparameters = {
                "order": (2, 2, 3),
                "seasonal_order": (0, 1, 2, 12),
            }

        # a local dictionary that store the best parameter found for each combination.
        # key to be the combination and value to be the hyperparameters
        # for now, this cache only live in the model instance, and will not be stored in anywhere.
        self._hyperparameters_cache = {}

    def full_model_name(
        self, order: tuple[int, int, int], seasonal_order: tuple[int, int, int]
    ) -> str:
        """Return the name of this model with given set of hyperparameters. Used to reduce hard-coding for logging."""
        return f"{self.name}{order}{seasonal_order}"

    def create_forecast(
        self,
        input_df: pd.DataFrame,
        months_to_forecast: int,
        start_month: Optional[datetime.date] = None,
        search_hyperparameters: Optional[bool] = None,
    ) -> pd.DataFrame:
        """Create forecast for the given spend data.
        This method will automatically split data into multiple Category/MarketSector combinations,
        fit a different model for every combination,
        and finally combine all the forecast output in one dataframe.

        Args:
            input_df (pd.DataFrame): Spend data
            months_to_forecast (int): The number of months to forecast for.
            start_month (datetime.date): The first month to forecast for. Should be in Python native date type. If omitted, will use the next month of today's date.

            search_hyperparameters (bool): If True, will run a search for hyperparameter for each combination.
                Note: This could potentially take a long time to run (>= 10min per combination.)
                If False, will use a pre-defined set of hyperparameter, which largely reduce computation time but may lead to much less accurate forecast.
                If omitted or given None, will refer to self._search_hyperparameters, which is defined during instantiation

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

        # prepare data by summing up each month and remove irrelavent columns
        prepared_data = self.prepare_input_data(input_df=input_df)

        output_df_list = []

        # Go through each combination and generate forecast.
        for combination, category_sector_spend in prepared_data.groupby(
            self.columns_to_consider, as_index=False
        ):
            # make sure the data are sorted by date in asc order. drop the index
            category_sector_spend = category_sector_spend.sort_values(
                self.date_column, ascending=True
            ).reset_index(drop=True)

            if not self._search_hyperparameters:
                # If _search_hyperparameters flag is False, will just use a hardcoded set of params.
                params = self._default_hyperparameters

            elif combination not in self._hyperparameters_cache:
                # If _search_hyperparameters flag is True and this combination wasn't searched before, will run a grid search.
                logger.debug(
                    f"{self.name}: Start searching for best hyperparameters for {combination}..."
                )
                try:
                    params = self.find_best_hyperparameter_for_single_combination(
                        category_sector_spend
                    )
                    self._hyperparameters_cache[combination] = params
                except Exception as err:
                    logger.error(f"Error while searching for hyperparameter: {err}")
                    continue
            else:
                # If _search_hyperparameters flag is True and already did a grid search for this combination in current job session,
                # will reuse the best params to create forecast
                params = self._hyperparameters_cache[combination]
                logger.debug(
                    f"{self.name}: Reuse the hyperparameters {params} for {combination}..."
                )

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

        # combine the forecast for all combinations into one dataframe
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
                    use_exact_diffuse=True,
                    trend_offset=0,
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
        """Create forecast with a given set of hyper parameters for one particular combination.

        Args:
            category_sector_spend (pd.DataFrame): Spend data under one particular combination. Assumed to be sorted by date in asc order, with only one row per month.
            order: parameter (p,d,q) for SARIMA model.
            seasonal_order: parameter (P,D,Q,m) for SARIMA model.
            start_month (datetime.date): The first month to make forecast for.
            months_to_forecast (int): The number of months to make forecast for.

        Returns:
            pd.Dataframe: A dataframe that contains the forecast for one particular combination.
        """
        spend = category_sector_spend[self.amount_column]

        # determine the whole time period to forecast for.
        # here we calculate the total number of months from the earliest record of this combination, until the end of forecast period.
        # later use this information to correctly align forecast values with date.
        input_data_start_date = category_sector_spend[self.date_column].iloc[0]
        forecast_end_date = start_month + relativedelta(months=months_to_forecast - 1)
        date_delta = relativedelta(forecast_end_date, input_data_start_date)
        total_number_of_months = date_delta.months + date_delta.years * 12

        # Feed the spend data and hyperparameters to model.
        sarima_model = SARIMAX(
            spend,
            order=order,
            seasonal_order=seasonal_order,
            simple_differencing=False,
            use_exact_diffuse=True,
            trend_offset=0,
        )

        # Fit the model and get forecast.
        try:
            sarima_model_fit = sarima_model.fit(disp=False)
            sarima_pred = sarima_model_fit.get_prediction(
                start=0, end=total_number_of_months
            ).predicted_mean
        except Exception as err:
            # If the model raise exception, log the error, fill the forecast column with NaN
            logger.error(f"Exception raised when trying to create forecast {err}")
            logger.error(
                f"Model name: {self.name}, combinations: {category_sector_spend.head(1)[self.columns_to_consider]}"
            )
            sarima_pred = pd.Series(np.NaN, index=range(total_number_of_months + 1))

        # As prediction came out as a non-labelled list of values, add dates to the output of model
        forecast_with_dates = sarima_pred.to_frame(name="ForecastSpend")
        forecast_with_dates[self.date_column] = pd.date_range(
            start=input_data_start_date, end=forecast_end_date, freq="1MS"
        )
        forecast_with_dates[self.date_column] = forecast_with_dates[
            self.date_column
        ].transform(lambda datetime: datetime.date())

        # Combine the forecasts with input data
        combined_df = category_sector_spend.merge(
            forecast_with_dates, how="outer", on=self.date_column
        )
        combined_df[self.columns_to_consider] = combined_df[
            self.columns_to_consider
        ].fillna(method="ffill")

        # drop the EvidenceSpend column to avoid confusion
        combined_df = combined_df.drop(columns=self.amount_column)

        # Select only the rows of relevent time period and return
        # For example, if start_month = 2023 01 01 and months_to_forecast = 12, the model will produce prediction/forecast for all time up to 2023 12 01.
        # Then we select only the period within 2023 01 01 to 2023 12 01 as the output.
        output_df = combined_df[
            (combined_df[self.date_column] >= start_month)
            & (combined_df[self.date_column] <= forecast_end_date)
        ]

        return output_df

    def hyperparameter_range(self) -> dict:
        """Return a default set of search range for hyperparameters for this model.
        For SARIMA model we use (p,d,q)(P,D,Q,m)
        Here we search p, q, seasonal P, seasonal Q within 0~3.
        d and seasonal D are to be found by stationarity test, and m is fixed to be 12.
        Returns:
            dict: A dictionary containing the search range for hyperparameters.
        """

        return {
            # p,d,q
            "p": range(0, 4, 1),  # p
            "q": range(0, 4, 1),  # q
            "d": "auto",  # d
            "seasonal_P": range(0, 4, 1),  # P
            "seasonal_Q": range(0, 4, 1),  # Q
            "seasonal_D": "auto",  # D
            "seasonal_period": 12,  # m or s, default to be 12
        }

    def find_best_hyperparameter_for_single_combination(
        self, category_sector_spend: pd.DataFrame
    ) -> dict:
        """Find a best set of hyperparameter for one single Category/MarketSector combination,
        by comparing the AIC score.

        Args:
            category_sector_spend (pd.DataFrame): Spend data of one single Category and MarketSector combination.


        Returns:
            dict: a dictionary containing the hyperparameter, in the form of {"order": (p,d,q), "seasonal_order": (P,D,Q,m)}
        """

        spend = category_sector_spend[self.amount_column]
        default = self.hyperparameter_range()

        # Setting up hyperparameter search range

        ps = default["p"]
        qs = default["q"]
        Ps = default["seasonal_P"]
        Qs = default["seasonal_Q"]

        s = default["seasonal_period"]

        if default["d"] == "auto":
            try:
                d = find_integration_order(spend)
            except Exception as err:
                logger.error(f"Error while trying to find integration order: {err}")
                d = 0
        else:
            d = default["d"]

        if default["seasonal_D"] == "auto":
            try:
                D = find_seasonal_integration_order(spend, seasonal_order=s)
            except Exception as err:
                logger.error(
                    f"Error while trying to find seasonal integration order: {err}"
                )
                D = 0
        else:
            D = default["seasonal_D"]

        # check params are valid int or int range before go on to searching.

        for param in (ps, qs, Ps, Qs, s, d, D):
            if not self.is_param_valid(param):
                raise ValueError(
                    f"All hyperparameters should be either int or a range of int. Got: {(ps, qs, Ps, Qs, s, d, D)}"
                )

        pqPQ_combinations = list(product(ps, qs, Ps, Qs))

        raw_aic_scores = self.sarima_aic_scores(spend, pqPQ_combinations, d, D, s)

        aic_scores = self.clean_aic_scores(df=raw_aic_scores)

        lowest_aic_score = aic_scores.iloc[0]
        best_p = int(lowest_aic_score["p"])
        best_q = int(lowest_aic_score["q"])
        best_P = int(lowest_aic_score["P"])
        best_Q = int(lowest_aic_score["Q"])
        logger.debug(
            f"Best p is: {best_p}\nBest q is {best_q}\nBest P is: {best_P}\nBest Q is: {best_Q}"
        )

        order = (best_p, d, best_q)
        seasonal_order = (best_P, D, best_Q, s)

        # Analysis residuals and store to logs
        sarima_model = SARIMAX(
            spend,
            order=order,
            seasonal_order=seasonal_order,
            simple_differencing=False,
            use_exact_diffuse=True,
            trend_offset=0,
        )
        sarima_model_fit = sarima_model.fit(disp=False)
        logger.debug(f"Model fit summary:\n{sarima_model_fit.summary()}")

        # For development purpose.
        # Uncomment below codes to view diagnostic graphs for troubleshooting.
        # try:
        #     sarima_model_fit.plot_diagnostics(figsize=(10, 8))
        # except Exception as e:
        #     logger.error(f"Exception occurred due to {e}")

        # logger.debug("Performing Ljung-Box test on for the residuals, on 10 lags")
        # residuals = sarima_model_fit.resid
        # is_residual_white_noise = ljung_box_residual_test(residuals)
        # logger.debug(
        #     "Is {self.full_model_name} residual just random error?",
        #     is_residual_white_noise,
        # )

        return {"order": order, "seasonal_order": seasonal_order}

    def is_param_valid(self, param) -> bool:
        """Check that all hyperparameters are integer or list of intgers. Return True if all input hyperparameters are valid."""
        if isinstance(param, int):
            return True
        if isinstance(param, Iterable) and all(isinstance(elem, int) for elem in param):
            return True
        return False

    def clean_aic_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """A function to remove exceptionally low AIC scores.

            # Note: We have experimented on different physical machine, where on some machine a particular set of hyperparamter
            # results in LU decomposition error while on other machine it results very low aic score which seems wrong. The
            # below code filters those erroneous aic scores

        Args:
            df (pd.DataFrame): DataFrame containing one column named "aic"

        Returns:
            pd.DataFrame: DataFrame with exceptionally low AIC scores removed
        """
        logger.debug("Raw AIC Scores before removing exceptional values:")
        logger.debug(df)

        q1 = df["aic"].quantile(0.25)
        q3 = df["aic"].quantile(0.75)
        iqr = q3 - q1
        whisker_width = 1.5

        df = df.loc[
            (df["aic"] >= q1 - whisker_width * iqr)
            & (df["aic"] <= q3 + whisker_width * iqr)
        ]
        output_df = df.sort_values(by="aic", ascending=True).reset_index(drop=True)

        logger.debug("Raw AIC Scores after removing exceptional values:")
        logger.debug(output_df)

        return output_df
