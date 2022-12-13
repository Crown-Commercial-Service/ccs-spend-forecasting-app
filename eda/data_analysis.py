import argparse
from itertools import product
from typing import Union, Tuple

import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
from pandas.errors import SettingWithCopyWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from utils import get_logger, get_database_connection

pd.set_option('display.max_rows', 0)
pd.set_option('display.max_columns', 0)
pd.set_option('expand_frame_repr', False)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

logger = get_logger()


def get_data_for_analysis(local_data_path: str = None) -> pd.DataFrame:
    """Get the data from local machine if local_data_path is set or else will try to fetch from the database.
    Note: Fetching data from database is expensive and local stored data should be in parquet format.

    Args:
        local_data_path: Path of the data folder storing data in parquet format.

    Returns:
        Returns the DataFrame

    """
    sql = """
        SELECT
        DATEADD(month,3,CONVERT(date,CONCAT(spend.FYMonthKey,'01'),112)) as SpendMonth,
        spend.CustomerURN,
        spend.CustomerName,
        cust.Status AS CustomerStatus,
        spend.Pillar,
        spend.Category,
        spend.SubCategory,
        spend.FrameworkNumber,
        spend.FrameworkKey,
        spend.FYMonthKey,
        spend.MonthName,
        spend.FinancialMonth,
        spend.FinancialYear,
        spend.EvidencedSpend,
        spend.Quantity,
        spend.CommissionRate,
        spend.CommissionValue,
        spend.CountOfTransactions,
        spend.SpendType,
        spend.Sector AS spend_Sector,
        spend.[Group] AS spend_Group,
        cust.[Group] AS customer_Group,
        cust.Sector as customer_Sector,
        ISNULL(cust.MarketSector, 'Unassigned') AS MarketSector
        FROM dbo.SpendAggregated AS spend
        INNER JOIN dbo.FrameworkCategoryPillar frame ON spend.FrameworkNumber = frame.FrameworkNumber
        LEFT JOIN dbo.Customers cust ON spend.CustomerURN = cust.CustomerKey
        WHERE frame.STATUS IN ('Live', 'Expired - Data Still Received')
        ORDER BY spend.Category, cust.MarketSector, spend.FYMonthKey
    """

    if local_data_path:
        logger.debug(f"As local_data_path is set, so using local dataset.")
        df = pd.read_parquet(local_data_path)
    else:
        with get_database_connection() as con:
            logger.debug("As local_data_path is not set so reading from database, this can take a while.")
            df = pd.read_sql(sql, con)

    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares the data for the analysis and to be used for training the model. It aggregates the data by month,
    category and market sector.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame to be used t

    """
    return df.groupby(["SpendMonth", "Category", "MarketSector"], as_index=False).agg({"EvidencedSpend": "sum"})


def get_category_sector(index: int = 0) -> Tuple[str, str]:
    """A helper method to return category and market sector from a list. User needs pass the index and category and
    market sector stored at that index will be returned. This is used to test model for various category and market
    sector combinations.
    Args:
        index: Index of the list

    Returns:
        Category and market sector

    """
    category_sector_list = [("Workforce Health & Education", "Health"),
                            ("Document Management & Logistics", "Education"),
                            ("Financial Services", "Local Community and Housing"), ("Financial Services", "Education"),
                            ("Network Services", "Local Community and Housing"),
                            ("Document Management & Logistics", "Local Community and Housing"),
                            ("Network Services", "Health"), ("Fleet", "Health"),
                            ("Document Management & Logistics", "Health"), ("Network Services", "Education"),
                            ("Financial Services", "Health"), ("Energy", "Health"),
                            ("Energy", "Local Community and Housing"), ("Fleet", "Local Community and Housing"),
                            ("Energy", "Education"), ("Technology Products & Services", "Local Community and Housing"),
                            ("Digital Future", "Government Policy"),
                            ("Document Management & Logistics", "Defence and Security"),
                            ("Technology Solutions & Outcomes", "Local Community and Housing"),
                            ("Network Services", "Defence and Security"), ("Digital Future", "Health"),
                            ("Technology Products & Services", "Health"),
                            ("Technology Products & Services", "Defence and Security"),
                            ("Digital Future", "Defence and Security"),
                            ("Digital Future", "Local Community and Housing"), ("Fleet", "Defence and Security"),
                            ("Document Management & Logistics", "Government Policy"), ("Construction", "Education"),
                            ("Network Services", "Government Policy"), ("Construction", "Local Community and Housing"),
                            ("Financial Services", "Defence and Security"),
                            ("Technology Products & Services", "Education"), ("People Services", "Government Policy"),
                            ("Energy", "Defence and Security"), ("Travel", "Health"),
                            ("Construction", "Defence and Security"),
                            ("Technology Products & Services", "Government Policy"),
                            ("Workforce Health & Education", "Education"), ("Energy", "Government Policy"),
                            ("Travel", "Government Policy"), ("Workplace", "Health"),
                            ("Technology Solutions & Outcomes", "Health"),
                            ("Professional Services", "Government Policy"), ("Financial Services", "Government Policy"),
                            ("Digital Future", "Education"), ("Workplace", "Defence and Security"),
                            ("Digital Future", "Infrastructure"), ("People Services", "Defence and Security"),
                            ("Fleet", "Government Policy"), ("Workforce Health & Education", "Government Policy"),
                            ("Professional Services", "Health"), ("Travel", "Local Community and Housing"),
                            ("Construction", "Health"), ("PSR & Permanent Recruitment", "Defence and Security"),
                            ("Fleet", "Education"), ("Network Services", "Infrastructure"),
                            ("Document Management & Logistics", "Culture, Media and Sport"),
                            ("Travel", "Defence and Security"), ("People Services", "Health"),
                            ("Document Management & Logistics", "Infrastructure"),
                            ("Financial Services", "Infrastructure"),
                            ("Professional Services", "Local Community and Housing"),
                            ("Document Management & Logistics", "Unassigned"),
                            ("Financial Services", "Culture, Media and Sport"),
                            ("Network Services", "Culture, Media and Sport"),
                            ("People Services", "Local Community and Housing"), ("Energy", "Infrastructure"),
                            ("Technology Products & Services", "Infrastructure"),
                            ("Marcomms & Research", "Government Policy"), ("Workplace", "Local Community and Housing"),
                            ("Energy", "Culture, Media and Sport"), ("Fleet", "Infrastructure"),
                            ("Workforce Health & Education", "Defence and Security"),
                            ("Network Services", "Unassigned"), ("Professional Services", "Defence and Security"),
                            ("Technology Solutions & Outcomes", "Defence and Security"),
                            ("Professional Services", "Infrastructure"),
                            ("Technology Solutions & Outcomes", "Government Policy"),
                            ("Construction", "Government Policy"), ("Workplace", "Government Policy"),
                            ("PSR & Permanent Recruitment", "Government Policy"),
                            ("Workforce Health & Education", "Local Community and Housing"),
                            ("Workforce Health & Education", "Unassigned"), ("Travel", "Education"),
                            ("Energy", "Unassigned"), ("Travel", "Infrastructure"),
                            ("Financial Services", "Unassigned"),
                            ("Technology Products & Services", "Culture, Media and Sport"),
                            ("Digital Future", "Culture, Media and Sport"), ("Workplace", "Education"),
                            ("Professional Services", "Education"), ("People Services", "Infrastructure"),
                            ("People Services", "Education"), ("Travel", "Culture, Media and Sport"),
                            ("Workplace", "Infrastructure"), ("Construction", "Infrastructure"),
                            ("Technology Solutions & Outcomes", "Education"), ("PSR & Permanent Recruitment", "Health"),
                            ("Fleet", "Unassigned"), ("Marcomms & Research", "Defence and Security"),
                            ("Technology Solutions & Outcomes", "Infrastructure"),
                            ("Fleet", "Culture, Media and Sport"), ("Marcomms & Research", "Health"),
                            ("Marcomms & Research", "Infrastructure"), ("Technology Products & Services", "Unassigned"),
                            ("Managed Service", "Defence and Security"),
                            ("People Services", "Culture, Media and Sport"),
                            ("Technology Solutions & Outcomes", "Culture, Media and Sport"),
                            ("Workforce Health & Education", "Infrastructure"), ("Marcomms & Research", "Education"),
                            ("Construction", "Unassigned"), ("Professional Services", "Culture, Media and Sport"),
                            ("Workforce Health & Education", "Culture, Media and Sport"),
                            ("PSR & Permanent Recruitment", "Infrastructure"),
                            ("Marcomms & Research", "Culture, Media and Sport"),
                            ("Marcomms & Research", "Local Community and Housing"),
                            ("Construction", "Culture, Media and Sport"), ("Workplace", "Culture, Media and Sport"),
                            ("Digital Future", "Unassigned"),
                            ("PSR & Permanent Recruitment", "Local Community and Housing"), ("Travel", "Unassigned"),
                            ("PSR & Permanent Recruitment", "Education"),
                            ("PSR & Permanent Recruitment", "Culture, Media and Sport"), ("Workplace", "Unassigned"),
                            ("Contact Centres", "Government Policy"), ("Technology Solutions & Outcomes", "Unassigned"),
                            ("People Services", "Unassigned"), ("Contact Centres", "Defence and Security"),
                            ("Professional Services", "Unassigned"), ("Marcomms & Research", "Unassigned"),
                            ("Contact Centres", "Health"), ("Financial Planning & Estates", "Government Policy"),
                            ("Contact Centres", "Education"), ("Managed Service", "Government Policy"),
                            ("PSR & Permanent Recruitment", "Unassigned"),
                            ("Commodities and Innovation", "Government Policy"), ("Contact Centres", "Infrastructure"),
                            ("Contact Centres", "Local Community and Housing"),
                            ("Financial Planning & Estates", "Defence and Security"),
                            ("Financial Planning & Estates", "Culture, Media and Sport"),
                            ("Financial Planning & Estates", "Education"),
                            ("Financial Planning & Estates", "Local Community and Housing"),
                            ("Contact Centres", "Unassigned"), ("Contact Centres", "Culture, Media and Sport"),
                            ("Managed Service", "Education"), ("Managed Service", "Unassigned"),
                            ("Below Threshold", "Health"), ("Financial Planning & Estates", "Unassigned"),
                            ("Managed Service", "Infrastructure"), ("Below Threshold", "Defence and Security"),
                            ("Below Threshold", "Local Community and Housing"),
                            ("Below Threshold", "Government Policy"), ("Below Threshold", "Education"),
                            ("Below Threshold", "Culture, Media and Sport")]
    return category_sector_list[index]


def visualize_raw_data(df: pd.DataFrame, category: str, sector: str, run: bool = False):
    """Plot the spend data for the given category and market sector across all the months
    Args:
        df: DataFrame containing data
        category: Category for which spend data should be plot
        sector: Market sector for which spend data should be plot
        run: A flag to run this function, pass true to run this function.

    Returns:
        None

    """
    if run:
        category_sector_spend = df[(df["Category"] == category) & (df["MarketSector"] == sector)].reset_index(drop=True)
        spend = category_sector_spend["EvidencedSpend"]
        labels = pd.date_range(start=category_sector_spend["SpendMonth"].min(axis=0),
                               end=category_sector_spend["SpendMonth"].max(axis=0),
                               freq="MS").tolist()[::4]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(category_sector_spend["SpendMonth"], spend / 1e6, label=f"{category}:{sector}")
        ax.set_xlabel("Year-Month")
        ax.set_ylabel("Monthly Spend (Millions £)")
        plt.xticks(labels)
        plt.title(f"Monthly Spend (Millions £) for\n{category} : {sector}")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.legend()
        plt.show()


def adf_stationary_test(series: Union[pd.DataFrame, pd.Series, np.ndarray]) -> bool:
    """
    Test stationarity of a series bases on Augmented Dickey-Fuller test and returns true if series is stationary
    otherwise false

    Args:
        series: Timeseries to be tested for stationarity.

    Returns:
        True is series is stationary otherwise False

    """

    result = adfuller(series)
    adf_statistics = result[0]
    p_value = result[1]
    critical_values = result[4]
    return adf_statistics < critical_values["1%"] and p_value < 0.05


def is_spend_random_walk(df: pd.DataFrame, category: str, sector: str, run: bool = False):
    """A function that plots the graph to show is the data is random

    Args:
        df: DataFrame containing data
        category: Category for which spend data should be plot
        sector: Market sector for which spend data should be plot
        run: A flag to run this function, pass true to run this function.

    Returns:
        None

    """
    if run:
        category_sector_spend = df[(df["Category"] == category) & (df["MarketSector"] == sector)].reset_index(drop=True)
        spend = category_sector_spend["EvidencedSpend"]
        labels = pd.date_range(start=category_sector_spend["SpendMonth"].min(axis=0),
                               end=category_sector_spend["SpendMonth"].max(axis=0),
                               freq="MS").tolist()[::4]
        fig_size = (12, 6)
        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(category_sector_spend["SpendMonth"], spend / 1e6, label=f"{category}:{sector}")
        ax.set_xlabel("Year-Month")
        ax.set_ylabel("Monthly Spend (Millions £)")
        plt.xticks(labels)
        plt.title(f"Monthly Spend (Millions £) for\n{category} : {sector}")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.legend()
        plt.show()

        logger.debug(f"Look for the trend in the above plot and see for sudden or sharp changes.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
        fig.suptitle(f"ACF plot and PACF plot of Spend across\n{category} : {sector}")
        plot_acf(spend, lags=20, ax=ax1)
        ax1.title.set_text(f"ACF plot of Spend across\n{category} : {sector}")
        plot_pacf(spend, lags=20, ax=ax2)
        ax2.title.set_text(f"PACF plot of Spend across\n{category} : {sector}")
        plt.tight_layout()
        plt.show()

        logger.debug(f"Look for the pattern in the above graph.")

        is_stationary = adf_stationary_test(spend)
        logger.info(f"Spend for {category}: {sector} is stationary? {is_stationary}")

        if not is_stationary:
            diff = np.diff(spend, n=1)
            is_stationary = adf_stationary_test(diff)
            logger.debug(f"1st order differencing for spend of {category}: {sector} is stationary? {is_stationary}")
            if is_stationary:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
                fig.suptitle(f"ACF plot and PACF plot of 1st order differencing spend across\n{category} : {sector}")
                plot_acf(diff, lags=20, ax=ax1)
                ax1.title.set_text(f"ACF plot of 1st order differencing")
                plot_pacf(diff, lags=20, ax=ax2)
                ax2.title.set_text(f"PACF plot of 1st order differencing")
                plt.tight_layout()
                plt.show()
            else:
                diff = np.diff(spend, n=1)
                is_stationary = adf_stationary_test(diff)
                logger.debug(f"2nd order differencing for spend of {category}: {sector} is stationary? {is_stationary}")
                if is_stationary:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
                    fig.suptitle(
                        f"ACF plot and PACF plot of 2nd order differencing spend across\n{category} : {sector}")
                    plot_acf(diff, lags=20, ax=ax1)
                    ax1.title.set_text(f"ACF plot of 2nd order differencing")
                    plot_pacf(diff, lags=20, ax=ax2)
                    ax2.title.set_text(f"PACF plot of 2nd order differencing")
                    plt.tight_layout()
                    plt.show()


def arma_aic_scores(timeseries: Union[pd.Series, np.ndarray, list], pq_combinations: list) -> pd.DataFrame:
    """Returns Akike Information Criteria (AIC) in ascending order for all models trained using the given parameters.
    Args:
        timeseries: Timeseries data
        pq_combinations: List of all combination of p and q that you want to test

    Returns:
        DataFrame containing parameters and its respective aic score, sorted by aic score ascending.

    """
    aic_scores = []
    for p, q in pq_combinations:
        d = 0
        model = SARIMAX(timeseries, order=(p, d, q), simple_differencing=False).fit(disp=False)
        aic = model.aic
        aic_scores.append([p, q, aic])

    df = pd.DataFrame(data=aic_scores, columns=["p", "q", "aic"]) \
        .sort_values(by="aic", ascending=True) \
        .reset_index(drop=True)

    return df


def arima_aic_scores(timeseries: Union[pd.Series, np.ndarray, list], pq_combinations: list, d: int) -> pd.DataFrame:
    """Returns Akike Information Criteria (AIC) in ascending order for all models trained using the given parameters.

    Args:
        timeseries: Timeseries data
        pq_combinations: List of all combination of p and q that you want to test
        d: Integration order

    Returns:
        DataFrame containing parameters and its respective aic score, sorted by aic score ascending.

    """
    aic_scores = []
    for p, q in pq_combinations:
        model = SARIMAX(timeseries, order=(p, d, q), simple_differencing=False).fit(disp=False)
        aic = model.aic
        aic_scores.append([p, q, aic])

    df = pd.DataFrame(data=aic_scores, columns=["p", "q", "aic"]) \
        .sort_values(by="aic", ascending=True) \
        .reset_index(drop=True)

    return df


def sarima_aic_scores(timeseries: Union[pd.Series, np.ndarray, list], pqPQ_combinations: list, d: int, D: int,
                      s: int) -> pd.DataFrame:
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
            model = SARIMAX(timeseries, order=(p, d, q), seasonal_order=(P, D, Q, s), simple_differencing=False).fit(
                disp=False)
            aic = model.aic
            aic_scores.append([p, q, d, P, Q, D, s, aic])
        except Exception as e:
            logger.error(f"Error while calculating SARIMA aic scores: {e}")

    df = pd.DataFrame(data=aic_scores, columns=["p", "q", "d", "P", "Q", "D", "s", "aic"]) \
        .sort_values(by="aic", ascending=True) \
        .reset_index(drop=True)

    return df


def ljung_box_residual_test(residuals: pd.Series) -> bool:
    """Perform the Ljung–Box test on the residual and test if the residuals are white noise.
    Args:
        residuals: Residuals from the model

    Returns:
        True if the residuals is white noise else returns False

    """
    ljung_box_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
    is_residual_white_noise = (ljung_box_result["lb_pvalue"] > 0.05).all()
    return is_residual_white_noise


def find_integration_order(timeseries: Union[pd.Series, np.ndarray, list]) -> int:
    """Finds the integration order (denoted by d) of the timeseries
    Args:
        timeseries: Timeseries for which you wish to find the integration order

    Returns:
        Integration order as integer

    """

    def integration_finder(ts: Union[pd.Series, np.ndarray, list], d: int = 0) -> int:
        return d if adf_stationary_test(ts) else integration_finder(np.diff(ts, n=1), d + 1)

    return integration_finder(timeseries)


def find_seasonal_integration_order(timeseries: Union[pd.Series, np.ndarray, list], s=0) -> int:
    """Finds the integration order (denoted by d) of the timeseries
    Args:
        timeseries: Timeseries for which you wish to find the integration order
        s: Seasonal Order. E.g. for monthly it os 12.

    Returns:
        Seasonal Integration order as integer

    """

    def integration_finder(ts: Union[pd.Series, np.ndarray, list], d: int = 0) -> int:
        return d if adf_stationary_test(ts) else integration_finder(np.diff(ts, n=s), d + 1)

    return integration_finder(timeseries)


def rolling_forecast(df: pd.DataFrame, train_size: int, prediction_size: int, window: int, method: str, p: int,
                     q: int) -> list:
    """A method that repeatedly fit a model and forecast over a specified window, until all the future (horizon)
        prediction is done.

    Args:
        df: DataFrame containing timeseries
        train_size: Size of the train set that can be used to fit a model
        prediction_size: Number of future timesteps that needs to be predicted.
        window: The window of prediction or the order of the AR(p) process for SARIMAX i.e. how many steps are predicted
                at a time.
        method: A string to tell which model to calculate i.e Historical mean (mean), last know (last_value) value or
                ARMA(p,q)
        p: Order of AR process
        q: Order of MA process

    Returns:
        List of predictions

    """

    total_size = train_size + prediction_size
    if method == "mean":
        pred_mean = []
        for i in range(train_size, total_size, window):
            mean = np.mean(df[:i].values)
            pred_mean.extend(mean for _ in range(window))
        return pred_mean

    elif method == "last_value":
        pred_last_value = []
        for i in range(train_size, total_size, window):
            last_value = df[:i].iloc[-1].values[0]
            pred_last_value.extend(last_value for _ in range(window))
        return pred_last_value

    elif method == "ARMA":
        pred_ARMA = []
        for i in range(train_size, total_size, window):
            model = SARIMAX(df[:i], order=(p, 0, q))
            result = model.fit(disp=False)
            predictions = result.get_prediction(i, i + window - 1)
            oos_pred = predictions.predicted_mean.iloc[-window:]
            pred_ARMA.extend(oos_pred)
        return pred_ARMA
    else:
        return []


def mean_absolute_percentage_error(y_true: Union[pd.Series, np.ndarray, list],
                                   y_pred: Union[pd.Series, np.ndarray, list]) -> float:
    """Calculates the Mean Absolute Percentage Error between true value and predicted value
    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Mean Absolute Percentage Error

    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def model_arma(df: pd.DataFrame, category: str, sector: str, run: bool = False):
    """Forecast the data using ARMA(p,q) model.

    Args:
        df: DataFrame containing data
        category: Category for which spend data should be forecasted
        sector: Market sector for which spend data should be forecasted
        run: A flag to run this function, pass true to run this function.

    Returns:
        None

    """

    if run:
        category_sector_spend = df[(df["Category"] == category) & (df["MarketSector"] == sector)].reset_index(drop=True)
        spend = category_sector_spend["EvidencedSpend"]
        labels = pd.date_range(start=category_sector_spend["SpendMonth"].min(axis=0),
                               end=category_sector_spend["SpendMonth"].max(axis=0),
                               freq="MS").tolist()[::4]
        fig_size = (12, 6)
        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(category_sector_spend["SpendMonth"], spend / 1e6, label=f"{category}:{sector}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Spend (Millions £)")
        plt.xticks(labels)
        plt.title(f"Monthly Spend (Millions £) for\n{category} : {sector}")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.legend()
        plt.show()

        is_stationary = adf_stationary_test(spend)
        logger.debug(f"Spend for {category} : {sector} is stationary? {is_stationary}")

        if not is_stationary:
            diff = np.diff(spend, n=1)
            is_stationary = adf_stationary_test(diff)
            logger.debug(f"Is 1st order differencing for {category} : {sector} stationary? {is_stationary}")

            fig, ax = plt.subplots(figsize=fig_size)
            ax.plot(category_sector_spend["SpendMonth"].iloc[1:], diff / 1e6, label=f"{category}:{sector}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Monthly Spend (Millions £) - Diff")
            plt.xticks(labels)
            plt.title(f"Spend across {category} : {sector}")
            plt.title("Differenced Data")
            plt.title(f"1st order differencing for\n{category} : {sector}")
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.show()

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
            fig.suptitle(f"ACF plot and PACF plot of 1st order differencing spend across\n{category} : {sector}")
            plot_acf(diff, lags=20, ax=ax1)
            ax1.title.set_text(f"ACF plot of 1st order differencing")
            plot_pacf(diff, lags=20, ax=ax2)
            ax2.title.set_text(f"PACF plot of 1st order differencing")
            plt.tight_layout()
            plt.show()

            df_diff = pd.DataFrame({"diff": diff})
            dataset_size = len(df_diff)
            train_ratio = 0.9
            train_size = int(dataset_size * train_ratio)
            test_size = dataset_size - train_size
            train = df_diff[:train_size]
            test = df_diff[train_size:]

            logger.debug(f"Total dataset size...: {dataset_size}")
            logger.debug(f"Training dataset size: {train_size}")
            logger.debug(f"Test dataset size....: {test_size}")

            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex="all", figsize=fig_size)
            ax1.plot(category_sector_spend["SpendMonth"], spend / 1e6, label=f"{category}:{sector}")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Monthly Spend (Millions £)")
            ax1.axvspan(date2num(category_sector_spend["SpendMonth"].iloc[train_size]),
                        date2num(category_sector_spend["SpendMonth"].iloc[dataset_size]),
                        color="#808080", alpha=0.2)
            ax2.plot(category_sector_spend["SpendMonth"].iloc[1:], diff / 1e6, label=f"Diff {category}:{sector}")
            ax2.set_xlabel("Month")
            ax2.set_ylabel("Monthly Spend (Millions £) - Diff")
            ax2.axvspan(date2num(category_sector_spend['SpendMonth'].iloc[train_size]),
                        date2num(category_sector_spend['SpendMonth'].iloc[dataset_size]),
                        color="#808080", alpha=0.2)
            plt.xticks(labels)
            fig.suptitle("Train and test sets for the original and differenced series")
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.show()

            ps = range(0, 4, 1)  # Possible value of order p of AR(p)
            qs = range(0, 4, 1)  # Possible value of order q of MA(q)
            pq_combinations = list(product(ps, qs))
            aic_scores = arma_aic_scores(train["diff"], pq_combinations)
            logger.debug(f"AIC scores are:\n{aic_scores.head(len(pq_combinations))}")
            lowest_aic_score = aic_scores.iloc[0]
            best_p = int(lowest_aic_score["p"])
            best_q = int(lowest_aic_score["q"])
            logger.debug(f"Best p is: {best_p} and best q is {best_q}")
            d = 0
            model = SARIMAX(train["diff"], order=(best_p, d, best_q), simple_differencing=False)
            model_fit = model.fit(disp=False)
            logger.debug(f"Model fit summary:\n{model_fit.summary()}")
            fig = model_fit.plot_diagnostics(figsize=(10, 8))
            fig.suptitle(f"Model diagnostics of ARMA({best_p},{best_q})")
            plt.show()

            residuals = model_fit.resid
            is_residual_white_noise = ljung_box_residual_test(residuals)
            logger.debug(f"Is residual just random error? {is_residual_white_noise}")

            window = best_p
            pred_mean = rolling_forecast(df_diff, train_size, test_size, window, "mean", best_p, best_q)
            pred_last_value = rolling_forecast(df_diff, train_size, test_size, window, "last_value", best_p, best_q)
            pred_ARMA = rolling_forecast(df_diff, train_size, test_size, window, "ARMA", best_p, best_q)

            test.loc[:, "pred_mean"] = pred_mean
            test.loc[:, "pred_last_value"] = pred_last_value
            test.loc[:, "pred_ARMA"] = pred_ARMA
            logger.debug(f"Test comparisons (diff):\n{test.tail(test_size)}")

            fig, ax = plt.subplots(figsize=fig_size)
            ax.plot(category_sector_spend["SpendMonth"].iloc[1:], diff / 1e6, label=f"Diff {category}:{sector}")
            ax.plot(category_sector_spend["SpendMonth"].iloc[-test_size:], test["diff"] / 1e6, "b-",
                    label="Actual")
            ax.plot(category_sector_spend["SpendMonth"].iloc[-test_size:], test["pred_mean"] / 1e6, "g:",
                    label="Mean")
            ax.plot(category_sector_spend["SpendMonth"].iloc[-test_size:], test["pred_last_value"] / 1e6, "r-.",
                    label="Last")
            ax.plot(category_sector_spend["SpendMonth"].iloc[-test_size:], test["pred_ARMA"] / 1e6, "k--",
                    label=f"ARMA({best_p},{best_q})")
            ax.legend(loc=2)
            ax.set_xlabel("Month")
            ax.set_ylabel("Monthly Spend (Millions £) - Diff")
            ax.axvspan(date2num(category_sector_spend['SpendMonth'].iloc[train_size]),
                       date2num(category_sector_spend['SpendMonth'].iloc[dataset_size]),
                       color="#808080", alpha=0.2)
            plt.xticks(labels)
            plt.title("Forecasts of the differenced monthly spend using \nthe mean, the last known value, and \n"
                      f"an ARMA({best_p},{best_q}) model for\n{category}:{sector}")
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.show()

            mse_mean = mean_squared_error(test["diff"], test["pred_mean"])
            mse_last_value = mean_squared_error(test["diff"], test["pred_last_value"])
            mse_ARMA = mean_squared_error(test["diff"], test["pred_ARMA"])
            logger.debug(f"Mean Squared Error of historical mean: {mse_mean:.2f}")
            logger.debug(f"Mean Squared Error of last know value: {mse_last_value:.2f}")
            logger.debug(f"Mean Squared Error of ARMA({best_p},{best_q})......: {mse_ARMA:.2f}")

            mae_mean = mean_absolute_error(test["diff"], test["pred_mean"])
            mae_last_value = mean_absolute_error(test["diff"], test["pred_last_value"])
            mae_ARMA = mean_absolute_error(test["diff"], test["pred_ARMA"])
            logger.debug(f"Mean Absolute Error of historical mean: {mae_mean:.2f}")
            logger.debug(f"Mean Absolute Error of last know value: {mae_last_value:.2f}")
            logger.debug(f"Mean Absolute Error of ARMA({best_p},{best_q})......: {mae_ARMA:.2f}")

            category_sector_spend["forecast_mean"] = pd.Series(dtype=float)
            category_sector_spend["forecast_mean"][-test_size:] = test["pred_mean"]
            category_sector_spend["forecast_last_value"] = pd.Series(dtype=float)
            category_sector_spend["forecast_last_value"][-test_size:] = test["pred_last_value"]
            category_sector_spend["forecast"] = pd.Series(dtype=float)
            category_sector_spend["forecast"][-test_size:] = \
                category_sector_spend["EvidencedSpend"].iloc[-test_size] + test["pred_ARMA"].cumsum()

            fig, ax = plt.subplots(figsize=fig_size)
            ax.plot(category_sector_spend["SpendMonth"], category_sector_spend["EvidencedSpend"] / 1e6,
                    label=f"{category}:{sector}")
            ax.plot(category_sector_spend["SpendMonth"], category_sector_spend["forecast"] / 1e6, "k--",
                    label=f"ARMA({best_p},{best_q})")
            ax.legend(loc=2)
            ax.set_xlabel("Month")
            ax.set_ylabel("Monthly Spend (Millions £)")
            ax.axvspan(date2num(category_sector_spend["SpendMonth"].iloc[train_size]),
                       date2num(category_sector_spend["SpendMonth"].iloc[dataset_size]),
                       color="#808080", alpha=0.2)
            plt.xticks(labels)
            plt.title(f"Forecast for\n{category} : {sector}")
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.show()

            mae_mean_original = mean_absolute_error(category_sector_spend["EvidencedSpend"][-test_size:],
                                                    category_sector_spend["forecast_mean"][-test_size:])
            mae_last_value = mean_absolute_error(category_sector_spend["EvidencedSpend"][-test_size:],
                                                 category_sector_spend["forecast_last_value"][-test_size:])
            mae_arma = mean_absolute_error(category_sector_spend["EvidencedSpend"][-test_size:],
                                           category_sector_spend["forecast"][-test_size:])
            logger.debug(f"Mean Absolute Error of mean on original data: {mae_mean_original:.2f}")
            logger.debug(f"Mean Absolute Error of last on original data: {mae_last_value:.2f}")
            logger.debug(f"Mean Absolute Error of ARMA on original data: {mae_arma:.2f}")

            mape_mean = mean_absolute_percentage_error(category_sector_spend["EvidencedSpend"][-test_size:],
                                                       category_sector_spend["forecast_mean"][-test_size:])
            mape_last = mean_absolute_percentage_error(category_sector_spend["EvidencedSpend"][-test_size:],
                                                       category_sector_spend["forecast_last_value"][-test_size:])
            mape_arma = mean_absolute_percentage_error(category_sector_spend["EvidencedSpend"][-test_size:],
                                                       category_sector_spend["forecast"][-test_size:])
            logger.debug(f"Mean Absolute Percentage Error of mean on original data: {mape_mean:.2f}")
            logger.debug(f"Mean Absolute Percentage Error of last on original data: {mape_last:.2f}")
            logger.debug(f"Mean Absolute Percentage Error of ARMA on original data: {mape_arma:.2f}")

            fig, ax = plt.subplots()
            x = ["Mean", "Last Month", f"ARIMA({best_p},{d},{best_q})"]
            y = [mape_mean, mape_last, mape_arma]
            ax.bar(x, y, width=0.4)
            ax.set_xlabel("Models")
            ax.set_ylabel("Mean Absolute Percentage Error (%)")
            for index, value in enumerate(y):
                plt.text(x=index, y=value + 1, s=str(round(value, 2)), ha="center")
            plt.title("Mean Absolute Percentage Error")
            plt.tight_layout()
            plt.show()

            fig, ax = plt.subplots()
            x = ["Mean", "Last Month", f"ARIMA({best_p},{d},{best_q})"]
            y = [mae_mean / 1e6, mae_last_value / 1e6, mae_arma / 1e6]
            ax.bar(x, y, width=0.4)
            ax.set_xlabel("Models")
            ax.set_ylabel("Mean Absolute Error (Millions £)")
            for index, value in enumerate(y):
                plt.text(x=index, y=value + 1, s=str(round(value, 2)), ha="center")
            plt.title("Mean Absolute Error")
            plt.tight_layout()
            plt.show()


def model_arima(df: pd.DataFrame, category: str, sector: str, run: bool = False):
    """Forecast the data using ARIMA(p,d,q) model.

    Args:
        df: DataFrame containing data
        category: Category for which spend data should be forecasted
        sector: Market sector for which spend data should be forecasted
        run: A flag to run this function, pass true to run this function

    Returns:
        None

    """
    if run:
        category_sector_spend = df[(df["Category"] == category) & (df["MarketSector"] == sector)].reset_index(drop=True)
        spend = category_sector_spend["EvidencedSpend"]
        labels = pd.date_range(start=category_sector_spend["SpendMonth"].min(axis=0),
                               end=category_sector_spend["SpendMonth"].max(axis=0),
                               freq="MS").tolist()[::4]
        fig_size = (12, 6)
        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(category_sector_spend["SpendMonth"], spend / 1e6, label=f"{category}:{sector}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Spend (Millions £)")
        plt.xticks(labels)
        plt.title(f"Monthly Spend (Millions £) for\n{category} : {sector}")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.legend()
        plt.show()

        d = find_integration_order(spend)
        logger.debug(f"Integration order is {d}")

        dataset_size = len(category_sector_spend)
        train_ratio = 0.9
        train_size = int(dataset_size * train_ratio)
        test_size = dataset_size - train_size

        train = category_sector_spend[:train_size]
        test = category_sector_spend[train_size:]

        logger.debug(f"Size of dataset is {dataset_size} training set is {train_size} and test size is: {test_size}")

        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(category_sector_spend["SpendMonth"], spend / 1e6, label=f"{category}:{sector}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Spend (Millions £)")
        ax.axvspan(date2num(category_sector_spend["SpendMonth"].iloc[-test_size]),
                   date2num(category_sector_spend["SpendMonth"].iloc[-1]),
                   color="#808080", alpha=0.2)
        plt.xticks(labels)
        plt.title("Train and test sets")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

        ps = range(0, 4, 1)  # Possible value of order p of AR(p)
        qs = range(0, 4, 1)  # Possible value of order q of MA(q)
        pq_combinations = list(product(ps, qs))
        logger.info(f"Possible pair of p and q are: {pq_combinations}")
        aic_scores = arima_aic_scores(train["EvidencedSpend"], pq_combinations, d)
        logger.debug(f"AIC scores are:\n{aic_scores.head(len(pq_combinations))}")
        lowest_aic_score = aic_scores.iloc[0]
        best_p = int(lowest_aic_score["p"])
        best_q = int(lowest_aic_score["q"])
        print(f"Best p is: {best_p} and best q is {best_q}")

        model = SARIMAX(train["EvidencedSpend"], order=(best_p, d, best_q), simple_differencing=False)
        model_fit = model.fit(disp=False)
        logger.debug(f"Model fit summary:\n{model_fit.summary()}")
        fig = model_fit.plot_diagnostics(figsize=(10, 8))
        fig.suptitle(f"Model diagnostics of ARIMA({best_p},{d},{best_q})")
        plt.show()

        residuals = model_fit.resid
        is_residual_white_noise = ljung_box_residual_test(residuals)
        logger.debug(f"Is residual just random error? {is_residual_white_noise}")

        test["last_month"] = category_sector_spend["EvidencedSpend"].iloc[train_size - 1:dataset_size - 1].values
        pred = model_fit.get_prediction(train_size, dataset_size - 1).predicted_mean
        test["forecast"] = pred
        logger.debug(f"Prediction on test\n{test.head(test_size)}")

        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(category_sector_spend["SpendMonth"], spend / 1e6, label=f"{category}:{sector}")
        ax.plot(test["SpendMonth"], test["last_month"] / 1e6, "r:", label="Last Month")
        ax.plot(test["SpendMonth"], test["forecast"] / 1e6, "k--", label=f"ARIMA({best_p},{d},{best_q})")
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Spend (Millions £)")
        ax.axvspan(date2num(category_sector_spend["SpendMonth"].iloc[-test_size]),
                   date2num(category_sector_spend["SpendMonth"].iloc[-1]),
                   color="#808080", alpha=0.2)
        ax.legend(loc=2)
        plt.xticks(labels)
        plt.title("Forecast")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

        last_month_mean_absolute_percentage_error = mean_absolute_percentage_error(test["EvidencedSpend"],
                                                                                   test["last_month"])
        arima_mean_absolute_percentage_error = mean_absolute_percentage_error(test["EvidencedSpend"], test["forecast"])
        logger.debug(f"Last Month mean absolute percentage error....: {last_month_mean_absolute_percentage_error:.2f}")
        logger.debug(f"ARIMA seasonal mean absolute percentage error: {arima_mean_absolute_percentage_error:.2f}")

        last_month_mae = mean_absolute_error(test["EvidencedSpend"], test["last_month"])
        arima_mae = mean_absolute_error(test["EvidencedSpend"], test["forecast"])
        logger.debug(f"Last Month mean absolute error....: {last_month_mae:.2f}")
        logger.debug(f"ARIMA seasonal mean absolute error: {arima_mae:.2f}")

        fig, ax = plt.subplots()
        x = ["Last Month", f"ARIMA({best_p},{d},{best_q})"]
        y = [last_month_mean_absolute_percentage_error, arima_mean_absolute_percentage_error]
        ax.bar(x, y, width=0.4)
        ax.set_xlabel("Models")
        ax.set_ylabel("Mean Absolute Percentage Error (%)")
        for index, value in enumerate(y):
            plt.text(x=index, y=value + 1, s=str(round(value, 2)), ha="center")
        plt.title("Mean Absolute Percentage Error")
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots()
        x = ["Last Month", f"ARIMA({best_p},{d},{best_q})"]
        y = [last_month_mae / 1e6, arima_mae / 1e6]
        ax.bar(x, y, width=0.4)
        ax.set_xlabel("Models")
        ax.set_ylabel("Mean Absolute Error (Millions £)")
        for index, value in enumerate(y):
            plt.text(x=index, y=value + 1, s=str(round(value, 2)), ha="center")
        plt.title("Mean Absolute Error")
        plt.tight_layout()
        plt.show()


def model_sarima(df: pd.DataFrame, category: str, sector: str, run: bool = False):
    """Forecast the data using SARIMA(p,d,q)(P,D,Q)m model.

    Args:
        df: DataFrame containing data
        category: Category for which spend data should be forecasted
        sector: Market sector for which spend data should be forecasted
        run: A flag to run this function, pass true to run this function

    Returns:
        None

    """

    if run:
        category_sector_spend = df[(df["Category"] == category) & (df["MarketSector"] == sector)].reset_index(drop=True)
        logger.debug(f"Dataset size is: {category_sector_spend.shape[0]}")
        spend = category_sector_spend["EvidencedSpend"]
        labels = pd.date_range(start=category_sector_spend["SpendMonth"].min(axis=0),
                               end=category_sector_spend["SpendMonth"].max(axis=0),
                               freq="MS").tolist()[::4]
        fig_size = (12, 6)
        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(category_sector_spend["SpendMonth"], spend / 1e6, label=f"{category}:{sector}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Spend (Millions £)")
        plt.xticks(labels)
        plt.title(f"Monthly Spend (Millions £) for\n{category} : {sector}")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.legend()
        plt.show()

        d = find_integration_order(spend)
        logger.debug(f"Integration order is {d}")

        dataset_size = len(category_sector_spend)
        train_ratio = 0.9
        train_size = int(dataset_size * train_ratio)
        test_size = dataset_size - train_size

        train = category_sector_spend[:train_size]
        test = category_sector_spend[train_size:]

        logger.debug(f"Size of dataset is {dataset_size} training set is {train_size} and test size is: {test_size}")

        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(category_sector_spend["SpendMonth"], spend / 1e6, label=f"{category}:{sector}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Spend (Millions £)")
        ax.axvspan(date2num(category_sector_spend["SpendMonth"].iloc[-test_size]),
                   date2num(category_sector_spend["SpendMonth"].iloc[-1]),
                   color="#808080", alpha=0.2)
        plt.xticks(labels)
        plt.title("Train and test sets")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

        test["last_month"] = category_sector_spend["EvidencedSpend"].iloc[train_size - 1:dataset_size - 1].values

        ps = range(0, 4, 1)  # Possible value of order p of AR(p)
        qs = range(0, 4, 1)  # Possible value of order q of MA(q)
        Ps = [0]  # Seasonality is 0 for ARIMA
        Qs = [0]  # Seasonality is 0 for ARIMA
        D = 0
        s = 12  # s is same as m
        pqPQ_combinations = list(product(ps, qs, Ps, Qs))
        aic_scores = sarima_aic_scores(train["EvidencedSpend"], pqPQ_combinations, d, D, s)
        logger.debug(f"AIC scores are:\n{aic_scores.head(len(pqPQ_combinations))}")
        lowest_aic_score = aic_scores.iloc[0]
        best_p = int(lowest_aic_score["p"])
        best_q = int(lowest_aic_score["q"])
        logger.debug(f"ARMIA({best_p},{d},{best_q}) best p is: {best_p} and best q is {best_q}")

        model = SARIMAX(train["EvidencedSpend"], order=(best_p, d, best_q), seasonal_order=(Ps[0], D, Qs[0], s),
                        simple_differencing=False)
        model_fit = model.fit(disp=False)
        logger.debug(f"Model fit summary:\n{model_fit.summary()}")
        fig = model_fit.plot_diagnostics(figsize=(10, 8))
        fig.suptitle(f"ARMIA({best_p},{d},{best_q}) model diagnostics")
        plt.show()

        residuals = model_fit.resid
        is_residual_white_noise = ljung_box_residual_test(residuals)
        logger.debug(f"Is ARMIA({best_p},{d},{best_q}) residual just random error? {is_residual_white_noise}")
        arima_pred = model_fit.get_prediction(train_size, dataset_size - 1).predicted_mean
        test["arima_forecast"] = arima_pred

        decomposition = STL(spend / 1e6, period=s).fit()
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex="all", figsize=(10, 8))
        ax1.plot(decomposition.observed)
        ax1.set_ylabel("Observed")
        ax2.plot(decomposition.trend)
        ax2.set_ylabel("Trend")
        ax3.plot(decomposition.seasonal)
        ax3.set_ylabel("Seasonal")
        ax4.plot(decomposition.resid)
        ax4.set_ylabel("Residuals")
        # plt.xticks(labels)
        plt.suptitle("Decomposing the dataset")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

        D = find_seasonal_integration_order(spend, s=s)
        ps = range(0, 4, 1)
        qs = range(0, 4, 1)
        Ps = range(0, 4, 1)
        Qs = range(0, 4, 1)
        pqPQ_combinations = list(product(ps, qs, Ps, Qs))
        # d = 1
        aic_scores = sarima_aic_scores(train["EvidencedSpend"], pqPQ_combinations, d, D, s)
        logger.debug(f"AIC scores are:\n{aic_scores.head(len(pqPQ_combinations))}")
        lowest_aic_score = aic_scores.iloc[0]
        best_p = int(lowest_aic_score["p"])
        best_q = int(lowest_aic_score["q"])
        best_P = int(lowest_aic_score["P"])
        best_Q = int(lowest_aic_score["Q"])
        logger.debug(f"Best p is: {best_p}\nBest q is {best_q}\nBest P is: {best_P}\nBest Q is: {best_Q}")
        sarima_model = SARIMAX(train["EvidencedSpend"], order=(best_p, d, best_q),
                               seasonal_order=(best_P, D, best_Q, s),
                               simple_differencing=False)
        sarima_model_fit = sarima_model.fit(disp=False)
        logger.debug(f"Model fit summary:\n{sarima_model_fit.summary()}")
        try:
            fig = sarima_model_fit.plot_diagnostics(figsize=(10, 8))
            fig.suptitle(f"Residuals diagnostics of the SARIMA({best_p},{d},{best_q})({best_P},{D},{best_Q}){s} model")
            plt.show()
        except Exception as e:
            logger.error(f"Exception occurred due to {e}")

        logger.debug("Performing Ljung-Box test on for the residuals, on 10 lags")
        residuals = sarima_model_fit.resid
        is_residual_white_noise = ljung_box_residual_test(residuals)
        logger.debug("Is SARIMA({best_p},{d},{best_q})({best_P},{D},{best_Q}){s} residual just random error?",
                     is_residual_white_noise)
        sarima_pred = sarima_model_fit.get_prediction(train_size, dataset_size - 1).predicted_mean
        test["sarima_forecast"] = sarima_pred
        logger.debug(f"SARIMA({best_p},{d},{best_q})({best_P},{D},{best_Q}){s} prediction are:\n{test.head(test_size)}")

        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(category_sector_spend["SpendMonth"], spend / 1e6, label=f"{category}:{sector}")
        ax.plot(test["SpendMonth"], test["last_month"] / 1e6, "r:", label="Previous Month")
        ax.plot(test["SpendMonth"], test["arima_forecast"] / 1e6, "k--", label=f"ARIMA({best_p},{d},{best_q})")
        ax.plot(test["SpendMonth"], test["sarima_forecast"] / 1e6, "g--",
                label=f"SARIMA({best_p},{d},{best_q})({best_P},{D},{best_Q}){s})")
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Spend (Millions £)")
        ax.axvspan(date2num(category_sector_spend["SpendMonth"].iloc[-test_size]),
                   date2num(category_sector_spend["SpendMonth"].iloc[-1]),
                   color="#808080", alpha=0.2)
        ax.legend(loc=2)
        plt.xticks(labels)
        plt.title("Forecast")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

        last_month_mean_absolute_percentage_error = mean_absolute_percentage_error(test["EvidencedSpend"],
                                                                                   test["last_month"])
        arima_mean_absolute_percentage_error = mean_absolute_percentage_error(test["EvidencedSpend"],
                                                                              test["arima_forecast"])
        sarima_mean_absolute_percentage_error = mean_absolute_percentage_error(test["EvidencedSpend"],
                                                                               test["sarima_forecast"])
        logger.debug(f"Last Month absolute percentage error..........: {last_month_mean_absolute_percentage_error:.2f}")
        logger.debug(f"ARIMA seasonal mean absolute percentage error.: {arima_mean_absolute_percentage_error:.2f}")
        logger.debug(f"SARIMA seasonal mean absolute percentage error: {sarima_mean_absolute_percentage_error:.2f}")

        fig, ax = plt.subplots()
        x = ["Last Month", f"ARIMA({best_p},{d},{best_q})", f"SARIMA({best_p},{d},{best_q})({best_P},{D},{best_Q}){s})"]
        y = [last_month_mean_absolute_percentage_error, arima_mean_absolute_percentage_error,
             sarima_mean_absolute_percentage_error]
        ax.bar(x, y, width=0.4)
        ax.set_xlabel("Models")
        ax.set_ylabel('Mean Absolute Percentage Error (%)')
        ax.set_ylim(0, 100)
        for index, value in enumerate(y):
            plt.text(x=index, y=value + 1, s=str(round(value, 2)), ha='center')
        plt.title("Mean Absolute Percentage Error")
        plt.tight_layout()
        plt.show()

        last_month_mae = mean_absolute_error(test["EvidencedSpend"], test["last_month"])
        arima_mae = mean_absolute_error(test["EvidencedSpend"], test["arima_forecast"])
        sarima_mae = mean_absolute_error(test["EvidencedSpend"], test["sarima_forecast"])
        logger.debug(f"Last Month MAE.....: {last_month_mae:.2f}")
        logger.debug(f"ARIMA seasonal MAE.: {arima_mae:.2f}")
        logger.debug(f"SARIMA seasonal MAE: {sarima_mae:.2f}")

        fig, ax = plt.subplots()
        x = ["Last Month", f"ARIMA({best_p},{d},{best_q})", f"SARIMA({best_p},{d},{best_q})({best_P},{D},{best_Q}){s})"]
        y = [last_month_mae / 1e6, arima_mae / 1e6, sarima_mae / 1e6]
        ax.bar(x, y, width=0.4)
        ax.set_xlabel("Models")
        ax.set_ylabel("Mean Absolute Error (Millions £)")
        for index, value in enumerate(y):
            plt.text(x=index, y=value + 1, s=str(round(value, 2)), ha='center')
        plt.title("Mean Absolute Error")
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(usage="data_analysis.py [path to local dataset (optional)] ",
                                     description="Exploratory Data Analysis")
    parser.add_argument("--local_data_path", default=None, metavar="Path to the local dataset folder", type=str)
    parsed = parser.parse_args()
    local_data_path = parsed.local_data_path

    raw_df = get_data_for_analysis(local_data_path)
    prepared_df = prepare_data(raw_df)

    category, sector = get_category_sector(index=0)
    visualize_raw_data(df=prepared_df, category=category, sector=sector, run=False)
    is_spend_random_walk(prepared_df, category=category, sector=sector, run=False)
    model_arma(prepared_df, category=category, sector=sector, run=False)
    model_arima(prepared_df, category=category, sector=sector, run=False)
    model_sarima(prepared_df, category=category, sector=sector, run=False)


if __name__ == '__main__':
    main()
