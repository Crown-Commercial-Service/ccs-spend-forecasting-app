import argparse
import glob
import os
import shutil
from datetime import date
from itertools import product
from typing import Union, Tuple

import warnings

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
from pandas.errors import SettingWithCopyWarning
from prophet import Prophet
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from utils import get_logger, get_database_connection

pd.set_option("display.max_rows", 0)
pd.set_option("display.max_columns", 0)
pd.set_option("expand_frame_repr", False)
pd.set_option("display.max_rows", None)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

logger = get_logger()
local_category_sector_path = os.path.join("eda", "category_sector.csv")


def get_data_for_analysis(
    local_data_path: str = None, category: str = None, sector: str = None
) -> pd.DataFrame:
    """Get the data from local machine if local_data_path is set or else will try to fetch from the database.
    Note: Fetching data from database is expensive and local stored data should be in parquet format.

    Args:
        local_data_path: Path of the data folder storing data in parquet format.
        category: Category for which spend data should be fetched
        sector: Market sector for which spend data should be fetched

    Returns:
        Returns the DataFrame

    """
    sql_filter_condition = " "
    if category:
        sql_filter_condition = (
            sql_filter_condition + f"AND spend.Category = '{category}' "
        )
    if sector:
        sql_filter_condition = sql_filter_condition + f"AND MarketSector = '{sector}' "
    sql = f"""
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
        WHERE DATEADD(month,3,CONVERT(date,CONCAT(spend.FYMonthKey,'01'),112)) < DATEADD(month, -2, GETDATE())  
        {sql_filter_condition}
        ORDER BY spend.Category, cust.MarketSector, spend.FYMonthKey
    """

    if local_data_path:
        logger.debug(f"As local_data_path is set, so using local dataset.")
        df = pd.read_parquet(local_data_path)
        if category and sector:
            df = df[(df["Category"] == category) & (df["MarketSector"] == sector)]
        elif category:
            df = df[df["Category"] == category]
        elif sector:
            df = df[df["MarketSector"] == sector]

        df.reset_index(drop=True, inplace=True)
    else:
        with get_database_connection() as con:
            logger.debug(
                "As local_data_path is not set so reading from database, this can take a while."
            )
            df = pd.read_sql(sql, con)

    df["SpendMonth"] = pd.to_datetime(df["SpendMonth"])
    return df


def aggregate_spend(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares the data for the analysis and to be used for training the model. It aggregates the data by month,
    category and market sector.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame after aggregating by Month, Category and MarketSector

    """
    return df.groupby(["SpendMonth", "Category", "MarketSector"], as_index=False).agg(
        {"EvidencedSpend": "sum"}
    )


def prepare_data(df: pd.DataFrame, category: str, sector: str) -> pd.DataFrame:
    """
    Generate a correct data range and index for the Dataframe to be used for training.
    Args:
        df: Input DataFrame
        category: Category for which spend data should be fetched
        sector: Market sector for which spend data should be fetched

    Returns:
        Generate the correct date range to be used to train models.

    """
    category_sector_spend = df[
        (df["Category"] == category) & (df["MarketSector"] == sector)
    ].reset_index(drop=True)
    category_sector_spend.set_index("SpendMonth", inplace=True, drop=False)
    end_date = max(
        [
            category_sector_spend["SpendMonth"].max(axis=0),
            date.today() - relativedelta(months=2),
        ]
    )
    complete_data = pd.DataFrame(
        data=pd.date_range(
            start=category_sector_spend["SpendMonth"].min(axis=0),
            end=end_date,
            freq="MS",
        ),
        columns=["SpendMonth"],
    )
    complete_data["Category"] = category
    complete_data["MarketSector"] = sector
    complete_data["EvidencedSpend"] = 0.0
    complete_data.set_index("SpendMonth", inplace=True, drop=False)
    complete_data.update(category_sector_spend)
    complete_data.reset_index(drop=True, inplace=True)
    return complete_data


def get_all_category_sector(latest: bool = False) -> list[Tuple[str, str]]:
    """
    Get list of all the live categories and sector. If the file eda/category_sector.csv did not exist data will be
    fetched from the database and saved to the csv file. This will help user to avoid unnecessary data fetch unlees
    they want to refresh the csv file.
    Args:
        latest: If True refreshes the eda/category_sector.csv file from the database. Default value is False

    Returns:

    """
    category_sector_list = []
    if not os.path.exists(local_category_sector_path):
        latest = True
    else:
        df = pd.read_csv(local_category_sector_path)
        category_sector_list = list(df.itertuples(index=False, name=None))

    if latest:
        sql = """
            SELECT COUNT (a.SpendMonth) AS RecordCount,a.Category,a.MarketSector
            FROM 
            (SELECT
               DATEADD(month,3,CONVERT(date,CONCAT(spend.FYMonthKey,'01'),112)) as SpendMonth,
               spend.Category,
               ISNULL(cust.MarketSector,'Unassigned') AS MarketSector
               FROM dbo.SpendAggregated AS spend
               INNER JOIN dbo.FrameworkCategoryPillar frame ON spend.FrameworkNumber = frame.FrameworkNumber
               LEFT JOIN dbo.Customers cust ON spend.CustomerURN = cust.CustomerKey
               WHERE frame.STATUS IN ('Live', 'Expired - Data Still Received') AND DATEADD
               (month,3,CONVERT(date,CONCAT(spend.FYMonthKey,'01'),112)) < DATEADD(month,-2,GETDATE())
               GROUP BY spend.Category, ISNULL(cust.MarketSector,'Unassigned'), spend.FYMonthKey
            ) AS a
            GROUP BY a.Category,a.MarketSector
            ORDER BY RecordCount DESC,a.Category,a.MarketSector
            """
        with get_database_connection() as con:
            logger.debug(
                "Fetching the latest list of categories and sector from database."
            )
            df = pd.read_sql(sql, con)
            df = df[["Category", "MarketSector"]]
            df.to_csv(local_category_sector_path, index=False)
            category_sector_list = list(df.itertuples(index=False, name=None))

    return category_sector_list


def get_category_sector(index: int = 0) -> Tuple[str, str]:
    """A helper method to return category and market sector from a list. User needs pass the index and category and
    market sector stored at that index will be returned. This is used to test model for various category and market
    sector combinations. This uses a local file called eda/category_sector.csv, if file didn't exist, data will be
    fetched from the database and saved to the csv file.
    Args:
        index: Index of the list

    Returns:
        Category and market sector

    Raises:
        Value Error if index is our of range.

    """

    if not os.path.exists(local_category_sector_path):
        get_all_category_sector(latest=True)

    df = pd.read_csv(local_category_sector_path)
    category_sector_list = list(df.itertuples(index=False, name=None))

    if index < 0 or index > len(category_sector_list):
        raise ValueError("Index or of range")
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
        category_sector_spend = prepare_data(df=df, category=category, sector=sector)
        spend = category_sector_spend["EvidencedSpend"]
        labels = pd.date_range(
            start=category_sector_spend["SpendMonth"].min(axis=0),
            end=category_sector_spend["SpendMonth"].max(axis=0),
            freq="MS",
        ).tolist()[::4]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            category_sector_spend["SpendMonth"],
            spend / 1e6,
            label=f"{category}:{sector}",
        )
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


def is_spend_random_walk(
    df: pd.DataFrame, category: str, sector: str, run: bool = False
):
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
        category_sector_spend = prepare_data(df=df, category=category, sector=sector)
        spend = category_sector_spend["EvidencedSpend"]
        labels = pd.date_range(
            start=category_sector_spend["SpendMonth"].min(axis=0),
            end=category_sector_spend["SpendMonth"].max(axis=0),
            freq="MS",
        ).tolist()[::4]
        fig_size = (12, 6)
        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(
            category_sector_spend["SpendMonth"],
            spend / 1e6,
            label=f"{category}:{sector}",
        )
        ax.set_xlabel("Year-Month")
        ax.set_ylabel("Monthly Spend (Millions £)")
        plt.xticks(labels)
        plt.title(f"Monthly Spend (Millions £) for\n{category} : {sector}")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.legend()
        plt.show()

        logger.debug(
            f"Look for the trend in the above plot and see for sudden or sharp changes."
        )

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
            logger.debug(
                f"1st order differencing for spend of {category}: {sector} is stationary? {is_stationary}"
            )
            if is_stationary:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
                fig.suptitle(
                    f"ACF plot and PACF plot of 1st order differencing spend across\n{category} : {sector}"
                )
                plot_acf(diff, lags=20, ax=ax1)
                ax1.title.set_text(f"ACF plot of 1st order differencing")
                plot_pacf(diff, lags=20, ax=ax2)
                ax2.title.set_text(f"PACF plot of 1st order differencing")
                plt.tight_layout()
                plt.show()
            else:
                diff = np.diff(spend, n=1)
                is_stationary = adf_stationary_test(diff)
                logger.debug(
                    f"2nd order differencing for spend of {category}: {sector} is stationary? {is_stationary}"
                )
                if is_stationary:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
                    fig.suptitle(
                        f"ACF plot and PACF plot of 2nd order differencing spend across\n{category} : {sector}"
                    )
                    plot_acf(diff, lags=20, ax=ax1)
                    ax1.title.set_text(f"ACF plot of 2nd order differencing")
                    plot_pacf(diff, lags=20, ax=ax2)
                    ax2.title.set_text(f"PACF plot of 2nd order differencing")
                    plt.tight_layout()
                    plt.show()


def get_aic_scores(
    timeseries: Union[pd.Series, np.ndarray, list],
    ps: Union[list, range] = None,
    qs: Union[list, range] = None,
    Ps: Union[list, range] = None,
    Qs: Union[list, range] = None,
    d: int = 0,
    D: int = 0,
    s: int = 0,
) -> pd.DataFrame:
    """
    Calculate the AIC scores of using all the provided parameter
    Args:
        timeseries: Timeseries data
        ps: List of all possible values of p that you want to test
        qs: List of all possible values of q that you want to test
        Ps: List of all possible values of P that you want to test
        Qs: List of all possible values of Q that you want to test
        d: Integration order for the series
        D: Integration order for seasonality
        s: Seasonality
    Returns:
        DataFrame containing parameters and its respective aic score, sorted by aic score ascending.
    """
    ps = ps if ps else [0]
    qs = qs if qs else [0]
    Ps = Ps if Ps else [0]
    Qs = Qs if Qs else [0]
    pqPQ_combinations = list(product(ps, qs, Ps, Qs))

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
            logger.error(f"Error while calculating aic scores: {e}")

    df = (
        pd.DataFrame(
            data=aic_scores, columns=["p", "q", "d", "P", "Q", "D", "s", "aic"]
        )
        .sort_values(by="aic", ascending=True)
        .reset_index(drop=True)
    )

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
        return (
            d
            if adf_stationary_test(ts)
            else integration_finder(np.diff(ts, n=1), d + 1)
        )

    return integration_finder(timeseries)


def find_seasonal_integration_order(
    timeseries: Union[pd.Series, np.ndarray, list], s=0
) -> int:
    """Finds the integration order (denoted by d) of the timeseries
    Args:
        timeseries: Timeseries for which you wish to find the integration order
        s: Seasonal Order. E.g. for monthly it os 12.

    Returns:
        Seasonal Integration order as integer

    """

    def integration_finder(ts: Union[pd.Series, np.ndarray, list], d: int = 0) -> int:
        return (
            d
            if adf_stationary_test(ts)
            else integration_finder(np.diff(ts, n=s), d + 1)
        )

    return integration_finder(timeseries)


def rolling_forecast(
    df: pd.DataFrame,
    train_size: int,
    prediction_size: int,
    window: int,
    method: str,
    p: int,
    q: int,
) -> list:
    """A method that repeatedly fit a model and forecast over a specified window, until all the future (horizon)
        prediction is done.

    Args:
        df: DataFrame containing timeseries
        train_size: Size of the train set that can be used to fit a model
        prediction_size: Number of future timesteps that needs to be predicted.
        window: The window of prediction or the order of the AR(p) process for SARIMAX i.e. how many steps are predicted
                at a time.
        method: A string to tell which model to calculate i.e. Historical mean (mean), last know (last_value) value or
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
        category_sector_spend = prepare_data(df=df, category=category, sector=sector)
        spend = category_sector_spend["EvidencedSpend"]
        labels = pd.date_range(
            start=category_sector_spend["SpendMonth"].min(axis=0),
            end=category_sector_spend["SpendMonth"].max(axis=0),
            freq="MS",
        ).tolist()[::4]
        fig_size = (12, 6)
        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(
            category_sector_spend["SpendMonth"],
            spend / 1e6,
            label=f"{category}:{sector}",
        )
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
            logger.debug(
                f"Is 1st order differencing for {category} : {sector} stationary? {is_stationary}"
            )

            fig, ax = plt.subplots(figsize=fig_size)
            ax.plot(
                category_sector_spend["SpendMonth"].iloc[1:],
                diff / 1e6,
                label=f"{category}:{sector}",
            )
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
            fig.suptitle(
                f"ACF plot and PACF plot of 1st order differencing spend across\n{category} : {sector}"
            )
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

            fig, (ax1, ax2) = plt.subplots(
                nrows=2, ncols=1, sharex="all", figsize=fig_size
            )
            ax1.plot(
                category_sector_spend["SpendMonth"],
                spend / 1e6,
                label=f"{category}:{sector}",
            )
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Monthly Spend (Millions £)")
            ax1.axvspan(
                date2num(category_sector_spend["SpendMonth"].iloc[train_size]),
                date2num(category_sector_spend["SpendMonth"].iloc[dataset_size]),
                color="#808080",
                alpha=0.2,
            )
            ax2.plot(
                category_sector_spend["SpendMonth"].iloc[1:],
                diff / 1e6,
                label=f"Diff {category}:{sector}",
            )
            ax2.set_xlabel("Month")
            ax2.set_ylabel("Monthly Spend (Millions £) - Diff")
            ax2.axvspan(
                date2num(category_sector_spend["SpendMonth"].iloc[train_size]),
                date2num(category_sector_spend["SpendMonth"].iloc[dataset_size]),
                color="#808080",
                alpha=0.2,
            )
            plt.xticks(labels)
            fig.suptitle("Train and test sets for the original and differenced series")
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.show()

            ps = range(0, 4, 1)  # Possible value of order p of AR(p)
            qs = range(0, 4, 1)  # Possible value of order q of MA(q)

            aic_scores = get_aic_scores(train["diff"], ps=ps, qs=qs)
            logger.debug(f"AIC scores are:\n{aic_scores}")
            lowest_aic_score = aic_scores.iloc[0]
            best_p = int(lowest_aic_score["p"])
            best_q = int(lowest_aic_score["q"])
            logger.debug(f"Best p is: {best_p} and best q is {best_q}")

            d = 0
            model = SARIMAX(
                train["diff"], order=(best_p, d, best_q), simple_differencing=False
            )
            model_fit = model.fit(disp=False)
            logger.debug(f"Model fit summary:\n{model_fit.summary()}")
            fig = model_fit.plot_diagnostics(figsize=(10, 8))
            fig.suptitle(f"Model diagnostics of ARMA({best_p},{best_q})")
            plt.show()

            residuals = model_fit.resid
            is_residual_white_noise = ljung_box_residual_test(residuals)
            logger.debug(f"Is residual just random error? {is_residual_white_noise}")

            window = best_p
            pred_mean = rolling_forecast(
                df_diff, train_size, test_size, window, "mean", best_p, best_q
            )
            pred_last_value = rolling_forecast(
                df_diff, train_size, test_size, window, "last_value", best_p, best_q
            )
            pred_ARMA = rolling_forecast(
                df_diff, train_size, test_size, window, "ARMA", best_p, best_q
            )

            test.loc[:, "pred_mean"] = pred_mean
            test.loc[:, "pred_last_value"] = pred_last_value
            test.loc[:, "pred_ARMA"] = pred_ARMA
            logger.debug(f"Test comparisons (diff):\n{test.tail(test_size)}")

            fig, ax = plt.subplots(figsize=fig_size)
            ax.plot(
                category_sector_spend["SpendMonth"].iloc[1:],
                diff / 1e6,
                label=f"Diff {category}:{sector}",
            )
            ax.plot(
                category_sector_spend["SpendMonth"].iloc[-test_size:],
                test["diff"] / 1e6,
                "b-",
                label="Actual",
            )
            ax.plot(
                category_sector_spend["SpendMonth"].iloc[-test_size:],
                test["pred_mean"] / 1e6,
                "g:",
                label="Mean",
            )
            ax.plot(
                category_sector_spend["SpendMonth"].iloc[-test_size:],
                test["pred_last_value"] / 1e6,
                "r-.",
                label="Last",
            )
            ax.plot(
                category_sector_spend["SpendMonth"].iloc[-test_size:],
                test["pred_ARMA"] / 1e6,
                "k--",
                label=f"ARMA({best_p},{best_q})",
            )
            ax.legend(loc=2)
            ax.set_xlabel("Month")
            ax.set_ylabel("Monthly Spend (Millions £) - Diff")
            ax.axvspan(
                date2num(category_sector_spend["SpendMonth"].iloc[train_size]),
                date2num(category_sector_spend["SpendMonth"].iloc[dataset_size]),
                color="#808080",
                alpha=0.2,
            )
            plt.xticks(labels)
            plt.title(
                "Forecasts of the differenced monthly spend using \nthe mean, the last known value, and \n"
                f"an ARMA({best_p},{best_q}) model for\n{category}:{sector}"
            )
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.show()

            mse_mean = mean_squared_error(test["diff"], test["pred_mean"])
            mse_last_value = mean_squared_error(test["diff"], test["pred_last_value"])
            mse_ARMA = mean_squared_error(test["diff"], test["pred_ARMA"])
            logger.debug(f"Mean Squared Error of historical mean: {mse_mean:.2f}")
            logger.debug(f"Mean Squared Error of last know value: {mse_last_value:.2f}")
            logger.debug(
                f"Mean Squared Error of ARMA({best_p},{best_q})......: {mse_ARMA:.2f}"
            )

            mae_mean = mean_absolute_error(test["diff"], test["pred_mean"])
            mae_last_value = mean_absolute_error(test["diff"], test["pred_last_value"])
            mae_ARMA = mean_absolute_error(test["diff"], test["pred_ARMA"])
            logger.debug(f"Mean Absolute Error of historical mean: {mae_mean:.2f}")
            logger.debug(
                f"Mean Absolute Error of last know value: {mae_last_value:.2f}"
            )
            logger.debug(
                f"Mean Absolute Error of ARMA({best_p},{best_q})......: {mae_ARMA:.2f}"
            )

            category_sector_spend["forecast_mean"] = pd.Series(dtype=float)
            category_sector_spend["forecast_mean"][-test_size:] = test["pred_mean"]
            category_sector_spend["forecast_last_value"] = pd.Series(dtype=float)
            category_sector_spend["forecast_last_value"][-test_size:] = test[
                "pred_last_value"
            ]
            category_sector_spend["forecast"] = pd.Series(dtype=float)
            category_sector_spend["forecast"][-test_size:] = (
                category_sector_spend["EvidencedSpend"].iloc[-test_size]
                + test["pred_ARMA"].cumsum()
            )

            fig, ax = plt.subplots(figsize=fig_size)
            ax.plot(
                category_sector_spend["SpendMonth"],
                category_sector_spend["EvidencedSpend"] / 1e6,
                label=f"{category}:{sector}",
            )
            ax.plot(
                category_sector_spend["SpendMonth"],
                category_sector_spend["forecast"] / 1e6,
                "k--",
                label=f"ARMA({best_p},{best_q})",
            )
            ax.legend(loc=2)
            ax.set_xlabel("Month")
            ax.set_ylabel("Monthly Spend (Millions £)")
            ax.axvspan(
                date2num(category_sector_spend["SpendMonth"].iloc[train_size]),
                date2num(category_sector_spend["SpendMonth"].iloc[dataset_size]),
                color="#808080",
                alpha=0.2,
            )
            plt.xticks(labels)
            plt.title(f"Forecast for\n{category} : {sector}")
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.show()

            mae_mean_original = mean_absolute_error(
                category_sector_spend["EvidencedSpend"][-test_size:],
                category_sector_spend["forecast_mean"][-test_size:],
            )
            mae_last_value = mean_absolute_error(
                category_sector_spend["EvidencedSpend"][-test_size:],
                category_sector_spend["forecast_last_value"][-test_size:],
            )
            mae_arma = mean_absolute_error(
                category_sector_spend["EvidencedSpend"][-test_size:],
                category_sector_spend["forecast"][-test_size:],
            )
            logger.debug(
                f"Mean Absolute Error of mean on original data: {mae_mean_original:.2f}"
            )
            logger.debug(
                f"Mean Absolute Error of last on original data: {mae_last_value:.2f}"
            )
            logger.debug(
                f"Mean Absolute Error of ARMA on original data: {mae_arma:.2f}"
            )

            mape_mean = (
                mean_absolute_percentage_error(
                    category_sector_spend["EvidencedSpend"][-test_size:],
                    category_sector_spend["forecast_mean"][-test_size:],
                )
                * 100.0
            )
            mape_last = (
                mean_absolute_percentage_error(
                    category_sector_spend["EvidencedSpend"][-test_size:],
                    category_sector_spend["forecast_last_value"][-test_size:],
                )
                * 100.0
            )
            mape_arma = (
                mean_absolute_percentage_error(
                    category_sector_spend["EvidencedSpend"][-test_size:],
                    category_sector_spend["forecast"][-test_size:],
                )
                * 100.0
            )
            logger.debug(
                f"Mean Absolute Percentage Error of mean on original data: {mape_mean:.2f}"
            )
            logger.debug(
                f"Mean Absolute Percentage Error of last on original data: {mape_last:.2f}"
            )
            logger.debug(
                f"Mean Absolute Percentage Error of ARMA on original data: {mape_arma:.2f}"
            )

            fig, ax = plt.subplots()
            x = ["Mean", "Last Month", f"ARMA({best_p},{best_q})"]
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
            x = ["Mean", "Last Month", f"ARMA({best_p},{best_q})"]
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
        category_sector_spend = prepare_data(df=df, category=category, sector=sector)
        spend = category_sector_spend["EvidencedSpend"]
        labels = pd.date_range(
            start=category_sector_spend["SpendMonth"].min(axis=0),
            end=category_sector_spend["SpendMonth"].max(axis=0),
            freq="MS",
        ).tolist()[::4]
        fig_size = (12, 6)
        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(
            category_sector_spend["SpendMonth"],
            spend / 1e6,
            label=f"{category}:{sector}",
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Spend (Millions £)")
        plt.xticks(labels)
        plt.title(f"Monthly Spend (Millions £) for\n{category} : {sector}")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.legend()
        plt.show()

        dataset_size = len(category_sector_spend)
        train_ratio = 0.9
        train_size = int(dataset_size * train_ratio)
        test_size = dataset_size - train_size

        train = category_sector_spend[:train_size]
        test = category_sector_spend[train_size:]
        logger.debug(
            f"Size of dataset is {dataset_size} training set is {train_size} and test size is: {test_size}"
        )

        d = 0
        try:
            d = find_integration_order(train["EvidencedSpend"])
            logger.debug(f"Integration order is {d}")
        except ValueError as e:
            logger.error(f"Error while calculating d: {e}")

        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(
            category_sector_spend["SpendMonth"],
            spend / 1e6,
            label=f"{category}:{sector}",
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Spend (Millions £)")
        ax.axvspan(
            date2num(category_sector_spend["SpendMonth"].iloc[-test_size]),
            date2num(category_sector_spend["SpendMonth"].iloc[-1]),
            color="#808080",
            alpha=0.2,
        )
        plt.xticks(labels)
        plt.title("Train and test sets")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

        ps = range(0, 4, 1)  # Possible value of order p of AR(p)
        qs = range(0, 4, 1)  # Possible value of order q of MA(q)
        aic_scores = get_aic_scores(train["EvidencedSpend"], ps=ps, qs=qs, d=d)
        logger.debug(f"AIC scores are:\n{aic_scores}")
        lowest_aic_score = aic_scores.iloc[0]
        best_p = int(lowest_aic_score["p"])
        best_q = int(lowest_aic_score["q"])
        logger.debug(f"Best p is: {best_p} and best q is {best_q}")

        model = SARIMAX(
            train["EvidencedSpend"],
            order=(best_p, d, best_q),
            simple_differencing=False,
        )
        model_fit = model.fit(disp=False)
        logger.debug(f"Model fit summary:\n{model_fit.summary()}")
        fig = model_fit.plot_diagnostics(figsize=(10, 8))
        fig.suptitle(f"Model diagnostics of ARIMA({best_p},{d},{best_q})")
        plt.show()

        residuals = model_fit.resid
        is_residual_white_noise = ljung_box_residual_test(residuals)
        logger.debug(f"Is residual just random error? {is_residual_white_noise}")

        test["last_month"] = (
            category_sector_spend["EvidencedSpend"]
            .iloc[test.index.min() - 1 : test.index.max()]
            .values
        )
        test["forecast"] = model_fit.get_prediction(
            test.index.min(), test.index.max()
        ).predicted_mean
        logger.debug(f"Prediction on test\n{test}")

        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(
            category_sector_spend["SpendMonth"],
            spend / 1e6,
            label=f"{category}:{sector}",
        )
        ax.plot(test["SpendMonth"], test["last_month"] / 1e6, "r:", label="Last Month")
        ax.plot(
            test["SpendMonth"],
            test["forecast"] / 1e6,
            "k--",
            label=f"ARIMA({best_p},{d},{best_q})",
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Spend (Millions £)")
        ax.axvspan(
            date2num(category_sector_spend["SpendMonth"].iloc[-test_size]),
            date2num(category_sector_spend["SpendMonth"].iloc[-1]),
            color="#808080",
            alpha=0.2,
        )
        ax.legend(loc=2)
        plt.xticks(labels)
        plt.title("Forecast")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

        last_month_mean_absolute_percentage_error = (
            mean_absolute_percentage_error(test["EvidencedSpend"], test["last_month"])
            * 100.0
        )
        arima_mean_absolute_percentage_error = (
            mean_absolute_percentage_error(test["EvidencedSpend"], test["forecast"])
            * 100.0
        )
        logger.debug(
            f"Last Month mean absolute percentage error....: {last_month_mean_absolute_percentage_error:.2f}"
        )
        logger.debug(
            f"ARIMA seasonal mean absolute percentage error: {arima_mean_absolute_percentage_error:.2f}"
        )

        last_month_mae = mean_absolute_error(test["EvidencedSpend"], test["last_month"])
        arima_mae = mean_absolute_error(test["EvidencedSpend"], test["forecast"])
        logger.debug(f"Last Month mean absolute error....: {last_month_mae:.2f}")
        logger.debug(f"ARIMA seasonal mean absolute error: {arima_mae:.2f}")

        fig, ax = plt.subplots()
        x = ["Last Month", f"ARIMA({best_p},{d},{best_q})"]
        y = [
            last_month_mean_absolute_percentage_error,
            arima_mean_absolute_percentage_error,
        ]
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


def model_sarima_arima(df: pd.DataFrame, category: str, sector: str, run: bool = False):
    """Forecast the data using SARIMA(p,d,q)(P,D,Q)m model and compares it with ARIMA(p,d,q) model.

    Args:
        df: DataFrame containing data
        category: Category for which spend data should be forecasted
        sector: Market sector for which spend data should be forecasted
        run: A flag to run this function, pass true to run this function

    Returns:
        None

    """

    if run:
        category_sector_spend = prepare_data(df=df, category=category, sector=sector)
        logger.debug(f"Dataset size is: {category_sector_spend.shape[0]}")
        spend = category_sector_spend["EvidencedSpend"]
        labels = pd.date_range(
            start=category_sector_spend["SpendMonth"].min(axis=0),
            end=category_sector_spend["SpendMonth"].max(axis=0),
            freq="MS",
        ).tolist()[::4]
        fig_size = (12, 6)
        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(
            category_sector_spend["SpendMonth"],
            spend / 1e6,
            label=f"{category}:{sector}",
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Spend (Millions £)")
        plt.xticks(labels)
        plt.title(f"Monthly Spend (Millions £) for\n{category} : {sector}")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.legend()
        plt.show()

        dataset_size = len(category_sector_spend)
        train_ratio = 0.9
        train_size = int(dataset_size * train_ratio)
        test_size = dataset_size - train_size

        train = category_sector_spend[:train_size]
        test = category_sector_spend[train_size:]

        logger.debug(
            f"Size of dataset is {dataset_size} training set is {train_size} and test size is: {test_size}"
        )

        d = 0
        try:
            d = find_integration_order(train["EvidencedSpend"])
            logger.debug(f"Integration order is {d}")
        except ValueError as e:
            logger.error(f"Error while calculating d: {e}")

        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(
            category_sector_spend["SpendMonth"],
            spend / 1e6,
            label=f"{category}:{sector}",
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Spend (Millions £)")
        ax.axvspan(
            date2num(category_sector_spend["SpendMonth"].iloc[-test_size]),
            date2num(category_sector_spend["SpendMonth"].iloc[-1]),
            color="#808080",
            alpha=0.2,
        )
        plt.xticks(labels)
        plt.title("Train and test sets")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

        test["last_month"] = (
            category_sector_spend["EvidencedSpend"]
            .iloc[test.index.min() - 1 : test.index.max()]
            .values
        )

        ps = range(0, 4, 1)  # Possible value of order p of AR(p)
        qs = range(0, 4, 1)  # Possible value of order q of MA(q)
        Ps = [0]  # Seasonality is 0 for ARIMA
        Qs = [0]  # Seasonality is 0 for ARIMA
        D = 0
        s = 0  # s is same as m
        aic_scores = get_aic_scores(
            train["EvidencedSpend"], ps=ps, qs=qs, Ps=Ps, Qs=Qs, d=d, D=D, s=s
        )
        logger.debug(f"AIC scores are:\n{aic_scores}")
        lowest_aic_score = aic_scores.iloc[0]
        best_p = int(lowest_aic_score["p"])
        best_q = int(lowest_aic_score["q"])
        logger.debug(
            f"ARMIA({best_p},{d},{best_q}) best p is: {best_p} and best q is {best_q}"
        )

        arima_model = SARIMAX(
            train["EvidencedSpend"],
            order=(best_p, d, best_q),
            seasonal_order=(Ps[0], D, Qs[0], s),
            simple_differencing=False,
        )
        arima_model_fit = arima_model.fit(disp=False)
        logger.debug(f"Model fit summary:\n{arima_model_fit.summary()}")
        fig = arima_model_fit.plot_diagnostics(figsize=(10, 8))
        fig.suptitle(f"ARMIA({best_p},{d},{best_q}) model diagnostics")
        plt.show()

        residuals = arima_model_fit.resid
        is_residual_white_noise = ljung_box_residual_test(residuals)
        logger.debug(
            f"Is ARMIA({best_p},{d},{best_q}) residual just random error? {is_residual_white_noise}"
        )

        test["arima_forecast"] = arima_model_fit.get_prediction(
            test.index.min(), test.index.max()
        ).predicted_mean

        s = 12  # s is same as m
        decomposition = STL(spend / 1e6, period=s).fit()
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            nrows=4, ncols=1, sharex="all", figsize=(10, 8)
        )
        ax1.plot(decomposition.observed)
        ax1.set_ylabel("Observed")
        ax2.plot(decomposition.trend)
        ax2.set_ylabel("Trend")
        ax3.plot(decomposition.seasonal)
        ax3.set_ylabel("Seasonal")
        ax4.plot(decomposition.resid)
        ax4.set_ylabel("Residuals")
        plt.suptitle("Decomposing the dataset")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

        D = 0
        try:
            D = find_seasonal_integration_order(train["EvidencedSpend"], s=s)
        except ValueError as e:
            logger.error(f"Error while calculating D: {e}")
        ps = range(0, 4, 1)
        qs = range(0, 4, 1)
        Ps = range(0, 4, 1)
        Qs = range(0, 4, 1)
        aic_scores = get_aic_scores(
            train["EvidencedSpend"], ps=ps, qs=qs, Ps=Ps, Qs=Qs, d=d, D=D, s=s
        )
        logger.debug(f"AIC scores are:\n{aic_scores}")
        lowest_aic_score = aic_scores.iloc[0]
        best_p = int(lowest_aic_score["p"])
        best_q = int(lowest_aic_score["q"])
        best_P = int(lowest_aic_score["P"])
        best_Q = int(lowest_aic_score["Q"])
        logger.debug(
            f"Best p is:\n{best_p}\nBest q is {best_q}\nBest P is: {best_P}\nBest Q is: {best_Q}"
        )

        sarima_model = SARIMAX(
            train["EvidencedSpend"],
            order=(best_p, d, best_q),
            seasonal_order=(best_P, D, best_Q, s),
            simple_differencing=False,
        )
        sarima_model_fit = sarima_model.fit(disp=False)
        logger.debug(f"Model fit summary:\n{sarima_model_fit.summary()}")
        try:
            fig = sarima_model_fit.plot_diagnostics(figsize=(10, 8))
            fig.suptitle(
                f"Residuals diagnostics of the SARIMA({best_p},{d},{best_q})({best_P},{D},{best_Q}){s} model"
            )
            plt.show()
        except Exception as e:
            logger.error(f"Exception occurred due to {e}")

        logger.debug("Performing Ljung-Box test on for the residuals, on 10 lags")
        residuals = sarima_model_fit.resid
        is_residual_white_noise = ljung_box_residual_test(residuals)
        logger.debug(
            f"Is SARIMA({best_p},{d},{best_q})({best_P},{D},{best_Q}){s} residual just random error? "
            + f"{is_residual_white_noise}"
        )
        test["sarima_forecast"] = sarima_model_fit.get_prediction(
            test.index.min(), test.index.max()
        ).predicted_mean
        logger.debug(
            f"SARIMA({best_p},{d},{best_q})({best_P},{D},{best_Q}){s} prediction are:\n{test}"
        )

        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(
            category_sector_spend["SpendMonth"],
            spend / 1e6,
            label=f"{category}:{sector}",
        )
        ax.plot(
            test["SpendMonth"], test["last_month"] / 1e6, "r:", label="Previous Month"
        )
        ax.plot(
            test["SpendMonth"],
            test["arima_forecast"] / 1e6,
            "k--",
            label=f"ARIMA({best_p},{d},{best_q})",
        )
        ax.plot(
            test["SpendMonth"],
            test["sarima_forecast"] / 1e6,
            "g--",
            label=f"SARIMA({best_p},{d},{best_q})({best_P},{D},{best_Q}){s})",
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Spend (Millions £)")
        ax.axvspan(
            date2num(category_sector_spend["SpendMonth"].iloc[-test_size]),
            date2num(category_sector_spend["SpendMonth"].iloc[-1]),
            color="#808080",
            alpha=0.2,
        )
        ax.legend(loc=2)
        plt.xticks(labels)
        plt.title("Forecast")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

        last_month_mean_absolute_percentage_error = (
            mean_absolute_percentage_error(test["EvidencedSpend"], test["last_month"])
            * 100.0
        )
        arima_mean_absolute_percentage_error = (
            mean_absolute_percentage_error(
                test["EvidencedSpend"], test["arima_forecast"]
            )
            * 100.0
        )
        sarima_mean_absolute_percentage_error = (
            mean_absolute_percentage_error(
                test["EvidencedSpend"], test["sarima_forecast"]
            )
            * 100.0
        )
        logger.debug(
            f"Last Month absolute percentage error..........: {last_month_mean_absolute_percentage_error:.2f}"
        )
        logger.debug(
            f"ARIMA seasonal mean absolute percentage error.: {arima_mean_absolute_percentage_error:.2f}"
        )
        logger.debug(
            f"SARIMA seasonal mean absolute percentage error: {sarima_mean_absolute_percentage_error:.2f}"
        )

        fig, ax = plt.subplots()
        x = [
            "Last Month",
            f"ARIMA({best_p},{d},{best_q})",
            f"SARIMA({best_p},{d},{best_q})({best_P},{D},{best_Q}){s}",
        ]
        y = [
            last_month_mean_absolute_percentage_error,
            arima_mean_absolute_percentage_error,
            sarima_mean_absolute_percentage_error,
        ]
        ax.bar(x, y, width=0.4)
        ax.set_xlabel("Models")
        ax.set_ylabel("Mean Absolute Percentage Error (%)")
        ax.set_ylim(0, 100)
        for index, value in enumerate(y):
            plt.text(x=index, y=value + 1, s=str(round(value, 2)), ha="center")
        plt.title("Mean Absolute Percentage Error")
        plt.tight_layout()
        plt.show()

        last_month_mae = mean_absolute_error(test["EvidencedSpend"], test["last_month"])
        arima_mae = mean_absolute_error(test["EvidencedSpend"], test["arima_forecast"])
        sarima_mae = mean_absolute_error(
            test["EvidencedSpend"], test["sarima_forecast"]
        )
        logger.debug(f"Last Month MAE.....: {last_month_mae:.2f}")
        logger.debug(f"ARIMA seasonal MAE.: {arima_mae:.2f}")
        logger.debug(f"SARIMA seasonal MAE: {sarima_mae:.2f}")

        fig, ax = plt.subplots()
        x = [
            "Last Month",
            f"ARIMA({best_p},{d},{best_q})",
            f"SARIMA({best_p},{d},{best_q})({best_P},{D},{best_Q}){s}",
        ]
        y = [last_month_mae / 1e6, arima_mae / 1e6, sarima_mae / 1e6]
        ax.bar(x, y, width=0.4)
        ax.set_xlabel("Models")
        ax.set_ylabel("Mean Absolute Error (Millions £)")
        for index, value in enumerate(y):
            plt.text(x=index, y=value + 1, s=str(round(value, 2)), ha="center")
        plt.title("Mean Absolute Error")
        plt.tight_layout()
        plt.show()


def model_prophet_sarima_arima(df: pd.DataFrame, category: str, sector: str, run: bool = False):
    """Forecast the data using Prophet and compares it with SARIMA(p,d,q)(P,D,Q)m model and ARIMA(p,d,q) model.

    Args:
        df: DataFrame containing data
        category: Category for which spend data should be forecasted
        sector: Market sector for which spend data should be forecasted
        run: A flag to run this function, pass true to run this function

    Returns:
        None

    """
    if run:
        category_sector_spend = prepare_data(df=df, category=category, sector=sector)
        logger.debug(f"Dataset size is: {category_sector_spend.shape[0]}")
        spend = category_sector_spend["EvidencedSpend"]
        labels = pd.date_range(
            start=category_sector_spend["SpendMonth"].min(axis=0),
            end=category_sector_spend["SpendMonth"].max(axis=0),
            freq="MS",
        ).tolist()[::4]
        fig_size = (12, 6)
        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(
            category_sector_spend["SpendMonth"],
            spend / 1e6,
            label=f"{category}:{sector}",
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Spend (Millions £)")
        plt.xticks(labels)
        plt.title(f"Monthly Spend (Millions £) for\n{category} : {sector}")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.legend()
        plt.show()

        dataset_size = len(category_sector_spend)
        train_ratio = 0.9
        train_size = int(dataset_size * train_ratio)
        test_size = dataset_size - train_size

        logger.debug(
            f"Size of dataset is {dataset_size} training set is {train_size} and test size is: {test_size}"
        )

        train = category_sector_spend[:train_size]
        test = category_sector_spend[train_size:]
        test["last_month"] = (
            category_sector_spend["EvidencedSpend"]
            .iloc[test.index.min() - 1 : test.index.max()]
            .values
        )

        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(
            category_sector_spend["SpendMonth"],
            spend / 1e6,
            label=f"{category}:{sector}",
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Spend (Millions£)")
        ax.axvspan(
            date2num(category_sector_spend["SpendMonth"].iloc[-test_size]),
            date2num(category_sector_spend["SpendMonth"].iloc[-1]),
            color="#808080",
            alpha=0.2,
        )
        plt.xticks(labels)
        plt.title("Train and test sets")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

        d = 0
        try:
            d = find_integration_order(train["EvidencedSpend"])
            logger.debug(f"Integration order is {d}")
        except ValueError as e:
            logger.error(f"Error while calculating d: {e}")

        ps = range(0, 4, 1)  # Possible value of order p of AR(p)
        qs = range(0, 4, 1)  # Possible value of order q of MA(q)
        Ps = [0]  # Seasonality is 0 for ARIMA
        Qs = [0]  # Seasonality is 0 for ARIMA
        D = 0
        s = 0  # s is same as m
        aic_scores = get_aic_scores(
            train["EvidencedSpend"], ps=ps, qs=qs, Ps=Ps, Qs=Qs, d=d, D=D, s=s
        )
        logger.debug(f"AIC scores are:\n{aic_scores}")
        lowest_aic_score = aic_scores.iloc[0]
        best_p = int(lowest_aic_score["p"])
        best_q = int(lowest_aic_score["q"])
        logger.debug(f"Best p is: {best_p} and best q is {best_q}")

        arima_model = SARIMAX(
            train["EvidencedSpend"],
            order=(best_p, d, best_q),
            seasonal_order=(Ps[0], D, Qs[0], s),
            simple_differencing=False,
        )
        arima_model_fit = arima_model.fit(disp=False)
        logger.debug(f"Model fir summary:\n{arima_model_fit.summary()}")
        fig = arima_model_fit.plot_diagnostics(figsize=(10, 8))
        fig.suptitle(f"ARMIA({best_p},{d},{best_q}) model diagnostics")
        plt.show()

        residuals = arima_model_fit.resid
        is_residual_white_noise = ljung_box_residual_test(residuals)
        logger.debug(
            f"Is ARMIA({best_p},{d},{best_q}) residual just random error? {is_residual_white_noise}"
        )

        test["arima_forecast"] = arima_model_fit.get_prediction(
            test.index.min(), test.index.max()
        ).predicted_mean

        s = 12  # s is same as m
        D = 0
        try:
            D = find_seasonal_integration_order(train["EvidencedSpend"], s=s)
        except ValueError as e:
            logger.error(f"Error while calculating D: {e}")

        Ps = range(0, 4, 1)
        Qs = range(0, 4, 1)
        aic_scores = get_aic_scores(
            train["EvidencedSpend"], ps=ps, qs=qs, Ps=Ps, Qs=Qs, d=d, D=D, s=s
        )
        logger.debug(f"AIC scores are:\n{aic_scores}")
        lowest_aic_score = aic_scores.iloc[0]
        best_p = int(lowest_aic_score["p"])
        best_q = int(lowest_aic_score["q"])
        best_P = int(lowest_aic_score["P"])
        best_Q = int(lowest_aic_score["Q"])
        logger.debug(
            f"\nBest p is: {best_p}\nBest q is {best_q}\nBest P is: {best_P}\nBest Q is: {best_Q}"
        )

        sarima_model = SARIMAX(
            train["EvidencedSpend"],
            order=(best_p, d, best_q),
            seasonal_order=(best_P, D, best_Q, s),
            simple_differencing=False,
        )
        sarima_model_fit = sarima_model.fit(disp=False)
        try:
            fig = sarima_model_fit.plot_diagnostics(figsize=(10, 8))
            fig.suptitle(
                f"Residuals diagnostics of the SARIMA({best_p},{d},{best_q})({best_P},{D},{best_Q}){s} model"
            )
            plt.show()
        except Exception as e:
            logger.error(f"Exception occured due to {e}")

        logger.debug("Performing Ljung-Box test on for the residuals, on 10 lags")
        residuals = sarima_model_fit.resid
        is_residual_white_noise = ljung_box_residual_test(residuals)
        logger.debug(
            f"Is SARIMA({best_p},{d},{best_q})({best_P},{D},{best_Q}){s} residual just random error? "
            + f"{is_residual_white_noise}"
        )

        test["sarima_forecast"] = sarima_model_fit.get_prediction(
            test.index.min(), test.index.max()
        ).predicted_mean

        train_prophet = train[["SpendMonth", "EvidencedSpend"]]
        test_prophet = test[["SpendMonth", "EvidencedSpend"]]
        train_prophet.columns = ["ds", "y"]
        test_prophet.columns = ["ds", "y"]

        changepoint_prior_scale_values = [0.001, 0.01, 0.1, 0.5]
        seasonality_prior_scale_values = [0.01, 0.1, 1.0, 10.0]

        lowest_mape = []

        for changepoint_prior_scale, seasonality_prior_scale in list(
            product(changepoint_prior_scale_values, seasonality_prior_scale_values)
        ):
            m = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
            )
            m.add_country_holidays(country_name="UK")
            m.fit(train_prophet.copy(deep=True))

            forecast = m.predict(train_prophet)
            forecast.index = train_prophet.index
            train_prophet_copy = train_prophet.copy(deep=True)
            train_prophet_copy[["yhat", "yhat_lower", "yhat_upper"]] = forecast[
                ["yhat", "yhat_lower", "yhat_upper"]
            ]
            prophet_mean_absolute_percentage_error = (
                mean_absolute_percentage_error(
                    train_prophet_copy["y"], train_prophet_copy["yhat"]
                )
                * 100.0
            )
            lowest_mape.append(
                (
                    changepoint_prior_scale,
                    seasonality_prior_scale,
                    prophet_mean_absolute_percentage_error,
                )
            )

        mape_scores = (
            pd.DataFrame(
                data=lowest_mape,
                columns=[
                    "changepoint_prior_scale",
                    "seasonality_prior_scale",
                    "min_score",
                ],
            )
            .sort_values(by="min_score", ascending=True)
            .reset_index(drop=True)
        )
        logger.debug(f"MAPE Scores:\n{mape_scores}")
        best_params = mape_scores.iloc[0]
        best_changepoint_prior_scale = float(best_params["changepoint_prior_scale"])
        best_seasonality_prior_scale = float(best_params["seasonality_prior_scale"])
        logger.debug(f"Best changepoint_prior_scale: {best_changepoint_prior_scale}")
        logger.debug(f"Best seasonality_prior_scale: {best_seasonality_prior_scale}")

        m = Prophet(
            changepoint_prior_scale=best_changepoint_prior_scale,
            seasonality_prior_scale=best_seasonality_prior_scale,
        )
        m.add_country_holidays(country_name="UK")
        m.fit(train_prophet)
        forecast = m.predict(test_prophet)
        forecast.index = test_prophet.index
        test["prophet_forecast"] = forecast["yhat"]
        logger.debug(f"Prophet forecast details:\n{forecast}")

        fig, ax = plt.subplots()
        ax.plot(
            category_sector_spend["SpendMonth"],
            spend / 1e6,
            label=f"{category}:{sector}",
        )
        ax.plot(
            test["SpendMonth"], test["last_month"] / 1e6, "r:", label="Previous Month"
        )
        ax.plot(
            test["SpendMonth"],
            test["arima_forecast"] / 1e6,
            "k--",
            label=f"ARIMA({best_p},{d},{best_q})",
        )
        ax.plot(
            test["SpendMonth"],
            test["sarima_forecast"] / 1e6,
            "g--",
            label=f"SARIMA({best_p},{d},{best_q})({best_P},{D},{best_Q}){s}",
        )
        ax.plot(
            test["SpendMonth"], test["prophet_forecast"] / 1e6, "m--", label="Prophet"
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Monthly Spend (Millions £)")
        ax.axvspan(
            date2num(category_sector_spend["SpendMonth"].iloc[-test_size]),
            date2num(category_sector_spend["SpendMonth"].iloc[-1]),
            color="#808080",
            alpha=0.2,
        )
        ax.legend(loc=2)
        plt.xticks(labels)
        plt.title("Forecast")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

        last_month_mean_absolute_percentage_error = (
            mean_absolute_percentage_error(test["EvidencedSpend"], test["last_month"])
            * 100.0
        )
        arima_mean_absolute_percentage_error = (
            mean_absolute_percentage_error(
                test["EvidencedSpend"], test["arima_forecast"]
            )
            * 100.0
        )
        sarima_mean_absolute_percentage_error = (
            mean_absolute_percentage_error(
                test["EvidencedSpend"], test["sarima_forecast"]
            )
            * 100.0
        )
        prophet_mean_absolute_percentage_error = (
            mean_absolute_percentage_error(
                test["EvidencedSpend"], test["prophet_forecast"]
            )
            * 100.0
        )
        logger.debug(
            f"Last Month absolute percentage error.........: {last_month_mean_absolute_percentage_error:.2f}"
        )
        logger.debug(
            f"ARIMA seasonal mean absolute percentage error: {arima_mean_absolute_percentage_error:.2f}"
        )
        logger.debug(
            f"SARIMA seasonal mean absolute percentage error: {sarima_mean_absolute_percentage_error:.2f}"
        )
        logger.debug(
            f"Prophet mean absolute percentage error........: {prophet_mean_absolute_percentage_error:.2f}"
        )

        fig, ax = plt.subplots()
        x = [
            "Last Month",
            f"ARIMA({best_p},{d},{best_q})",
            f"SARIMA({best_p},{d},{best_q})({best_P},{D},{best_Q}){s})",
            "Prophet",
        ]
        y = [
            last_month_mean_absolute_percentage_error,
            arima_mean_absolute_percentage_error,
            sarima_mean_absolute_percentage_error,
            prophet_mean_absolute_percentage_error,
        ]
        ax.bar(x, y, width=0.4)
        ax.set_xlabel("Models")
        ax.set_ylabel("Mean Absolute Percentage Error (%)")
        ax.set_ylim(0, 100)
        for index, value in enumerate(y):
            plt.text(x=index, y=value + 1, s=str(round(value, 2)), ha="center")
        plt.title("Mean Absolute Percentage Error")
        plt.tight_layout()
        plt.show()

        last_month_mae = mean_absolute_error(test["EvidencedSpend"], test["last_month"])
        arima_mae = mean_absolute_error(test["EvidencedSpend"], test["arima_forecast"])
        sarima_mae = mean_absolute_error(
            test["EvidencedSpend"], test["sarima_forecast"]
        )
        prophet_mae = mean_absolute_error(
            test["EvidencedSpend"], test["prophet_forecast"]
        )
        logger.debug(f"Last Month MAE.....: {last_month_mae:.2f}")
        logger.debug(f"ARIMA seasonal MAE.: {arima_mae:.2f}")
        logger.debug(f"SARIMA seasonal MAE: {sarima_mae:.2f}")
        logger.debug(f"Prophet MAE........: {prophet_mae:.2f}")

        fig, ax = plt.subplots()
        x = [
            "Last Month",
            f"ARIMA({best_p},{d},{best_q})",
            f"SARIMA({best_p},{d},{best_q})({best_P},{D},{best_Q}){s}",
            "Prophet",
        ]
        y = [last_month_mae / 1e6, arima_mae / 1e6, sarima_mae / 1e6, prophet_mae / 1e6]
        ax.bar(x, y, width=0.4)
        ax.set_xlabel("Models")
        ax.set_ylabel("Mean Absolute Error")
        # ax.set_ylim(0, 100)
        for index, value in enumerate(y):
            plt.text(x=index, y=value + 1, s=str(round(value, 2)), ha="center")
        plt.title("Mean Absolute Error (Millions £)")
        plt.tight_layout()
        plt.show()

        logger.debug(f"Forecast are:\n{test}")


def forecast_future_spend(
    df: pd.DataFrame,
    category: str,
    sector: str,
    forecast_end_date: date,
    run: bool = False,
) -> pd.DataFrame:
    """Generate the best forecast by comparing various models.

    Args:
        df: DataFrame containing data
        category: Category for which spend data should be forecasted
        sector: Market sector for which spend data should be forecasted
        forecast_end_date: The last date till which forecast should be done
        run: A flag to run this function, pass true to run this function

    Returns:
        DataFrame with the best forecast

    """
    if run:
        category_sector_spend = prepare_data(df=df, category=category, sector=sector)
        dataset_size = len(category_sector_spend)
        train_ratio = 0.9
        train_size = int(dataset_size * train_ratio)
        test_size = dataset_size - train_size

        logger.debug(
            f"Size of dataset is {dataset_size} training set is {train_size} and test size is: {test_size}"
        )

        train = category_sector_spend[:train_size]
        test = category_sector_spend[train_size:]
        spend = train["EvidencedSpend"]

        d = 0
        try:
            d = find_integration_order(spend)
        except ValueError as e:
            logger.error(f"Error while calculating d: {e}")

        ps = range(0, 4, 1)  # Possible value of order p of AR(p)
        qs = range(0, 4, 1)  # Possible value of order q of MA(q)
        Ps = [0]  # Seasonality is 0 for ARIMA
        Qs = [0]  # Seasonality is 0 for ARIMA
        D = 0
        s = 0  # s is same as m
        aic_scores = get_aic_scores(
            train["EvidencedSpend"], ps=ps, qs=qs, Ps=Ps, Qs=Qs, d=d, D=D, s=s
        )
        lowest_aic_score = aic_scores.iloc[0]
        best_p = int(lowest_aic_score["p"])
        best_q = int(lowest_aic_score["q"])

        arima_model = SARIMAX(
            train["EvidencedSpend"],
            order=(best_p, d, best_q),
            seasonal_order=(Ps[0], D, Qs[0], s),
            simple_differencing=False,
        )
        best_arima_model = SARIMAX(
            category_sector_spend["EvidencedSpend"],
            order=(best_p, d, best_q),
            seasonal_order=(Ps[0], D, Qs[0], s),
            simple_differencing=False,
        )
        arima_model_fit = arima_model.fit(disp=False)
        test["arima_forecast"] = arima_model_fit.get_prediction(
            test.index.min(), test.index.max()
        ).predicted_mean

        s = 12  # s is same as m
        D = 0
        try:
            D = find_seasonal_integration_order(spend, s=s)
        except ValueError as e:
            logger.error(f"Error while calculating D: {e}")
        Ps = range(0, 4, 1)
        Qs = range(0, 4, 1)
        aic_scores = get_aic_scores(
            train["EvidencedSpend"], ps=ps, qs=qs, Ps=Ps, Qs=Qs, d=d, D=D, s=s
        )
        lowest_aic_score = aic_scores.iloc[0]
        best_p = int(lowest_aic_score["p"])
        best_q = int(lowest_aic_score["q"])
        best_P = int(lowest_aic_score["P"])
        best_Q = int(lowest_aic_score["Q"])
        sarima_model = SARIMAX(
            train["EvidencedSpend"],
            order=(best_p, d, best_q),
            seasonal_order=(best_P, D, best_Q, s),
            simple_differencing=False,
        )
        best_sarima_model = SARIMAX(
            category_sector_spend["EvidencedSpend"],
            order=(best_p, d, best_q),
            seasonal_order=(best_P, D, best_Q, s),
            simple_differencing=False,
        )
        sarima_model_fit = sarima_model.fit(disp=False)
        test["sarima_forecast"] = sarima_model_fit.get_prediction(
            test.index.min(), test.index.max()
        ).predicted_mean

        train_prophet = train[["SpendMonth", "EvidencedSpend"]]
        test_prophet = test[["SpendMonth", "EvidencedSpend"]]
        train_prophet.columns = ["ds", "y"]
        test_prophet.columns = ["ds", "y"]

        changepoint_prior_scale_values = [0.001, 0.01, 0.1, 0.5]
        seasonality_prior_scale_values = [0.01, 0.1, 1.0, 10.0]

        lowest_mape = []
        for changepoint_prior_scale, seasonality_prior_scale in list(
            product(changepoint_prior_scale_values, seasonality_prior_scale_values)
        ):
            m = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
            )
            m.add_country_holidays(country_name="UK")
            m.fit(train_prophet.copy(deep=True))
            forecast = m.predict(train_prophet)
            forecast.index = train_prophet.index
            train_prophet_copy = train_prophet.copy(deep=True)
            train_prophet_copy[["yhat", "yhat_lower", "yhat_upper"]] = forecast[
                ["yhat", "yhat_lower", "yhat_upper"]
            ]
            prophet_mean_absolute_percentage_error = (
                mean_absolute_percentage_error(
                    train_prophet_copy["y"], train_prophet_copy["yhat"]
                )
                * 100.0
            )

            lowest_mape.append(
                (
                    changepoint_prior_scale,
                    seasonality_prior_scale,
                    prophet_mean_absolute_percentage_error,
                )
            )

        mape_scores = (
            pd.DataFrame(
                data=lowest_mape,
                columns=[
                    "changepoint_prior_scale",
                    "seasonality_prior_scale",
                    "min_score",
                ],
            )
            .sort_values(by="min_score", ascending=True)
            .reset_index(drop=True)
        )
        best_params = mape_scores.iloc[0]
        best_changepoint_prior_scale = float(best_params["changepoint_prior_scale"])
        best_seasonality_prior_scale = float(best_params["seasonality_prior_scale"])

        m = Prophet(
            changepoint_prior_scale=best_changepoint_prior_scale,
            seasonality_prior_scale=best_seasonality_prior_scale,
        )
        m.add_country_holidays(country_name="UK")
        m.fit(train_prophet)
        forecast = m.predict(test_prophet)
        forecast.index = test_prophet.index
        test["prophet_forecast"] = forecast["yhat"]

        arima_mape = (
            mean_absolute_percentage_error(
                test["EvidencedSpend"], test["arima_forecast"]
            )
            * 100.0
        )
        sarima_mape = (
            mean_absolute_percentage_error(
                test["EvidencedSpend"], test["sarima_forecast"]
            )
            * 100.0
        )
        prophet_mape = (
            mean_absolute_percentage_error(
                test["EvidencedSpend"], test["prophet_forecast"]
            )
            * 100.0
        )
        logger.debug(f"ARIMA seasonal MAPE.: {arima_mape:.2f}")
        logger.debug(f"SARIMA seasonal MAPE: {sarima_mape:.2f}")
        logger.debug(f"Prophet MAPE........: {prophet_mape:.2f}")

        best_mape = np.argmin(np.array([arima_mape, sarima_mape, prophet_mape]))

        forecast_dates = pd.date_range(
            start=category_sector_spend["SpendMonth"].min(axis=0),
            end=forecast_end_date,
            freq="MS",
        )
        forecast = pd.DataFrame(data=forecast_dates, columns=["SpendMonth"])
        forecast["Category"] = category
        forecast["MarketSector"] = sector
        forecast = forecast[
            forecast["SpendMonth"] > category_sector_spend["SpendMonth"].max(axis=0)
        ]
        forecast["Forecast"] = np.NAN

        best_arima_model_fit = best_arima_model.fit(disp=False)
        best_sarima_model_fit = best_sarima_model.fit(disp=False)
        best_prophet = Prophet(
            changepoint_prior_scale=best_changepoint_prior_scale,
            seasonality_prior_scale=best_seasonality_prior_scale,
        )
        best_prophet.add_country_holidays(country_name="UK")

        forecast["ARIMA_Forecast"] = best_arima_model_fit.get_prediction(
            forecast.index.min(), forecast.index.max()
        ).predicted_mean.values
        forecast["SARIMA_Forecast"] = best_sarima_model_fit.get_prediction(
            forecast.index.min(), forecast.index.max()
        ).predicted_mean.values
        prophet_data = category_sector_spend[["SpendMonth", "EvidencedSpend"]].copy(
            deep=True
        )
        prophet_data.columns = ["ds", "y"]
        best_prophet.fit(prophet_data)
        prophet_forecast_dates = forecast[["SpendMonth"]]
        prophet_forecast_dates.columns = ["ds"]
        best_prophet_forecast = best_prophet.predict(prophet_forecast_dates)
        best_prophet_forecast.index = forecast.index
        forecast["Prophet_Forecast"] = best_prophet_forecast["yhat"]

        if best_mape == 0:
            forecast["Forecast"] = forecast["ARIMA_Forecast"]
        elif best_mape == 1:
            forecast["Forecast"] = forecast["SARIMA_Forecast"]
        elif best_mape == 2:
            forecast["Forecast"] = forecast["Prophet_Forecast"]

        return forecast


def main():
    default_forecast_end_date = date.today() + relativedelta(years=2)
    parser = argparse.ArgumentParser(
        usage="data_analysis.py [path to local dataset (optional)] ",
        description="Exploratory Data Analysis",
    )
    parser.add_argument(
        "--local_data_path",
        default=None,
        metavar="Path to the local dataset folder",
        type=str,
    )
    parser.add_argument(
        "--forecast_output_path",
        default=None,
        metavar="Path of the folder to save the forecast",
        type=str,
    )
    parser.add_argument(
        "--forecast_end_date",
        default=default_forecast_end_date,
        metavar="Forecast end date in yyyy-MM-dd format",
        type=date.fromisoformat,
    )

    parsed = parser.parse_args()
    local_data_path = parsed.local_data_path
    forecast_output_path = parsed.forecast_output_path
    forecast_end_date = parsed.forecast_end_date

    category, sector = get_category_sector(index=0)
    raw_df = get_data_for_analysis(
        local_data_path=local_data_path, category=category, sector=sector
    )
    aggregated_df = aggregate_spend(raw_df)
    visualize_raw_data(df=aggregated_df, category=category, sector=sector, run=False)
    is_spend_random_walk(aggregated_df, category=category, sector=sector, run=False)
    model_arma(aggregated_df, category=category, sector=sector, run=False)
    model_arima(aggregated_df, category=category, sector=sector, run=False)
    model_sarima_arima(aggregated_df, category=category, sector=sector, run=False)
    model_prophet_sarima_arima(aggregated_df, category=category, sector=sector, run=False)
    future_forecast = forecast_future_spend(
        df=aggregated_df,
        category=category,
        sector=sector,
        forecast_end_date=forecast_end_date,
        run=False,
    )
    logger.debug(f"Future forecast is:\n{future_forecast}")

    if forecast_output_path:
        forecast_file_name = "forecast.csv"
        full_file_path = os.path.join(forecast_output_path, forecast_file_name)
        individual_forecast_path = os.path.join(forecast_output_path, "individuals")
        if os.path.exists(forecast_output_path):
            logger.info(
                f"Deleting existing output directory and its contents: {forecast_output_path}"
            )
            shutil.rmtree(individual_forecast_path)
        os.makedirs(individual_forecast_path, exist_ok=True)

        for i, (cat, sect) in enumerate(get_all_category_sector()):
            logger.debug(
                f"{i}. Forecasting for category = '{cat}' and sector = '{sect}'"
            )
            try:
                raw_df = get_data_for_analysis(
                    local_data_path=local_data_path, category=cat, sector=sect
                )
                aggregated_df = aggregate_spend(raw_df)
                category_sector_forecast = forecast_future_spend(
                    df=aggregated_df,
                    category=cat,
                    sector=sect,
                    forecast_end_date=forecast_end_date,
                    run=True,
                )
                category_sector_forecast.to_csv(
                    os.path.join(individual_forecast_path, f"{i}.csv"), index=False
                )
            except Exception as e:
                msg = (
                    f"Error in generation forecast for ({i}) category = '{cat}' and sector = '{sect}'"
                    + f"due to {e}"
                )
                logger.error(msg)

        forecast_files = glob.glob(os.path.join(individual_forecast_path, "*.csv"))
        forecast = pd.concat(
            [pd.read_csv(f) for f in forecast_files], ignore_index=True
        ).reset_index(drop=True)
        forecast.to_csv(full_file_path, index=False)


if __name__ == "__main__":
    main()
