""" Utility functions used for the machine learning process. Extracted from `eda/data_analysis.py`"""

from typing import Union
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox


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
    timeseries: Union[pd.Series, np.ndarray, list], seasonal_order: int = 0
) -> int:
    """Finds the integration order (denoted by d) of the timeseries
    Args:
        timeseries: Timeseries for which you wish to find the integration order
        seasonal_order: Seasonal Order E.g. for monthly it is 12.

    Returns:
        Seasonal Integration order as integer

    """

    def integration_finder(ts: Union[pd.Series, np.ndarray, list], d: int = 0) -> int:
        return (
            d
            if adf_stationary_test(ts)
            else integration_finder(np.diff(ts, n=seasonal_order), d + 1)
        )

    return integration_finder(timeseries)


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
