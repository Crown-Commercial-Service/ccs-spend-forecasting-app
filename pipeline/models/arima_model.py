from typing import Optional
from pipeline.models.sarima_model import SarimaModel


class ArimaModel(SarimaModel):
    """Forecast the data using ARIMA(p,d,q) model.

    Args:
        name (str):
            Name of the model instance. Default to be "ARIMA"

        search_hyperparameters (bool):
            If True, will run a search for hyperparameter for each combination.
            Note: This could potentially take a very long time to run (>= 10min per combination.)
            If False, will use a pre-defined set of hyperparameter, which largely reduce computation time but may lead to much less accurate forecast.
            The same name argument of method #create_forecast will take precedence over this value.

        default_hyperparameters (dict): A dict that contains the necessary hyperparameters for the model.
            For SARIMA, it requires input with the format {"order": (p, d, q)}
            If `search_hyperparameters` is False and this value is given, will use this instead of the pre-defined ones.
    """

    def __init__(
        self,
        name: str = "ARIMA",
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
                "seasonal_order": (0, 0, 0, 0),
            }

    def full_model_name(self, order: tuple[int, int, int]) -> str:
        """Return the name of this model with given set of hyperparameters. Used to reduce hard-coding for logging."""
        return f"{self.name}({order})"

    def hyperparameter_range(self) -> dict:
        """Return a default set of search range for hyperparameters.
        Returns:
            dict:
        """

        return {
            # p,d,q
            "p": range(0, 4, 1),  # p
            "q": range(0, 4, 1),  # q
            "d": "auto",  # d
            # For ARIMA model, the seasonal order are confined to be 0
            "seasonal_P": [0],  # P
            "seasonal_Q": [0],  # Q
            "seasonal_D": 0,  # D
            "seasonal_period": 0,  # m or s, default to be 0
        }
