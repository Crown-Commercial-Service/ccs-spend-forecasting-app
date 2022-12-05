import pandas as pd
from typing import Callable


def create_models_comparison(
    input_df: pd.DataFrame, train_size: int, models: dict[str, Callable]
) -> pd.DataFrame:

    return pd.DataFrame(data={'a': [1,2,3], 'b': [2,3,4]})
