import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def neutralize_series(
    target: pd.Series, by: pd.Series, proportion: float = 1.0
) -> pd.Series:
    """
    Neutralizes the target series against the 'by' series (e.g., DXY, SPX).
    Removes the linear component of 'by' from 'target'.
    """
    scores = target.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # Fit linear model
    model = LinearRegression()
    model.fit(exposures, scores)

    # Predict the component to remove
    predictions = model.predict(exposures)

    # Subtract (neutralize)
    neutralized = scores - (proportion * predictions)

    return pd.Series(neutralized.flatten(), index=target.index)


def neutralize_dataframe(
    df: pd.DataFrame, columns: list, by: pd.Series, proportion: float = 1.0
) -> pd.DataFrame:
    """
    Neutralizes multiple columns in a DataFrame against a series.
    """
    out = df.copy()
    for col in columns:
        out[col] = neutralize_series(out[col], by, proportion)
    return out
