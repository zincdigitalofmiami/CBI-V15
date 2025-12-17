"""
AutoGluon TimeSeriesPredictor wrapper with Mitra fallback for Mac M4.

This module provides a unified interface for time series forecasting that:
1. Attempts to use AutoGluon TimeSeriesPredictor with Chronos-Bolt
2. Falls back to Mitra (Metal-accelerated) if Chronos hangs/fails on Mac M4
3. Supports probabilistic forecasts (P10/P50/P90 quantiles)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import warnings

try:
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

    AG_AVAILABLE = True
except ImportError:
    AG_AVAILABLE = False
    TimeSeriesPredictor = None
    TimeSeriesDataFrame = None

try:
    from .mitra_trainer import MitraForecastWrapper, MITRA_AVAILABLE
except ImportError:
    MITRA_AVAILABLE = False
    MitraForecastWrapper = None


# Configuration for time series training
TIMESERIES_CONFIG = {
    "primary_model": "autogluon",  # 'autogluon' or 'mitra'
    "fallback_on_hang": True,  # Use Mitra if Chronos hangs
    "use_mps": True,  # Mac M4 Metal acceleration for Mitra
    "exclude_chronos": False,  # Set True to skip Chronos entirely (Mac M4 workaround)
}


def train_timeseries(
    train_df: Union[pd.DataFrame, TimeSeriesDataFrame],
    target_col: str = "target",
    known_covariates: Optional[List[str]] = None,
    prediction_length: int = 14,
    model_path: Optional[str] = None,
    use_mitra_fallback: bool = True,
    exclude_chronos: bool = True,  # Default True for Mac M4 compatibility
    time_limit: int = 3600,
    verbosity: int = 1,
):
    """
    Train time series predictor with AutoGluon (Chronos) or Mitra fallback.

    This function attempts to use AutoGluon TimeSeriesPredictor first.
    If Chronos hangs (common on Mac M4) or fails, it falls back to Mitra.

    Parameters
    ----------
    train_df : pd.DataFrame or TimeSeriesDataFrame
        Training data in time series format
    target_col : str, default='target'
        Target column name
    known_covariates : list, optional
        List of known covariate columns (e.g., Big 8 bucket scores)
    prediction_length : int, default=14
        Forecast horizon (days)
    model_path : str, optional
        Path to save model artifacts
    use_mitra_fallback : bool, default=True
        If True, fall back to Mitra if Chronos fails/hangs
    exclude_chronos : bool, default=True
        If True, exclude Chronos from AutoGluon training (Mac M4 workaround)
    time_limit : int, default=3600
        Training time limit in seconds
    verbosity : int, default=1
        Verbosity level (0=silent, 1=info, 2=debug)

    Returns
    -------
    predictor : TimeSeriesPredictor or MitraForecastWrapper
        Trained predictor (type depends on which model succeeded)
    """
    if model_path is None:
        model_path = f"data/models/timeseries_{target_col}"

    # Try AutoGluon first (unless explicitly excluded)
    if AG_AVAILABLE and not exclude_chronos:
        try:
            if verbosity >= 1:
                print("Attempting AutoGluon TimeSeriesPredictor with Chronos...")

            # Convert to TimeSeriesDataFrame if needed
            if isinstance(train_df, pd.DataFrame):
                # Assume DataFrame has 'item_id' and 'timestamp' columns
                if "item_id" not in train_df.columns:
                    # Single series - add item_id
                    train_df = train_df.copy()
                    train_df["item_id"] = "default"

                ts_df = TimeSeriesDataFrame.from_data_frame(
                    train_df,
                    id_column="item_id",
                    timestamp_column=(
                        "timestamp"
                        if "timestamp" in train_df.columns
                        else train_df.index.name
                    ),
                )
            else:
                ts_df = train_df

            # Configure hyperparameters (exclude Chronos if requested)
            if exclude_chronos:
                hyperparameters = {
                    "Naive": {},
                    "SeasonalNaive": {},
                    "RecursiveTabular": {},
                    "DirectTabular": {},
                    "ETS": {},
                    "Theta": {},
                    # 'Chronos': {},  # Excluded - hangs on Mac M4
                }
            else:
                hyperparameters = {
                    "Chronos": {},  # Chronos-Bolt zero-shot baseline
                    "Naive": {},
                    "SeasonalNaive": {},
                    "RecursiveTabular": {},
                }

            predictor = TimeSeriesPredictor(
                target=target_col,
                known_covariates_names=known_covariates,
                prediction_length=prediction_length,
                path=model_path,
                quantile_levels=[0.1, 0.5, 0.9],
                eval_metric="MASE",
                verbosity=verbosity,
            )

            predictor.fit(
                train_data=ts_df, hyperparameters=hyperparameters, time_limit=time_limit
            )

            if verbosity >= 1:
                print("✅ AutoGluon TimeSeriesPredictor training complete")

            return predictor

        except Exception as e:
            if verbosity >= 1:
                print(f"⚠️  AutoGluon TimeSeriesPredictor failed: {e}")

            if not use_mitra_fallback:
                raise

    # Fall back to Mitra
    if MITRA_AVAILABLE and use_mitra_fallback:
        if verbosity >= 1:
            print("Falling back to Mitra (Metal-accelerated on Mac M4)...")

        # Determine device
        import torch

        device = (
            "mps"
            if (torch.backends.mps.is_available() and TIMESERIES_CONFIG["use_mps"])
            else "cpu"
        )

        mitra = MitraForecastWrapper(
            prediction_length=prediction_length, device=device, verbosity=verbosity
        )

        # Mitra doesn't need training (zero-shot), but we can "fit" for compatibility
        mitra.fit(train_df)

        if verbosity >= 1:
            print("✅ Mitra predictor ready (zero-shot, no training required)")

        return mitra

    # No options available
    raise RuntimeError(
        "Neither AutoGluon TimeSeriesPredictor nor Mitra is available. "
        "Install with: pip install autogluon.timeseries[all] mitra-forecast"
    )


def predict_timeseries(
    predictor: Union[TimeSeriesPredictor, MitraForecastWrapper],
    data: Union[pd.DataFrame, TimeSeriesDataFrame],
    item_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate forecasts using trained predictor (AutoGluon or Mitra).

    Parameters
    ----------
    predictor : TimeSeriesPredictor or MitraForecastWrapper
        Trained predictor
    data : pd.DataFrame or TimeSeriesDataFrame
        Input time series data
    item_id : str, optional
        Item ID for multi-series data

    Returns
    -------
    pd.DataFrame
        Forecasts with quantiles (P10, P50, P90)
    """
    # Check if it's AutoGluon or Mitra
    if isinstance(predictor, TimeSeriesPredictor):
        # AutoGluon prediction
        predictions = predictor.predict(data)
        return predictions
    elif isinstance(predictor, MitraForecastWrapper):
        # Mitra prediction
        return predictor.predict(data, item_id=item_id)
    else:
        raise TypeError(f"Unsupported predictor type: {type(predictor)}")


# Convenience function for quick training
def quick_timeseries_train(
    train_df: pd.DataFrame,
    target_col: str = "target",
    prediction_length: int = 14,
    use_mitra: bool = True,  # Default to Mitra for Mac M4
) -> Union[TimeSeriesPredictor, MitraForecastWrapper]:
    """
    Quick training function that defaults to Mitra on Mac M4.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data
    target_col : str
        Target column
    prediction_length : int
        Forecast horizon
    use_mitra : bool
        If True, use Mitra directly (bypasses AutoGluon)

    Returns
    -------
    Trained predictor
    """
    if use_mitra and MITRA_AVAILABLE:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        return MitraForecastWrapper(prediction_length=prediction_length, device=device)
    else:
        return train_timeseries(
            train_df=train_df,
            target_col=target_col,
            prediction_length=prediction_length,
            exclude_chronos=True,  # Skip Chronos on Mac M4
            use_mitra_fallback=True,
        )








