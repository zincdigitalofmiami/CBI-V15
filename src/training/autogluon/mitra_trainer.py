"""
Mitra Time Series Foundation Model Wrapper for Mac M4 (Metal/MPS).

Mitra is Salesforce's time-series foundation model that works reliably
on Apple Silicon via PyTorch's Metal Performance Shaders (MPS) backend.

This wrapper provides:
- Metal-accelerated inference on Mac M4
- Probabilistic forecasts (P10/P50/P90 quantiles)
- Compatible interface with AutoGluon TimeSeriesPredictor
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional, Union, Dict, Any
from pathlib import Path

try:
    from mitra.models import MitraForecast

    MITRA_AVAILABLE = True
except ImportError:
    MITRA_AVAILABLE = False
    MitraForecast = None

try:
    from autogluon.timeseries import TimeSeriesDataFrame

    AG_AVAILABLE = True
except ImportError:
    AG_AVAILABLE = False
    TimeSeriesDataFrame = None


class MitraForecastWrapper:
    """
    Wrapper for Mitra foundation model with Metal (MPS) support on Mac M4.

    Mitra is a zero-shot time series forecasting model that works well
    for inference and light fine-tuning on Apple Silicon.
    """

    def __init__(
        self,
        model_name: str = "salesforce/mitra-base",
        device: Optional[str] = None,
        prediction_length: int = 14,
        num_samples: int = 100,
        verbosity: int = 1,
    ):
        """
        Initialize Mitra forecast wrapper.

        Parameters
        ----------
        model_name : str, default="salesforce/mitra-base"
            HuggingFace model name for Mitra
        device : str, optional
            Device to use ('mps', 'cpu', or None for auto-detect)
        prediction_length : int, default=14
            Number of steps to forecast ahead
        num_samples : int, default=100
            Number of samples for probabilistic forecasting
        verbosity : int, default=1
            Verbosity level (0=silent, 1=info, 2=debug)
        """
        if not MITRA_AVAILABLE:
            raise ImportError(
                "mitra-forecast package not installed. "
                "Install with: pip install mitra-forecast"
            )

        # Auto-detect device if not specified
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model_name = model_name
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.verbosity = verbosity

        # Load model
        if self.verbosity >= 1:
            print(f"Loading Mitra model '{model_name}' on device: {device}")

        self.model = MitraForecast.from_pretrained(model_name, device=device)

        if self.verbosity >= 1:
            print(f"✅ Mitra model loaded successfully")

    def predict(
        self,
        data: Union[pd.DataFrame, TimeSeriesDataFrame, np.ndarray],
        item_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate forecasts using Mitra.

        Parameters
        ----------
        data : pd.DataFrame, TimeSeriesDataFrame, or np.ndarray
            Time series data. If DataFrame, must have 'timestamp' and 'target' columns.
            If TimeSeriesDataFrame, will extract target series.
            If ndarray, treated as single time series.
        item_id : str, optional
            Item ID for multi-series data (if using DataFrame format)

        Returns
        -------
        pd.DataFrame
            Forecasts with columns: ['timestamp', 'mean', 'P10', 'P50', 'P90']
        """
        # Convert input to numpy array
        if isinstance(data, np.ndarray):
            series = data.astype(np.float32)
        elif isinstance(data, TimeSeriesDataFrame) if AG_AVAILABLE else False:
            # Extract target series from TimeSeriesDataFrame
            if item_id is None:
                item_id = data.item_ids[0]
            series = data.loc[item_id]["target"].values.astype(np.float32)
        elif isinstance(data, pd.DataFrame):
            # DataFrame with 'target' column
            if "target" not in data.columns:
                raise ValueError("DataFrame must have 'target' column")
            series = data["target"].values.astype(np.float32)
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. "
                "Expected: np.ndarray, pd.DataFrame, or TimeSeriesDataFrame"
            )

        # Generate forecast
        if self.verbosity >= 1:
            print(f"Generating {self.prediction_length}-step forecast...")

        with torch.no_grad():
            forecast_samples = self.model.predict(
                series, horizon=self.prediction_length, num_samples=self.num_samples
            )

        # Convert to quantiles (P10, P50, P90)
        # forecast_samples shape: (num_samples, prediction_length)
        if isinstance(forecast_samples, torch.Tensor):
            forecast_samples = forecast_samples.cpu().numpy()

        # Compute quantiles
        p10 = np.percentile(forecast_samples, 10, axis=0)
        p50 = np.percentile(forecast_samples, 50, axis=0)
        p90 = np.percentile(forecast_samples, 90, axis=0)
        mean = np.mean(forecast_samples, axis=0)

        # Create output DataFrame
        if isinstance(data, pd.DataFrame) and "timestamp" in data.columns:
            # Use last timestamp as base
            last_timestamp = pd.to_datetime(data["timestamp"].iloc[-1])
            timestamps = pd.date_range(
                start=last_timestamp + pd.Timedelta(days=1),
                periods=self.prediction_length,
                freq="D",
            )
        else:
            # Generate timestamps from today
            timestamps = pd.date_range(
                start=pd.Timestamp.now().normalize(),
                periods=self.prediction_length,
                freq="D",
            )

        result = pd.DataFrame(
            {"timestamp": timestamps, "mean": mean, "P10": p10, "P50": p50, "P90": p90}
        )

        if self.verbosity >= 1:
            print(f"✅ Forecast complete: {len(result)} steps")

        return result

    def fit(self, train_data: Union[pd.DataFrame, TimeSeriesDataFrame]):
        """
        Placeholder for compatibility with AutoGluon interface.

        Mitra is a zero-shot model and doesn't require training.
        This method exists for API compatibility only.
        """
        if self.verbosity >= 1:
            print("⚠️  Mitra is a zero-shot model - no training required")
        return self


def create_mitra_predictor(
    prediction_length: int = 14, device: Optional[str] = None, verbosity: int = 1
) -> MitraForecastWrapper:
    """
    Factory function to create a Mitra predictor.

    Parameters
    ----------
    prediction_length : int, default=14
        Number of steps to forecast ahead
    device : str, optional
        Device to use ('mps', 'cpu', or None for auto-detect)
    verbosity : int, default=1
        Verbosity level

    Returns
    -------
    MitraForecastWrapper
        Configured Mitra predictor
    """
    return MitraForecastWrapper(
        prediction_length=prediction_length, device=device, verbosity=verbosity
    )









