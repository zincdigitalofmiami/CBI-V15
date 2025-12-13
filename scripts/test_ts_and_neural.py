#!/usr/bin/env python3
"""Test AutoGluon TimeSeriesPredictor and force neural network models."""

import pandas as pd
import numpy as np
import sys

print("=" * 80)
print("AUTOGLUON 1.4 TIMESERIES + NEURAL NETWORK TEST")
print("=" * 80)
print("")

# ============================================================================
# TEST 1: TimeSeriesPredictor
# ============================================================================
print("TEST 1: TIMESERIES PREDICTOR")
print("=" * 60)

try:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    import tempfile

    # Create synthetic time series data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    n_items = 2

    data = []
    for item_id in range(n_items):
        for i, date in enumerate(dates):
            data.append(
                {
                    "item_id": f"item_{item_id}",
                    "timestamp": date,
                    "target": 100
                    + np.sin(i / 30) * 10
                    + np.random.randn() * 3
                    + item_id * 20,
                }
            )

    df = pd.DataFrame(data)
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df, id_column="item_id", timestamp_column="timestamp"
    )

    print(
        f"✅ TimeSeriesDataFrame created: {len(ts_df)} rows, {len(ts_df.item_ids)} items"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Training TimeSeriesPredictor (60s limit)...")
        predictor = TimeSeriesPredictor(
            path=tmpdir,
            target="target",
            prediction_length=7,
            eval_metric="MASE",
            verbosity=1,
        )

        predictor.fit(train_data=ts_df, presets="fast_training", time_limit=60)

        lb = predictor.leaderboard()
        print("")
        print("TIMESERIES MODEL LEADERBOARD:")
        print(lb[["model", "score_val", "pred_time_val"]].to_string())
        print("")
        print(f"✅ TimeSeriesPredictor: {len(lb)} models trained")
        print(f'   Best: {lb.iloc[0]["model"]} (score: {lb.iloc[0]["score_val"]:.4f})')

except Exception as e:
    print(f"❌ TimeSeriesPredictor FAILED: {e}")
    import traceback

    traceback.print_exc()

print("")
print("")

# ============================================================================
# TEST 2: Force Neural Network Models (TabularPredictor)
# ============================================================================
print("TEST 2: NEURAL NETWORK MODELS (TabularPredictor)")
print("=" * 60)

try:
    from autogluon.tabular import TabularPredictor
    import tempfile

    # Create tabular dataset
    np.random.seed(42)
    n = 500
    df_tab = pd.DataFrame(
        {
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
            "f4": np.random.randn(n),
            "f5": np.random.randn(n),
            "target": np.random.randn(n),
        }
    )

    print(f"Dataset: {len(df_tab)} rows, 5 features")

    # Try to force neural networks with hyperparameters
    with tempfile.TemporaryDirectory() as tmpdir:
        print("Attempting to train neural network models...")

        hyperparameters = {
            "NN_TORCH": {},  # PyTorch neural network
            "FASTAI": {},  # FastAI neural network
        }

        predictor = TabularPredictor(
            label="target", path=tmpdir, problem_type="regression", verbosity=2
        ).fit(
            train_data=df_tab,
            hyperparameters=hyperparameters,
            time_limit=120,
            num_gpus=0,  # Force CPU
        )

        lb = predictor.leaderboard(silent=True)
        print("")
        print("NEURAL NETWORK MODEL RESULTS:")
        print(lb[["model", "score_val", "pred_time_val", "fit_time"]].to_string())
        print("")
        print(f"✅ Neural models: {len(lb)} trained")

except Exception as e:
    print(f"❌ Neural network test FAILED: {e}")
    import traceback

    traceback.print_exc()

print("")
print("")

# ============================================================================
# TEST 3: Mitra Foundation Model (Metal-Accelerated on Mac M4)
# ============================================================================
print("TEST 3: MITRA FOUNDATION MODEL (METAL/MPS ON MAC M4)")
print("=" * 60)

try:
    import torch
    from src.training.autogluon.mitra_trainer import MitraForecastWrapper

    # Check MPS availability
    mps_available = torch.backends.mps.is_available()
    device = "mps" if mps_available else "cpu"
    print(f"PyTorch MPS available: {mps_available}")
    print(f"Using device: {device}")
    print("")

    # Create synthetic time series data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=256, freq="D")
    series = 100 + np.sin(np.arange(256) / 30) * 10 + np.random.randn(256) * 3

    df_mitra = pd.DataFrame({"timestamp": dates, "target": series})

    print(f"Time series data: {len(df_mitra)} points")
    print("Loading Mitra model...")

    # Create Mitra predictor
    mitra = MitraForecastWrapper(
        model_name="salesforce/mitra-base",
        device=device,
        prediction_length=14,
        verbosity=1,
    )

    print("Generating forecasts...")
    forecasts = mitra.predict(df_mitra)

    print("")
    print("MITRA FORECAST RESULTS:")
    print(forecasts[["timestamp", "mean", "P10", "P50", "P90"]].head(10).to_string())
    print("")
    print(f"✅ Mitra: {len(forecasts)} forecasts generated")
    print(
        f'   Mean forecast range: [{forecasts["P10"].min():.2f}, {forecasts["P90"].max():.2f}]'
    )
    print(f'   P50 median: {forecasts["P50"].median():.2f}')

except ImportError as e:
    print(f"⚠️  Mitra not available: {e}")
    print("   Install with: pip install mitra-forecast")
except Exception as e:
    print(f"❌ Mitra test FAILED: {e}")
    import traceback

    traceback.print_exc()

print("")
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
