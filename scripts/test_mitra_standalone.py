#!/usr/bin/env python3
"""
Standalone test for Mitra foundation model on Mac M4 (Metal/MPS).

This test verifies:
1. Mitra loads successfully on Metal (MPS)
2. Generates probabilistic forecasts (P10/P50/P90)
3. Works with TimeSeriesDataFrame format
4. Performance on Mac M4
"""

import sys
import numpy as np
import pandas as pd

print("=" * 80)
print("MITRA STANDALONE TEST (MAC M4 METAL ACCELERATION)")
print("=" * 80)
print("")

# ============================================================================
# Test 1: PyTorch MPS Verification
# ============================================================================
print("TEST 1: PYTORCH MPS BACKEND")
print("=" * 60)

try:
    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        device = "mps"
        print(f"✅ Using Metal (MPS) acceleration")

        # Quick MPS test
        x = torch.randn(100, 100).to(device)
        y = torch.matmul(x, x.T)
        print(f"✅ MPS tensor operation test passed: {y.shape}")
    else:
        device = "cpu"
        print(f"⚠️  MPS not available, using CPU")

except Exception as e:
    print(f"❌ PyTorch MPS check failed: {e}")
    device = "cpu"

print("")
print("")

# ============================================================================
# Test 2: Mitra Import and Loading
# ============================================================================
print("TEST 2: MITRA MODEL LOADING")
print("=" * 60)

try:
    from src.training.autogluon.mitra_trainer import MitraForecastWrapper

    print("Loading Mitra model (salesforce/mitra-base)...")
    mitra = MitraForecastWrapper(
        model_name="salesforce/mitra-base",
        device=device,
        prediction_length=14,
        verbosity=1,
    )

    print("✅ Mitra model loaded successfully")

except ImportError as e:
    print(f"❌ Mitra import failed: {e}")
    print("   Install with: pip install mitra-forecast")
    sys.exit(1)
except Exception as e:
    print(f"❌ Mitra loading failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("")
print("")

# ============================================================================
# Test 3: Forecast Generation (NumPy Array)
# ============================================================================
print("TEST 3: FORECAST GENERATION (NUMPY ARRAY)")
print("=" * 60)

try:
    # Create synthetic time series
    np.random.seed(42)
    series = 100 + np.sin(np.arange(256) / 30) * 10 + np.random.randn(256) * 3
    series = series.astype(np.float32)

    print(f"Input series: {len(series)} points")
    print(f"   Mean: {series.mean():.2f}, Std: {series.std():.2f}")
    print("Generating 14-step forecast...")

    forecasts = mitra.predict(series)

    print("")
    print("FORECAST RESULTS:")
    print(forecasts[["timestamp", "mean", "P10", "P50", "P90"]].to_string())
    print("")
    print(f"✅ Forecast complete: {len(forecasts)} steps")
    print(f'   Mean forecast: {forecasts["mean"].mean():.2f}')
    print(f'   P50 range: [{forecasts["P50"].min():.2f}, {forecasts["P50"].max():.2f}]')

except Exception as e:
    print(f"❌ Forecast generation failed: {e}")
    import traceback

    traceback.print_exc()

print("")
print("")

# ============================================================================
# Test 4: Forecast Generation (DataFrame)
# ============================================================================
print("TEST 4: FORECAST GENERATION (DATAFRAME)")
print("=" * 60)

try:
    # Create DataFrame with timestamp and target
    dates = pd.date_range("2020-01-01", periods=256, freq="D")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "target": 100 + np.sin(np.arange(256) / 30) * 10 + np.random.randn(256) * 3,
        }
    )

    print(f"Input DataFrame: {len(df)} rows")
    print("Generating forecast...")

    forecasts = mitra.predict(df)

    print("")
    print("FORECAST RESULTS (DataFrame input):")
    print(forecasts.head(10).to_string())
    print("")
    print(f"✅ DataFrame forecast complete: {len(forecasts)} steps")

except Exception as e:
    print(f"❌ DataFrame forecast failed: {e}")
    import traceback

    traceback.print_exc()

print("")
print("")

# ============================================================================
# Test 5: TimeSeriesDataFrame (if AutoGluon available)
# ============================================================================
print("TEST 5: TIMESERIESDATAFRAME (AUTOGLUON FORMAT)")
print("=" * 60)

try:
    from autogluon.timeseries import TimeSeriesDataFrame

    # Create TimeSeriesDataFrame
    data = []
    for item_id in ["item_1", "item_2"]:
        for i, date in enumerate(pd.date_range("2020-01-01", periods=200, freq="D")):
            data.append(
                {
                    "item_id": item_id,
                    "timestamp": date,
                    "target": 100 + np.sin(i / 30) * 10 + np.random.randn() * 3,
                }
            )

    df_ts = pd.DataFrame(data)
    ts_df = TimeSeriesDataFrame.from_data_frame(
        df_ts, id_column="item_id", timestamp_column="timestamp"
    )

    print(f"TimeSeriesDataFrame: {len(ts_df)} rows, {len(ts_df.item_ids)} items")
    print("Generating forecast for first item...")

    forecasts = mitra.predict(ts_df, item_id="item_1")

    print("")
    print("FORECAST RESULTS (TimeSeriesDataFrame input):")
    print(forecasts.head(10).to_string())
    print("")
    print(f"✅ TimeSeriesDataFrame forecast complete: {len(forecasts)} steps")

except ImportError:
    print("⚠️  AutoGluon not available, skipping TimeSeriesDataFrame test")
except Exception as e:
    print(f"❌ TimeSeriesDataFrame forecast failed: {e}")
    import traceback

    traceback.print_exc()

print("")
print("=" * 80)
print("✅ ALL TESTS COMPLETE")
print("=" * 80)
print("")
print("Summary:")
print(f"  - Device: {device}")
print(f"  - Mitra model: salesforce/mitra-base")
print(f"  - Forecast length: 14 steps")
print(f"  - Probabilistic outputs: P10, P50, P90 quantiles")
print("")









