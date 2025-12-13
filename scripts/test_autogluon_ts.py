#!/usr/bin/env python3
"""Test AutoGluon TimeSeriesPredictor."""

import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import tempfile

print("=" * 80)
print("AUTOGLUON 1.4 TIMESERIES PREDICTOR TEST")
print("=" * 80)

# Create synthetic time series data
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=500, freq="D")
n_items = 3

data = []
for item_id in range(n_items):
    for i, date in enumerate(dates):
        data.append(
            {
                "item_id": f"item_{item_id}",
                "timestamp": date,
                "target": 100
                + np.sin(i / 30) * 10
                + np.random.randn() * 5
                + item_id * 20,
            }
        )

df = pd.DataFrame(data)
ts_df = TimeSeriesDataFrame.from_data_frame(
    df, id_column="item_id", timestamp_column="timestamp"
)

print(f"TimeSeriesDataFrame: {len(ts_df)} rows, {len(ts_df.item_ids)} items")
print("")

# Test TimeSeriesPredictor
print("=" * 60)
print("TIMESERIES PREDICTOR TEST (2 min)")
print("=" * 60)
print("NOTE: Excluding Chronos[bolt_small] - hangs on Mac M4 with mutex lock")
print("")

with tempfile.TemporaryDirectory() as tmpdir:
    predictor = TimeSeriesPredictor(
        path=tmpdir,
        target="target",
        prediction_length=14,
        eval_metric="MASE",
        verbosity=2,
    )

    # Custom hyperparameters excluding Chronos (hangs on Mac M4)
    hyperparameters = {
        "Naive": {},
        "SeasonalNaive": {},
        "RecursiveTabular": {},
        "DirectTabular": {},
        "ETS": {},
        "Theta": {},
        # 'Chronos': {},  # Excluded - hangs on Mac M4 with mutex lock
    }

    predictor.fit(train_data=ts_df, hyperparameters=hyperparameters, time_limit=120)

    print("")
    print("=" * 60)
    print("TIMESERIES MODEL LEADERBOARD")
    print("=" * 60)
    lb = predictor.leaderboard()
    print(lb.to_string())

    print("")
    print(f"Total models: {len(lb)}")
    print(f'Best model: {lb.iloc[0]["model"]}')
