#!/usr/bin/env python3
"""
Export ZL train/val/test splits from BigQuery training.daily_ml_matrix
into local Parquet files expected by the LightGBM baseline:

  TrainingData/exports/zl_training_minimal_{horizon}_{split}.parquet

Horizons: 1w, 1m, 3m, 6m
Splits:
  train: 2010-01-01 to 2018-12-31
  val:   2019-01-01 to 2021-12-31
  test:  2022-01-01 onward
"""

import os
from pathlib import Path

import pandas as pd
from google.cloud import bigquery


PROJECT_ID = "cbi-v15"
TABLE_ID = "cbi-v15.training.daily_ml_matrix"

OUT_DIR = Path("TrainingData/exports")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_zl_panel() -> pd.DataFrame:
    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
    SELECT *
    FROM `{TABLE_ID}`
    WHERE symbol = 'ZL'
      AND target_1w IS NOT NULL
    ORDER BY date
    """
    df = client.query(query).to_dataframe()
    if df.empty:
        raise RuntimeError("No ZL rows with target_1w found in training.daily_ml_matrix")
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_target_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    # The baseline trainer expects target_{horizon}_price columns
    df = df.copy()
    horizon_map = {
        "1w": "target_1w",
        "1m": "target_1m",
        "3m": "target_3m",
        "6m": "target_6m",
    }
    for h, col in horizon_map.items():
        target_price_col = f"target_{h}_price"
        if target_price_col not in df.columns:
            df[target_price_col] = df[col]
    # For backward compatibility with baseline script, add price_current
    if "price_current" not in df.columns and "price" in df.columns:
        df["price_current"] = df["price"]
    return df


def split_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.to_datetime(start)
    if end is None:
        mask = df["date"] >= start_ts
    else:
        end_ts = pd.to_datetime(end)
        mask = (df["date"] >= start_ts) & (df["date"] <= end_ts)
    return df[mask].copy()


def main() -> None:
    df = fetch_zl_panel()
    df = add_target_price_columns(df)

    splits = {
        "train": ("2010-01-01", "2018-12-31"),
        "val": ("2019-01-01", "2021-12-31"),
        "test": ("2022-01-01", None),
    }

    horizons = ["1w", "1m", "3m", "6m"]

    for split_name, (start, end) in splits.items():
        split_df = split_by_date(df, start, end)
        for h in horizons:
            out_path = OUT_DIR / f"zl_training_minimal_{h}_{split_name}.parquet"
            split_df.to_parquet(out_path, index=False)

    print("✅ Exported ZL splits for all horizons to TrainingData/exports/")


if __name__ == "__main__":
    main()

