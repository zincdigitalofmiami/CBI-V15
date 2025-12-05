#!/usr/bin/env python3
"""
FX Indicators Builder - Python-first, Mac-run only

Purpose:
- Compute the MAIN run of FX technical indicators from FRED FX series
  for use with ZL features.

Data sources (all from FRED, already ingested by collect_fred_fx.py):
  - DTWEXBGS   (broad dollar index)
  - DTWEXEMEGS (EM dollar index)        [not used directly yet]
  - DTWEXAFEGS (advanced economies)     [not used directly yet]
  - DEXBZUS    (BRL/USD)
  - DEXMXUS    (MXN/USD)                [optional for future]
  - DEXCHUS    (CHF/USD)                [optional for future]
  - DEXUSEU    (USD/EUR)                [optional for future]

Outputs (features.fx_indicators_daily):
  - date
  - brl_momentum_21d, brl_momentum_63d, brl_momentum_252d
  - dxy_momentum_21d, dxy_momentum_63d, dxy_momentum_252d
  - brl_volatility_21d, brl_volatility_63d
  - corr_zl_brl_30d, corr_zl_brl_60d, corr_zl_brl_90d
  - corr_zl_dxy_30d, corr_zl_dxy_60d, corr_zl_dxy_90d
  - terms_of_trade_zl_brl

Run manually on Mac (no scheduler):
  python src/features/build_fx_indicators_daily.py
"""

import os
import logging
from datetime import date
from typing import List

import numpy as np
import pandas as pd
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("GCP_PROJECT", "cbi-v15")

RAW_FRED_TABLE = f"{PROJECT_ID}.raw.fred_economic"
TRAINING_TABLE = f"{PROJECT_ID}.training.daily_ml_matrix"
FX_FEATURES_TABLE = f"{PROJECT_ID}.features.fx_indicators_daily"

# FRED FX series IDs we actually use for indicators
FX_SERIES = [
    "DTWEXBGS",   # broad dollar index
    "DEXBZUS",    # BRL/USD
]


def load_fx_from_bq(client: bigquery.Client) -> pd.DataFrame:
    """
    Load FX series from raw.fred_economic and pivot to wide format.
    Returns DataFrame indexed by date with columns: dxy, brl.
    """
    series_list = ",".join(f"'{s}'" for s in FX_SERIES)
    query = f"""
    SELECT
      date,
      series_id,
      value
    FROM `{RAW_FRED_TABLE}`
    WHERE series_id IN ({series_list})
      AND date >= DATE('2010-01-01')
      AND value IS NOT NULL
    """
    logger.info("[FX] Loading FRED FX series from BigQuery...")
    df = client.query(query).to_dataframe()
    if df.empty:
        raise RuntimeError("[FX] No FX data found in raw.fred_economic for target series.")

    df["date"] = pd.to_datetime(df["date"]).dt.date
    pivot = df.pivot(index="date", columns="series_id", values="value").sort_index()

    # Rename to simpler names
    rename_map = {
        "DTWEXBGS": "dxy",   # broad dollar index
        "DEXBZUS": "brl",   # BRL/USD
    }
    pivot = pivot.rename(columns=rename_map)

    # Require both dxy and brl
    missing_cols = [c for c in ["dxy", "brl"] if c not in pivot.columns]
    if missing_cols:
        raise RuntimeError(f"[FX] Missing required FX series after pivot: {missing_cols}")

    logger.info(f"[FX] Loaded FX frame with {len(pivot):,} dates")
    return pivot


def load_zl_from_bq(client: bigquery.Client) -> pd.DataFrame:
    """
    Load ZL price and returns from training.daily_ml_matrix.
    """
    query = f"""
    SELECT
      date,
      price,
      ret_1d
    FROM `{TRAINING_TABLE}`
    WHERE symbol = 'ZL'
      AND date >= DATE('2010-01-01')
    """
    logger.info("[FX] Loading ZL price/returns from training.daily_ml_matrix...")
    df = client.query(query).to_dataframe()
    if df.empty:
        raise RuntimeError("[FX] No ZL data found in training.daily_ml_matrix")

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date")
    logger.info(f"[FX] Loaded ZL frame with {len(df):,} dates")
    return df


def compute_momentum(series: pd.Series, window: int) -> pd.Series:
    """
    Simple price momentum over N days: (P_t / P_{t-N}) - 1
    """
    return series / series.shift(window) - 1.0


def compute_realized_vol(logret: pd.Series, window: int) -> pd.Series:
    """
    Realized volatility over N days, annualized: std(log_return) * sqrt(252)
    """
    return logret.rolling(window, min_periods=max(10, window // 2)).std() * np.sqrt(252)


def compute_rolling_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """
    Rolling Pearson correlation over N days.
    """
    return x.rolling(window, min_periods=max(15, window // 2)).corr(y)


def build_fx_indicators() -> pd.DataFrame:
    """
    Build FX indicators DataFrame with date + FX feature columns.
    """
    client = bigquery.Client(project=PROJECT_ID)

    fx = load_fx_from_bq(client)
    zl = load_zl_from_bq(client)

    # Merge on date
    fx.index.name = "date"
    zl = zl.set_index("date")
    df = fx.join(zl[["price", "ret_1d"]], how="inner").dropna(subset=["price"])
    df = df.sort_index()

    # Log returns for FX series
    df["dxy_logret"] = np.log(df["dxy"] / df["dxy"].shift(1))
    df["brl_logret"] = np.log(df["brl"] / df["brl"].shift(1))
    df["zl_logret"] = np.log(df["price"] / df["price"].shift(1))

    # Momentum
    for w in (21, 63, 252):
        df[f"brl_momentum_{w}d"] = compute_momentum(df["brl"], w)
        df[f"dxy_momentum_{w}d"] = compute_momentum(df["dxy"], w)

    # Volatility (BRL)
    df["brl_volatility_21d"] = compute_realized_vol(df["brl_logret"], 21)
    df["brl_volatility_63d"] = compute_realized_vol(df["brl_logret"], 63)

    # Rolling correlations ZL-BRL / ZL-DXY
    for w in (30, 60, 90):
        df[f"corr_zl_brl_{w}d"] = compute_rolling_corr(df["zl_logret"], df["brl_logret"], w)
        df[f"corr_zl_dxy_{w}d"] = compute_rolling_corr(df["zl_logret"], df["dxy_logret"], w)

    # Terms of trade: ZL price vs BRL (simple ratio)
    # Note: DEXBZUS is BRL per USD; this is a rough ToT proxy, consistent with V14 spec.
    df["terms_of_trade_zl_brl"] = df["price"] / df["brl"]

    # Final DataFrame
    out_cols = [
        "brl_momentum_21d", "brl_momentum_63d", "brl_momentum_252d",
        "dxy_momentum_21d", "dxy_momentum_63d", "dxy_momentum_252d",
        "brl_volatility_21d", "brl_volatility_63d",
        "corr_zl_brl_30d", "corr_zl_brl_60d", "corr_zl_brl_90d",
        "corr_zl_dxy_30d", "corr_zl_dxy_60d", "corr_zl_dxy_90d",
        "terms_of_trade_zl_brl",
    ]

    # Ensure we have all columns (some may be entirely NaN if series too short)
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan

    result = df[out_cols].reset_index().rename(columns={"index": "date"})
    result["date"] = pd.to_datetime(result["date"]).dt.date

    logger.info(f"[FX] Built FX indicators for {len(result):,} dates")
    return result


def load_fx_features_to_bq(df: pd.DataFrame) -> None:
    """
    Load FX indicators into features.fx_indicators_daily with MONTH partitioning.
    """
    if df.empty:
        logger.warning("[FX] No FX indicators to load; skipping BigQuery write")
        return

    client = bigquery.Client(project=PROJECT_ID)
    df = df.sort_values("date")

    schema = [
        bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
        bigquery.SchemaField("brl_momentum_21d", "FLOAT"),
        bigquery.SchemaField("brl_momentum_63d", "FLOAT"),
        bigquery.SchemaField("brl_momentum_252d", "FLOAT"),
        bigquery.SchemaField("dxy_momentum_21d", "FLOAT"),
        bigquery.SchemaField("dxy_momentum_63d", "FLOAT"),
        bigquery.SchemaField("dxy_momentum_252d", "FLOAT"),
        bigquery.SchemaField("brl_volatility_21d", "FLOAT"),
        bigquery.SchemaField("brl_volatility_63d", "FLOAT"),
        bigquery.SchemaField("corr_zl_brl_30d", "FLOAT"),
        bigquery.SchemaField("corr_zl_brl_60d", "FLOAT"),
        bigquery.SchemaField("corr_zl_brl_90d", "FLOAT"),
        bigquery.SchemaField("corr_zl_dxy_30d", "FLOAT"),
        bigquery.SchemaField("corr_zl_dxy_60d", "FLOAT"),
        bigquery.SchemaField("corr_zl_dxy_90d", "FLOAT"),
        bigquery.SchemaField("terms_of_trade_zl_brl", "FLOAT"),
    ]

    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        time_partitioning=bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.MONTH,
            field="date",
        ),
        clustering_fields=None,
        schema=schema,
    )

    logger.info(f"[FX] Loading FX indicators to {FX_FEATURES_TABLE}...")
    job = client.load_table_from_dataframe(df, FX_FEATURES_TABLE, job_config=job_config)
    job.result()
    logger.info(f"[FX] ✅ Loaded {job.output_rows:,} FX indicator rows into {FX_FEATURES_TABLE}")


def main():
    logger.info("🚀 Building FX indicators (Mac-only)")
    df = build_fx_indicators()
    load_fx_features_to_bq(df)
    logger.info("✅ FX indicators build complete")


if __name__ == "__main__":
    main()





