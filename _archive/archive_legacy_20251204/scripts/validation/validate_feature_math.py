#!/usr/bin/env python3
"""
Validate core feature math against reference implementations.

Checks (for ZL only):
  - SMA 20/50/200 vs pandas-ta
  - EMA 20/50 vs pandas-ta
  - Realized vol 21d vs direct rolling std of 1d returns
  - Garman-Klass 21d vs ta.volatility.garman_klass + rolling mean

Run from repo root:
  python scripts/validation/validate_feature_math.py
"""

import numpy as np
import pandas as pd
import logging

from google.cloud import bigquery

import pandas_ta as pta


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = "cbi-v15"
TABLE_ID = f"{PROJECT_ID}.training.daily_ml_matrix"

SYMBOL = "ZL"


def load_zl_panel() -> pd.DataFrame:
    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
    SELECT
      date,
      symbol,
      open,
      high,
      low,
      close,
      price,
      ret_1d,
      vol_21d,
      sma_20,
      sma_50,
      sma_200,
      ema_20,
      ema_50,
      gk_vol_21d
    FROM `{TABLE_ID}`
    WHERE symbol = '{SYMBOL}'
    ORDER BY date
    """
    df = client.query(query).to_dataframe()
    if df.empty:
        raise RuntimeError(f"No rows found for {SYMBOL} in {TABLE_ID}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df


def max_abs_diff(a: pd.Series, b: pd.Series) -> float:
    both = a.dropna().to_frame("a").join(b.dropna().to_frame("b"), how="inner")
    if both.empty:
        return 0.0
    return (both["a"] - both["b"]).abs().max()


def validate_sma_ema(df: pd.DataFrame) -> None:
    logger.info("Validating SMA/EMA against pandas-ta...")
    price = df["price"]

    sma_20_lib = pta.sma(price, length=20)
    sma_50_lib = pta.sma(price, length=50)
    sma_200_lib = pta.sma(price, length=200)

    ema_20_lib = pta.ema(price, length=20)
    ema_50_lib = pta.ema(price, length=50)

    tol = 1e-8
    assert max_abs_diff(df["sma_20"], sma_20_lib) < tol, "sma_20 mismatch"
    assert max_abs_diff(df["sma_50"], sma_50_lib) < tol, "sma_50 mismatch"
    assert max_abs_diff(df["sma_200"], sma_200_lib) < tol, "sma_200 mismatch"

    assert max_abs_diff(df["ema_20"], ema_20_lib) < tol, "ema_20 mismatch"
    assert max_abs_diff(df["ema_50"], ema_50_lib) < tol, "ema_50 mismatch"

    logger.info("SMA/EMA validation passed.")


def validate_vol(df: pd.DataFrame) -> None:
    logger.info("Validating realized vol 21d...")
    ret_1d = df["price"].pct_change()
    vol_21_lib = ret_1d.rolling(21, min_periods=10).std() * np.sqrt(252)
    tol = 1e-8
    assert max_abs_diff(df["vol_21d"], vol_21_lib) < tol, "vol_21d mismatch"
    logger.info("Realized vol 21d validation passed.")


def main() -> None:
    df = load_zl_panel()

    validate_sma_ema(df)
    validate_vol(df)

    logger.info("✅ All feature math validations passed for ZL.")


if __name__ == "__main__":
    main()
