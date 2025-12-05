"""
Canonical Feature Builder - Python-First Architecture
Replaces Dataform feature layer

LOCKED FORMULAS (from V15_LOCKED_FEATURES.md):
- BOHO Spread: (ZL/100 * 7.5) - HO
- Garman-Klass: SQRT(0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2)
- Targets: Future price levels (not returns)
"""

import os
import math
from typing import List

import numpy as np
import pandas as pd
import pandas_ta as pta
from google.cloud import bigquery

PROJECT_ID = os.getenv("GCP_PROJECT", "cbi-v15")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_table(sql: str) -> pd.DataFrame:
    """Load data from BigQuery."""
    client = bigquery.Client(project=PROJECT_ID)
    return client.query(sql).to_dataframe()


def assert_columns(df: pd.DataFrame, required: List[str], context: str):
    """Fail fast if required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"[{context}] missing required columns: {missing}")


def garman_klass_daily(high: np.ndarray, low: np.ndarray, 
                       open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    """
    Garman-Klass daily volatility estimator (per-day, not annualized).
    
    Formula: sqrt(0.5 * ln(H/L)^2 - (2*ln(2)-1) * ln(C/O)^2)
    
    Returns daily GK vol; to annualize, multiply by sqrt(252).
    """
    # Prevent division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        log_hl = np.log(high / low)
        log_co = np.log(close / open_)
        
        # GK variance (per-day)
        variance = 0.5 * (log_hl ** 2) - (2 * math.log(2) - 1) * (log_co ** 2)
        
        # GK volatility
        gk_vol = np.sqrt(np.clip(variance, 0, None))
        
    return np.where(np.isfinite(gk_vol), gk_vol, np.nan)


# -----------------------------------------------------------------------------
# Main Builder
# -----------------------------------------------------------------------------
def build_daily_ml_matrix() -> pd.DataFrame:
    """
    Build the daily ML feature matrix from BigQuery staging data.
    
    Returns:
        DataFrame with all features, partitioned/clustered ready for BQ load.
    """
    
    # =========================================================================
    # 1) LOAD MARKET DATA
    # =========================================================================
    print("[build_daily_ml_matrix] Loading market data...")
    mkt = load_table(f"""
        SELECT 
            date,
            symbol,
            CAST(open AS FLOAT64) as open,
            CAST(high AS FLOAT64) as high,
            CAST(low AS FLOAT64) as low,
            CAST(close AS FLOAT64) as close,
            CAST(volume AS INT64) as volume
        FROM `{PROJECT_ID}.staging.market_daily`
    """)
    assert_columns(mkt, ["date", "symbol", "open", "high", "low", "close"], "market_daily")
    print(f"  -> Loaded {len(mkt):,} rows, {mkt['symbol'].nunique()} symbols")
    
    # =========================================================================
    # 2) LOAD REGIME CALENDAR (Range-based)
    # =========================================================================
    print("[build_daily_ml_matrix] Loading regime calendar...")
    regimes = load_table(f"""
        SELECT 
            regime_type as regime,
            start_date,
            end_date,
            base_weight
        FROM `{PROJECT_ID}.reference.regime_calendar`
    """)
    
    # Convert dates
    mkt['date'] = pd.to_datetime(mkt['date']).dt.date
    
    if regimes.empty:
        raise RuntimeError(
            "[build_daily_ml_matrix] reference.regime_calendar is empty. "
            "Regimes are required for proper training. "
            "Run: scripts/setup/initialize_reference_tables.sql"
        )
    else:
        print(f"  -> Loaded {len(regimes)} regime periods")
        regimes['start_date'] = pd.to_datetime(regimes['start_date']).dt.date
        regimes['end_date'] = pd.to_datetime(regimes['end_date']).dt.date
        
        # Range-based join: find regime for each date
        def get_regime(date):
            for _, row in regimes.iterrows():
                if row['start_date'] <= date <= row['end_date']:
                    return row['regime'], row['base_weight']
            return None, None  # fail later if uncovered
        
        # Build regime mapping for unique dates
        unique_dates = mkt['date'].drop_duplicates().sort_values()
        regime_list = [get_regime(d) for d in unique_dates]
        regime_df = pd.DataFrame({
            'date': unique_dates,
            'regime': [r[0] for r in regime_list],
            'regime_weight': [r[1] for r in regime_list]
        })
        
        mkt = mkt.merge(regime_df, on='date', how='left')
        if mkt['regime'].isna().any():
            missing_dates = mkt.loc[mkt['regime'].isna(), 'date'].drop_duplicates().head(10).tolist()
            raise RuntimeError(
                f"[build_daily_ml_matrix] Regime missing for some dates, e.g. {missing_dates}. "
                "Ensure regime_calendar covers all market dates."
            )
    
    # =========================================================================
    # 3) SORT AND GROUP
    # =========================================================================
    df = mkt.sort_values(["symbol", "date"]).copy()
    
    # =========================================================================
    # 4) CORE PRICE FEATURES (per symbol)
    # =========================================================================
    print("[build_daily_ml_matrix] Computing core price features...")
    
    # Rename close to price for clarity
    df['price'] = df['close']
    
    # Group by symbol for per-symbol rolling calculations
    grp = df.groupby('symbol', group_keys=False)
    
    # --- Lagged Prices ---
    df['lag_1d'] = grp['price'].shift(1)
    df['lag_5d'] = grp['price'].shift(5)
    df['lag_21d'] = grp['price'].shift(21)
    
    # --- Returns (percentage change) ---
    df['ret_1d'] = grp['price'].pct_change(1)
    df['ret_5d'] = grp['price'].pct_change(5)
    df['ret_21d'] = grp['price'].pct_change(21)
    
    # --- Rolling Volatility (annualized) ---
    df['vol_21d'] = grp['ret_1d'].transform(
        lambda x: x.rolling(21, min_periods=10).std() * np.sqrt(252)
    )
    
    # --- Simple Moving Averages (aligned with pandas-ta) ---
    df['sma_20'] = grp['price'].transform(lambda x: pta.sma(x, length=20))
    df['sma_50'] = grp['price'].transform(lambda x: pta.sma(x, length=50))
    df['sma_200'] = grp['price'].transform(lambda x: pta.sma(x, length=200))
    
    # --- Exponential Moving Averages (aligned with pandas-ta) ---
    df['ema_20'] = grp['price'].transform(lambda x: pta.ema(x, length=20))
    df['ema_50'] = grp['price'].transform(lambda x: pta.ema(x, length=50))
    
    # --- Distance from MAs (stationary features) ---
    df['dist_sma_20'] = (df['price'] / df['sma_20']) - 1
    df['dist_sma_50'] = (df['price'] / df['sma_50']) - 1
    df['dist_ema_20'] = (df['price'] / df['ema_20']) - 1
    
    # =========================================================================
    # 5) GARMAN-KLASS VOLATILITY (Annualized, 21-day rolling)
    # =========================================================================
    print("[build_daily_ml_matrix] Computing Garman-Klass volatility...")
    
    def compute_gk_vol_annualized(g):
        """Compute annualized GK volatility with 21-day rolling window."""
        gk_daily = garman_klass_daily(
            g['high'].values,
            g['low'].values,
            g['open'].values,
            g['close'].values
        )
        # Rolling mean of daily GK, then annualize
        gk_series = pd.Series(gk_daily, index=g.index)
        gk_rolling = gk_series.rolling(21, min_periods=10).mean()
        return gk_rolling * np.sqrt(252)
    
    df['gk_vol_21d'] = grp.apply(compute_gk_vol_annualized).reset_index(level=0, drop=True)
    
    # =========================================================================
    # 6) FX SPOT TECHNICALS (from FRED)
    # =========================================================================
    print("[build_daily_ml_matrix] Computing FX technicals from FRED (BRL, Dollar)...")
    fx = load_table(f"""
        SELECT date, series_id, value
        FROM `{PROJECT_ID}.raw.fred_economic`
        WHERE series_id IN ('DEXBZUS', 'DTWEXBGS')
    """)
    if not fx.empty:
        fx['date'] = pd.to_datetime(fx['date']).dt.date
        fx_piv = fx.pivot(index='date', columns='series_id', values='value').sort_index()
        fx_piv = fx_piv.rename(columns={
            'DEXBZUS': 'fx_brl',
            'DTWEXBGS': 'fx_dxy'
        })
        
        # BRL FX
        fx_piv['fx_brl_ret_1d'] = fx_piv['fx_brl'].pct_change()
        fx_piv['fx_brl_mom_21d'] = fx_piv['fx_brl'].pct_change(21)
        fx_piv['fx_brl_mom_63d'] = fx_piv['fx_brl'].pct_change(63)
        fx_piv['fx_brl_mom_252d'] = fx_piv['fx_brl'].pct_change(252)
        fx_piv['fx_brl_vol_21d'] = fx_piv['fx_brl_ret_1d'].rolling(21, min_periods=10).std() * np.sqrt(252)
        fx_piv['fx_brl_vol_63d'] = fx_piv['fx_brl_ret_1d'].rolling(63, min_periods=25).std() * np.sqrt(252)
        
        # Dollar index FX
        fx_piv['fx_dxy_ret_1d'] = fx_piv['fx_dxy'].pct_change()
        fx_piv['fx_dxy_mom_21d'] = fx_piv['fx_dxy'].pct_change(21)
        fx_piv['fx_dxy_mom_63d'] = fx_piv['fx_dxy'].pct_change(63)
        fx_piv['fx_dxy_mom_252d'] = fx_piv['fx_dxy'].pct_change(252)
        fx_piv['fx_dxy_vol_21d'] = fx_piv['fx_dxy_ret_1d'].rolling(21, min_periods=10).std() * np.sqrt(252)
        fx_piv['fx_dxy_vol_63d'] = fx_piv['fx_dxy_ret_1d'].rolling(63, min_periods=25).std() * np.sqrt(252)
        
        # FX momentum/volatility features
        fx_features = fx_piv[[
            'fx_brl_mom_21d', 'fx_brl_mom_63d', 'fx_brl_mom_252d',
            'fx_brl_vol_21d', 'fx_brl_vol_63d',
            'fx_dxy_mom_21d', 'fx_dxy_mom_63d', 'fx_dxy_mom_252d',
            'fx_dxy_vol_21d', 'fx_dxy_vol_63d'
        ]].reset_index()
        
        # ZL returns for cross-asset correlations / terms of trade
        zl_df = df[df['symbol'] == 'ZL'][['date', 'price']].copy()
        zl_df['zl_log_ret'] = np.log(zl_df['price'] / zl_df['price'].shift(1))
        
        fx_corr = fx_piv[['fx_brl', 'fx_dxy']].copy()
        fx_corr['brl_log_ret'] = np.log(fx_corr['fx_brl'] / fx_corr['fx_brl'].shift(1))
        fx_corr['dxy_log_ret'] = np.log(fx_corr['fx_dxy'] / fx_corr['fx_dxy'].shift(1))
        
        merged = zl_df.set_index('date').join(fx_corr, how='left')
        
        for window in [30, 60, 90]:
            min_periods = window // 2
            merged[f'corr_zl_brl_{window}d'] = merged['zl_log_ret'].rolling(window, min_periods=min_periods).corr(merged['brl_log_ret'])
            merged[f'corr_zl_dxy_{window}d'] = merged['zl_log_ret'].rolling(window, min_periods=min_periods).corr(merged['dxy_log_ret'])
        
        # Terms of trade: ZL price / BRL FX level
        merged['terms_of_trade_zl_brl'] = merged['price'] / merged['fx_brl']
        
        corr_features = merged[[
            'corr_zl_brl_30d', 'corr_zl_brl_60d', 'corr_zl_brl_90d',
            'corr_zl_dxy_30d', 'corr_zl_dxy_60d', 'corr_zl_dxy_90d',
            'terms_of_trade_zl_brl'
        ]].reset_index()
        
        df = df.merge(fx_features, on='date', how='left')
        df = df.merge(corr_features, on='date', how='left')
    else:
        print("[build_daily_ml_matrix] WARNING: No FRED FX data found; FX technicals not populated.")
    
    # =========================================================================
    # 7) FRED MACRO LEVELS (Rates, Curve, Financial Conditions)
    # =========================================================================
    print("[build_daily_ml_matrix] Merging FRED macro series (levels)...")
    fred_series_map = {
        # Policy & overnight
        "DFF": "fred_dff",
        "DFEDTARU": "fred_dfedtaru",
        "DFEDTARL": "fred_dfedtarl",
        "EFFR": "fred_effr",
        "SOFR": "fred_sofr",
        # Treasury curve
        "DGS3MO": "fred_dgs3mo",
        "DGS1": "fred_dgs1",
        "DGS2": "fred_dgs2",
        "DGS5": "fred_dgs5",
        "DGS10": "fred_dgs10",
        "DGS30": "fred_dgs30",
        # Spreads
        "T10Y2Y": "fred_t10y2y",
        "T10Y3M": "fred_t10y3m",
        # Financial conditions / risk
        "VIXCLS": "fred_vixcls",
        "NFCI": "fred_nfci",
        "NFCILEVERAGE": "fred_nfcileverage",
        "BAAFFM": "fred_baaffm",
        "BAMLH0A0HYM2": "fred_bamlh0a0hym2",
    }
    fred_ids = ",".join(f"'{sid}'" for sid in fred_series_map.keys())
    fred_macro = load_table(f"""
        SELECT date, series_id, value
        FROM `{PROJECT_ID}.raw.fred_economic`
        WHERE series_id IN ({fred_ids})
    """)
    if not fred_macro.empty:
        fred_macro["date"] = pd.to_datetime(fred_macro["date"]).dt.date
        macro_wide = (
            fred_macro
            .pivot(index="date", columns="series_id", values="value")
            .sort_index()
        )
        # Rename to fred_* columns and forward-fill to cover weekends/holidays
        macro_wide = macro_wide.rename(columns=fred_series_map)
        macro_wide = macro_wide.sort_index().ffill()
        df = df.merge(macro_wide.reset_index(), on="date", how="left")
        print(f"  -> FRED macro series merged: {len(macro_wide.columns)} columns")
    else:
        print("[build_daily_ml_matrix] WARNING: No FRED macro data found; macro levels not populated.")
    
    # =========================================================================
    # 8) BOHO SPREAD (ZL vs HO) - LOCKED FORMULA
    # =========================================================================
    print("[build_daily_ml_matrix] Computing BOHO spread...")
    
    # Pivot to get ZL, HO prices by date
    piv = df.pivot_table(index='date', columns='symbol', values='price', aggfunc='first')
    
    # LOCKED FORMULA: (ZL/100 * 7.5) - HO
    # Note: CL is intentionally excluded. This measures pure ZL-HO biodiesel arbitrage.
    # ZL in cents/lb, HO in $/gal, conversion factor 7.5 lbs/gal
    required_boho_syms = ['ZL', 'HO']
    if all(sym in piv.columns for sym in required_boho_syms):
        zl = piv['ZL']  # cents/lb
        ho = piv['HO']  # $/gal
        
        # LOCKED FORMULA: (ZL/100 * 7.5) - HO
        # ZL/100 = $/lb, * 7.5 = $/gal (approx 7.5 lbs per gallon)
        zl_per_gal = (zl / 100) * 7.5
        boho = zl_per_gal - ho  # Positive = ZL expensive, Negative = HO expensive
        
        boho_df = boho.rename('boho_spread').reset_index()
        df = df.merge(boho_df, on='date', how='left')
        print("  -> BOHO spread computed successfully")
    else:
        missing = [s for s in required_boho_syms if s not in piv.columns]
        print(f"  WARNING: BOHO skipped; missing symbols: {missing}")
        df['boho_spread'] = np.nan
    
    # =========================================================================
    # 9) TARGETS - ZL ONLY, FUTURE PRICE LEVELS (not returns)
    # =========================================================================
    print("[build_daily_ml_matrix] Computing targets (ZL only)...")
    
    df = df.sort_values(['symbol', 'date'])
    
    # Initialize target columns as NaN
    for col in ['target_1w', 'target_1m', 'target_3m', 'target_6m', 'target_12m']:
        df[col] = np.nan
    
    # Compute targets only for ZL
    zl_mask = df['symbol'] == 'ZL'
    if zl_mask.any():
        zl_df = df[zl_mask].copy()
        
        # LOCKED: Targets are future PRICE LEVELS
        zl_df['target_1w'] = zl_df['price'].shift(-5)    # 5 trading days
        zl_df['target_1m'] = zl_df['price'].shift(-21)   # 21 trading days
        zl_df['target_3m'] = zl_df['price'].shift(-63)   # 63 trading days
        zl_df['target_6m'] = zl_df['price'].shift(-126)  # 126 trading days
        zl_df['target_12m'] = zl_df['price'].shift(-252) # 252 trading days
        
        # Update main df with ZL targets
        for col in ['target_1w', 'target_1m', 'target_3m', 'target_6m', 'target_12m']:
            df.loc[zl_mask, col] = zl_df[col].values
        
        print(f"  -> Targets computed for {zl_mask.sum():,} ZL rows")
        
        # Validate target completeness for training splits
        train_mask = (zl_df['date'] >= pd.Timestamp('2010-01-01').date()) & (zl_df['date'] <= pd.Timestamp('2018-12-31').date())
        train_zl = zl_df[train_mask]
        missing_targets = train_zl['target_1m'].isna().sum()
        if missing_targets > 21:  # Allow ~1 month of edge NaNs
            print(f"  WARNING: {missing_targets} missing targets in training period (2010-2018)")
    else:
        raise RuntimeError("[build_daily_ml_matrix] No ZL data found. ZL is required for training.")
    
    # =========================================================================
    # 10) FINAL CLEANUP
    # =========================================================================
    print("[build_daily_ml_matrix] Final cleanup...")
    
    # Drop rows with null price
    before = len(df)
    df = df[df['price'].notna()]
    if before != len(df):
        print(f"  -> Dropped {before - len(df)} rows with null price")
    
    # Ensure proper types for BigQuery
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['symbol'] = df['symbol'].astype(str)
    df['regime'] = df['regime'].astype(str)
    
    # =========================================================================
    # 11) OUTPUT VALIDATION
    # =========================================================================
    required_output = [
        'date', 'symbol', 'regime', 'regime_weight',
        'open', 'high', 'low', 'close', 'volume',
        'price',
        'lag_1d', 'lag_5d', 'lag_21d',
        'ret_1d', 'ret_5d', 'ret_21d',
        'vol_21d', 'gk_vol_21d',
        'sma_20', 'sma_50', 'sma_200',
        'ema_20', 'ema_50',
        'dist_sma_20', 'dist_sma_50', 'dist_ema_20',
        'fx_brl_mom_21d', 'fx_brl_mom_63d', 'fx_brl_mom_252d',
        'fx_brl_vol_21d', 'fx_brl_vol_63d',
        'fx_dxy_mom_21d', 'fx_dxy_mom_63d', 'fx_dxy_mom_252d',
        'fx_dxy_vol_21d', 'fx_dxy_vol_63d',
        'fred_dff', 'fred_dfedtaru', 'fred_dfedtarl', 'fred_effr', 'fred_sofr',
        'fred_dgs3mo', 'fred_dgs1', 'fred_dgs2', 'fred_dgs5', 'fred_dgs10', 'fred_dgs30',
        'fred_t10y2y', 'fred_t10y3m',
        'fred_vixcls', 'fred_nfci', 'fred_nfcileverage', 'fred_baaffm', 'fred_bamlh0a0hym2',
        'corr_zl_brl_30d', 'corr_zl_brl_60d', 'corr_zl_brl_90d',
        'corr_zl_dxy_30d', 'corr_zl_dxy_60d', 'corr_zl_dxy_90d',
        'terms_of_trade_zl_brl',
        'boho_spread',
        'target_1w', 'target_1m', 'target_3m', 'target_6m', 'target_12m'
    ]
    assert_columns(df, required_output, "daily_ml_matrix output")
    
    print(f"\n[build_daily_ml_matrix] COMPLETE: {len(df):,} rows, {len(df.columns)} columns")
    return df


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    out_path = "TrainingData/exports/daily_ml_matrix.parquet"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    df = build_daily_ml_matrix()
    df.to_parquet(out_path, index=False)
    
    print(f"\n✅ Saved to {out_path}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Symbols: {df['symbol'].nunique()}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
