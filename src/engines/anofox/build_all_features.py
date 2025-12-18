"""
Build All Features: Populate Feature Tables from SQL Macros

This script:
1. Loads all SQL macros (technical indicators, cross-asset, Big 8 buckets)
2. Executes macros to compute features
3. Populates feature tables in DuckDB/MotherDuck
    4. Builds final daily_ml_matrix_zl table

Run this daily after raw data ingestion.
"""

import os
from datetime import datetime
from pathlib import Path

import duckdb

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")
ROOT_DIR = Path(__file__).resolve().parents[3]
MACROS_DIR = ROOT_DIR / "database" / "macros"


def _load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ[key] = value


def _load_local_env() -> None:
    _load_dotenv_file(ROOT_DIR / ".env")
    _load_dotenv_file(ROOT_DIR / ".env.local")


def _iter_motherduck_tokens():
    candidates = [
        ("MOTHERDUCK_TOKEN", os.getenv("MOTHERDUCK_TOKEN")),
        ("motherduck_storage_MOTHERDUCK_TOKEN", os.getenv("motherduck_storage_MOTHERDUCK_TOKEN")),
        ("MOTHERDUCK_READ_SCALING_TOKEN", os.getenv("MOTHERDUCK_READ_SCALING_TOKEN")),
        (
            "motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN",
            os.getenv("motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN"),
        ),
    ]
    for _, value in candidates:
        if not value:
            continue
        token = value.strip().strip('"').strip("'")
        if token.count(".") != 2:
            continue
        yield token


def get_connection():
    """Get MotherDuck connection - NO LOCAL FALLBACK"""
    _load_local_env()
    for token in _iter_motherduck_tokens():
        try:
            con = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={token}")
            con.execute("SELECT 1").fetchone()
            return con
        except Exception:
            continue
    raise ValueError(
        "MotherDuck token required (set MOTHERDUCK_TOKEN or motherduck_storage_MOTHERDUCK_TOKEN)"
    )


def load_macros(con: duckdb.DuckDBPyConnection):
    """Load all SQL macros into DuckDB session"""
    print("📦 Loading SQL macros...")

    macro_files = [
        "features.sql",  # Original price/return macros
        "technical_indicators_all_symbols.sql",
        "cross_asset_features.sql",
        "big8_bucket_features.sql",
        "master_feature_matrix.sql",
    ]

    for macro_file in macro_files:
        macro_path = MACROS_DIR / macro_file
        if macro_path.exists():
            print(f"  Loading {macro_file}...")
            with open(macro_path, "r") as f:
                sql = f.read()
                # Execute all CREATE MACRO statements
                con.execute(sql)
        else:
            print(f"  ⚠️  {macro_file} not found, skipping")

    print("✅ Macros loaded\n")

def build_staging_ohlcv_daily(con: duckdb.DuckDBPyConnection) -> None:
    """
    Segment raw Databento daily bars into a cleaned staging panel.
    Gap-fills missing weekdays (point-in-time safe; forward-fill only).
    """
    print("🧱 Building staging.ohlcv_daily...")

    con.execute("DELETE FROM staging.ohlcv_daily")
    con.execute(
        """
        INSERT INTO staging.ohlcv_daily (
          symbol, as_of_date, open, high, low, close, volume, open_interest,
          gap_filled, outlier_adjusted, updated_at
        )
        WITH bounds AS (
          SELECT
            r.symbol,
            MIN(r.as_of_date) AS min_date,
            MAX(r.as_of_date) AS max_date
          FROM raw.databento_futures_ohlcv_1d r
          INNER JOIN reference.symbols s
            ON s.symbol = r.symbol
          GROUP BY r.symbol
        ),
        cal AS (
          SELECT
            b.symbol,
            gs.dt::DATE AS as_of_date
          FROM bounds b
          CROSS JOIN LATERAL generate_series(b.min_date, b.max_date, INTERVAL 1 DAY) AS gs(dt)
          WHERE EXTRACT(DOW FROM gs.dt) NOT IN (0, 6)
        ),
        joined AS (
          SELECT
            c.symbol,
            c.as_of_date,
            r.open,
            r.high,
            r.low,
            r.close,
            r.volume,
            r.open_interest,
            CASE WHEN r.close IS NULL THEN TRUE ELSE FALSE END AS gap_filled
          FROM cal c
          LEFT JOIN raw.databento_futures_ohlcv_1d r
            ON r.symbol = c.symbol
           AND r.as_of_date = c.as_of_date
        ),
        filled AS (
          SELECT
            symbol,
            as_of_date,
            last_value(close IGNORE NULLS) OVER w AS close_filled,
            last_value(open_interest IGNORE NULLS) OVER w AS oi_filled,
            open,
            high,
            low,
            volume,
            open_interest,
            gap_filled
          FROM joined
          WINDOW w AS (
            PARTITION BY symbol
            ORDER BY as_of_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
          )
        )
        SELECT
          symbol,
          as_of_date,
          COALESCE(open, close_filled) AS open,
          COALESCE(high, close_filled) AS high,
          COALESCE(low, close_filled) AS low,
          close_filled AS close,
          COALESCE(volume, 0) AS volume,
          COALESCE(open_interest, oi_filled) AS open_interest,
          gap_filled,
          FALSE AS outlier_adjusted,
          CURRENT_TIMESTAMP AS updated_at
        FROM filled
        WHERE close_filled IS NOT NULL
        """
    )

    row_count = con.execute("SELECT COUNT(*) FROM staging.ohlcv_daily").fetchone()[0]
    print(f"✅ staging.ohlcv_daily built: {row_count:,} rows\n")


def log_data_quality_ohlcv_daily(con: duckdb.DuckDBPyConnection) -> None:
    """
    Log basic data-quality metrics for staging.ohlcv_daily.
    Keeps this lightweight and deterministic; no look-ahead or model logic.
    """
    import uuid

    report_id = str(uuid.uuid4())
    report_date = con.execute("SELECT CURRENT_DATE").fetchone()[0]

    total_rows = con.execute("SELECT COUNT(*) FROM staging.ohlcv_daily").fetchone()[0]
    null_count = con.execute(
        "SELECT COUNT(*) FROM staging.ohlcv_daily WHERE close IS NULL"
    ).fetchone()[0]
    dup_count = con.execute(
        """
        SELECT COUNT(*) - COUNT(DISTINCT symbol || '|' || CAST(as_of_date AS VARCHAR))
        FROM staging.ohlcv_daily
        """
    ).fetchone()[0]
    max_date = con.execute("SELECT MAX(as_of_date) FROM staging.ohlcv_daily").fetchone()[0]
    days_stale = con.execute(
        "SELECT DATE_DIFF('day', ?, CURRENT_DATE)", [max_date]
    ).fetchone()[0]
    gap_count = con.execute(
        "SELECT COUNT(*) FROM staging.ohlcv_daily WHERE gap_filled"
    ).fetchone()[0]
    max_gap_days = con.execute(
        """
        WITH gaps AS (
          SELECT
            r.symbol,
            r.as_of_date,
            LEAD(r.as_of_date) OVER (PARTITION BY r.symbol ORDER BY r.as_of_date) - r.as_of_date AS date_gap
          FROM raw.databento_futures_ohlcv_1d r
          INNER JOIN reference.symbols s
            ON s.symbol = r.symbol
        )
        SELECT COALESCE(MAX(date_gap), 0) FROM gaps
        """
    ).fetchone()[0]

    # Use ZL close distribution as a representative sanity check (close-only).
    mean_val, std_val, min_val, max_val = con.execute(
        """
        SELECT
          AVG(CAST(close AS DOUBLE)) AS mean_val,
          STDDEV_SAMP(CAST(close AS DOUBLE)) AS std_val,
          MIN(CAST(close AS DOUBLE)) AS min_val,
          MAX(CAST(close AS DOUBLE)) AS max_val
        FROM staging.ohlcv_daily
        WHERE symbol = 'ZL'
        """
    ).fetchone()

    null_ratio = (float(null_count) / float(total_rows)) if total_rows else None
    status = "PASSED"
    if total_rows == 0 or max_date is None:
        status = "FAILED"
    elif days_stale is not None and days_stale > 3:
        status = "WARNING"
    elif null_ratio is not None and null_ratio > 0.01:
        status = "WARNING"
    elif dup_count and dup_count > 0:
        status = "FAILED"

    try:
        con.execute(
            """
            INSERT INTO ops.data_quality_log (
              report_id,
              report_date,
              schema_name,
              table_name,
              total_rows,
              null_count,
              null_ratio,
              duplicate_count,
              max_date,
              days_stale,
              gap_count,
              max_gap_days,
              mean_val,
              std_val,
              min_val,
              max_val,
              status,
              issues
            ) VALUES (?, ?, 'staging', 'ohlcv_daily', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            [
                report_id,
                report_date,
                int(total_rows),
                int(null_count),
                float(null_ratio) if null_ratio is not None else None,
                int(dup_count) if dup_count is not None else 0,
                max_date,
                int(days_stale) if days_stale is not None else None,
                int(gap_count),
                int(max_gap_days) if max_gap_days is not None else 0,
                float(mean_val) if mean_val is not None else None,
                float(std_val) if std_val is not None else None,
                float(min_val) if min_val is not None else None,
                float(max_val) if max_val is not None else None,
                status,
            ],
        )
        print("🧾 Logged ops.data_quality_log for staging.ohlcv_daily\n")
    except Exception as e:
        print(f"⚠️  Could not log ops.data_quality_log: {e}\n")

def build_staging_market_daily(con: duckdb.DuckDBPyConnection) -> None:
    """
    Segment cleaned OHLCV + select FRED series into a wide daily panel.
    """
    print("🧱 Building staging.market_daily...")

    con.execute("DELETE FROM staging.market_daily")
    con.execute(
        """
        WITH base AS (
          SELECT
            as_of_date AS date,
            MAX(CASE WHEN symbol = 'ZL' THEN close END) AS zl_close,
            MAX(CASE WHEN symbol = 'ZL' THEN volume END) AS zl_volume,
            MAX(CASE WHEN symbol = 'ZL' THEN open_interest END) AS zl_open_interest,
            MAX(CASE WHEN symbol = 'ZS' THEN close END) AS zs_close,
            MAX(CASE WHEN symbol = 'ZM' THEN close END) AS zm_close,
            MAX(CASE WHEN symbol = 'CL' THEN close END) AS cl_close,
            MAX(CASE WHEN symbol = 'HO' THEN close END) AS ho_close,
            MAX(CASE WHEN symbol = 'RB' THEN close END) AS rb_close,
            MAX(CASE WHEN symbol = 'DX' THEN close END) AS dx_close,
            MAX(CASE WHEN symbol = 'FCPO' THEN close END) AS fcpo_close,
            MAX(CASE WHEN symbol = 'HG' THEN close END) AS hg_close,
            MAX(CASE WHEN symbol = 'GC' THEN close END) AS gc_close
          FROM staging.ohlcv_daily
          GROUP BY as_of_date
        ),
        vix AS (
          SELECT date, value AS vix_close
          FROM raw.fred_economic
          WHERE series_id = 'VIXCLS'
        ),
        with_returns AS (
          SELECT
            b.*,
            LN(b.zl_close / NULLIF(LAG(b.zl_close) OVER (ORDER BY b.date), 0)) AS zl_return
          FROM base b
        )
        INSERT INTO staging.market_daily (
          date,
          zl_close,
          zl_volume,
          zl_open_interest,
          zl_return,
          zs_close,
          zm_close,
          cl_close,
          ho_close,
          rb_close,
          dx_close,
          fcpo_close,
          hg_close,
          gc_close,
          vix_close,
          updated_at
        )
        SELECT
          r.date,
          r.zl_close,
          r.zl_volume,
          r.zl_open_interest,
          r.zl_return,
          r.zs_close,
          r.zm_close,
          r.cl_close,
          r.ho_close,
          r.rb_close,
          r.dx_close,
          r.fcpo_close,
          r.hg_close,
          r.gc_close,
          v.vix_close,
          CURRENT_TIMESTAMP AS updated_at
        FROM with_returns r
        LEFT JOIN vix v
          ON v.date = r.date
        """
    )

    row_count = con.execute("SELECT COUNT(*) FROM staging.market_daily").fetchone()[0]
    print(f"✅ staging.market_daily built: {row_count:,} rows\n")


def build_staging_daily_returns(con: duckdb.DuckDBPyConnection) -> None:
    """
    Build per-symbol returns and rolling volatility on the daily grain.
    """
    print("🧱 Building staging.daily_returns...")

    con.execute("DELETE FROM staging.daily_returns")
    con.execute(
        """
        WITH base AS (
          SELECT
            symbol,
            as_of_date AS date,
            close,
            volume,
            open_interest,
            LAG(close) OVER w AS prev_close,
            LAG(open_interest) OVER w AS prev_open_interest
          FROM staging.ohlcv_daily
          WINDOW w AS (PARTITION BY symbol ORDER BY as_of_date)
        ),
        returns AS (
          SELECT
            symbol,
            date,
            close,
            LN(close / NULLIF(prev_close, 0)) AS log_return,
            (close / NULLIF(prev_close, 0)) - 1 AS simple_return,
            (open_interest - prev_open_interest) AS oi_change,
            (volume - AVG(volume) OVER vw) / NULLIF(STDDEV(volume) OVER vw, 0) AS volume_zscore
          FROM base
          WINDOW vw AS (PARTITION BY symbol ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW)
        ),
        with_vol AS (
          SELECT
            *,
            STDDEV(log_return) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS volatility_5d,
            STDDEV(log_return) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 20 PRECEDING AND CURRENT ROW) AS volatility_21d,
            STDDEV(log_return) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 62 PRECEDING AND CURRENT ROW) AS volatility_63d
          FROM returns
        )
        INSERT INTO staging.daily_returns (
          symbol,
          date,
          close,
          log_return,
          simple_return,
          volatility_5d,
          volatility_21d,
          volatility_63d,
          volume_zscore,
          oi_change,
          updated_at
        )
        SELECT
          symbol,
          date,
          close,
          log_return,
          simple_return,
          volatility_5d,
          volatility_21d,
          volatility_63d,
          volume_zscore,
          oi_change,
          CURRENT_TIMESTAMP AS updated_at
        FROM with_vol
        """
    )

    row_count = con.execute("SELECT COUNT(*) FROM staging.daily_returns").fetchone()[0]
    print(f"✅ staging.daily_returns built: {row_count:,} rows\n")


def build_staging_fred_macro_clean(con: duckdb.DuckDBPyConnection) -> None:
    """
    Segment + forward-fill key FRED series into a wide macro panel.
    This is point-in-time safe (only uses past observations).
    """
    print("🧱 Building staging.fred_macro_clean...")

    con.execute("DELETE FROM staging.fred_macro_clean")

    con.execute(
        """
        WITH bounds AS (
          SELECT MIN(date) AS min_date, MAX(date) AS max_date
          FROM raw.fred_economic
          WHERE series_id IN (
            'DFF','DGS10','DGS2','T10Y2Y',
            'DTWEXBGS','DEXBZUS','DEXCHUS','DEXMXUS',
            'VIXCLS','STLFSI4','NFCI',
            'CPIAUCSL','PCEPI',
            'DCOILWTICO'
          )
        ),
        cal AS (
          SELECT * FROM generate_series(
            (SELECT min_date FROM bounds),
            (SELECT max_date FROM bounds),
            INTERVAL 1 DAY
          ) AS t(date)
        ),
        daily_series AS (
          SELECT date,
            MAX(CASE WHEN series_id='DFF' THEN value END) AS fed_funds_raw,
            MAX(CASE WHEN series_id='DGS10' THEN value END) AS treasury_10y_raw,
            MAX(CASE WHEN series_id='DGS2' THEN value END) AS treasury_2y_raw,
            MAX(CASE WHEN series_id='T10Y2Y' THEN value END) AS yield_curve_10y2y_raw,
            MAX(CASE WHEN series_id='DTWEXBGS' THEN value END) AS dxy_raw,
            MAX(CASE WHEN series_id='DEXBZUS' THEN value END) AS brl_usd_raw,
            MAX(CASE WHEN series_id='DEXCHUS' THEN value END) AS cny_usd_raw,
            MAX(CASE WHEN series_id='DEXMXUS' THEN value END) AS mxn_usd_raw,
            MAX(CASE WHEN series_id='VIXCLS' THEN value END) AS vix_raw,
            MAX(CASE WHEN series_id='STLFSI4' THEN value END) AS stlfsi4_raw,
            MAX(CASE WHEN series_id='NFCI' THEN value END) AS nfci_raw,
            MAX(CASE WHEN series_id='DCOILWTICO' THEN value END) AS wti_crude_raw
          FROM raw.fred_economic
          WHERE series_id IN (
            'DFF','DGS10','DGS2','T10Y2Y',
            'DTWEXBGS','DEXBZUS','DEXCHUS','DEXMXUS',
            'VIXCLS','STLFSI4','NFCI',
            'DCOILWTICO'
          )
          GROUP BY date
        ),
        cpi_yoy AS (
          SELECT
            date,
            CASE
              WHEN LAG(value, 12) OVER (ORDER BY date) IS NULL THEN NULL
              ELSE (value / LAG(value, 12) OVER (ORDER BY date)) - 1
            END AS cpi_yoy_raw
          FROM raw.fred_economic
          WHERE series_id = 'CPIAUCSL'
        ),
        pce_yoy AS (
          SELECT
            date,
            CASE
              WHEN LAG(value, 12) OVER (ORDER BY date) IS NULL THEN NULL
              ELSE (value / LAG(value, 12) OVER (ORDER BY date)) - 1
            END AS pce_yoy_raw
          FROM raw.fred_economic
          WHERE series_id = 'PCEPI'
        ),
        base AS (
          SELECT
            c.date,
            d.fed_funds_raw,
            d.treasury_10y_raw,
            d.treasury_2y_raw,
            d.yield_curve_10y2y_raw,
            d.dxy_raw,
            d.brl_usd_raw,
            d.cny_usd_raw,
            d.mxn_usd_raw,
            d.vix_raw,
            d.stlfsi4_raw,
            d.nfci_raw,
            d.wti_crude_raw,
            cy.cpi_yoy_raw,
            py.pce_yoy_raw
          FROM cal c
          LEFT JOIN daily_series d ON d.date = c.date
          LEFT JOIN cpi_yoy cy ON cy.date = c.date
          LEFT JOIN pce_yoy py ON py.date = c.date
        ),
        ff AS (
          SELECT
            date,
            last_value(fed_funds_raw IGNORE NULLS) OVER w AS fed_funds,
            last_value(treasury_10y_raw IGNORE NULLS) OVER w AS treasury_10y,
            last_value(treasury_2y_raw IGNORE NULLS) OVER w AS treasury_2y,
            last_value(yield_curve_10y2y_raw IGNORE NULLS) OVER w AS yield_curve_10y2y,
            last_value(dxy_raw IGNORE NULLS) OVER w AS dxy,
            last_value(brl_usd_raw IGNORE NULLS) OVER w AS brl_usd,
            last_value(cny_usd_raw IGNORE NULLS) OVER w AS cny_usd,
            last_value(mxn_usd_raw IGNORE NULLS) OVER w AS mxn_usd,
            last_value(vix_raw IGNORE NULLS) OVER w AS vix,
            last_value(stlfsi4_raw IGNORE NULLS) OVER w AS stlfsi4,
            last_value(nfci_raw IGNORE NULLS) OVER w AS nfci,
            last_value(cpi_yoy_raw IGNORE NULLS) OVER w AS cpi_yoy,
            last_value(pce_yoy_raw IGNORE NULLS) OVER w AS pce_yoy,
            last_value(wti_crude_raw IGNORE NULLS) OVER w AS wti_crude
          FROM base
          WINDOW w AS (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        )
        INSERT INTO staging.fred_macro_clean (
          date, fed_funds, treasury_10y, treasury_2y, yield_curve_10y2y,
          dxy, brl_usd, cny_usd, mxn_usd,
          vix, stlfsi4, nfci,
          cpi_yoy, pce_yoy,
          wti_crude,
          updated_at
        )
        SELECT
          date,
          fed_funds,
          treasury_10y,
          treasury_2y,
          yield_curve_10y2y,
          dxy,
          brl_usd,
          cny_usd,
          mxn_usd,
          vix,
          stlfsi4,
          nfci,
          cpi_yoy,
          pce_yoy,
          wti_crude,
          CURRENT_TIMESTAMP AS updated_at
        FROM ff
        """
    )

    row_count = con.execute("SELECT COUNT(*) FROM staging.fred_macro_clean").fetchone()[0]
    print(f"✅ staging.fred_macro_clean built: {row_count:,} rows\n")


def build_technical_indicators(con: duckdb.DuckDBPyConnection):
    """Build technical indicators for all symbols"""
    print("🔧 Building technical indicators for all 33 futures symbols...")

    symbols = [
        # Agricultural/Softs (11) - VERIFIED CME ONLY
        "ZL",  # Soybean Oil (PRIMARY - RIN-driven)
        "ZS",  # Soybeans
        "ZM",  # Soybean Meal
        "ZC",  # Corn
        "ZW",  # Wheat
        "ZO",  # Oats
        "ZR",  # Rough Rice (VERIFIED CME)
        "HE",  # Lean Hogs
        "LE",  # Live Cattle (INFLATION HEDGE)
        "GF",  # Feeder Cattle
        "FCPO",  # Crude Palm Oil (Bursa - CRITICAL)
        # Energy (4) - HO includes ULSD
        "CL",  # WTI Crude Oil
        "HO",  # Heating Oil / ULSD (no separate UL symbol)
        "RB",  # RBOB Gasoline
        "NG",  # Natural Gas
        # Metals (5) - CME/COMEX/NYMEX only
        "HG",  # Copper (CHINA GREEN INFRASTRUCTURE PROXY - 2022 structural break)
        "GC",  # Gold
        "SI",  # Silver
        "PL",  # Platinum
        "PA",  # Palladium
        # Treasuries (3) - ZN is 10Y (TY is floor symbol)
        "ZF",  # 5-Year Treasury Note
        "ZN",  # 10-Year Treasury Note (use ZN, not TY)
        "ZB",  # 30-Year Treasury Bond
        # FX Futures (10)
        "6E",  # Euro FX
        "6J",  # Japanese Yen
        "6B",  # British Pound
        "6C",  # Canadian Dollar
        "6A",  # Australian Dollar
        "6N",  # New Zealand Dollar
        "6M",  # Mexican Peso
        "6L",  # Brazilian Real
        "6S",  # Swiss Franc
        "DX",  # U.S. Dollar Index
    ]

    # NOTE: Removed symbols (not available via CME/Databento):
    # - OJ (Orange Juice) - trades on ICE U.S., not CME
    # - UL (ULSD) - no separate symbol, use HO
    # - AL (Aluminum) - trades on LME, not CME
    # - TY (10Y Treasury) - use ZN instead (same contract)

    for symbol in symbols:
        print(f"  Processing {symbol}...")
        try:
            con.execute(
                f"""
                INSERT OR REPLACE INTO features.technical_indicators_all_symbols
                SELECT 
                    *,
                    CURRENT_TIMESTAMP as updated_at
                FROM calc_all_technical_indicators('{symbol}')
            """
            )
        except Exception as e:
            print(f"    ⚠️  Error: {e}")

    row_count = con.execute(
        "SELECT COUNT(*) FROM features.technical_indicators_all_symbols"
    ).fetchone()[0]
    print(
        f"✅ Technical indicators built: {row_count:,} rows ({len(symbols)} symbols)\n"
    )


def build_cross_asset_features(con: duckdb.DuckDBPyConnection):
    """Build cross-asset correlations and spreads"""
    print("🔗 Building cross-asset features...")

    # Correlation matrix
    print("  Computing correlation matrix...")
    con.execute(
        """
        INSERT OR REPLACE INTO features.cross_asset_correlations
        SELECT * FROM calc_correlation_matrix(60)
    """
    )

    # Fundamental spreads
    print("  Computing fundamental spreads...")
    con.execute(
        """
        INSERT OR REPLACE INTO features.fundamental_spreads
        SELECT * FROM calc_fundamental_spreads()
    """
    )

    corr_count = con.execute(
        "SELECT COUNT(*) FROM features.cross_asset_correlations"
    ).fetchone()[0]
    spread_count = con.execute(
        "SELECT COUNT(*) FROM features.fundamental_spreads"
    ).fetchone()[0]
    print(
        f"✅ Cross-asset features built: {corr_count:,} correlations, {spread_count:,} spreads\n"
    )


def build_bucket_scores(con: duckdb.DuckDBPyConnection):
    """Build Big 8 bucket scores"""
    print("📊 Building Big 8 bucket scores...")

    con.execute(
        """
        INSERT OR REPLACE INTO features.bucket_scores
        SELECT 
            *,
            CURRENT_TIMESTAMP as updated_at
        FROM calc_all_bucket_scores()
    """
    )

    row_count = con.execute("SELECT COUNT(*) FROM features.bucket_scores").fetchone()[0]
    print(f"✅ Bucket scores built: {row_count:,} rows\n")


def build_master_feature_matrix(con: duckdb.DuckDBPyConnection):
    """Build final daily_ml_matrix_zl table"""
    print("🎯 Building master feature matrix...")

    # For now, just build for ZL (primary symbol)
    # Later can expand to all symbols

    # Delete existing ZL data then insert fresh
    # (Can't use INSERT OR REPLACE because table has no PRIMARY KEY)
    con.execute("DELETE FROM features.daily_ml_matrix_zl WHERE symbol = 'ZL'")
    con.execute(
        """
        INSERT INTO features.daily_ml_matrix_zl
        SELECT 
            *,
            CURRENT_TIMESTAMP as updated_at
        FROM build_symbol_features('ZL')
    """
    )

    row_count = con.execute(
        "SELECT COUNT(*) FROM features.daily_ml_matrix_zl"
    ).fetchone()[0]
    feature_count = con.execute(
        """
        SELECT COUNT(*) 
        FROM information_schema.columns 
        WHERE table_schema = 'features' 
        AND table_name = 'daily_ml_matrix_zl'
    """
    ).fetchone()[0]

    print(
        f"✅ Master feature matrix built: {row_count:,} rows × {feature_count} features\n"
    )


def main():
    """Main build pipeline"""
    print("=" * 80)
    print("BUILDING ALL FEATURES FOR 30+ SYMBOLS")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    con = get_connection()

    try:
        # Step 1: Load macros
        load_macros(con)

        # Step 2: Segment raw data into staging panels
        build_staging_ohlcv_daily(con)
        log_data_quality_ohlcv_daily(con)
        build_staging_market_daily(con)
        build_staging_daily_returns(con)
        build_staging_fred_macro_clean(con)

        # Step 3: Build technical indicators
        build_technical_indicators(con)

        # Step 4: Build cross-asset features (DISABLED - tables don't exist yet)
        # build_cross_asset_features(con)

        # Step 5: Build bucket scores (DISABLED - table schema doesn't match macro)
        # build_bucket_scores(con)

        # Step 6: Build master feature matrix (uses build_symbol_features macro)
        build_master_feature_matrix(con)

        print("=" * 80)
        print("✅ ALL FEATURES BUILT SUCCESSFULLY")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise

    finally:
        con.close()


if __name__ == "__main__":
    main()
