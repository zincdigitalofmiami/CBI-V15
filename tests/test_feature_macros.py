"""Tests for feature engineering SQL macros.

These tests are intentionally lightweight and only verify that:
- Macros compile and execute against minimal stub raw tables.
- Key technical indicator macros return data.
"""

from pathlib import Path

import duckdb
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
MACROS_DIR = ROOT_DIR / "database" / "macros"


def test_load_macros() -> None:
    """Test that all macros load without errors."""
    print("Testing macro loading...", flush=True)

    con = duckdb.connect(":memory:")

    # Create schemas first
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.execute("CREATE SCHEMA IF NOT EXISTS staging")
    con.execute("CREATE SCHEMA IF NOT EXISTS features")

    # Create dummy raw data table for prices
    con.execute(
        """
        CREATE TABLE raw.databento_ohlcv_daily (
            as_of_date DATE,
            symbol TEXT,
            close DOUBLE,
            high DOUBLE,
            low DOUBLE,
            volume DOUBLE
        )
        """
    )

    # Insert sample price data
    con.execute(
        """
        INSERT INTO raw.databento_ohlcv_daily VALUES
        ('2024-01-01', 'ZL', 45.50, 46.00, 45.00, 100000),
        ('2024-01-02', 'ZL', 45.75, 46.25, 45.50, 110000),
        ('2024-01-03', 'ZL', 45.60, 46.00, 45.40, 105000),
        ('2024-01-04', 'ZL', 45.80, 46.10, 45.60, 108000),
        ('2024-01-05', 'ZL', 45.90, 46.20, 45.70, 112000)
        """
    )

    # Stub dependent raw tables required for macro binding
    con.execute(
        """
        CREATE TABLE raw.cftc_cot_disaggregated (
            report_date DATE,
            symbol TEXT,
            managed_money_net_pct_oi DOUBLE,
            prod_merc_net_pct_oi DOUBLE
        )
        """
    )
    con.execute(
        """
        INSERT INTO raw.cftc_cot_disaggregated VALUES
        ('2024-01-01', 'ZL', 5.0, -2.0),
        ('2024-01-02', 'ZS', -3.0, 1.5),
        ('2024-01-03', 'ZM', 2.5, -1.0),
        ('2024-01-04', 'HG', 1.0, -0.5)
        """
    )

    # EIA biodiesel / biofuels table (no RIN prices here)
    con.execute(
        """
        CREATE TABLE raw.eia_biofuels (
            date DATE,
            series_id TEXT,
            value DOUBLE
        )
        """
    )
    con.execute(
        """
        INSERT INTO raw.eia_biofuels VALUES
        ('2024-01-01', 'biodiesel_production', 10.0)
        """
    )

    # EPA RIN price table (canonical home for RIN prices)
    con.execute(
        """
        CREATE TABLE raw.epa_rin_prices (
            date DATE,
            series_id TEXT,
            value DOUBLE
        )
        """
    )
    con.execute(
        """
        INSERT INTO raw.epa_rin_prices VALUES
        ('2024-01-01', 'rin_d4_price', 1.5),
        ('2024-01-01', 'rin_d6_price', 1.0)
        """
    )

    # FRED observations table (canonical source for all FRED data)
    con.execute(
        """
        CREATE TABLE raw.fred_observations (
            date DATE,
            series_id TEXT,
            value DOUBLE
        )
        """
    )
    con.execute(
        """
        INSERT INTO raw.fred_observations VALUES
        ('2024-01-01', 'DGS10', 4.0),
        ('2024-01-01', 'DGS2', 4.5),
        ('2024-01-01', 'DFEDTARU', 5.5),
        ('2024-01-01', 'VIXCLS', 15.0)
        """
    )

    # News buckets stub (for policy / tariff features)
    con.execute(
        """
        CREATE TABLE raw.scrapecreators_news_buckets (
            date DATE,
            zl_sentiment TEXT,
            is_trump_related BOOLEAN,
            policy_axis TEXT
        )
        """
    )
    con.execute(
        """
        INSERT INTO raw.scrapecreators_news_buckets VALUES
        ('2024-01-01', 'BULLISH_ZL', TRUE, 'TRADE_CHINA'),
        ('2024-01-01', 'BEARISH_ZL', TRUE, 'TRADE_TARIFFS')
        """
    )

    # Load macros
    macro_files = [
        "features.sql",
        "technical_indicators_all_symbols.sql",
        "cross_asset_features.sql",
        "big8_bucket_features.sql",
        "master_feature_matrix.sql",
    ]

    for macro_file in macro_files:
        macro_path = MACROS_DIR / macro_file
        assert macro_path.exists(), f"{macro_file} not found in {MACROS_DIR}"

        print(f"  Loading {macro_file}...", flush=True)
        with open(macro_path, "r") as f:
            try:
                con.execute(f.read())
                print(f"    Loaded: {macro_file}", flush=True)
            except Exception as exc:  # pragma: no cover - debug detail
                pytest.fail(f"{macro_file} failed to load: {exc}")

    print("All macros loaded successfully\n", flush=True)


def test_technical_indicators() -> None:
    """Smoke test a few technical indicator macros."""
    print("Testing technical indicators...", flush=True)

    con = duckdb.connect(":memory:")

    # Create schemas first
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.execute("CREATE SCHEMA IF NOT EXISTS staging")
    con.execute("CREATE SCHEMA IF NOT EXISTS features")

    # Create sample data
    con.execute(
        """
        CREATE TABLE raw.databento_ohlcv_daily AS
        SELECT 
            DATE '2024-01-01' + INTERVAL (i) DAY AS as_of_date,
            'ZL' AS symbol,
            45.0 + (i * 0.1) + (RANDOM() * 0.5) AS close,
            45.5 + (i * 0.1) + (RANDOM() * 0.5) AS high,
            44.5 + (i * 0.1) + (RANDOM() * 0.5) AS low,
            100000 + (RANDOM() * 10000) AS volume
        FROM generate_series(0, 100) AS t(i)
        """
    )

    # Load macros
    with open(MACROS_DIR / "features.sql", "r") as f:
        con.execute(f.read())

    with open(MACROS_DIR / "technical_indicators_all_symbols.sql", "r") as f:
        con.execute(f.read())

    rsi = con.execute("SELECT * FROM calc_rsi('ZL', 14) LIMIT 5").df()
    assert len(rsi) == 5, "RSI macro returned unexpected row count"
    print(f"  RSI macro works: {len(rsi)} rows", flush=True)

    macd = con.execute("SELECT * FROM calc_macd('ZL', 12, 26, 9) LIMIT 5").df()
    assert len(macd) == 5, "MACD macro returned unexpected row count"
    print(f"  MACD macro works: {len(macd)} rows", flush=True)

    boll = con.execute("SELECT * FROM calc_bollinger('ZL', 20, 2) LIMIT 5").df()
    assert len(boll) == 5, "Bollinger macro returned unexpected row count"
    print(f"  Bollinger macro works: {len(boll)} rows", flush=True)

    all_ind = con.execute(
        "SELECT * FROM calc_all_technical_indicators('ZL') LIMIT 5"
    ).df()
    assert not all_ind.empty, "All indicators macro returned no rows"
    print(
        f"  All indicators macro works: {len(all_ind)} rows and {len(all_ind.columns)} columns",
        flush=True,
    )

    print("All technical indicator macros work\n", flush=True)


def test_feature_counts() -> None:
    """Test that we get at least a baseline number of features."""
    print("Testing feature counts...", flush=True)

    con = duckdb.connect(":memory:")

    # Create schemas first
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.execute("CREATE SCHEMA IF NOT EXISTS staging")
    con.execute("CREATE SCHEMA IF NOT EXISTS features")

    # Create sample data for multiple symbols
    con.execute(
        """
        CREATE TABLE raw.databento_ohlcv_daily AS
        SELECT 
            DATE '2024-01-01' + INTERVAL (i) DAY AS as_of_date,
            symbol,
            45.0 + (i * 0.1) + (RANDOM() * 0.5) AS close,
            45.5 + (i * 0.1) + (RANDOM() * 0.5) AS high,
            44.5 + (i * 0.1) + (RANDOM() * 0.5) AS low,
            100000 + (RANDOM() * 10000) AS volume
        FROM generate_series(0, 100) AS t(i)
        CROSS JOIN (VALUES ('ZL'), ('ZS'), ('CL')) AS s(symbol)
        """
    )

    # Load macros
    with open(MACROS_DIR / "features.sql", "r") as f:
        con.execute(f.read())

    with open(MACROS_DIR / "technical_indicators_all_symbols.sql", "r") as f:
        con.execute(f.read())

    # Count features
    result = con.execute(
        "SELECT * FROM calc_all_technical_indicators('ZL') LIMIT 1"
    ).df()
    feature_count = len(result.columns)

    print(f"  Technical indicators: {feature_count} columns")

    expected_features = [
        "as_of_date",
        "symbol",
        "close",
        "lag_close_1d",
        "lag_close_5d",
        "lag_close_21d",
        "log_ret_1d",
        "log_ret_5d",
        "log_ret_21d",
        "sma_5",
        "sma_10",
        "sma_21",
        "sma_50",
        "sma_200",
        "volatility_21d",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_position",
        "bb_width_pct",
        "atr_14",
        "tr_pct",
        "stoch_k",
        "stoch_d",
        "roc_10d",
        "roc_21d",
        "roc_63d",
        "momentum_10d",
        "momentum_21d",
        "volume",
        "avg_volume_21d",
        "volume_ratio",
        "volume_zscore",
        "obv",
    ]

    print(f"  Expected (approx): {len(expected_features)} features")
    assert (
        feature_count >= len(expected_features) - 5
    ), f"Expected at least {len(expected_features) - 5} features, got {feature_count}"

    print("Feature counts validated\n", flush=True)


if __name__ == "__main__":  # pragma: no cover - manual debug helper
    print("=" * 80)
    print("TESTING FEATURE ENGINEERING MACROS")
    print("=" * 80)
    print()

    all_passed = True

    try:
        test_load_macros()
    except Exception as exc:  # pragma: no cover
        all_passed = False
        print(f"test_load_macros failed: {exc}")

    try:
        test_technical_indicators()
    except Exception as exc:  # pragma: no cover
        all_passed = False
        print(f"test_technical_indicators failed: {exc}")

    try:
        test_feature_counts()
    except Exception as exc:  # pragma: no cover
        all_passed = False
        print(f"test_feature_counts failed: {exc}")

    print("=" * 80)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 80)
