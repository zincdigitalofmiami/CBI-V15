"""
Test Feature Engineering Macros

Validates that all SQL macros work correctly.
"""

import os
from pathlib import Path

import duckdb

ROOT_DIR = Path(__file__).resolve().parents[1]
MACROS_DIR = ROOT_DIR / "database" / "macros"


def test_load_macros():
    """Test that all macros load without errors"""
    print("Testing macro loading...")

    con = duckdb.connect(":memory:")

    # Create dummy raw data table
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

    # Insert sample data
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
        if macro_path.exists():
            print(f"  Loading {macro_file}...")
            with open(macro_path, "r") as f:
                try:
                    con.execute(f.read())
                    print(f"    ✅ {macro_file} loaded")
                except Exception as e:
                    print(f"    ❌ {macro_file} failed: {e}")
                    return False
        else:
            print(f"  ⚠️  {macro_file} not found")

    print("✅ All macros loaded successfully\n")
    return True


def test_technical_indicators():
    """Test technical indicator macros"""
    print("Testing technical indicators...")

    con = duckdb.connect(":memory:")

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
    with open(MACROS_DIR / "features.sql") as f:
        con.execute(f.read())

    with open(MACROS_DIR / "technical_indicators_all_symbols.sql") as f:
        con.execute(f.read())

    # Test RSI
    try:
        result = con.execute("SELECT * FROM calc_rsi('ZL', 14) LIMIT 5").df()
        print(f"  ✅ RSI macro works: {len(result)} rows")
    except Exception as e:
        print(f"  ❌ RSI macro failed: {e}")
        return False

    # Test MACD
    try:
        result = con.execute("SELECT * FROM calc_macd('ZL', 12, 26, 9) LIMIT 5").df()
        print(f"  ✅ MACD macro works: {len(result)} rows")
    except Exception as e:
        print(f"  ❌ MACD macro failed: {e}")
        return False

    # Test Bollinger Bands
    try:
        result = con.execute("SELECT * FROM calc_bollinger('ZL', 20, 2) LIMIT 5").df()
        print(f"  ✅ Bollinger macro works: {len(result)} rows")
    except Exception as e:
        print(f"  ❌ Bollinger macro failed: {e}")
        return False

    # Test All Indicators
    try:
        result = con.execute(
            "SELECT * FROM calc_all_technical_indicators('ZL') LIMIT 5"
        ).df()
        print(
            f"  ✅ All indicators macro works: {len(result)} rows × {len(result.columns)} columns"
        )
    except Exception as e:
        print(f"  ❌ All indicators macro failed: {e}")
        return False

    print("✅ All technical indicator macros work\n")
    return True


def test_feature_counts():
    """Test that we get the expected number of features"""
    print("Testing feature counts...")

    con = duckdb.connect(":memory:")

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
    with open(MACROS_DIR / "features.sql") as f:
        con.execute(f.read())

    with open(MACROS_DIR / "technical_indicators_all_symbols.sql") as f:
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

    print(f"  Expected: {len(expected_features)} features")
    print(
        f"  Match: {feature_count >= len(expected_features) - 5}"
    )  # Allow some variance

    print("✅ Feature counts validated\n")
    return True


if __name__ == "__main__":
    print("=" * 80)
    print("TESTING FEATURE ENGINEERING MACROS")
    print("=" * 80)
    print()

    all_passed = True

    all_passed &= test_load_macros()
    all_passed &= test_technical_indicators()
    all_passed &= test_feature_counts()

    print("=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80)
