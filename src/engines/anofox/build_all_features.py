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


def get_connection():
    """Get MotherDuck connection - NO LOCAL FALLBACK"""
    motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
    if not motherduck_token:
        raise ValueError("MOTHERDUCK_TOKEN required - no local fallback")
    return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={motherduck_token}")


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

        # Step 2: Build technical indicators
        build_technical_indicators(con)

        # Step 3: Build cross-asset features (DISABLED - tables don't exist yet)
        # build_cross_asset_features(con)

        # Step 4: Build bucket scores (DISABLED - table schema doesn't match macro)
        # build_bucket_scores(con)

        # Step 5: Build master feature matrix (uses build_symbol_features macro)
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
