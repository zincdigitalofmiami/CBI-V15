#!/usr/bin/env python3
"""
Databento Options Daily OHLCV Collection
Collects daily options bars for all futures with listed options.
Target: raw.databento_options_ohlcv_1d

Options symbology: Databento uses instrument IDs for specific contracts.
For options chains, we request data with parent="underlying_symbol" parameter.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database config
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")

# GLBX.MDP3 dataset
DATASET = "GLBX.MDP3"
DATASET_START = "2010-06-06"

# Symbols with active options markets (prioritize soy complex + energy + metals)
OPTIONS_SYMBOLS = [
    # Soy complex (CRITICAL)
    "ZL",  # Soybean Oil options
    "ZS",  # Soybean options
    "ZM",  # Soybean Meal options
    "ZC",  # Corn options
    "ZW",  # Wheat options
    # Energy (HIGH VOLUME)
    "CL",  # Crude Oil options
    "NG",  # Natural Gas options
    "HO",  # Heating Oil options
    "RB",  # RBOB Gasoline options
    # Metals
    "GC",  # Gold options
    "SI",  # Silver options
    "HG",  # Copper options
    # Rates (volatility/macro)
    "ZN",  # 10-Year Note options
    "ZB",  # 30-Year Bond options
    "ZF",  # 5-Year Note options
    # Equity indices
    "ES",  # S&P 500 options
    "NQ",  # Nasdaq 100 options
    # FX
    "6E",  # Euro options
    "6J",  # Yen options
]


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
    _load_dotenv_file(project_root / ".env")
    _load_dotenv_file(project_root / ".env.local")


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
    """Get MotherDuck connection"""
    _load_local_env()
    db_name = os.getenv("MOTHERDUCK_DB", "cbi_v15")
    last_error: Exception | None = None
    for token in _iter_motherduck_tokens():
        try:
            con = duckdb.connect(f"md:{db_name}?motherduck_token={token}")
            con.execute("SELECT 1").fetchone()
            return con
        except Exception as e:
            last_error = e
    raise RuntimeError(
        f"MotherDuck token not set/invalid (set MOTHERDUCK_TOKEN or motherduck_storage_MOTHERDUCK_TOKEN): {last_error}"
    )


def get_databento_client():
    """Initialize Databento client"""
    try:
        import databento as db

        _load_local_env()
        api_key = os.getenv("DATABENTO_API_KEY")
        if not api_key:
            raise ValueError("DATABENTO_API_KEY environment variable not set")
        client = db.Historical(key=api_key)
        logger.info("Initialized Databento client")
        return client
    except ImportError:
        logger.error("databento package not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Databento client: {e}")
        return None


def collect_options_for_symbol(
    client, symbol: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Collect options chain data for a specific underlying symbol.

    NOTE: Databento options require specific instrument IDs or parent filtering.
    This is a simplified version - full implementation requires:
    1. Query symbology API to get options instrument IDs for the underlying
    2. Request each options contract individually or use parent filter
    3. Parse options metadata (strike, expiration, type)
    """
    logger.info(f"Collecting options for {symbol} from {start_date} to {end_date}")

    try:
        # Method 1: Try to get options using parent parameter (if supported)
        # This may not work on all Databento subscriptions
        data = client.timeseries.get_range(
            dataset=DATASET,
            symbols=[f"{symbol}.OPT"],  # Options suffix (check Databento docs)
            schema="ohlcv-1d",
            start=start_date,
            end=end_date,
            stype_in="parent",  # Filter by parent underlying
        )

        df = data.to_df().reset_index()

        if df.empty:
            logger.warning(f"No options data for {symbol}")
            return pd.DataFrame()

        # Preserve Databento's contract identifier if present
        contract_col = None
        for c in ["symbol", "instrument_id", "instrument", "raw_symbol"]:
            if c in df.columns:
                contract_col = c
                break

        if contract_col is None:
            logger.warning(f"No contract identifier column returned for {symbol}")
            return pd.DataFrame()

        df_out = pd.DataFrame()
        df_out["symbol"] = symbol  # underlying root
        df_out["contract_symbol"] = df[contract_col].astype(str)
        df_out["as_of_date"] = pd.to_datetime(df.get("ts_event"), errors="coerce").dt.date

        def pick(colnames):
            for c in colnames:
                if c in df.columns:
                    return df[c]
            return None

        strike = pick(["strike_price", "strike_px", "strike", "strike_price_px"])
        exp = pick(["expiration_date", "expiration", "exp_date", "expiry"])
        opt_type = pick(["option_type", "put_call", "cp_flag", "call_put"])

        df_out["strike_price"] = pd.to_numeric(strike, errors="coerce") if strike is not None else None
        df_out["expiration_date"] = pd.to_datetime(exp, errors="coerce").dt.date if exp is not None else None
        if opt_type is not None:
            s = opt_type.astype(str).str.upper().str.slice(0, 1)
            df_out["option_type"] = s.where(s.isin(["C", "P"]), None)
        else:
            df_out["option_type"] = None

        df_out["open"] = pd.to_numeric(df.get("open"), errors="coerce")
        df_out["high"] = pd.to_numeric(df.get("high"), errors="coerce")
        df_out["low"] = pd.to_numeric(df.get("low"), errors="coerce")
        df_out["close"] = pd.to_numeric(df.get("close"), errors="coerce")
        df_out["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")
        df_out["open_interest"] = pd.to_numeric(df.get("open_interest"), errors="coerce")

        # Optional greeks/iv
        df_out["implied_volatility"] = pd.to_numeric(pick(["implied_volatility", "iv"]), errors="coerce") if pick(["implied_volatility", "iv"]) is not None else None
        df_out["delta"] = pd.to_numeric(df.get("delta"), errors="coerce") if "delta" in df.columns else None
        df_out["gamma"] = pd.to_numeric(df.get("gamma"), errors="coerce") if "gamma" in df.columns else None
        df_out["theta"] = pd.to_numeric(df.get("theta"), errors="coerce") if "theta" in df.columns else None
        df_out["vega"] = pd.to_numeric(df.get("vega"), errors="coerce") if "vega" in df.columns else None

        df_out["source"] = "databento_options"
        df_out["ingested_at"] = datetime.utcnow()

        df_out = df_out.dropna(subset=["contract_symbol", "as_of_date"])
        return df_out

    except Exception as e:
        logger.error(f"Failed to collect options for {symbol}: {e}")
        return pd.DataFrame()


def main(days_back: int = 30, dry_run: bool = False):
    """
    Main options ingestion function.

    Args:
        days_back: Days of history to fetch (default 30 for daily incremental)
        dry_run: If True, only log what would be done
    """
    logger.info("=" * 60)
    logger.info("DATABENTO OPTIONS DAILY COLLECTION")
    logger.info("=" * 60)

    _load_local_env()
    client = get_databento_client()
    if not client:
        return

    con = get_connection()
    logger.info(f"Target: MotherDuck")
    logger.info(f"Symbols: {len(OPTIONS_SYMBOLS)}")

    # Determine date range (incremental from last date)
    try:
        result = con.execute(
            """
            SELECT MAX(as_of_date) as last_date
            FROM raw.databento_options_ohlcv_1d
            WHERE symbol = 'ZL'
        """
        ).df()

        if not result.empty and result["last_date"].iloc[0] is not None:
            last_date = pd.to_datetime(result["last_date"].iloc[0]).date()
            start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(f"Resuming from {start_date}")
        else:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime(
                "%Y-%m-%d"
            )
            logger.info(f"No existing data. Starting from {start_date}")
    except Exception as e:
        logger.warning(f"Could not check existing data: {e}")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    end_date = datetime.now().strftime("%Y-%m-%d")

    if start_date >= end_date:
        logger.info("Options data is already up to date")
        con.close()
        return

    if dry_run:
        logger.info(f"DRY RUN - would collect {start_date} to {end_date}")
        con.close()
        return

    # Collect options for each symbol
    all_options = []
    for symbol in OPTIONS_SYMBOLS:
        options_df = collect_options_for_symbol(client, symbol, start_date, end_date)
        if not options_df.empty:
            all_options.append(options_df)

    if not all_options:
        logger.warning("No options data collected")
        con.close()
        return

    # Combine and load
    combined = pd.concat(all_options, ignore_index=True)
    logger.info(f"Total options contracts: {len(combined):,}")

    # Load to MotherDuck (best-effort; depends on Databento subscription and symbology support)
    con.register("options_staging", combined)
    con.execute(
        """
        DELETE FROM raw.databento_options_ohlcv_1d
        WHERE EXISTS (
          SELECT 1
          FROM options_staging s
          WHERE s.contract_symbol = raw.databento_options_ohlcv_1d.contract_symbol
            AND s.as_of_date = raw.databento_options_ohlcv_1d.as_of_date
        )
    """
    )
    con.execute(
        """
        INSERT INTO raw.databento_options_ohlcv_1d
            (symbol, contract_symbol, as_of_date, strike_price, expiration_date, option_type,
             open, high, low, close, volume, open_interest,
             implied_volatility, delta, gamma, theta, vega, source, ingested_at)
        SELECT
               symbol, contract_symbol, as_of_date, strike_price, expiration_date, option_type,
               open, high, low, close, volume, open_interest,
               implied_volatility, delta, gamma, theta, vega, source, ingested_at
        FROM options_staging
    """
    )

    logger.info(
        f"Successfully loaded {len(combined):,} rows to raw.databento_options_ohlcv_1d"
    )
    con.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30, help="Days to fetch")
    parser.add_argument("--dry-run", action="store_true", help="Dry run only")
    args = parser.parse_args()

    main(days_back=args.days, dry_run=args.dry_run)

