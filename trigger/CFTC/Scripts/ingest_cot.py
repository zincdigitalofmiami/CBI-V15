"""
CFTC Commitment of Traders (COT) Data Ingestion

Downloads and processes weekly COT reports from CFTC.

Data sources:
- Disaggregated Reports (commodities): https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm
- Traders in Financial Futures (TFF): https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm

Reports released: Every Friday at 3:30 PM ET (data as of prior Tuesday)

Usage:
    python trigger/CFTC/Scripts/ingest_cot.py --start-date 2020-01-01 --end-date 2024-12-31
    python trigger/CFTC/Scripts/ingest_cot.py --backfill  # Download all historical data
"""

import os
import sys
import argparse
import requests
import pandas as pd
import duckdb
from pathlib import Path
from datetime import datetime, timedelta
import zipfile
import io

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")

# CFTC URLs for historical compressed data
DISAGGREGATED_FUTURES_URL = (
    "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
)
DISAGGREGATED_COMBINED_URL = (
    "https://www.cftc.gov/files/dea/history/com_disagg_txt_{year}.zip"
)
TFF_FUTURES_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
TFF_COMBINED_URL = "https://www.cftc.gov/files/dea/history/com_fin_txt_{year}.zip"

# Symbol mapping: CFTC contract names to our symbols
# NOTE: Only CME/CBOT/NYMEX/COMEX symbols (verified Databento availability)
SYMBOL_MAPPING = {
    # Agricultural (11 symbols - CME/CBOT only)
    "SOYBEAN OIL": "ZL",
    "SOYBEANS": "ZS",
    "SOYBEAN MEAL": "ZM",
    "CORN": "ZC",
    "WHEAT": "ZW",
    "OATS": "ZO",
    "ROUGH RICE": "ZR",  # VERIFIED CME
    "LEAN HOGS": "HE",
    "LIVE CATTLE": "LE",
    "FEEDER CATTLE": "GF",
    "CRUDE PALM OIL": "FCPO",  # Bursa Malaysia (if available)
    # Energy (4 symbols - NYMEX)
    "CRUDE OIL, LIGHT SWEET": "CL",
    "WTI-PHYSICAL": "CL",
    "HEATING OIL": "HO",  # Includes ULSD (no separate UL symbol)
    "NY HARBOR ULSD": "HO",
    "RBOB GASOLINE": "RB",
    "NATURAL GAS": "NG",
    # Metals (5 symbols - COMEX/NYMEX only, NO LME)
    "COPPER": "HG",
    "GOLD": "GC",
    "SILVER": "SI",
    "PLATINUM": "PL",
    "PALLADIUM": "PA",
    # NOTE: Aluminum (AL) removed - trades on LME, not CME
    # Treasuries (3 symbols - CBOT)
    "5-YEAR U.S. TREASURY NOTES": "ZF",
    "10-YEAR U.S. TREASURY NOTES": "ZN",  # Use ZN (TY is floor symbol)
    "U.S. TREASURY BONDS": "ZB",
    # FX Futures (10 symbols - CME)
    "EURO FX": "6E",
    "JAPANESE YEN": "6J",
    "BRITISH POUND": "6B",
    "CANADIAN DOLLAR": "6C",
    "AUSTRALIAN DOLLAR": "6A",
    "NEW ZEALAND DOLLAR": "6N",
    "MEXICAN PESO": "6M",
    "BRAZILIAN REAL": "6L",
    "SWISS FRANC": "6S",
    "U.S. DOLLAR INDEX": "DX",
}


def get_connection():
    """Get DuckDB/MotherDuck connection"""
    motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
    if motherduck_token:
        return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={motherduck_token}")
    else:
        # Local fallback
        db_path = ROOT_DIR / "data" / "duckdb" / "cbi_v15.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(db_path))


def download_cot_file(url: str, year: int) -> pd.DataFrame:
    """Download and extract COT data file"""
    print(f"  Downloading {url.format(year=year)}...")

    try:
        response = requests.get(url.format(year=year), timeout=60)
        response.raise_for_status()

        # Extract ZIP file
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Get the first .txt file in the archive
            txt_files = [f for f in z.namelist() if f.endswith(".txt")]
            if not txt_files:
                print(f"    ⚠️  No .txt file found in ZIP")
                return None

            # Read the text file
            with z.open(txt_files[0]) as f:
                df = pd.read_csv(f, low_memory=False)
                print(f"    ✅ Downloaded {len(df):,} rows")
                return df

    except requests.exceptions.RequestException as e:
        print(f"    ❌ Error downloading: {e}")
        return None
    except Exception as e:
        print(f"    ❌ Error processing: {e}")
        return None


def process_disaggregated_report(df: pd.DataFrame) -> pd.DataFrame:
    """Process disaggregated COT report"""
    if df is None or df.empty:
        return None

    # Map contract names to our symbols
    df["symbol"] = df["Market_and_Exchange_Names"].str.upper().map(SYMBOL_MAPPING)

    # Filter to only our symbols
    df = df[df["symbol"].notna()].copy()

    if df.empty:
        return None

    # Convert report date
    df["report_date"] = pd.to_datetime(df["Report_Date_as_YYYY-MM-DD"])

    # Select and rename columns
    processed = df[
        [
            "report_date",
            "symbol",
            "Open_Interest_All",
            "Prod_Merc_Positions_Long_All",
            "Prod_Merc_Positions_Short_All",
            "Swap_Positions_Long_All",
            "Swap_Positions_Short_All",
            "M_Money_Positions_Long_All",
            "M_Money_Positions_Short_All",
            "Other_Rept_Positions_Long_All",
            "Other_Rept_Positions_Short_All",
            "NonRept_Positions_Long_All",
            "NonRept_Positions_Short_All",
        ]
    ].copy()

    # Rename columns
    processed.columns = [
        "report_date",
        "symbol",
        "open_interest",
        "prod_merc_long",
        "prod_merc_short",
        "swap_long",
        "swap_short",
        "managed_money_long",
        "managed_money_short",
        "other_rept_long",
        "other_rept_short",
        "nonrept_long",
        "nonrept_short",
    ]

    # Calculate net positions
    processed["prod_merc_net"] = (
        processed["prod_merc_long"] - processed["prod_merc_short"]
    )
    processed["swap_net"] = processed["swap_long"] - processed["swap_short"]
    processed["managed_money_net"] = (
        processed["managed_money_long"] - processed["managed_money_short"]
    )
    processed["other_rept_net"] = (
        processed["other_rept_long"] - processed["other_rept_short"]
    )
    processed["nonrept_net"] = processed["nonrept_long"] - processed["nonrept_short"]

    # Calculate as % of open interest
    processed["managed_money_net_pct_oi"] = (
        processed["managed_money_net"] / processed["open_interest"] * 100
    ).round(2)
    processed["prod_merc_net_pct_oi"] = (
        processed["prod_merc_net"] / processed["open_interest"] * 100
    ).round(2)

    return processed


def process_tff_report(df: pd.DataFrame) -> pd.DataFrame:
    """Process Traders in Financial Futures (TFF) report"""
    if df is None or df.empty:
        return None

    # Map contract names to our symbols
    df["symbol"] = df["Market_and_Exchange_Names"].str.upper().map(SYMBOL_MAPPING)

    # Filter to only our symbols
    df = df[df["symbol"].notna()].copy()

    if df.empty:
        return None

    # Convert report date
    df["report_date"] = pd.to_datetime(df["Report_Date_as_YYYY-MM-DD"])

    # Select and rename columns
    processed = df[
        [
            "report_date",
            "symbol",
            "Open_Interest_All",
            "Dealer_Positions_Long_All",
            "Dealer_Positions_Short_All",
            "Asset_Mgr_Positions_Long_All",
            "Asset_Mgr_Positions_Short_All",
            "Lev_Money_Positions_Long_All",
            "Lev_Money_Positions_Short_All",
            "Other_Rept_Positions_Long_All",
            "Other_Rept_Positions_Short_All",
            "NonRept_Positions_Long_All",
            "NonRept_Positions_Short_All",
        ]
    ].copy()

    # Rename columns
    processed.columns = [
        "report_date",
        "symbol",
        "open_interest",
        "dealer_long",
        "dealer_short",
        "asset_mgr_long",
        "asset_mgr_short",
        "lev_money_long",
        "lev_money_short",
        "other_rept_long",
        "other_rept_short",
        "nonrept_long",
        "nonrept_short",
    ]

    # Calculate net positions
    processed["dealer_net"] = processed["dealer_long"] - processed["dealer_short"]
    processed["asset_mgr_net"] = (
        processed["asset_mgr_long"] - processed["asset_mgr_short"]
    )
    processed["lev_money_net"] = (
        processed["lev_money_long"] - processed["lev_money_short"]
    )
    processed["other_rept_net"] = (
        processed["other_rept_long"] - processed["other_rept_short"]
    )
    processed["nonrept_net"] = processed["nonrept_long"] - processed["nonrept_short"]

    # Calculate as % of open interest
    processed["lev_money_net_pct_oi"] = (
        processed["lev_money_net"] / processed["open_interest"] * 100
    ).round(2)
    processed["asset_mgr_net_pct_oi"] = (
        processed["asset_mgr_net"] / processed["open_interest"] * 100
    ).round(2)

    return processed


def ingest_cot_data(start_year: int, end_year: int):
    """Ingest COT data for specified year range"""
    print(f"\n{'=' * 80}")
    print(f"CFTC COT DATA INGESTION")
    print(f"{'=' * 80}")
    print(f"Years: {start_year} - {end_year}\n")

    con = get_connection()

    all_disagg_data = []
    all_tff_data = []

    # Download data for each year
    for year in range(start_year, end_year + 1):
        print(f"\n📅 Processing year {year}...")

        # Disaggregated Futures Only
        print("  Disaggregated Futures Only...")
        df_disagg = download_cot_file(DISAGGREGATED_FUTURES_URL, year)
        if df_disagg is not None:
            processed = process_disaggregated_report(df_disagg)
            if processed is not None and not processed.empty:
                all_disagg_data.append(processed)
                print(f"    ✅ Processed {len(processed):,} rows for our symbols")

        # Traders in Financial Futures (for FX and Treasuries)
        print("  Traders in Financial Futures...")
        df_tff = download_cot_file(TFF_FUTURES_URL, year)
        if df_tff is not None:
            processed = process_tff_report(df_tff)
            if processed is not None and not processed.empty:
                all_tff_data.append(processed)
                print(f"    ✅ Processed {len(processed):,} rows for our symbols")

    # Combine all data
    if all_disagg_data:
        print(f"\n📊 Combining disaggregated data...")
        combined_disagg = pd.concat(all_disagg_data, ignore_index=True)
        combined_disagg = combined_disagg.sort_values(["symbol", "report_date"])
        print(f"  Total rows: {len(combined_disagg):,}")

        # Insert into database
        print(f"\n💾 Inserting disaggregated data into database...")
        con.execute(
            """
            INSERT OR REPLACE INTO raw.cftc_cot_disaggregated
            SELECT * FROM combined_disagg
        """
        )
        print(f"  ✅ Inserted {len(combined_disagg):,} rows")

    if all_tff_data:
        print(f"\n📊 Combining TFF data...")
        combined_tff = pd.concat(all_tff_data, ignore_index=True)
        combined_tff = combined_tff.sort_values(["symbol", "report_date"])
        print(f"  Total rows: {len(combined_tff):,}")

        # Insert into database
        print(f"\n💾 Inserting TFF data into database...")
        con.execute(
            """
            INSERT OR REPLACE INTO raw.cftc_cot_tff
            SELECT * FROM combined_tff
        """
        )
        print(f"  ✅ Inserted {len(combined_tff):,} rows")

    con.close()

    print(f"\n{'=' * 80}")
    print(f"✅ COT DATA INGESTION COMPLETE")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Ingest CFTC COT data")
    parser.add_argument(
        "--start-year", type=int, default=2020, help="Start year (default: 2020)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=datetime.now().year,
        help="End year (default: current year)",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Download all historical data (2006-present)",
    )

    args = parser.parse_args()

    if args.backfill:
        # Disaggregated reports start in 2006
        start_year = 2006
        end_year = datetime.now().year
    else:
        start_year = args.start_year
        end_year = args.end_year

    ingest_cot_data(start_year, end_year)


if __name__ == "__main__":
    main()
