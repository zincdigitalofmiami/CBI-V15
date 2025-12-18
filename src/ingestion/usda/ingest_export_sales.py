"""
USDA Export Sales Data Ingestion

Downloads weekly export sales reports from USDA FAS.

Data source: USDA FAS Export Sales Reports (REAL DATA)
URL: https://apps.fas.usda.gov/export-sales/wkHistData.htm
Coverage: Soybeans (h801), Soybean Oil (h902), Soybean Meal (h901)
Frequency: Weekly (released every Thursday at 8:30 AM ET)
History: Data available from 1990-present

Usage:
    python src/ingestion/usda/ingest_export_sales.py --start-date 2020-01-01
    python src/ingestion/usda/ingest_export_sales.py --backfill
"""

import os
import sys
import argparse
import requests
import pandas as pd
import duckdb
from pathlib import Path
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT_DIR))

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")

# USDA FAS Export Sales Historical Data URLs
USDA_EXPORT_SALES_BASE = "https://apps.fas.usda.gov/export-sales/"
COMMODITY_PAGES = {
    "Soybeans": "h801.htm",
    "Soybean Oil": "h902.htm",
    "Soybean Meal": "h901.htm",
}


def get_connection():
    """Get MotherDuck connection (CLOUD ONLY - no local fallback)"""
    motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
    if not motherduck_token:
        raise RuntimeError("MOTHERDUCK_TOKEN not set - cannot proceed (CLOUD ONLY)")
    return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={motherduck_token}")


def parse_usda_table(url: str, commodity: str) -> pd.DataFrame:
    """
    Parse USDA FAS export sales table from HTML.

    Returns DataFrame with columns:
    - report_date
    - commodity
    - weekly_exports_mt
    - accumulated_exports_mt
    - net_sales_mt
    - outstanding_sales_mt
    """
    logger.info(f"  Fetching {commodity} from {url}")

    try:
        response = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        table = soup.find("table")

        if not table:
            logger.warning(f"No table found for {commodity}")
            return pd.DataFrame()

        rows = table.find_all("tr")
        logger.info(f"    Found {len(rows)} rows")

        data = []
        for row in rows[3:]:  # Skip header rows
            cells = [td.get_text().strip() for td in row.find_all(["td", "th"])]

            if len(cells) >= 5 and cells[0]:
                try:
                    # Parse date (MM/DD/YYYY format)
                    date_str = cells[0]
                    if "/" in date_str and len(date_str) == 10:
                        report_date = datetime.strptime(date_str, "%m/%d/%Y").date()

                        # Parse numeric values (remove commas)
                        def parse_num(s):
                            if not s or s == "-":
                                return 0.0
                            return float(s.replace(",", ""))

                        data.append(
                            {
                                "report_date": report_date,
                                "commodity": commodity,
                                "weekly_exports_mt": (
                                    parse_num(cells[1]) if len(cells) > 1 else 0
                                ),
                                "accumulated_exports_mt": (
                                    parse_num(cells[2]) if len(cells) > 2 else 0
                                ),
                                "net_sales_mt": (
                                    parse_num(cells[3]) if len(cells) > 3 else 0
                                ),
                                "outstanding_sales_mt": (
                                    parse_num(cells[4]) if len(cells) > 4 else 0
                                ),
                            }
                        )
                except (ValueError, IndexError) as e:
                    continue  # Skip malformed rows

        df = pd.DataFrame(data)
        logger.info(f"    Parsed {len(df)} valid records")
        return df

    except Exception as e:
        logger.error(f"Error fetching {commodity}: {e}")
        return pd.DataFrame()


def fetch_export_sales(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch REAL export sales data from USDA FAS"""

    logger.info(f"Fetching USDA FAS export sales data...")
    logger.info(f"Date range: {start_date} to {end_date}")

    all_data = []

    for commodity, page in COMMODITY_PAGES.items():
        url = USDA_EXPORT_SALES_BASE + page
        df = parse_usda_table(url, commodity)

        if not df.empty:
            # Filter to date range
            start_dt = pd.to_datetime(start_date).date()
            end_dt = pd.to_datetime(end_date).date()
            df = df[(df["report_date"] >= start_dt) & (df["report_date"] <= end_dt)]
            all_data.append(df)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)

        # Deduplicate: keep first occurrence per date/commodity
        # (USDA data may have multiple marketing year entries per date)
        combined = combined.drop_duplicates(
            subset=["report_date", "commodity"], keep="first"
        )
        combined = combined.sort_values(["commodity", "report_date"])

        logger.info(f"Total records after dedup: {len(combined)}")
        return combined
    else:
        logger.warning("No data fetched from USDA FAS")
        return pd.DataFrame()


def ingest_export_sales(start_date: str, end_date: str):
    """Ingest export sales data from USDA FAS"""
    logger.info("=" * 70)
    logger.info("USDA EXPORT SALES DATA INGESTION (REAL DATA)")
    logger.info("=" * 70)

    con = get_connection()

    df = fetch_export_sales(start_date, end_date)

    if df is not None and not df.empty:
        logger.info(f"Data summary:")
        logger.info(f"  Total rows: {len(df):,}")
        logger.info(
            f"  Date range: {df['report_date'].min()} to {df['report_date'].max()}"
        )
        logger.info(f"  Commodities: {df['commodity'].nunique()}")

        # Add destination_country = 'TOTAL' for aggregate data
        # (USDA historical data is aggregate, not per-country)
        df["destination_country"] = "TOTAL"
        df["exports_mt"] = df["weekly_exports_mt"]  # Map to DDL column

        # Register DataFrame
        con.register("staging_export", df)

        # Delete existing data for this date range (idempotent upsert)
        min_date = df["report_date"].min()
        max_date = df["report_date"].max()
        con.execute(
            f"""
            DELETE FROM raw.usda_export_sales
            WHERE report_date >= '{min_date}' AND report_date <= '{max_date}'
              AND destination_country = 'TOTAL'
        """
        )

        # Insert new data (match DDL columns: report_date, commodity, destination_country,
        #                   net_sales_mt, exports_mt, outstanding_sales_mt)
        con.execute(
            """
            INSERT INTO raw.usda_export_sales (
                report_date, commodity, destination_country,
                net_sales_mt, exports_mt, outstanding_sales_mt
            )
            SELECT 
                report_date, commodity, destination_country,
                net_sales_mt, exports_mt, outstanding_sales_mt
            FROM staging_export
        """
        )

        # Verify
        count = con.execute("SELECT COUNT(*) FROM raw.usda_export_sales").fetchone()[0]
        logger.info(
            f"✅ Loaded {len(df)} rows to raw.usda_export_sales (total: {count})"
        )
    else:
        logger.warning("No data to ingest")

    con.close()

    logger.info("=" * 70)
    logger.info("USDA EXPORT SALES INGESTION COMPLETE")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Ingest USDA export sales data")
    parser.add_argument(
        "--start-date", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Download all historical data (2015-present)",
    )

    args = parser.parse_args()

    if args.backfill:
        start_date = "2015-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
    else:
        start_date = args.start_date
        end_date = args.end_date

    ingest_export_sales(start_date, end_date)


if __name__ == "__main__":
    main()
