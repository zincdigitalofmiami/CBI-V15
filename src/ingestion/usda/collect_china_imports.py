#!/usr/bin/env python3
"""
UN Comtrade - China Soybean Imports (FREE)
API: https://comtradeapi.un.org/
Cost: FREE (100 requests/hour)

Collects:
- China monthly soybean imports by origin (US, Brazil, Argentina)
- Historical: 2010-present (15+ years)
- HS Codes: 1201 (soybeans), 1507 (soybean oil)

Target: raw.un_comtrade_china_imports
"""

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List

import duckdb
import pandas as pd
import requests
import time

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")
UN_COMTRADE_KEY = os.getenv("UN_COMTRADE_API_KEY")

API_BASE = "https://comtradeapi.un.org/data/v1/get/C/M/HS"

# Country codes
CHINA = "156"
US = "842"
BRAZIL = "076"
ARGENTINA = "032"

# HS commodity codes
HS_SOYBEANS = "1201"
HS_SOYBEAN_OIL = "1507"


def generate_periods(start_year: int = 2010) -> List[str]:
    """Generate period strings (2010M01, 2010M02, etc.)."""
    current = datetime.now()
    periods = []

    for year in range(start_year, current.year + 1):
        end_month = 12 if year < current.year else current.month
        for month in range(1, end_month + 1):
            periods.append(f"{year}M{month:02d}")

    return periods


def fetch_china_imports(
    commodity_code: str, periods: List[str], batch_size: int = 12
) -> List[Dict[str, Any]]:
    """
    Fetch China import data for commodity.
    Batches periods to respect rate limits (100 req/hour).
    """
    headers = {"Ocp-Apim-Subscription-Key": UN_COMTRADE_KEY}

    all_data = []

    # Process in batches (12 months per request)
    for i in range(0, len(periods), batch_size):
        batch = periods[i : i + batch_size]
        period_str = ",".join(batch)

        params = {
            "reporterCode": CHINA,
            "cmdCode": commodity_code,
            "period": period_str,
            "partnerCode": f"{US},{BRAZIL},{ARGENTINA}",
            "flowCode": "M",  # Imports
        }

        print(f"  Fetching {batch[0]} to {batch[-1]}...")

        try:
            resp = requests.get(API_BASE, headers=headers, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            records = data.get("data", [])
            all_data.extend(records)

            print(f"    ✅ {len(records)} records")

            # Rate limit: 100 req/hour = 1 req per 36 seconds
            time.sleep(1)  # Be nice to the API

        except Exception as e:
            print(f"    ❌ Error: {e}")

    return all_data


def parse_to_rows(
    records: List[Dict[str, Any]], commodity: str
) -> List[Dict[str, Any]]:
    """Convert UN Comtrade records to database rows."""
    rows = []

    for rec in records:
        # Parse period (2024M01 → 2024-01-01)
        period = rec.get("period", "")
        if "M" in period:
            year = int(period[:4])
            month = int(period[5:])
            date = datetime(year, month, 1).date()
        else:
            continue

        rows.append(
            {
                "report_date": date,
                "commodity": commodity,
                "origin_country": rec.get("partnerDesc", ""),
                "origin_country_code": rec.get("partnerCode", ""),
                "import_value_usd": rec.get("primaryValue", 0),
                "import_weight_kg": rec.get("netWgt", 0),
                "import_weight_mt": (
                    rec.get("netWgt", 0) / 1000 if rec.get("netWgt") else 0
                ),
                "trade_flow": rec.get("flowDesc", ""),
                "source": "un_comtrade",
                "ingested_at": datetime.now(),
            }
        )

    return rows


def load_to_motherduck(rows: List[Dict[str, Any]]) -> int:
    """Load to MotherDuck."""
    if not rows:
        return 0

    if not MOTHERDUCK_TOKEN:
        raise ValueError("MOTHERDUCK_TOKEN required")

    con = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")

    # Create table if not exists
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS raw.un_comtrade_china_imports (
            report_date DATE,
            commodity VARCHAR,
            origin_country VARCHAR,
            origin_country_code VARCHAR,
            import_value_usd DOUBLE,
            import_weight_kg DOUBLE,
            import_weight_mt DOUBLE,
            trade_flow VARCHAR,
            source VARCHAR,
            ingested_at TIMESTAMP,
            PRIMARY KEY (report_date, commodity, origin_country_code)
        )
    """
    )

    df = pd.DataFrame(rows)
    con.register("temp_data", df)

    con.execute(
        """
        INSERT INTO raw.un_comtrade_china_imports
        SELECT * FROM temp_data
        ON CONFLICT (report_date, commodity, origin_country_code) DO UPDATE SET
            import_value_usd = EXCLUDED.import_value_usd,
            import_weight_kg = EXCLUDED.import_weight_kg,
            import_weight_mt = EXCLUDED.import_weight_mt,
            ingested_at = EXCLUDED.ingested_at
    """
    )

    count = con.execute(
        "SELECT COUNT(*) FROM raw.un_comtrade_china_imports"
    ).fetchone()[0]
    con.close()

    return count


def main():
    """Main collection routine."""
    print("=" * 80)
    print("UN COMTRADE - CHINA SOYBEAN IMPORTS")
    print("=" * 80)

    if not UN_COMTRADE_KEY:
        print("⚠️  UN_COMTRADE_API_KEY not set")
        print("Register free at: https://comtradeapi.un.org/")
        return

    # Generate periods (2010-present, ~180 months)
    periods = generate_periods(start_year=2010)
    print(f"\nCollecting {len(periods)} months (2010-present)")

    all_rows = []

    # 1. Soybeans (HS 1201)
    print("\n[1/2] Soybeans (HS 1201)...")
    soy_records = fetch_china_imports(HS_SOYBEANS, periods)
    soy_rows = parse_to_rows(soy_records, "soybeans")
    all_rows.extend(soy_rows)
    print(f"  ✅ {len(soy_rows)} soybean import records")

    # 2. Soybean Oil (HS 1507)
    print("\n[2/2] Soybean Oil (HS 1507)...")
    oil_records = fetch_china_imports(HS_SOYBEAN_OIL, periods)
    oil_rows = parse_to_rows(oil_records, "soybean_oil")
    all_rows.extend(oil_rows)
    print(f"  ✅ {len(oil_rows)} soybean oil import records")

    # Load to MotherDuck
    print("\n" + "=" * 80)
    print(f"Total records: {len(all_rows)}")

    if all_rows:
        total_count = load_to_motherduck(all_rows)
        print(f"✅ raw.un_comtrade_china_imports: {total_count:,} total rows")

        # Show summary
        con = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")
        summary = con.execute(
            """
            SELECT 
                commodity,
                origin_country,
                COUNT(*) as months,
                SUM(import_weight_mt) as total_mt,
                MIN(report_date) as start_date,
                MAX(report_date) as end_date
            FROM raw.un_comtrade_china_imports
            GROUP BY commodity, origin_country
            ORDER BY total_mt DESC
        """
        ).fetchall()

        print("\nChina Import Summary:")
        for row in summary:
            print(
                f"  {row[0]:15} from {row[1]:15} {row[2]:3} months  {row[3]:12,.0f} MT"
            )

        con.close()


if __name__ == "__main__":
    main()
