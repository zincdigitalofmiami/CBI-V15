#!/usr/bin/env python3
"""
NOPA Crush Report Scraper - FREE
URL: https://nopa.org/nopa-crush-report/
Cost: FREE (public PDFs)

Collects monthly:
- Soybean crush volume (million bushels)
- Soybean oil stocks (million lbs)
- Soybean meal production (thousand short tons)

Release: 15th of each month
Target: raw.nopa_crush
"""

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd
import requests
from io import BytesIO

try:
    import PyPDF2

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️  PyPDF2 not installed. Install: pip install PyPDF2")

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")

NOPA_BASE = "https://nopa.org"
NOPA_REPORTS = f"{NOPA_BASE}/nopa-crush-report/"


def extract_crush_data_from_pdf(pdf_content: bytes) -> Optional[Dict[str, Any]]:
    """Extract crush data from NOPA PDF report."""
    if not PDF_AVAILABLE:
        return None

    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Parse key metrics (regex patterns for NOPA format)
        data = {}

        # Soybean crush (million bushels)
        crush_match = re.search(
            r"Crush.*?(\d+\.?\d*)\s*million bushels", text, re.IGNORECASE
        )
        if crush_match:
            data["crush_million_bushels"] = float(crush_match.group(1))

        # Soybean oil stocks (million lbs)
        oil_match = re.search(
            r"Soybean oil.*?stocks.*?(\d+\.?\d*)\s*million", text, re.IGNORECASE
        )
        if oil_match:
            data["oil_stocks_million_lbs"] = float(oil_match.group(1))

        # Soybean meal production (thousand tons)
        meal_match = re.search(
            r"Meal.*?production.*?(\d+\.?\d*)\s*thousand", text, re.IGNORECASE
        )
        if meal_match:
            data["meal_production_thousand_tons"] = float(meal_match.group(1))

        return data if data else None

    except Exception as e:
        print(f"  ❌ PDF parsing error: {e}")
        return None


def scrape_nopa_reports(months_back: int = 60) -> List[Dict[str, Any]]:
    """
    Scrape NOPA crush reports.
    Note: This is a placeholder - actual implementation needs to:
    1. Find PDF links on NOPA website
    2. Download each PDF
    3. Parse crush/oil/meal data
    """
    print("  ⚠️  NOPA scraper requires manual PDF download/parsing")
    print("  Alternative: Use USDA Oilseed Crushings data (free API)")

    # Fallback: Use USDA NASS Oilseed Crushings
    # This is available via USDA Quick Stats API (free)
    return []


def load_to_motherduck(rows: List[Dict[str, Any]]) -> int:
    """Load to MotherDuck."""
    if not rows:
        return 0

    if not MOTHERDUCK_TOKEN:
        raise ValueError("MOTHERDUCK_TOKEN required")

    con = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")

    # Create table
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS raw.nopa_crush (
            report_date DATE,
            crush_million_bushels DECIMAL(10,2),
            oil_stocks_million_lbs DECIMAL(10,2),
            meal_production_thousand_tons DECIMAL(10,2),
            capacity_utilization_pct DECIMAL(5,2),
            source VARCHAR,
            ingested_at TIMESTAMP,
            PRIMARY KEY (report_date)
        )
    """
    )

    df = pd.DataFrame(rows)
    con.register("temp_data", df)

    con.execute(
        """
        INSERT INTO raw.nopa_crush
        SELECT * FROM temp_data
        ON CONFLICT (report_date) DO UPDATE SET
            crush_million_bushels = EXCLUDED.crush_million_bushels,
            oil_stocks_million_lbs = EXCLUDED.oil_stocks_million_lbs,
            meal_production_thousand_tons = EXCLUDED.meal_production_thousand_tons,
            ingested_at = EXCLUDED.ingested_at
    """
    )

    count = con.execute("SELECT COUNT(*) FROM raw.nopa_crush").fetchone()[0]
    con.close()

    return count


def main():
    """Main collection routine."""
    print("=" * 80)
    print("NOPA CRUSH REPORT SCRAPER")
    print("=" * 80)
    print("\n⚠️  NOPA requires PDF scraping - using USDA alternative")
    print("\nAlternative: USDA NASS Oilseed Crushings API")
    print("URL: https://quickstats.nass.usda.gov/api")
    print("Cost: FREE")
    print("\nTo implement:")
    print("1. Register for USDA Quick Stats API key (free)")
    print("2. Query: Commodity=SOYBEANS, Data Item=CRUSHINGS")
    print("3. Get monthly crush volumes back to 2010")


if __name__ == "__main__":
    main()
