#!/usr/bin/env python3
"""
Backfill missing Databento symbols in small chunks to avoid 504 timeouts.
Runs 3 symbols per chunk with retry logic.
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.databento.collect_daily import (
    get_databento_client,
    get_connection,
    collect_batch_data,
    DATASET_START,
)

# Missing symbols - CORRECTED CME GLBX.MDP3 roots only (31 total)
# VX/VXM (CBOE), Z3N/FF/GD/LBS (invalid) removed
MISSING_SYMBOLS = [
    "10Y",
    "2YY",
    "30Y",
    "5YY",  # Yield Futures
    "ALI",
    "MSI",
    "QI",
    "QO",  # Metals
    "MCL",
    "QH",
    "QU",  # Energy micros
    "M6E",
    "M6A",
    "M6B",  # FX micros
    "EMD",
    "NIY",  # Equity
    "XC",
    "XW",
    "XK",
    "ZE",
    "CPO",  # Grains/Oilseeds
    "DC",
    "DY",
    "DA",  # Dairy
    "YO",
    "KT",
    "CJ",
    "TT",
    "LBR",  # Softs (CME alternatives)
    "ZQ",
    "TN",  # Rates
]

CHUNK_SIZE = 3  # Small chunks to avoid 504
RETRY_WAIT = 10  # seconds between retries


def main():
    start_date = DATASET_START  # 2010-06-06
    end_date = datetime.now().strftime("%Y-%m-%d")

    print("=" * 80)
    print(f"MISSING SYMBOLS BACKFILL: {len(MISSING_SYMBOLS)} symbols")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Chunk size: {CHUNK_SIZE} symbols")
    print("=" * 80)

    client = get_databento_client()
    if not client:
        print("❌ Failed to initialize Databento client")
        return

    con = get_connection()

    total_rows = 0
    failed_chunks = []

    # Process in small chunks
    for i in range(0, len(MISSING_SYMBOLS), CHUNK_SIZE):
        chunk = MISSING_SYMBOLS[i : i + CHUNK_SIZE]
        chunk_num = (i // CHUNK_SIZE) + 1
        total_chunks = (len(MISSING_SYMBOLS) + CHUNK_SIZE - 1) // CHUNK_SIZE

        print(f"\n{'='*80}")
        print(f"Chunk {chunk_num}/{total_chunks}: {chunk}")
        print(f"{'='*80}")

        success = False
        for attempt in range(3):
            try:
                print(f"Attempt {attempt + 1}/3...", flush=True)

                df = collect_batch_data(client, chunk, start_date, end_date)

                if df.empty:
                    print(f"  ⚠️ No data returned for {chunk}")
                    success = True  # Consider success (symbols may not exist)
                    break

                # Load to MotherDuck
                con.register("staging_df", df)
                con.execute(
                    """
                    DELETE FROM raw.databento_futures_ohlcv_1d
                    WHERE (symbol, as_of_date) IN (
                        SELECT symbol, as_of_date FROM staging_df
                    )
                """
                )
                con.execute(
                    """
                    INSERT INTO raw.databento_futures_ohlcv_1d 
                        (symbol, as_of_date, open, high, low, close, volume, open_interest)
                    SELECT symbol, as_of_date, open, high, low, close, volume, open_interest
                    FROM staging_df
                """
                )

                rows = len(df)
                total_rows += rows
                symbols_count = df["symbol"].nunique()
                print(f"  ✅ {rows:,} rows inserted ({symbols_count} symbols)")

                success = True
                break

            except Exception as e:
                error_str = str(e)
                if "504" in error_str or "timed out" in error_str:
                    print(f"  ⚠️ Timeout (attempt {attempt + 1}/3)")
                    if attempt < 2:
                        print(f"  Waiting {RETRY_WAIT}s before retry...")
                        time.sleep(RETRY_WAIT)
                        continue
                else:
                    print(f"  ❌ Error: {error_str}")
                    break

        if not success:
            failed_chunks.append(chunk)
            print(f"  ❌ Failed after 3 attempts")

        # Brief pause between chunks
        if i + CHUNK_SIZE < len(MISSING_SYMBOLS):
            time.sleep(3)

    con.close()

    print(f"\n{'='*80}")
    print(f"BACKFILL COMPLETE")
    print(f"{'='*80}")
    print(f"Total rows inserted: {total_rows:,}")

    if failed_chunks:
        print(f"\n⚠️ Failed chunks ({len(failed_chunks)}):")
        for chunk in failed_chunks:
            print(f"  {chunk}")
    else:
        print("\n✅ ALL CHUNKS SUCCESSFUL")


if __name__ == "__main__":
    main()



