#!/usr/bin/env python3
"""
Databento HOURLY OHLCV Collection
Targeting a rolling backfill window (default: last 365 days).
Schema: raw.databento_futures_ohlcv_1h

Usage:
    python src/ingestion/databento/collect_hourly.py --days 365
    python src/ingestion/databento/collect_hourly.py --days 730
    python src/ingestion/databento/collect_hourly.py --dry-run --days 365
"""

import os
import duckdb
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import databento as db
from pathlib import Path
import time

# Load environment
load_dotenv()

# Configuration
API_KEY = os.getenv("DATABENTO_API_KEY")
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DB_NAME = os.getenv("MOTHERDUCK_DB", "cbi_v15")

if not API_KEY:
    raise ValueError("DATABENTO_API_KEY not found in environment")
if not MOTHERDUCK_TOKEN:
    raise ValueError("MOTHERDUCK_TOKEN not found in environment")

# Hourly Collection Subset - High Liquidity / High Impact Only (GLBX.MDP3)
# Intentionally excludes non-GLBX roots (ICE softs, DX, etc.).
SYMBOLS = [
    # Grains/Ag
    "ZL",
    "ZS",
    "ZM",
    "ZC",
    "ZW",
    # Softs/Palm clones (CME/NYMEX cash-settled)
    "CPO",
    "YO",
    "KT",
    "CJ",
    "TT",
    "LBR",
    # Energy
    "CL",
    "BZ",
    "HO",
    "RB",
    "NG",
    # Metals
    "GC",
    "SI",
    "HG",
    "PA",
    "PL",
    # Indices
    "ES",
    "NQ",
    "YM",
    "RTY",
    "NIY",
    # Rates/FX
    "ZQ",
    "ZN",
    "ZB",
    "SR3",
    "6E",
    "6J",
    "6A",
    "6B",
    "6C",
    # Crypto
    "BTC",
    "ETH",
]

DATASET = "GLBX.MDP3"

DEFAULT_BATCH_OUTPUT_DIR = Path("data/raw/databento/batch")


def get_db_connection():
    return duckdb.connect(f"md:{DB_NAME}?motherduck_token={MOTHERDUCK_TOKEN}")


def estimate_cost(
    client: db.Historical, symbols: list[str], start_date: str, end_date: str
) -> float:
    """Estimate request cost (USD) before pulling."""
    try:
        continuous_symbols = [f"{s}.v.0" for s in symbols]
        return client.metadata.get_cost(
            dataset=DATASET,
            symbols=continuous_symbols,
            schema="ohlcv-1h",
            start=start_date,
            end=end_date,
            stype_in="continuous",
        )
    except Exception as e:
        print(f"[Cost] Could not estimate cost: {e}")
        return -1.0


def collect_hourly_data(
    days_back: int = 365, dry_run: bool = False, start_offset_days: int = 0
):
    print(f"=== Starting Hourly Data Collection (Last {days_back} days) ===")

    # Calculate date range (with offset for chunking)
    end_date = datetime.now(timezone.utc) - timedelta(days=start_offset_days)
    start_date = end_date - timedelta(days=days_back)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print(f"Range: {start_str} to {end_str}")
    print(f"Symbols: {len(SYMBOLS)}")

    # Initialize Databento client
    client = db.Historical(key=API_KEY)

    estimated_cost = estimate_cost(client, SYMBOLS, start_str, end_str)
    if estimated_cost >= 0:
        print(f"[Cost] Estimated: ${estimated_cost:.2f}")
    if dry_run:
        print("DRY RUN - skipping data fetch")
        return

    con = get_db_connection()
    total_rows = 0

    # Databento best practice: batch symbols in a single request.
    # Use continuous symbology + explicit stype_in.
    continuous_symbols = [f"{s}.v.0" for s in SYMBOLS]

    print(
        f"\nRequesting {len(continuous_symbols)} symbols from Databento...", flush=True
    )

    # Retry logic for transient errors (504 gateway timeout, rate limits)
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            data = client.timeseries.get_range(
                dataset=DATASET,
                symbols=continuous_symbols,
                schema="ohlcv-1h",
                start=start_str,
                end=end_str,
                stype_in="continuous",
            )
            df = data.to_df().reset_index()
            break  # Success, exit retry loop
        except Exception as e:
            error_str = str(e)
            if attempt < max_retries - 1:
                # Retry on 504, 429, or network errors
                if any(
                    code in error_str for code in ["504", "429", "timed out", "gateway"]
                ):
                    wait_time = retry_delay * (2**attempt)  # Exponential backoff
                    print(
                        f"⚠️  {error_str}. Retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    import time

                    time.sleep(wait_time)
                    continue
            # Final attempt failed or non-retryable error
            con.close()
            raise RuntimeError(
                f"Databento request failed after {max_retries} attempts: {e}"
            ) from e

    if df.empty:
        print("⚠️  No data returned.")
        con.close()
        return

    # Normalize symbol root (ZL.v.0 -> ZL)
    if "symbol" in df.columns:
        df["symbol"] = (
            df["symbol"].astype(str).str.replace(r"\.v\.\d+$", "", regex=True)
        )

    # Ensure open_interest column exists (ohlcv-1h typically doesn't include it)
    if "open_interest" not in df.columns:
        df["open_interest"] = None

    # Ensure ts_event is timestamp (store as naive UTC TIMESTAMP)
    # NOTE: tz_localize(None) preserves the UTC clock time while dropping tzinfo.
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True).dt.tz_localize(None)

    df = df[
        [
            "symbol",
            "ts_event",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
        ]
    ]

    print(
        f"Received {len(df):,} rows across {df['symbol'].nunique()} resolved symbols "
        f"({df['symbol'].min()}..{df['symbol'].max()})"
    )

    # Idempotent load: delete keys then insert
    con.register("staging_df", df)
    con.execute(
        """
        DELETE FROM raw.databento_futures_ohlcv_1h
        WHERE (symbol, ts_event) IN (
            SELECT symbol, ts_event FROM staging_df
        )
        """
    )
    con.execute(
        """
        INSERT INTO raw.databento_futures_ohlcv_1h
            (symbol, ts_event, open, high, low, close, volume, open_interest)
        SELECT symbol, ts_event, open, high, low, close, volume, open_interest
        FROM staging_df
        """
    )

    total_rows = len(df)

    print(f"\n\nTotal rows inserted: {total_rows:,}")
    con.close()


def submit_and_download_batch(
    symbols: list[str],
    start_str: str,
    end_str: str,
    output_dir: Path,
    split_duration: str = "month",
    encoding: str = "dbn",
    compression: str = "zstd",
) -> list[Path]:
    """
    Use Databento Batch API for faster, more reliable downloads.

    Databento docs note that `timeseries.get_range` can take a long time for large requests
    and recommends batch download for large pulls.
    """
    client = db.Historical(key=API_KEY)

    continuous_symbols = [f"{s}.v.0" for s in symbols]

    job = client.batch.submit_job(
        dataset=DATASET,
        symbols=continuous_symbols,
        schema="ohlcv-1h",
        start=start_str,
        end=end_str,
        stype_in="continuous",
        encoding=encoding,
        compression=compression,
        split_symbols=False,
        split_duration=split_duration,
        delivery="download",
    )

    job_id = job.get("id") or job.get("job_id")
    if not job_id:
        raise RuntimeError(f"Unexpected batch.submit_job response (missing job id): {job}")

    print(f"[Batch] Submitted job: {job_id}")

    # Poll until done
    deadline = time.time() + 60 * 60  # 1 hour
    poll_sleep_seconds = 5
    consecutive_errors = 0
    while True:
        try:
            jobs = client.batch.list_jobs(states="queued,processing,done,expired")
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            if time.time() > deadline:
                raise RuntimeError(
                    f"[Batch] Timeout waiting for job list (last error: {e})"
                ) from e
            # Databento can intermittently 504; treat as transient.
            backoff = min(poll_sleep_seconds * (2 ** min(consecutive_errors, 6)), 60)
            print(f"[Batch] ⚠️ list_jobs error ({consecutive_errors}): {e} - sleeping {backoff}s")
            time.sleep(backoff)
            continue

        info = next((j for j in jobs if j.get("id") == job_id), None)
        if info is None:
            # Newly submitted jobs can take a moment to appear in list_jobs
            if time.time() > deadline:
                raise RuntimeError(f"[Batch] Job {job_id} not visible after polling")
            time.sleep(3)
            continue

        state = info.get("state")
        if state == "done":
            break
        if state == "expired":
            raise RuntimeError(f"[Batch] Job {job_id} expired before download")

        if time.time() > deadline:
            raise RuntimeError(f"[Batch] Timeout waiting for job {job_id} (state={state})")
        time.sleep(poll_sleep_seconds)

    # List files (also can intermittently 504)
    for attempt in range(8):
        try:
            files = client.batch.list_files(job_id)
            break
        except Exception as e:
            if attempt == 7:
                raise
            backoff = min(2 ** attempt, 30)
            print(f"[Batch] ⚠️ list_files error: {e} - sleeping {backoff}s")
            time.sleep(backoff)

    print(f"[Batch] Job done. Files: {len(files)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(8):
        try:
            paths = client.batch.download(
                job_id=job_id, output_dir=str(output_dir), keep_zip=False
            )
            break
        except Exception as e:
            if attempt == 7:
                raise
            backoff = min(2 ** attempt, 30)
            print(f"[Batch] ⚠️ download error: {e} - sleeping {backoff}s")
            time.sleep(backoff)

    print(f"[Batch] Downloaded files: {len(paths)}")
    return paths


def download_existing_batch(
    job_id: str,
    output_dir: Path,
) -> list[Path]:
    """Download an existing Databento batch job without submitting a new one."""
    client = db.Historical(key=API_KEY)

    print(f"[Batch] Resuming existing job: {job_id}")

    deadline = time.time() + 60 * 60  # 1 hour
    poll_sleep_seconds = 5
    consecutive_errors = 0

    while True:
        try:
            jobs = client.batch.list_jobs(states="queued,processing,done,expired")
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            if time.time() > deadline:
                raise RuntimeError(
                    f"[Batch] Timeout waiting for job list (last error: {e})"
                ) from e
            backoff = min(poll_sleep_seconds * (2 ** min(consecutive_errors, 6)), 60)
            print(
                f"[Batch] ⚠️ list_jobs error ({consecutive_errors}): {e} - sleeping {backoff}s"
            )
            time.sleep(backoff)
            continue

        info = next((j for j in jobs if j.get("id") == job_id), None)
        if info is None:
            if time.time() > deadline:
                raise RuntimeError(f"[Batch] Job {job_id} not visible after polling")
            time.sleep(3)
            continue

        state = info.get("state")
        if state == "done":
            break
        if state == "expired":
            raise RuntimeError(f"[Batch] Job {job_id} expired before download")

        if time.time() > deadline:
            raise RuntimeError(f"[Batch] Timeout waiting for job {job_id} (state={state})")
        time.sleep(poll_sleep_seconds)

    for attempt in range(8):
        try:
            files = client.batch.list_files(job_id)
            break
        except Exception as e:
            if attempt == 7:
                raise
            backoff = min(2 ** attempt, 30)
            print(f"[Batch] ⚠️ list_files error: {e} - sleeping {backoff}s")
            time.sleep(backoff)

    print(f"[Batch] Job done. Files: {len(files)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(8):
        try:
            paths = client.batch.download(
                job_id=job_id, output_dir=str(output_dir), keep_zip=False
            )
            break
        except Exception as e:
            if attempt == 7:
                raise
            backoff = min(2 ** attempt, 30)
            print(f"[Batch] ⚠️ download error: {e} - sleeping {backoff}s")
            time.sleep(backoff)

    print(f"[Batch] Downloaded files: {len(paths)}")
    return paths


def ingest_ohlcv_1h_files_to_motherduck(paths: list[Path]) -> int:
    """
    Ingest batch-downloaded files into raw.databento_futures_ohlcv_1h.
    Files are expected to be DBN (default batch encoding).
    """
    con = get_db_connection()
    total_rows = 0

    for p in sorted(paths):
        store = db.DBNStore.from_file(str(p))
        df = store.to_df(pretty_ts=True, map_symbols=True)
        if df is None or len(df) == 0:
            continue

        df = df.reset_index()

        # Normalize symbol root (ZL.v.0 -> ZL) if needed
        if "symbol" in df.columns:
            df["symbol"] = (
                df["symbol"].astype(str).str.replace(r"\.v\.\d+$", "", regex=True)
            )

        # Ensure open_interest column exists (ohlcv-1h typically doesn't include it)
        if "open_interest" not in df.columns:
            df["open_interest"] = None

        # Prefer ts_event column; some exports use 'ts_event' already.
        if "ts_event" not in df.columns:
            raise RuntimeError(f"[Batch] Missing ts_event in file {p.name}: cols={list(df.columns)}")

        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True).dt.tz_localize(None)

        df = df[
            [
                "symbol",
                "ts_event",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_interest",
            ]
        ]

        con.register("staging_df", df)
        con.execute(
            """
            INSERT OR REPLACE INTO raw.databento_futures_ohlcv_1h
                (symbol, ts_event, open, high, low, close, volume, open_interest)
            SELECT symbol, ts_event, open, high, low, close, volume, open_interest
            FROM staging_df
            """
        )

        total_rows += len(df)
        print(f"[Batch] Ingested {len(df):,} rows from {p.name}")

    con.close()
    return total_rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--days", type=int, default=365, help="Days of history to fetch"
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=14,
        help="Days per chunk (prevents process aborts on large pulls)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Estimate cost but do not fetch data"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use Databento Batch API (faster/more reliable for large pulls)",
    )
    parser.add_argument(
        "--batch-output-dir",
        type=str,
        default=str(DEFAULT_BATCH_OUTPUT_DIR),
        help="Where to download batch files",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Resume an existing Databento batch job id (skip submission)",
    )
    args = parser.parse_args()

    if args.dry_run:
        # Just estimate total cost
        collect_hourly_data(days_back=args.days, dry_run=True)
    else:
        if args.batch:
            # Single batch job for the full requested window.
            output_dir = Path(args.batch_output_dir)

            if args.job_id:
                print(f"=== Batch Resume: job_id={args.job_id} output_dir={output_dir} ===")
                paths = download_existing_batch(job_id=args.job_id, output_dir=output_dir)
            else:
                end_dt = datetime.now(timezone.utc)
                start_dt = end_dt - timedelta(days=args.days)
                start_str = start_dt.strftime("%Y-%m-%d")
                end_str = end_dt.strftime("%Y-%m-%d")

                print(
                    f"=== Batch Backfill: {args.days} days ({start_str}..{end_str}) symbols={len(SYMBOLS)} ==="
                )

                paths = submit_and_download_batch(
                    symbols=SYMBOLS,
                    start_str=start_str,
                    end_str=end_str,
                    output_dir=output_dir,
                    split_duration="month",
                    encoding="dbn",
                    compression="zstd",
                )

            rows = ingest_ohlcv_1h_files_to_motherduck(paths)
            print(f"✅ Batch ingest complete. Rows upserted: {rows:,}")
            raise SystemExit(0)

        # Chunk the backfill to avoid process aborts
        total_days = args.days
        chunk_size = args.chunk_days
        chunks = (total_days + chunk_size - 1) // chunk_size

        print(
            f"=== Chunked Backfill: {total_days} days in {chunks} chunks of {chunk_size} days ==="
        )

        for i in range(chunks):
            chunk_start = i * chunk_size
            chunk_end = min((i + 1) * chunk_size, total_days)
            chunk_days = chunk_end - chunk_start

            print(f"\n{'=' * 80}")
            print(
                f"Chunk {i + 1}/{chunks}: Days {chunk_start} to {chunk_end} ({chunk_days} days)"
            )
            print(f"{'=' * 80}\n")

            collect_hourly_data(
                days_back=chunk_days, start_offset_days=chunk_start, dry_run=False
            )

        print(f"\n{'=' * 80}")
        print(f"✅ BACKFILL COMPLETE: {total_days} days ingested")
        print(f"{'=' * 80}")


