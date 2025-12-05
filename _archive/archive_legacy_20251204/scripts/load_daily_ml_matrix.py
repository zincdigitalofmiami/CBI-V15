"""
Canonical BigQuery Loader - Python-First Architecture
Loads feature matrix with proper partitioning and clustering.

CRITICAL: Uses .dt.date to ensure TRUE DATE type (not TIMESTAMP)
for proper BigQuery partitioning.
"""

import os
import argparse
import pandas as pd
from google.cloud import bigquery

PROJECT_ID = os.getenv("GCP_PROJECT", "cbi-v15")
DEFAULT_TABLE = "cbi-v15.training.daily_ml_matrix"

# Minimum required columns
REQUIRED_COLUMNS = ["date", "symbol", "regime", "price"]
# Full schema lock (32 columns)
LOCKED_SCHEMA = [
    bigquery.SchemaField("date", "DATE", mode="REQUIRED"),
    bigquery.SchemaField("symbol", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("open", "FLOAT64"),
    bigquery.SchemaField("high", "FLOAT64"),
    bigquery.SchemaField("low", "FLOAT64"),
    bigquery.SchemaField("close", "FLOAT64"),
    bigquery.SchemaField("volume", "INT64"),
    bigquery.SchemaField("regime", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("regime_weight", "FLOAT64"),
    bigquery.SchemaField("price", "FLOAT64"),
    bigquery.SchemaField("lag_1d", "FLOAT64"),
    bigquery.SchemaField("lag_5d", "FLOAT64"),
    bigquery.SchemaField("lag_21d", "FLOAT64"),
    bigquery.SchemaField("ret_1d", "FLOAT64"),
    bigquery.SchemaField("ret_5d", "FLOAT64"),
    bigquery.SchemaField("ret_21d", "FLOAT64"),
    bigquery.SchemaField("vol_21d", "FLOAT64"),
    bigquery.SchemaField("gk_vol_21d", "FLOAT64"),
    bigquery.SchemaField("sma_20", "FLOAT64"),
    bigquery.SchemaField("sma_50", "FLOAT64"),
    bigquery.SchemaField("sma_200", "FLOAT64"),
    bigquery.SchemaField("ema_20", "FLOAT64"),
    bigquery.SchemaField("ema_50", "FLOAT64"),
    bigquery.SchemaField("dist_sma_20", "FLOAT64"),
    bigquery.SchemaField("dist_sma_50", "FLOAT64"),
    bigquery.SchemaField("dist_ema_20", "FLOAT64"),
    bigquery.SchemaField("fx_brl_mom_21d", "FLOAT64"),
    bigquery.SchemaField("fx_brl_mom_63d", "FLOAT64"),
    bigquery.SchemaField("fx_brl_mom_252d", "FLOAT64"),
    bigquery.SchemaField("fx_brl_vol_21d", "FLOAT64"),
    bigquery.SchemaField("fx_brl_vol_63d", "FLOAT64"),
    bigquery.SchemaField("fx_dxy_mom_21d", "FLOAT64"),
    bigquery.SchemaField("fx_dxy_mom_63d", "FLOAT64"),
    bigquery.SchemaField("fx_dxy_mom_252d", "FLOAT64"),
    bigquery.SchemaField("fx_dxy_vol_21d", "FLOAT64"),
    bigquery.SchemaField("fx_dxy_vol_63d", "FLOAT64"),
    bigquery.SchemaField("fred_dff", "FLOAT64"),
    bigquery.SchemaField("fred_dfedtaru", "FLOAT64"),
    bigquery.SchemaField("fred_dfedtarl", "FLOAT64"),
    bigquery.SchemaField("fred_effr", "FLOAT64"),
    bigquery.SchemaField("fred_sofr", "FLOAT64"),
    bigquery.SchemaField("fred_dgs3mo", "FLOAT64"),
    bigquery.SchemaField("fred_dgs1", "FLOAT64"),
    bigquery.SchemaField("fred_dgs2", "FLOAT64"),
    bigquery.SchemaField("fred_dgs5", "FLOAT64"),
    bigquery.SchemaField("fred_dgs10", "FLOAT64"),
    bigquery.SchemaField("fred_dgs30", "FLOAT64"),
    bigquery.SchemaField("fred_t10y2y", "FLOAT64"),
    bigquery.SchemaField("fred_t10y3m", "FLOAT64"),
    bigquery.SchemaField("fred_vixcls", "FLOAT64"),
    bigquery.SchemaField("fred_nfci", "FLOAT64"),
    bigquery.SchemaField("fred_nfcileverage", "FLOAT64"),
    bigquery.SchemaField("fred_baaffm", "FLOAT64"),
    bigquery.SchemaField("fred_bamlh0a0hym2", "FLOAT64"),
    bigquery.SchemaField("corr_zl_brl_30d", "FLOAT64"),
    bigquery.SchemaField("corr_zl_brl_60d", "FLOAT64"),
    bigquery.SchemaField("corr_zl_brl_90d", "FLOAT64"),
    bigquery.SchemaField("corr_zl_dxy_30d", "FLOAT64"),
    bigquery.SchemaField("corr_zl_dxy_60d", "FLOAT64"),
    bigquery.SchemaField("corr_zl_dxy_90d", "FLOAT64"),
    bigquery.SchemaField("terms_of_trade_zl_brl", "FLOAT64"),
    bigquery.SchemaField("boho_spread", "FLOAT64"),
    bigquery.SchemaField("target_1w", "FLOAT64"),
    bigquery.SchemaField("target_1m", "FLOAT64"),
    bigquery.SchemaField("target_3m", "FLOAT64"),
    bigquery.SchemaField("target_6m", "FLOAT64"),
    bigquery.SchemaField("target_12m", "FLOAT64"),
]


def main(parquet_path: str, table_id: str, write_disposition: str):
    """Load parquet to BigQuery with partitioning and clustering."""
    
    client = bigquery.Client(project=PROJECT_ID)
    df = pd.read_parquet(parquet_path)
    
    print(f"[load_daily_ml_matrix] Loaded {len(df):,} rows from {parquet_path}")
    
    # Validate required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"[load_daily_ml_matrix] Missing required columns: {missing}")
    
    # Ensure proper types for partition/cluster keys
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['symbol'] = df['symbol'].astype(str)
    df['regime'] = df['regime'].astype(str)
    
    # Use MONTH partitioning to avoid 4000 partition limit with multi-year data
    job_config = bigquery.LoadJobConfig(
        schema=LOCKED_SCHEMA,
        write_disposition=write_disposition,
        time_partitioning=bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.MONTH,
            field="date",
        ),
        clustering_fields=["symbol", "regime"],
    )
    
    print(f"[load_daily_ml_matrix] Loading to {table_id}...")
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    
    print(f"\n✅ [load_daily_ml_matrix] Loaded {job.output_rows:,} rows")
    print(f"   Table: {table_id}")
    print(f"   Partitioned by: DATE_TRUNC(date, MONTH)")
    print(f"   Clustered by: symbol, regime")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Load feature matrix to BigQuery")
    ap.add_argument("--parquet", required=True, help="Path to input parquet file")
    ap.add_argument("--table", default=DEFAULT_TABLE, help="BigQuery table ID")
    ap.add_argument("--write-disposition", default="WRITE_TRUNCATE",
                    choices=["WRITE_TRUNCATE", "WRITE_APPEND"],
                    help="Write disposition (WRITE_TRUNCATE for full refresh)")
    args = ap.parse_args()
    main(args.parquet, args.table, args.write_disposition)
