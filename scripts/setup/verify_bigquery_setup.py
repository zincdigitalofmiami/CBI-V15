#!/usr/bin/env python3
"""
Verify BigQuery Setup - Ensures all datasets and tables are created correctly
"""

from google.cloud import bigquery
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = "cbi-v15"
LOCATION = "us-central1"

# Expected datasets
EXPECTED_DATASETS = [
    "raw",
    "staging",
    "features",
    "training",
    "forecasts",
    "api",
    "reference",
    "ops"
]

# Expected tables (42 total)
EXPECTED_TABLES = {
    "raw": [
        "databento_futures_ohlcv_1d",
        "fred_economic",
        "usda_reports",
        "cftc_cot",
        "eia_biofuels",
        "weather_noaa",
        "scrapecreators_trump",
        "scrapecreators_news_buckets"
    ],
    "staging": [
        "market_daily",
        "fred_macro_clean",
        "usda_reports_clean",
        "cftc_positions",
        "eia_biofuels_clean",
        "weather_regions_aggregated",
        "trump_policy_intelligence",
        "news_bucketed",
        "sentiment_buckets"
    ],
    "features": [
        "technical_indicators_us_oil_solutions",
        "fx_indicators_daily",
        "fundamental_spreads_daily",
        "pair_correlations_daily",
        "cross_asset_betas_daily",
        "lagged_features_daily",
        "daily_ml_matrix",
        "sentiment_features_daily",
        "regime_indicators_daily",
        "neural_signals_daily",
        "neural_master_score",
        "trump_news_features_daily"
    ],
    "reference": [
        "regime_calendar",
        "regime_weights",
        "neural_drivers",
        "train_val_test_splits"
    ],
    "ops": [
        "ingestion_completion"
    ],
    "training": [
        "zl_training_1w",
        "zl_training_1m",
        "zl_training_3m",
        "zl_training_6m"
    ],
    "forecasts": [
        "zl_predictions_1w",
        "zl_predictions_1m",
        "zl_predictions_3m",
        "zl_predictions_6m"
    ]
}


def verify_datasets(client: bigquery.Client) -> bool:
    """Verify all expected datasets exist"""
    logger.info("Verifying datasets...")
    
    datasets = list(client.list_datasets())
    dataset_ids = [d.dataset_id for d in datasets]
    
    missing = []
    for expected in EXPECTED_DATASETS:
        if expected not in dataset_ids:
            missing.append(expected)
            logger.error(f"❌ Missing dataset: {expected}")
        else:
            # Verify location
            dataset = client.get_dataset(expected)
            if dataset.location.lower() != LOCATION.lower():
                logger.error(f"❌ Dataset {expected} in wrong location: {dataset.location} (expected {LOCATION})")
                return False
            logger.info(f"✅ Dataset {expected} exists in {LOCATION}")
    
    if missing:
        logger.error(f"❌ Missing {len(missing)} datasets: {missing}")
        return False
    
    logger.info(f"✅ All {len(EXPECTED_DATASETS)} datasets verified")
    return True


def verify_tables(client: bigquery.Client) -> bool:
    """Verify all expected tables exist"""
    logger.info("Verifying tables...")
    
    all_passed = True
    total_tables = 0
    
    for dataset_id, expected_tables in EXPECTED_TABLES.items():
        logger.info(f"Checking dataset: {dataset_id}")
        
        tables = list(client.list_tables(dataset_id))
        table_ids = [t.table_id for t in tables]
        
        for expected_table in expected_tables:
            total_tables += 1
            if expected_table not in table_ids:
                logger.error(f"❌ Missing table: {dataset_id}.{expected_table}")
                all_passed = False
            else:
                # Verify partitioning and clustering
                table = client.get_table(f"{PROJECT_ID}.{dataset_id}.{expected_table}")
                
                # Check partitioning
                if table.time_partitioning:
                    partition_field = table.time_partitioning.field
                    logger.info(f"✅ Table {dataset_id}.{expected_table} partitioned by {partition_field}")
                else:
                    logger.warning(f"⚠️  Table {dataset_id}.{expected_table} not partitioned")
                
                # Check clustering
                if table.clustering_fields:
                    logger.info(f"✅ Table {dataset_id}.{expected_table} clustered by {table.clustering_fields}")
                
                logger.info(f"✅ Table {dataset_id}.{expected_table} exists")
    
    if all_passed:
        logger.info(f"✅ All {total_tables} tables verified")
    else:
        logger.error(f"❌ Some tables missing")
    
    return all_passed


def verify_reference_data(client: bigquery.Client) -> bool:
    """Verify reference tables are populated"""
    logger.info("Verifying reference data...")
    
    # Check regime_calendar
    query = "SELECT COUNT(*) as count FROM `cbi-v15.reference.regime_calendar`"
    result = client.query(query).result()
    count = next(result).count
    if count == 0:
        logger.error("❌ regime_calendar is empty")
        return False
    logger.info(f"✅ regime_calendar has {count} rows")
    
    # Check train_val_test_splits
    query = "SELECT COUNT(*) as count FROM `cbi-v15.reference.train_val_test_splits`"
    result = client.query(query).result()
    count = next(result).count
    if count == 0:
        logger.error("❌ train_val_test_splits is empty")
        return False
    logger.info(f"✅ train_val_test_splits has {count} rows")
    
    # Check neural_drivers
    query = "SELECT COUNT(*) as count FROM `cbi-v15.reference.neural_drivers`"
    result = client.query(query).result()
    count = next(result).count
    if count == 0:
        logger.error("❌ neural_drivers is empty")
        return False
    logger.info(f"✅ neural_drivers has {count} rows")
    
    return True


def main():
    """Main verification function"""
    logger.info(f"🔍 Verifying BigQuery setup for project: {PROJECT_ID} in location: {LOCATION}")
    
    client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
    
    # Verify datasets
    if not verify_datasets(client):
        logger.error("❌ Dataset verification failed")
        return False
    
    # Verify tables
    if not verify_tables(client):
        logger.error("❌ Table verification failed")
        return False
    
    # Verify reference data
    if not verify_reference_data(client):
        logger.error("❌ Reference data verification failed")
        return False
    
    logger.info("✅ BigQuery setup verification complete - All checks passed!")
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

