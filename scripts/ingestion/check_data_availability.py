#!/usr/bin/env python3
"""
Check data availability in BigQuery tables
Shows what data exists and what's missing
"""
from google.cloud import bigquery
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ID = "cbi-v15"

def check_table_data(dataset: str, table: str):
    """Check if table has data"""
    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{dataset}.{table}"
    
    try:
        query = f"SELECT COUNT(*) as count, MIN(date) as min_date, MAX(date) as max_date FROM `{table_id}`"
        result = client.query(query).to_dataframe()
        
        if not result.empty:
            count = result['count'].iloc[0]
            min_date = result['min_date'].iloc[0]
            max_date = result['max_date'].iloc[0]
            
            if count > 0:
                logger.info(f"✅ {dataset}.{table}: {count:,} rows ({min_date} to {max_date})")
                return True
            else:
                logger.warning(f"⚠️  {dataset}.{table}: Empty")
                return False
        else:
            logger.warning(f"⚠️  {dataset}.{table}: No data")
            return False
    except Exception as e:
        logger.error(f"❌ {dataset}.{table}: Error - {e}")
        return False

def main():
    """Check all critical tables"""
    logger.info("🔍 Checking Data Availability in BigQuery")
    logger.info("=" * 60)
    
    # Raw layer tables
    logger.info("\n📊 Raw Layer:")
    raw_tables = [
        ("raw", "databento_futures_ohlcv_1d"),
        ("raw", "fred_economic"),
        ("raw", "scrapecreators_trump_posts"),
        ("raw", "scrapecreators_news_buckets"),
    ]
    
    raw_has_data = False
    for dataset, table in raw_tables:
        if check_table_data(dataset, table):
            raw_has_data = True
    
    # Staging layer tables
    logger.info("\n📊 Staging Layer:")
    staging_tables = [
        ("staging", "market_daily"),
        ("staging", "fred_macro_clean"),
        ("staging", "news_bucketed"),
    ]
    
    staging_has_data = False
    for dataset, table in staging_tables:
        if check_table_data(dataset, table):
            staging_has_data = True
    
    # Features layer tables
    logger.info("\n📊 Features Layer:")
    feature_tables = [
        ("features", "daily_ml_matrix"),
        ("features", "technical_indicators_us_oil_solutions"),
    ]
    
    feature_has_data = False
    for dataset, table in feature_tables:
        if check_table_data(dataset, table):
            feature_has_data = True
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 Summary:")
    logger.info(f"  Raw Layer: {'✅ Has data' if raw_has_data else '⚠️  No data (ready for ingestion)'}")
    logger.info(f"  Staging Layer: {'✅ Has data' if staging_has_data else '⚠️  No data (run Dataform staging)'}")
    logger.info(f"  Features Layer: {'✅ Has data' if feature_has_data else '⚠️  No data (run Dataform features)'}")
    
    logger.info("\n🎯 Next Steps:")
    if not raw_has_data:
        logger.info("  1. Run data ingestion: python3 src/ingestion/databento/collect_daily.py")
    elif not staging_has_data:
        logger.info("  1. Run Dataform staging: cd dataform && npx dataform run --tags staging")
    elif not feature_has_data:
        logger.info("  1. Run Dataform features: cd dataform && npx dataform run --tags features")
    else:
        logger.info("  ✅ Data pipeline is operational!")

if __name__ == "__main__":
    main()

