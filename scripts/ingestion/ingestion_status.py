#!/usr/bin/env python3
"""
Check ingestion status and data freshness
"""
from google.cloud import bigquery
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ID = "cbi-v15"

def check_ingestion_status():
    """Check status of all ingestion sources"""
    client = bigquery.Client(project=PROJECT_ID)
    
    logger.info("📊 Ingestion Status Check")
    logger.info("=" * 60)
    
    # Check raw tables
    raw_tables = {
        "Databento": "raw.databento_futures_ohlcv_1d",
        "FRED": "raw.fred_economic",
        "ScrapeCreators News": "raw.scrapecreators_news_buckets",
        "ScrapeCreators Trump": "raw.scrapecreators_trump_posts",
    }
    
    logger.info("\n📥 Raw Layer:")
    for source, table_id in raw_tables.items():
        try:
            query = f"""
            SELECT 
                COUNT(*) as row_count,
                MIN(date) as min_date,
                MAX(date) as max_date
            FROM `{PROJECT_ID}.{table_id}`
            """
            result = client.query(query).to_dataframe()
            
            if not result.empty and result['row_count'].iloc[0] > 0:
                count = result['row_count'].iloc[0]
                max_date = result['max_date'].iloc[0]
                days_old = (datetime.now().date() - max_date).days if max_date else None
                
                status = "✅" if days_old is None or days_old <= 2 else "⚠️"
                logger.info(f"  {status} {source}: {count:,} rows, latest: {max_date} ({days_old} days ago)")
            else:
                logger.info(f"  ⚠️  {source}: No data")
        except Exception as e:
            logger.warning(f"  ❌ {source}: Error - {str(e)[:50]}")
    
    # Check staging tables
    logger.info("\n🔄 Staging Layer:")
    staging_tables = {
        "Market Daily": "staging.market_daily",
        "FRED Clean": "staging.fred_macro_clean",
        "News Bucketed": "staging.news_bucketed",
    }
    
    for table_name, table_id in staging_tables.items():
        try:
            query = f"SELECT COUNT(*) as count FROM `{PROJECT_ID}.{table_id}`"
            result = client.query(query).to_dataframe()
            count = result['count'].iloc[0] if not result.empty else 0
            
            if count > 0:
                logger.info(f"  ✅ {table_name}: {count:,} rows")
            else:
                logger.info(f"  ⚠️  {table_name}: Empty (run Dataform staging)")
        except Exception as e:
            logger.warning(f"  ❌ {table_name}: Error - {str(e)[:50]}")
    
    # Check features
    logger.info("\n🎯 Features Layer:")
    try:
        query = f"SELECT COUNT(*) as count FROM `{PROJECT_ID}.features.daily_ml_matrix`"
        result = client.query(query).to_dataframe()
        count = result['count'].iloc[0] if not result.empty else 0
        
        if count > 0:
            logger.info(f"  ✅ Daily ML Matrix: {count:,} rows")
        else:
            logger.info(f"  ⚠️  Daily ML Matrix: Empty (run Dataform features)")
    except Exception as e:
        logger.warning(f"  ❌ Daily ML Matrix: Error - {str(e)[:50]}")
    
    logger.info("\n" + "=" * 60)
    logger.info("📋 Recommendations:")
    logger.info("  - Check API keys: ./scripts/setup/verify_api_keys.sh")
    logger.info("  - Run ingestion: python3 src/ingestion/databento/collect_daily.py")
    logger.info("  - Run Dataform: cd dataform && npx dataform run --tags staging")

if __name__ == "__main__":
    check_ingestion_status()

