#!/usr/bin/env python3
"""
Check ingestion status and data freshness in MotherDuck
"""
import os
import duckdb
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MOTHERDUCK_TOKEN = os.getenv('MOTHERDUCK_TOKEN')
MOTHERDUCK_DB = os.getenv('MOTHERDUCK_DB', 'cbi_v15')

def get_connection():
    """Get MotherDuck connection"""
    if not MOTHERDUCK_TOKEN:
        raise ValueError("MOTHERDUCK_TOKEN environment variable not set")
    return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")

def check_ingestion_status():
    """Check status of all ingestion sources"""
    try:
        conn = get_connection()
        logger.info(f"✅ Connected to MotherDuck ({MOTHERDUCK_DB})")
    except Exception as e:
        logger.error(f"❌ Failed to connect: {e}")
        return
    
    logger.info("📊 Ingestion Status Check")
    logger.info("=" * 60)
    
    # Check raw tables
    raw_tables = {
        "Databento": "raw.databento_futures_ohlcv_1d",
        "FRED": "raw.fred_economic",
        "ScrapeCreators News": "raw.scrapecreators_news_buckets",
        "ScrapeCreators Trump": "raw.scrapecreators_trump",
        "EIA Biofuels": "raw.eia_biofuels",
        "CFTC COT": "raw.cftc_cot",
    }
    
    logger.info("\n📥 Raw Layer:")
    for source, table_id in raw_tables.items():
        try:
            # Use as_of_date for Databento, report_date for CFTC, date for others
            date_col = "as_of_date" if "databento" in table_id else ("report_date" if "cftc" in table_id else "date")
            query = f"""
            SELECT 
                COUNT(*) as row_count,
                MIN({date_col}) as min_date,
                MAX({date_col}) as max_date
            FROM {table_id}
            """
            result = conn.execute(query).fetchone()
            
            if result and result[0] > 0:
                count, min_date, max_date = result
                days_old = (datetime.now().date() - max_date).days if max_date else None
                
                status = "✅" if days_old is None or days_old <= 2 else "⚠️"
                logger.info(f"  {status} {source}: {count:,} rows, latest: {max_date} ({days_old} days ago)")
            else:
                logger.info(f"  ⚠️  {source}: No data")
        except Exception as e:
            logger.warning(f"  ❌ {source}: {str(e)[:50]}")
    
    # Check staging tables
    logger.info("\n🔄 Staging Layer:")
    staging_tables = {
        "Market Daily": "staging.market_daily",
        "FRED Clean": "staging.fred_macro_clean",
        "News Bucketed": "staging.news_bucketed",
    }
    
    for table_name, table_id in staging_tables.items():
        try:
            query = f"SELECT COUNT(*) as count FROM {table_id}"
            result = conn.execute(query).fetchone()
            count = result[0] if result else 0
            
            if count > 0:
                logger.info(f"  ✅ {table_name}: {count:,} rows")
            else:
                logger.info(f"  ⚠️  {table_name}: Empty")
        except Exception as e:
            logger.warning(f"  ❌ {table_name}: {str(e)[:50]}")
    
    # Check features
    logger.info("\n🎯 Features Layer:")
    try:
        query = f"SELECT COUNT(*) as count FROM features.daily_ml_matrix_zl"
        result = conn.execute(query).fetchone()
        count = result[0] if result else 0
        
        if count > 0:
            logger.info(f"  ✅ Daily ML Matrix: {count:,} rows")
        else:
            logger.info(f"  ⚠️  Daily ML Matrix: Empty")
    except Exception as e:
        logger.warning(f"  ❌ Daily ML Matrix: {str(e)[:50]}")
    
    logger.info("\n" + "=" * 60)
    logger.info("📋 Recommendations:")
    logger.info("  - Run ingestion: python3 trigger/DataBento/Scripts/collect_daily.py")
    logger.info("  - Check env vars: MOTHERDUCK_TOKEN, DATABENTO_API_KEY")
    
    conn.close()

if __name__ == "__main__":
    check_ingestion_status()
