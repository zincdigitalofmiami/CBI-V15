#!/usr/bin/env python3
"""
Check data availability in MotherDuck tables
Shows what data exists and what's missing
"""
import os
import duckdb
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi_v15")


def get_connection():
    """Get MotherDuck connection"""
    if not MOTHERDUCK_TOKEN:
        raise ValueError("MOTHERDUCK_TOKEN environment variable not set")
    return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")


def check_table_data(conn, schema: str, table: str):
    """Check if table has data"""
    try:
        # First check if table exists
        check_query = f"""
        SELECT COUNT(*) as exists 
        FROM information_schema.tables 
        WHERE table_schema = '{schema}' AND table_name = '{table}'
        """
        exists = conn.execute(check_query).fetchone()[0]

        if not exists:
            logger.warning(f"⚠️  {schema}.{table}: Table does not exist")
            return False

        # Check for data
        query = f"SELECT COUNT(*) as count FROM {schema}.{table}"
        result = conn.execute(query).fetchone()
        count = result[0] if result else 0

        if count > 0:
            # Try to get date range if date column exists
            try:
                date_query = f"SELECT MIN(date) as min_date, MAX(date) as max_date FROM {schema}.{table}"
                dates = conn.execute(date_query).fetchone()
                min_date, max_date = dates
                logger.info(
                    f"✅ {schema}.{table}: {count:,} rows ({min_date} to {max_date})"
                )
            except:
                logger.info(f"✅ {schema}.{table}: {count:,} rows")
            return True
        else:
            logger.warning(f"⚠️  {schema}.{table}: Empty")
            return False
    except Exception as e:
        logger.error(f"❌ {schema}.{table}: Error - {e}")
        return False


def main():
    """Check all critical tables"""
    logger.info("🔍 Checking Data Availability in MotherDuck")
    logger.info("=" * 60)

    try:
        conn = get_connection()
        logger.info(f"✅ Connected to MotherDuck ({MOTHERDUCK_DB})")
    except Exception as e:
        logger.error(f"❌ Failed to connect: {e}")
        return

    # Raw layer tables
    logger.info("\n📊 Raw Layer:")
    raw_tables = [
        ("raw", "databento_futures_ohlcv_1d"),
        ("raw", "fred_economic"),
        ("raw", "scrapecreators_trump"),
        ("raw", "scrapecreators_news_buckets"),
        ("raw", "eia_biofuels"),
    ]

    raw_has_data = False
    for schema, table in raw_tables:
        if check_table_data(conn, schema, table):
            raw_has_data = True

    # Staging layer tables
    logger.info("\n📊 Staging Layer:")
    staging_tables = [
        ("staging", "market_daily"),
        ("staging", "fred_macro_clean"),
        ("staging", "news_bucketed"),
    ]

    staging_has_data = False
    for schema, table in staging_tables:
        if check_table_data(conn, schema, table):
            staging_has_data = True

    # Features layer tables
    logger.info("\n📊 Features Layer:")
    feature_tables = [
        ("features", "daily_ml_matrix"),
    ]

    feature_has_data = False
    for schema, table in feature_tables:
        if check_table_data(conn, schema, table):
            feature_has_data = True

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 Summary:")
    logger.info(
        f"  Raw Layer: {'✅ Has data' if raw_has_data else '⚠️  No data (ready for ingestion)'}"
    )
    logger.info(
        f"  Staging Layer: {'✅ Has data' if staging_has_data else '⚠️  No data (run staging transforms)'}"
    )
    logger.info(
        f"  Features Layer: {'✅ Has data' if feature_has_data else '⚠️  No data (run feature transforms)'}"
    )

    logger.info("\n🎯 Next Steps:")
    if not raw_has_data:
        logger.info(
            "  1. Run data ingestion: python3 trigger/DataBento/Scripts/collect_daily.py"
        )
    elif not staging_has_data:
        logger.info("  1. Run staging transforms in MotherDuck")
    elif not feature_has_data:
        logger.info("  1. Run feature transforms in MotherDuck")
    else:
        logger.info("  ✅ Data pipeline is operational!")

    conn.close()


if __name__ == "__main__":
    main()
