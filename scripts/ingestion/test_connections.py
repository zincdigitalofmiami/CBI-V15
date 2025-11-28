#!/usr/bin/env python3
"""
Test connections to all data sources and BigQuery
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from google.cloud import bigquery
try:
    from src.utils.keychain_manager import get_api_key
except ImportError:
    # Fallback if module not found
    def get_api_key(key_name: str):
        return None
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = "cbi-v15"

def test_bigquery():
    """Test BigQuery connection"""
    try:
        client = bigquery.Client(project=PROJECT_ID)
        datasets = list(client.list_datasets())
        logger.info(f"✅ BigQuery connected: {len(datasets)} datasets found")
        return True
    except Exception as e:
        logger.error(f"❌ BigQuery connection failed: {e}")
        return False

def test_databento_key():
    """Test Databento API key retrieval"""
    try:
        key = get_api_key("DATABENTO_API_KEY")
        if key:
            logger.info("✅ Databento API key found in Keychain")
            return True
        else:
            logger.warning("⚠️  Databento API key not found (run store_api_keys.sh)")
            return False
    except Exception as e:
        logger.warning(f"⚠️  Databento key check: {e}")
        return False

def test_scrapecreators_key():
    """Test ScrapeCreators API key retrieval"""
    try:
        key = get_api_key("SCRAPECREATORS_API_KEY")
        if key:
            logger.info("✅ ScrapeCreators API key found in Keychain")
            return True
        else:
            logger.warning("⚠️  ScrapeCreators API key not found (run store_api_keys.sh)")
            return False
    except Exception as e:
        logger.warning(f"⚠️  ScrapeCreators key check: {e}")
        return False

def test_fred_key():
    """Test FRED API key retrieval"""
    try:
        key = get_api_key("FRED_API_KEY")
        if key:
            logger.info("✅ FRED API key found in Keychain")
            return True
        else:
            logger.warning("⚠️  FRED API key not found (optional)")
            return False
    except Exception as e:
        logger.warning(f"⚠️  FRED key check: {e}")
        return False

def main():
    """Run all connection tests"""
    logger.info("🔍 Testing CBI-V15 Connections")
    logger.info("=" * 50)
    
    results = {
        "BigQuery": test_bigquery(),
        "Databento Key": test_databento_key(),
        "ScrapeCreators Key": test_scrapecreators_key(),
        "FRED Key": test_fred_key(),
    }
    
    logger.info("")
    logger.info("=" * 50)
    logger.info("📊 Test Results:")
    for name, result in results.items():
        status = "✅" if result else "⚠️"
        logger.info(f"  {status} {name}")
    
    all_critical = results["BigQuery"]
    if all_critical:
        logger.info("")
        logger.info("✅ Critical connections working!")
        logger.info("⚠️  API keys can be added later with store_api_keys.sh")
    else:
        logger.error("")
        logger.error("❌ Critical connections failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

