#!/usr/bin/env python3
"""
Test connections to all data sources and MotherDuck
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MOTHERDUCK_TOKEN = os.getenv('MOTHERDUCK_TOKEN')
MOTHERDUCK_DB = os.getenv('MOTHERDUCK_DB', 'cbi_v15')

def test_motherduck():
    """Test MotherDuck connection"""
    try:
        import duckdb
        if not MOTHERDUCK_TOKEN:
            logger.warning("⚠️  MOTHERDUCK_TOKEN not set")
            return False
        
        conn = duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")
        schemas = conn.execute("SELECT schema_name FROM information_schema.schemata").fetchall()
        conn.close()
        logger.info(f"✅ MotherDuck connected: {len(schemas)} schemas found")
        return True
    except Exception as e:
        logger.error(f"❌ MotherDuck connection failed: {e}")
        return False

def test_env_var(name: str, required: bool = True):
    """Test if environment variable is set"""
    value = os.getenv(name)
    if value:
        # Mask the value for security
        masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
        logger.info(f"✅ {name} is set ({masked})")
        return True
    else:
        if required:
            logger.warning(f"⚠️  {name} not set (required)")
        else:
            logger.info(f"ℹ️  {name} not set (optional)")
        return not required

def test_keychain():
    """Test Keychain manager"""
    try:
        from src.utils.keychain_manager import get_api_key
        # Just test the import works
        logger.info("✅ Keychain manager importable")
        return True
    except ImportError as e:
        logger.warning(f"⚠️  Keychain manager not available: {e}")
        return False

def test_databento_key():
    """Test Databento API key"""
    key = os.getenv('DATABENTO_API_KEY')
    if key:
        logger.info("✅ DATABENTO_API_KEY found in environment")
        return True
    
    # Try keychain
    try:
        from src.utils.keychain_manager import get_api_key
        key = get_api_key("DATABENTO_API_KEY")
        if key:
            logger.info("✅ DATABENTO_API_KEY found in Keychain")
            return True
    except:
        pass
    
    logger.warning("⚠️  DATABENTO_API_KEY not found")
    return False

def test_openai_key():
    """Test OpenAI API key"""
    key = os.getenv('OPENAI_API_KEY')
    if key:
        logger.info("✅ OPENAI_API_KEY found in environment")
        return True
    
    logger.info("ℹ️  OPENAI_API_KEY not set (optional unless using AutoGluon)")
    return False

def main():
    """Run all connection tests"""
    logger.info("🔍 Testing CBI-V15 Connections")
    logger.info("=" * 50)
    
    results = {
        "MotherDuck": test_motherduck(),
        "MOTHERDUCK_TOKEN": test_env_var("MOTHERDUCK_TOKEN"),
        "Keychain Manager": test_keychain(),
        "Databento Key": test_databento_key(),
        "OpenAI Key": test_openai_key(),
    }
    
    logger.info("")
    logger.info("=" * 50)
    logger.info("📊 Test Results:")
    for name, result in results.items():
        status = "✅" if result else "⚠️"
        logger.info(f"  {status} {name}")
    
    critical_ok = results["MotherDuck"] and results["MOTHERDUCK_TOKEN"]
    if critical_ok:
        logger.info("")
        logger.info("✅ Critical connections working!")
    else:
        logger.error("")
        logger.error("❌ Critical connections failed!")
        logger.error("   Set MOTHERDUCK_TOKEN environment variable")
        sys.exit(1)

if __name__ == "__main__":
    main()

