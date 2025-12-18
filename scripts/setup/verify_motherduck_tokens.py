#!/usr/bin/env python3
"""
MotherDuck Token Verification Script

Verifies that MotherDuck tokens are properly configured and accessible.
Checks multiple sources and provides detailed diagnostics.

Usage:
    python scripts/setup/verify_motherduck_tokens.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import duckdb
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   Install: pip install duckdb python-dotenv")
    sys.exit(1)


def check_env_file():
    """Check if .env file exists and is readable"""
    env_path = PROJECT_ROOT / ".env"
    print(f"\n📁 Checking .env file: {env_path}")

    if not env_path.exists():
        print("   ⚠️  .env file not found")
        return False

    if not env_path.is_file():
        print("   ⚠️  .env exists but is not a file")
        return False

    print(f"   ✅ .env file exists")
    return True


def load_env():
    """Load .env file"""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"   ✅ Loaded .env from {env_path}")
    else:
        print(f"   ⚠️  .env not found, using environment variables only")


def check_token_sources():
    """Check all possible token sources"""
    print("\n🔑 Checking Token Sources:")

    sources = {
        "MOTHERDUCK_TOKEN": os.getenv("MOTHERDUCK_TOKEN"),
        "motherduck_storage_MOTHERDUCK_TOKEN": os.getenv(
            "motherduck_storage_MOTHERDUCK_TOKEN"
        ),
        "MOTHERDUCK_READ_SCALING_TOKEN": os.getenv("MOTHERDUCK_READ_SCALING_TOKEN"),
        "motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN": os.getenv(
            "motherduck_storage_MOTHERDUCK_READ_SCALING_TOKEN"
        ),
    }

    found_tokens = {}
    for name, value in sources.items():
        if value:
            # Mask token for display
            masked = f"{value[:20]}...{value[-4:]}" if len(value) > 24 else "***"
            print(f"   ✅ {name}: {masked}")
            found_tokens[name] = value
        else:
            print(f"   ⚠️  {name}: Not set")

    return found_tokens


def get_primary_token(found_tokens):
    """Get primary token with fallback logic"""
    # Priority 1: MOTHERDUCK_TOKEN
    token = found_tokens.get("MOTHERDUCK_TOKEN")
    if token:
        return token.strip().strip('"').strip("'"), "MOTHERDUCK_TOKEN"

    # Priority 2: motherduck_storage_MOTHERDUCK_TOKEN
    token = found_tokens.get("motherduck_storage_MOTHERDUCK_TOKEN")
    if token:
        return (
            token.strip().strip('"').strip("'"),
            "motherduck_storage_MOTHERDUCK_TOKEN",
        )

    return None, None


def test_connection(token, token_source):
    """Test MotherDuck connection"""
    print(f"\n🔌 Testing Connection (using {token_source}):")

    db_name = os.getenv("MOTHERDUCK_DB", "cbi_v15")
    print(f"   Database: {db_name}")

    try:
        conn = duckdb.connect(f"md:{db_name}?motherduck_token={token}")

        # Test query
        result = conn.execute("PRAGMA database_list;").fetchall()
        print(f"   ✅ Connection successful!")
        print(f"   Databases: {result}")

        # Test read access
        try:
            test_query = conn.execute("SELECT 1 as test").fetchone()
            print(f"   ✅ Read access verified: {test_query}")
        except Exception as e:
            print(f"   ⚠️  Read test failed: {e}")

        conn.close()
        return True

    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False


def check_shell_config():
    """Check if tokens are in shell config"""
    print("\n🐚 Checking Shell Configuration:")

    shell_configs = [
        Path.home() / ".bashrc",
        Path.home() / ".zshrc",
        Path.home() / ".bash_profile",
        Path.home() / ".zprofile",
    ]

    found_configs = []
    for config_path in shell_configs:
        if config_path.exists():
            try:
                content = config_path.read_text()
                if (
                    "MOTHERDUCK_TOKEN" in content
                    or "motherduck_storage_MOTHERDUCK_TOKEN" in content
                ):
                    print(f"   ✅ Found tokens in: {config_path}")
                    found_configs.append(config_path)
                else:
                    print(f"   ⚠️  No tokens in: {config_path}")
            except Exception as e:
                print(f"   ⚠️  Could not read {config_path}: {e}")

    if not found_configs:
        print("   ⚠️  No shell config files contain MOTHERDUCK tokens")

    return found_configs


def main():
    """Main verification routine"""
    print("=" * 80)
    print("🔍 MOTHERDUCK TOKEN VERIFICATION")
    print("=" * 80)

    # Step 1: Check .env file
    env_exists = check_env_file()

    # Step 2: Load environment
    load_env()

    # Step 3: Check token sources
    found_tokens = check_token_sources()

    # Step 4: Get primary token
    token, token_source = get_primary_token(found_tokens)

    if not token:
        print("\n❌ ERROR: No MOTHERDUCK_TOKEN found!")
        print("\n📋 Setup Instructions:")
        print("   1. Add to .env file:")
        print("      MOTHERDUCK_TOKEN=your_token_here")
        print("   2. Or export in shell:")
        print("      export MOTHERDUCK_TOKEN=your_token_here")
        print("   3. See: docs/ops/MOTHERDUCK_SETUP.md")
        return 1

    # Step 5: Test connection
    connection_ok = test_connection(token, token_source)

    # Step 6: Check shell config
    shell_configs = check_shell_config()

    # Summary
    print("\n" + "=" * 80)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"   .env file: {'✅ Found' if env_exists else '⚠️  Missing'}")
    print(f"   Token source: {token_source if token else '❌ None'}")
    print(f"   Connection: {'✅ Success' if connection_ok else '❌ Failed'}")
    print(f"   Shell configs: {len(shell_configs)} found")

    if connection_ok:
        print("\n✅ All checks passed! MotherDuck is properly configured.")
        return 0
    else:
        print("\n❌ Connection test failed. Check token validity and network.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
