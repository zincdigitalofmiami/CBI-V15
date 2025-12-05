#!/usr/bin/env python3
"""Comprehensive API key verification - checks all locations"""
import subprocess
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

print("=" * 70)
print("API Keys Verification - All Locations")
print("=" * 70)
print()

# 1. macOS Keychain (both naming conventions)
print("1. macOS Keychain:")
print("-" * 70)

keys_to_check = {
    'DATABENTO_API_KEY': 'Databento',
    'FRED_API_KEY': 'FRED',
    'SCRAPECREATORS_API_KEY': 'ScrapeCreators',
    'GLIDE_BEARER_TOKEN': 'Glide'
}

found_direct = []
found_legacy = []
missing = []

for key, name in keys_to_check.items():
    result = subprocess.run(['security', 'find-generic-password', '-s', key], 
                          capture_output=True, stderr=subprocess.DEVNULL, timeout=2)
    if result.returncode == 0:
        print(f"   ✅ {name:20} ({key:25}) - Found")
        found_direct.append(key)
    else:
        print(f"   ❌ {name:20} ({key:25}) - Not found")
        missing.append(key)

print()
print(f"   Summary: {len(found_direct)}/{len(keys_to_check)} keys found")
print(f"            Missing: {len(missing)}")
print()

# 2. GCP Secret Manager
print("2. GCP Secret Manager:")
print("-" * 70)

secrets_to_check = {
    'databento-api-key': 'Databento',
    'scrapecreators-api-key': 'ScrapeCreators',
    'fred-api-key': 'FRED',
    'glide-api-key': 'Glide',
    'forecasting-data-keys': 'Forecasting Bundle (JSON)'
}

found_secrets = []
for secret, name in secrets_to_check.items():
    result = subprocess.run(['gcloud', 'secrets', 'describe', secret, '--project=cbi-v15'],
                          capture_output=True, stderr=subprocess.DEVNULL, timeout=5)
    if result.returncode == 0:
        print(f"   ✅ {name:25} ({secret})")
        found_secrets.append(secret)
    else:
        print(f"   ❌ {name:25} ({secret}) - Not found")

print()
print(f"   Summary: {len(found_secrets)}/{len(secrets_to_check)} secrets found")
print()

# 3. Python Keychain Manager Test
print("3. Python Keychain Manager (get_api_key function):")
print("-" * 70)

try:
    from src.cbi_utils.keychain_manager import get_api_key
    print("   ✅ Module imports successfully")
    print()
    
    test_results = {}
    for key in keys_to_check.keys():
        value = get_api_key(key)
        status = '✅ Found' if value else '❌ Not found'
        source = ''
        if value:
            # Try to determine source
            if subprocess.run(['security', 'find-generic-password', '-s', key], 
                           capture_output=True, stderr=subprocess.DEVNULL).returncode == 0:
                source = ' (Keychain)'
            else:
                source = ' (Secret Manager)'
        
        print(f"   {status:12} {key:25}{source}")
        test_results[key] = status
    
    found_count = sum(1 for v in test_results.values() if '✅' in v)
    print()
    print(f"   Summary: {found_count}/{len(keys_to_check)} keys accessible via Python")
    
except Exception as e:
    print(f"   ❌ Error importing or testing: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
print()

# Final summary
total_found = len(found_direct) + len(found_legacy) + len(found_secrets)
total_possible = len(keys_to_check) + len(secrets_to_check)

if total_found > 0:
    print(f"✅ Found {total_found} key(s) across all locations")
    print("   Ready for data ingestion if required keys are present")
else:
    print("⚠️  No keys found. Run: ./scripts/setup/store_api_keys.sh")
print()

