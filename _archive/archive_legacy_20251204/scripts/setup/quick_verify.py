#!/usr/bin/env python3
"""Quick verification script for CBI-V15 connections"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

print("=" * 60)
print("CBI-V15 Quick Verification")
print("=" * 60)
print()

# 1. API Keys
print("1. API Keys (macOS Keychain):")
keys = ['DATABENTO_API_KEY', 'SCRAPECREATORS_API_KEY', 'FRED_API_KEY', 'GLIDE_API_KEY']
found = 0
for key in keys:
    try:
        result = subprocess.run(['security', 'find-generic-password', '-s', key], 
                              capture_output=True, stderr=subprocess.DEVNULL, timeout=2)
        if result.returncode == 0:
            print(f"   ✅ {key}")
            found += 1
        else:
            print(f"   ❌ {key}")
    except:
        print(f"   ⚠️  {key} (check failed)")
print(f"   Found: {found}/{len(keys)}")
print()

# 2. BigQuery
print("2. BigQuery Connection:")
try:
    from google.cloud import bigquery
    client = bigquery.Client(project='cbi-v15')
    datasets = list(client.list_datasets())
    print(f"   ✅ Connected - {len(datasets)} datasets")
    for d in sorted(datasets, key=lambda x: x.dataset_id):
        print(f"      - {d.dataset_id}")
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

# 3. Dataform
print("3. Dataform Repository:")
try:
    result = subprocess.run(['gcloud', 'dataform', 'repositories', 'list', 
                            '--project=cbi-v15', '--location=us-central1', '--format=json'],
                           capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        import json
        repos = json.loads(result.stdout) if result.stdout.strip() else []
        if repos:
            print(f"   ✅ {len(repos)} repository(ies) found")
            for repo in repos:
                name = repo.get('name', 'unknown')
                remote = repo.get('gitRemoteSettings', {}).get('remoteUri', 'not set')
                print(f"      - {name}: {remote}")
        else:
            print("   ⚠️  No repositories found")
    else:
        print(f"   ⚠️  Error: {result.stderr}")
except Exception as e:
    print(f"   ⚠️  Error: {e}")
print()

# 4. Python Utils
print("4. Python Utilities:")
try:
    from src.cbi_utils.keychain_manager import get_api_key
    from src.cbi_utils.bigquery_client import get_client
    print("   ✅ Utils importable")
except Exception as e:
    print(f"   ❌ Error: {e}")
print()

print("=" * 60)
if found >= 2:
    print("✅ Ready for data ingestion")
else:
    print("⚠️  Store API keys: ./scripts/setup/store_api_keys.sh")
print("=" * 60)






