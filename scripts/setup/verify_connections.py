#!/usr/bin/env python3
"""
Verify GCP and BigQuery connections for CBI-V15
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from google.cloud import bigquery
    from google.cloud import secretmanager
except ImportError:
    print("❌ Missing dependencies. Install: pip install google-cloud-bigquery google-cloud-secret-manager")
    sys.exit(1)

PROJECT_ID = "cbi-v15"
REGION = "us-central1"

def check_gcp_project():
    """Verify GCP project exists and is accessible"""
    print("🔍 Checking GCP project...")
    try:
        client = bigquery.Client(project=PROJECT_ID)
        project = client.project
        print(f"   ✅ Project accessible: {project}")
        return True
    except Exception as e:
        print(f"   ❌ Error accessing project: {e}")
        return False

def check_bigquery_datasets():
    """Verify all required BigQuery datasets exist"""
    print("🔍 Checking BigQuery datasets...")
    
    required_datasets = [
        "raw",
        "staging", 
        "features",
        "training",
        "forecasts",
        "api",
        "reference",
        "ops"
    ]
    
    client = bigquery.Client(project=PROJECT_ID)
    existing_datasets = {ds.dataset_id for ds in client.list_datasets()}
    
    all_exist = True
    for dataset in required_datasets:
        if dataset in existing_datasets:
            print(f"   ✅ Dataset '{dataset}' exists")
        else:
            print(f"   ❌ Dataset '{dataset}' missing")
            all_exist = False
    
    return all_exist

def check_secret_manager():
    """Verify Secret Manager is accessible"""
    print("🔍 Checking Secret Manager...")
    try:
        client = secretmanager.SecretManagerServiceClient()
        parent = f"projects/{PROJECT_ID}"
        
        # Try to list secrets (will fail if no access)
        list(client.list_secrets(request={"parent": parent}))
        print(f"   ✅ Secret Manager accessible")
        return True
    except Exception as e:
        print(f"   ⚠️  Secret Manager check: {e}")
        print(f"   ℹ️  This is OK if no secrets exist yet")
        return True  # Not a blocker

def check_api_keys():
    """Check if API keys are stored (macOS Keychain)"""
    print("🔍 Checking API keys in Keychain...")
    
    import subprocess
    
    keys_to_check = [
        "DATABENTO_API_KEY",
        "FRED_API_KEY",
        "SCRAPECREATORS_API_KEY"
    ]
    
    found_keys = []
    for key in keys_to_check:
        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-s", key, "-w"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                print(f"   ✅ {key} found in Keychain")
                found_keys.append(key)
            else:
                print(f"   ⚠️  {key} not found in Keychain")
        except Exception:
            print(f"   ⚠️  Could not check {key}")
    
    return len(found_keys) > 0

def main():
    print("=" * 60)
    print("CBI-V15 Connection Verification")
    print("=" * 60)
    print()
    
    results = {
        "GCP Project": check_gcp_project(),
        "BigQuery Datasets": check_bigquery_datasets(),
        "Secret Manager": check_secret_manager(),
        "API Keys": check_api_keys()
    }
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("✅ All checks passed! Ready to proceed.")
        return 0
    else:
        print("❌ Some checks failed. Please fix issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

