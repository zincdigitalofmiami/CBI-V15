#!/usr/bin/env python3
"""Check API keys and write results to workspace file"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cbi_utils.keychain_manager import get_api_key

keys = {
    'DATABENTO_API_KEY': 'Databento',
    'SCRAPECREATORS_API_KEY': 'ScrapeCreators',
    'FRED_API_KEY': 'FRED',
    'GLIDE_BEARER_TOKEN': 'Glide API'
}

output_file = Path(__file__).parent.parent / 'API_KEYS_STATUS.txt'

results = {}
with open(output_file, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("API KEY STATUS CHECK\n")
    f.write("=" * 60 + "\n\n")
    
    found_count = 0
    missing_count = 0
    
    for key_name, display_name in keys.items():
        value = get_api_key(key_name)
        found = bool(value)
        results[key_name] = found
        
        if found:
            # Show first 8 chars and last 4 chars for security
            masked = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            status = f"✅ {display_name:20} ({key_name:25}) - Found: {masked}"
            found_count += 1
        else:
            status = f"❌ {display_name:20} ({key_name:25}) - Missing"
            missing_count += 1
        
        f.write(status + "\n")
        print(status)  # Also try stdout
    
    f.write("\n" + "=" * 60 + "\n")
    f.write(f"Summary: {found_count} found, {missing_count} missing\n")
    f.write("=" * 60 + "\n")
    
    if missing_count > 0:
        f.write("\nTo store missing keys, run:\n")
        f.write("  ./scripts/setup/store_api_keys.sh\n")
    else:
        f.write("\n✅ All required API keys are available!\n")

print(f"\nResults written to: {output_file}")






