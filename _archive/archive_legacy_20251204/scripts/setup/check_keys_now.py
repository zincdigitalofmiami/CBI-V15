#!/usr/bin/env python3
"""Quick check of API keys - outputs to stdout and writes debug logs."""
import sys
import os
import json
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from cbi_utils.keychain_manager import get_api_key  # noqa: E402


# region agent log
DEBUG_LOG_PATH = "/Volumes/Satechi Hub/CBI-V15/.cursor/debug.log"
INGEST_ENDPOINT = "http://127.0.0.1:7242/ingest/7e84ff69-2ec3-4546-aa0d-a5191b254e43"


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    """Append a single NDJSON debug log line; never raises."""
    try:
        base_dir = os.path.dirname(DEBUG_LOG_PATH)
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

        payload = {
            "sessionId": "debug-session",
            "runId": os.getenv("CBI_DEBUG_RUN_ID", "pre-fix"),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(payload) + "\n")

        # Also send to ingest server so logs show up in central debug file
        try:
            req = urllib.request.Request(
                INGEST_ENDPOINT,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            # Fire-and-forget; ignore body
            urllib.request.urlopen(req, timeout=0.5).read()
        except Exception:
            # Never fail on logging
            pass
    except Exception:
        # Logging must never break runtime
        pass


# endregion

keys = {
    'DATABENTO_API_KEY': 'Databento',
    'SCRAPECREATORS_API_KEY': 'ScrapeCreators',
    'FRED_API_KEY': 'FRED',
}

_agent_debug_log(
    hypothesis_id="H1_TERMINAL_NO_OUTPUT",
    location="scripts/setup/check_keys_now.py:main:start",
    message="check_keys_now script started",
    data={"keys": list(keys.keys())},
)

print("=" * 60)
print("API KEY STATUS CHECK")
print("=" * 60)
print()

found_count = 0
missing_count = 0

for key_name, display_name in keys.items():
    value = get_api_key(key_name)
    if value:
        # Show first 8 chars and last 4 chars for security
        masked = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
        print(f"✅ {display_name:20} ({key_name:25}) - Found: {masked}")
        found_count += 1
    else:
        print(f"❌ {display_name:20} ({key_name:25}) - Missing")
        missing_count += 1

_agent_debug_log(
    hypothesis_id="H2_KEY_SOURCES",
    location="scripts/setup/check_keys_now.py:main:summary",
    message="check_keys_now script completed",
    data={"found_count": found_count, "missing_count": missing_count},
)

print()
print("=" * 60)
print(f"Summary: {found_count} found, {missing_count} missing")
print("=" * 60)

if missing_count > 0:
    print()
    print("To store missing keys, run:")
    print("  ./scripts/setup/store_api_keys.sh")
    sys.exit(1)
else:
    print()
    print("✅ All required API keys are available!")
    sys.exit(0)

