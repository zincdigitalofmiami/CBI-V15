#!/usr/bin/env python3
"""Quick training status check - run anytime to see progress"""

import os
import time
from pathlib import Path
from datetime import datetime

BUCKETS = ["crush", "china", "fx", "fed", "tariff", "biofuel", "energy", "volatility"]
HORIZONS = ["1w", "1m", "3m", "6m"]
TOTAL = 32


def main():
    base_path = Path("/Volumes/Satechi Hub/CBI-V15/models/bucket_specialists")

    print("=" * 80)
    print(
        f"🚀 BUCKET SPECIALIST TRAINING STATUS - {datetime.now().strftime('%H:%M:%S')}"
    )
    print("=" * 80)
    print()

    completed = []
    for bucket in BUCKETS:
        for horizon in HORIZONS:
            pkl = base_path / bucket / horizon / "predictor.pkl"
            if pkl.exists():
                completed.append((bucket, horizon, pkl.stat().st_mtime))

    progress = (len(completed) / TOTAL) * 100
    print(f"📊 Progress: {len(completed)}/{TOTAL} models ({progress:.1f}%)")
    print()

    # Progress bar
    filled = int(50 * len(completed) / TOTAL)
    bar = "█" * filled + "░" * (50 - filled)
    print(f"[{bar}] {progress:.1f}%")
    print()

    # Bucket status
    print("Bucket Status:")
    for bucket in BUCKETS:
        bucket_models = [c for c in completed if c[0] == bucket]
        status = f"{len(bucket_models)}/4"
        emoji = (
            "✅"
            if len(bucket_models) == 4
            else "🔄" if len(bucket_models) > 0 else "⏳"
        )
        horizons_done = [c[1] for c in bucket_models]
        print(f"  {emoji} {bucket:12s}: {status}  [{', '.join(horizons_done)}]")

    if completed:
        latest = max(completed, key=lambda x: x[2])
        time_ago = int(time.time() - latest[2])
        print()
        print(f"Latest: {latest[0]} {latest[1]} ({time_ago//60}m {time_ago%60}s ago)")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()


