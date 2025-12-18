#!/usr/bin/env python3
"""
Live Training Monitor - Watch bucket specialist training in real-time
"""

import os
import time
from pathlib import Path
from datetime import datetime
import subprocess

BUCKETS = ["crush", "china", "fx", "fed", "tariff", "biofuel", "energy", "volatility"]
HORIZONS = ["1w", "1m", "3m", "6m"]
TOTAL_MODELS = 32


def clear_screen():
    os.system("clear" if os.name != "nt" else "cls")


def get_completed_models():
    """Find all completed models"""
    base_path = Path("/Volumes/Satechi Hub/CBI-V15/models/bucket_specialists")
    completed = []

    for bucket in BUCKETS:
        for horizon in HORIZONS:
            predictor_file = base_path / bucket / horizon / "predictor.pkl"
            if predictor_file.exists():
                completed.append((bucket, horizon, predictor_file.stat().st_mtime))

    return sorted(completed, key=lambda x: x[2])


def format_time_ago(timestamp):
    """Format timestamp as 'X min ago'"""
    seconds_ago = time.time() - timestamp
    if seconds_ago < 60:
        return f"{int(seconds_ago)}s ago"
    elif seconds_ago < 3600:
        return f"{int(seconds_ago/60)}m ago"
    else:
        return f"{int(seconds_ago/3600)}h {int((seconds_ago % 3600)/60)}m ago"


def get_process_info():
    """Check if training process is running"""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        for line in result.stdout.split("\n"):
            if "bucket_specialist.py" in line and "python" in line:
                parts = line.split()
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                time_running = parts[9]
                return {"pid": pid, "cpu": cpu, "mem": mem, "time": time_running}
    except:
        pass
    return None


def monitor():
    start_time = time.time()
    iteration = 0

    while True:
        iteration += 1
        clear_screen()

        # Header
        print("=" * 80)
        print("🚀 L0 BUCKET SPECIALIST TRAINING - LIVE MONITOR")
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print()

        # Process Info
        proc_info = get_process_info()
        if proc_info:
            print(f"✅ Training Process ACTIVE")
            print(f"   PID: {proc_info['pid']}")
            print(f"   CPU: {proc_info['cpu']}%")
            print(f"   Memory: {proc_info['mem']}%")
            print(f"   Runtime: {proc_info['time']}")
        else:
            print("⚠️  Training process not detected")

        print()
        print("=" * 80)

        # Get completed models
        completed = get_completed_models()
        progress = (len(completed) / TOTAL_MODELS) * 100

        print(f"📊 PROGRESS: {len(completed)}/{TOTAL_MODELS} models ({progress:.1f}%)")
        print("=" * 80)
        print()

        # Progress bar
        bar_length = 50
        filled = int(bar_length * len(completed) / TOTAL_MODELS)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"[{bar}] {progress:.1f}%")
        print()

        # Bucket breakdown
        print("🧊 BUCKET STATUS:")
        print("-" * 80)
        for bucket in BUCKETS:
            bucket_models = [c for c in completed if c[0] == bucket]
            status = f"{len(bucket_models)}/4"
            emoji = (
                "✅"
                if len(bucket_models) == 4
                else "🔄" if len(bucket_models) > 0 else "⏳"
            )
            print(f"  {emoji} {bucket:12s}: {status:4s} ", end="")

            # Show which horizons
            completed_horizons = [c[1] for c in bucket_models]
            for h in HORIZONS:
                if h in completed_horizons:
                    print(f"✓{h} ", end="")
                else:
                    print(f"○{h} ", end="")
            print()

        print()

        # Recently completed
        if completed:
            print("🏁 RECENTLY COMPLETED:")
            print("-" * 80)
            recent = completed[-5:] if len(completed) > 5 else completed
            for bucket, horizon, timestamp in reversed(recent):
                time_str = format_time_ago(timestamp)
                print(f"  {bucket:12s} {horizon:4s} - {time_str}")
        else:
            print("⏳ Waiting for first model to complete...")

        print()

        # Time estimates
        if len(completed) >= 2:
            elapsed = time.time() - start_time
            avg_time_per_model = elapsed / len(completed)
            remaining_models = TOTAL_MODELS - len(completed)
            est_remaining_sec = avg_time_per_model * remaining_models
            est_remaining_min = int(est_remaining_sec / 60)

            print("⏱️  TIME ESTIMATE:")
            print("-" * 80)
            print(f"  Avg per model:    {avg_time_per_model/60:.1f} minutes")
            print(
                f"  Est. remaining:   {est_remaining_min} minutes (~{est_remaining_min/60:.1f} hours)"
            )
            print(
                f"  Est. completion:  {datetime.fromtimestamp(time.time() + est_remaining_sec).strftime('%H:%M:%S')}"
            )
            print()

        print("=" * 80)
        print(f"Press Ctrl+C to exit | Refresh #{iteration} | Next update in 10s...")
        print("=" * 80)

        # Sleep
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            clear_screen()
            print("\n✋ Monitoring stopped by user")
            print(f"\nFinal Status: {len(completed)}/{TOTAL_MODELS} models completed")
            break


if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped")


