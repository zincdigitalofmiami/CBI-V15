import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.anofox.anofox_bridge import AnofoxBridge
from src.tsci.curator import CuratorAgent
from src.tsci.planner import create_job


def verify():
    print("Verifying Pipeline...")

    # 1. Initialize Bridge (Local)
    bridge = AnofoxBridge()
    print("✅ Bridge Initialized")

    # 2. Check Data Access
    try:
        df = bridge.conn.execute(
            "SELECT * FROM features.daily_ml_matrix_zl_v15 LIMIT 5"
        ).df()
        print(f"✅ Read {len(df)} rows from features matrix")
    except Exception as e:
        print(f"❌ Failed to read features: {e}")
        return

    # 3. Simulate TSci Agent
    curator = CuratorAgent(bridge)
    quality = curator.analyze_data_quality("features.daily_ml_matrix_zl_v15")
    print(f"✅ Curator Analysis: {quality}")

    print("Pipeline Verification Complete.")


if __name__ == "__main__":
    verify()
