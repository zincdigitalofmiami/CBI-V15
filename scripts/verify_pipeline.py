"""
Pipeline verification script (TSci-free).

Checks that:
- AnofoxBridge can connect to DuckDB/MotherDuck.
- Core feature matrix exists and is readable.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.engines.anofox.anofox_bridge import AnofoxBridge  # type: ignore


def verify() -> None:
    """Verify core pipeline connectivity without TSci dependencies."""
    print("Verifying Pipeline...")

    # 1. Initialize Bridge (Local or MotherDuck)
    bridge = AnofoxBridge()
    print("✅ Bridge Initialized")

    # 2. Check Data Access
    try:
        df = bridge.conn.execute(
            "SELECT * FROM features.daily_ml_matrix_zl_v15 LIMIT 5"
        ).df()
        print(f"✅ Read {len(df)} rows from features matrix")
    except Exception as exc:
        print(f"❌ Failed to read features: {exc}")
        return

    print("✅ Pipeline Verification Complete.")


if __name__ == "__main__":
    verify()
