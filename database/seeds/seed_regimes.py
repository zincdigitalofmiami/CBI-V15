"""
Seed: Regime Definitions
Seeds the regime calendar with historical classifications
"""
import duckdb
from datetime import datetime

# Regime thresholds
REGIME_THRESHOLDS = {
    "CALM": {"vix_max": 20, "vol_zscore_max": 0.5},
    "STRESSED": {"vix_max": 30, "vol_zscore_max": 1.5},
    "CRISIS": {"vix_min": 30, "vol_zscore_min": 1.5},
}


def seed_regime_weights(conn: duckdb.DuckDBPyConnection) -> int:
    """Insert default regime weights."""
    # Default weights by regime
    WEIGHTS = [
        # CALM regime - fundamentals matter more
        ("CALM", "1w", "crush", 0.18),
        ("CALM", "1w", "china", 0.12),
        ("CALM", "1w", "fx", 0.10),
        ("CALM", "1w", "fed", 0.10),
        ("CALM", "1w", "tariff", 0.08),
        ("CALM", "1w", "biofuel", 0.15),
        ("CALM", "1w", "energy", 0.12),
        ("CALM", "1w", "volatility", 0.15),
        
        # STRESSED regime - balanced
        ("STRESSED", "1w", "crush", 0.14),
        ("STRESSED", "1w", "china", 0.12),
        ("STRESSED", "1w", "fx", 0.12),
        ("STRESSED", "1w", "fed", 0.12),
        ("STRESSED", "1w", "tariff", 0.10),
        ("STRESSED", "1w", "biofuel", 0.12),
        ("STRESSED", "1w", "energy", 0.12),
        ("STRESSED", "1w", "volatility", 0.16),
        
        # CRISIS regime - risk/vol dominates
        ("CRISIS", "1w", "crush", 0.08),
        ("CRISIS", "1w", "china", 0.10),
        ("CRISIS", "1w", "fx", 0.14),
        ("CRISIS", "1w", "fed", 0.14),
        ("CRISIS", "1w", "tariff", 0.12),
        ("CRISIS", "1w", "biofuel", 0.08),
        ("CRISIS", "1w", "energy", 0.10),
        ("CRISIS", "1w", "volatility", 0.24),
    ]
    
    count = 0
    for regime, horizon, bucket, weight in WEIGHTS:
        conn.execute("""
            INSERT INTO reference.regime_weights (regime, horizon_code, bucket_name, weight)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (regime, horizon_code, bucket_name) DO UPDATE SET
                weight = EXCLUDED.weight,
                updated_at = CURRENT_TIMESTAMP
        """, [regime, horizon, bucket, weight])
        count += 1
    
    return count


if __name__ == "__main__":
    import os
    
    token = os.getenv("MOTHERDUCK_TOKEN")
    if token:
        conn = duckdb.connect(f"md:usoil_intelligence?motherduck_token={token}")
    else:
        conn = duckdb.connect("data/duckdb/cbi_v15.duckdb")
    
    count = seed_regime_weights(conn)
    print(f"Seeded {count} regime weights")
    conn.close()

