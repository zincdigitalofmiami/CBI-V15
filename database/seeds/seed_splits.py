"""
Seed: Train/Validation/Test Splits
Seeds the canonical data splits for training
"""
import duckdb
from datetime import date

# Production splits
SPLITS = [
    {
        "split_id": "prod_v1",
        "split_name": "Production Split V1",
        "description": "5-year train, 1-year val, 5-day embargo",
        "train_start": date(2019, 1, 1),
        "train_end": date(2023, 12, 31),
        "embargo_days": 5,
        "val_start": date(2024, 1, 6),
        "val_end": date(2024, 12, 31),
        "test_start": None,
        "test_end": None,
    },
    {
        "split_id": "dev_v1",
        "split_name": "Development Split V1",
        "description": "3-year train, 6-month val for faster iteration",
        "train_start": date(2021, 1, 1),
        "train_end": date(2023, 12, 31),
        "embargo_days": 5,
        "val_start": date(2024, 1, 6),
        "val_end": date(2024, 6, 30),
        "test_start": None,
        "test_end": None,
    },
    {
        "split_id": "backtest_2023",
        "split_name": "2023 Backtest",
        "description": "Test on 2023 data",
        "train_start": date(2018, 1, 1),
        "train_end": date(2022, 12, 31),
        "embargo_days": 5,
        "val_start": date(2023, 1, 6),
        "val_end": date(2023, 12, 31),
        "test_start": None,
        "test_end": None,
    },
]


def seed_splits(conn: duckdb.DuckDBPyConnection) -> int:
    """Insert splits into reference.train_val_test_splits."""
    count = 0
    for split in SPLITS:
        conn.execute("""
            INSERT INTO reference.train_val_test_splits 
            (split_id, split_name, description, train_start, train_end, 
             embargo_days, val_start, val_end, test_start, test_end, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, TRUE)
            ON CONFLICT (split_id) DO UPDATE SET
                split_name = EXCLUDED.split_name,
                description = EXCLUDED.description,
                train_start = EXCLUDED.train_start,
                train_end = EXCLUDED.train_end,
                embargo_days = EXCLUDED.embargo_days,
                val_start = EXCLUDED.val_start,
                val_end = EXCLUDED.val_end
        """, [
            split["split_id"], split["split_name"], split["description"],
            split["train_start"], split["train_end"], split["embargo_days"],
            split["val_start"], split["val_end"], split["test_start"], split["test_end"]
        ])
        count += 1
    
    return count


if __name__ == "__main__":
    import os
    
    token = os.getenv("MOTHERDUCK_TOKEN")
    if token:
        conn = duckdb.connect(f"md:cbi_v15?motherduck_token={token}")
    else:
        conn = duckdb.connect("data/duckdb/cbi_v15.duckdb")
    
    count = seed_splits(conn)
    print(f"Seeded {count} splits")
    conn.close()

