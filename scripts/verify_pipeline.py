"""
Pipeline verification script (AutoGluon-free).

Checks that:
- MotherDuck connection works.
- Core tables exist and are readable.
- Training matrix + preconditioning outputs exist and are sane.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Avoid importing any training libraries here; keep this script "ops-safe".
from src.utils.motherduck_client import (  # type: ignore
    close_motherduck_connection,
    get_motherduck_connection,
)


def verify() -> None:
    """Verify core pipeline connectivity without AutoGluon dependencies."""
    print("Verifying Pipeline (MotherDuck + core tables)...")

    con = get_motherduck_connection()
    try:
        con.execute("SELECT 1").fetchone()
        print("✅ Connected to MotherDuck")

        checks = [
            ("raw.databento_futures_ohlcv_1d", "SELECT COUNT(*) FROM raw.databento_futures_ohlcv_1d"),
            ("raw.fred_economic", "SELECT COUNT(*) FROM raw.fred_economic"),
            ("features.daily_ml_matrix_zl", "SELECT COUNT(*) FROM features.daily_ml_matrix_zl"),
            ("training.daily_ml_matrix_zl", "SELECT COUNT(*) FROM training.daily_ml_matrix_zl"),
            ("training.feature_preconditioning_params_zl", "SELECT COUNT(*) FROM training.feature_preconditioning_params_zl"),
            ("training.daily_ml_matrix_zl_v15", "SELECT COUNT(*) FROM training.daily_ml_matrix_zl_v15"),
        ]

        for name, sql in checks:
            try:
                n = con.execute(sql).fetchone()[0]
                status = "✅" if n and n > 0 else "⚠️"
                print(f"{status} {name}: {n:,} rows")
            except Exception as exc:
                print(f"❌ {name}: {exc}")

        # Quick sanity checks on v15 matrix (no all-null features for ZL)
        try:
            non_target_cols = con.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema='training'
                  AND table_name='daily_ml_matrix_zl_v15'
                  AND column_name NOT IN ('as_of_date','symbol','train_val_test_split','training_weight','updated_at')
                  AND column_name NOT LIKE 'target_%'
                ORDER BY column_name
                """
            ).fetchall()
            non_target_cols = [c[0] for c in non_target_cols]

            worst = []
            for c in non_target_cols:
                nulls = con.execute(
                    f"""
                    SELECT SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END)
                    FROM training.daily_ml_matrix_zl_v15
                    WHERE symbol='ZL'
                    """
                ).fetchone()[0]
                worst.append((nulls, c))
            worst.sort(reverse=True)

            if worst and worst[0][0] == 0:
                print("✅ training.daily_ml_matrix_zl_v15: no NULL feature columns for ZL")
            else:
                print(f"⚠️ training.daily_ml_matrix_zl_v15: top NULL feature columns: {worst[:5]}")
        except Exception as exc:
            print(f"⚠️ Could not sanity-check v15 matrix: {exc}")

        print("✅ Pipeline Verification Complete.")
    finally:
        close_motherduck_connection()


if __name__ == "__main__":
    verify()
