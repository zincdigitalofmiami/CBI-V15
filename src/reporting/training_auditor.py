#!/usr/bin/env python3
"""
Hot-Audit Loop: Immediate Reporting After Each Bucket Training

Runs INSIDE the training loop to provide immediate feedback.
Compares AutoGluon performance vs AnoFox Structure baseline.
Logs to ops.training_logs immediately (no batching).

Usage:
    from src.reporting.training_auditor import audit_bucket_performance

    predictor = train_bucket_specialist('crush', train_df, val_df)
    audit_bucket_performance(predictor, 'crush', test_df, db_conn)
"""
import duckdb
from datetime import datetime
from typing import Optional

import pandas as pd


def audit_bucket_performance(
    predictor,
    bucket_name: str,
    test_data: pd.DataFrame,
    db_conn: duckdb.DuckDBPyConnection,
    log_to_motherduck: bool = True,
) -> dict:
    """
    Run immediate audit after bucket training completes.

    Args:
        predictor: Trained AutoGluon TabularPredictor
        bucket_name: Bucket name (crush, china, fx, etc.)
        test_data: Test dataset with targets
        db_conn: DuckDB connection (MotherDuck or local)
        log_to_motherduck: If True, log to ops.training_logs

    Returns:
        dict: Audit metrics (ag_score, baseline, lift, best_model)
    """
    print(f"\n{'='*80}")
    print(f"📢 HOT-AUDIT REPORT: {bucket_name.upper()}")
    print(f"{'='*80}")

    # 1. Get AutoGluon Performance
    leaderboard = predictor.leaderboard(test_data, silent=True)
    best_model = leaderboard.iloc[0]
    ag_score = best_model["score_test"]
    fit_time = best_model["fit_time"]
    model_name = best_model["model"]

    print(f"🏆 Best Model: {model_name}")
    print(f"   Training Time: {fit_time:.1f}s")
    print(f"   Test Score: {ag_score:.4f}")

    # 2. Get AnoFox Structure Baseline (Naive forecast)
    # Compare vs simple AutoETS or last-value-carry-forward
    # If AutoGluon isn't beating this, model is overfitting
    try:
        # Use simple persistence (last value) as baseline
        # This is the "structure check" - ML must beat naive
        baseline_query = f"""
            WITH last_values AS (
                SELECT 
                    as_of_date,
                    target_1w,
                    LAG(target_1w, 1) OVER (ORDER BY as_of_date) as naive_forecast
                FROM training.daily_ml_matrix_zl_v15
                WHERE as_of_date IN (SELECT as_of_date FROM test_data_temp)
            )
            SELECT AVG(ABS(target_1w - naive_forecast)) as mae
            FROM last_values
            WHERE naive_forecast IS NOT NULL
        """

        # Register test data as temp table
        db_conn.register("test_data_temp", test_data[["as_of_date", "target_1w"]])

        result = db_conn.execute(baseline_query).fetchone()
        baseline_mae = result[0] if result and result[0] else None

    except Exception as e:
        print(f"   ⚠️  Could not calculate AnoFox baseline: {e}")
        baseline_mae = None

    # 3. Calculate Lift (AutoGluon improvement over naive)
    if baseline_mae and baseline_mae > 0:
        # Convert AutoGluon score to MAE if using accuracy/pinball_loss
        # For quantile regression, lower is better
        lift = (
            (baseline_mae - ag_score) / baseline_mae if ag_score < baseline_mae else 0
        )

        print(f"\n📊 Structure Check:")
        print(f"   AutoGluon: {ag_score:.4f}")
        print(f"   Naive:     {baseline_mae:.4f}")
        print(f"   Lift:      {lift:.2%}")

        if lift <= 0:
            print(f"\n⚠️  WARNING: AutoGluon NOT beating naive baseline!")
            print(f"   Possible overfitting or insufficient training time")
    else:
        lift = None
        print(f"\n⚠️  Baseline comparison unavailable")

    # 4. Show top 5 models
    print(f"\n📋 Top 5 Models:")
    print(
        leaderboard[["model", "score_test", "fit_time"]].head(5).to_string(index=False)
    )

    # 5. Feature importance (top 10)
    try:
        fi = predictor.feature_importance(test_data)
        print(f"\n🎯 Top 10 Features:")
        print(fi.head(10).to_string())
    except Exception as e:
        print(f"\n⚠️  Feature importance unavailable: {e}")

    # 6. Push to MotherDuck ops.training_logs (immediate)
    if log_to_motherduck:
        try:
            log_entry = {
                "bucket_name": bucket_name,
                "trained_at": datetime.now().isoformat(),
                "best_model": model_name,
                "ag_score": float(ag_score),
                "baseline_score": float(baseline_mae) if baseline_mae else None,
                "lift": float(lift) if lift else None,
                "fit_time_seconds": float(fit_time),
                "model_count": len(leaderboard),
            }

            db_conn.execute(
                """
                INSERT INTO ops.training_logs (
                    bucket_name, trained_at, best_model, ag_score, 
                    baseline_score, lift, fit_time_seconds, model_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    log_entry["bucket_name"],
                    log_entry["trained_at"],
                    log_entry["best_model"],
                    log_entry["ag_score"],
                    log_entry["baseline_score"],
                    log_entry["lift"],
                    log_entry["fit_time_seconds"],
                    log_entry["model_count"],
                ],
            )

            print(f"\n✅ Logged to ops.training_logs in MotherDuck")

        except Exception as e:
            print(f"\n⚠️  Could not log to MotherDuck: {e}")

    print(f"\n{'='*80}\n")

    # Return audit results
    return {
        "bucket_name": bucket_name,
        "best_model": model_name,
        "ag_score": float(ag_score),
        "baseline_score": float(baseline_mae) if baseline_mae else None,
        "lift": float(lift) if lift else None,
        "fit_time": float(fit_time),
        "leaderboard": leaderboard,
    }
















