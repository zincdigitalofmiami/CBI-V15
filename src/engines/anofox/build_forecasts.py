"""
AnoFox: Build Forecasts

Applies trained models to generate predictions.
Writes to unversioned forecasts tables (e.g. forecasts.zl_predictions) and
updates model_registry (once wired to real models).
"""

import os
from pathlib import Path

import duckdb

MOTHERDUCK_DB = os.getenv("MOTHERDUCK_DB", "cbi-v15")


def build_forecasts(
    model_id: str, horizon: str, con: duckdb.DuckDBPyConnection = None
) -> None:
    """
    Generate forecasts for a specific model and horizon.

    Args:
        model_id: Model identifier from reference.model_registry
        horizon: Forecast horizon ('1w', '1m', '3m', '6m', '12m')
        con: DuckDB connection (optional)
    """
    if con is None:
        motherduck_token = os.getenv("MOTHERDUCK_TOKEN")
        if motherduck_token:
            con = duckdb.connect(
                f"md:{MOTHERDUCK_DB}?motherduck_token={motherduck_token}"
            )
        else:
            # Fallback to local
            ROOT_DIR = Path(__file__).resolve().parents[2]
            DB_PATH = ROOT_DIR / "data" / "duckdb" / "cbi_v15.duckdb"
            con = duckdb.connect(str(DB_PATH))

    print(f"Generating forecasts for {model_id} at {horizon} horizon...")

    # Load model from disk (placeholder - actual implementation will load pickled model)
    # model = load_model(model_id)

    # Get latest features
    # features = con.execute("SELECT * FROM features.daily_ml_matrix_zl WHERE as_of_date = (SELECT MAX(as_of_date) FROM features.daily_ml_matrix_zl)").df()

    # Generate predictions
    # predictions = model.predict(features)

    # Write to forecasts table (currently writes metadata only; predictions
    # stay NULL until real models are wired in).
    con.execute(
        f"""
        INSERT INTO forecasts.zl_predictions (
            as_of_date, symbol, model_id, horizon,
            y_pred, y_pred_lower, y_pred_upper,
            master_neural_score_at_run, regime_at_run
        )
        SELECT 
            as_of_date,
            symbol,
            '{model_id}' AS model_id,
            '{horizon}' AS horizon,
            NULL AS y_pred,  -- Placeholder until real models are wired
            NULL AS y_pred_lower,
            NULL AS y_pred_upper,
            master_neural_score AS master_neural_score_at_run,
            regime AS regime_at_run
        FROM features.daily_ml_matrix_zl
        WHERE as_of_date = (SELECT MAX(as_of_date) FROM features.daily_ml_matrix_zl)
    """
    )

    print(f"Forecasts written for {model_id}.")


if __name__ == "__main__":
    # Example usage (model_id is illustrative; ensure it matches
    # reference.model_registry once models are registered.)
    build_forecasts(model_id="lgbm_zl_1w", horizon="1w")
