import os
import time
import json
import pandas as pd
import psutil
import mlflow
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# ==========================================
# ⚙️ QUANT COMMAND CENTER: MAX FORCE (L0)
# ==========================================
PRESET = "best_quality"  # ENABLE: Bagging & Stacking
TIME_LIMIT = 3600  # ENABLE: 1 Hour Deep Search per Agent
RUN_ID = time.strftime("%Y%m%d_%H%M")

# 🧠 THE FEDERATION: CORE + BIG 8
BUCKETS = {
    "core_price_action": {
        "features": [
            "zl_close",
            "zl_volume",
            "zl_open_interest",
            "rsi_14",
            "macd",
            "bollinger_width",
        ],
        "known": ["options_expiration_date"],
    },
    "crush": {
        "features": [
            "soybean_close",
            "soymeal_close",
            "soyoil_close",
            "board_crush_margin",
        ],
        "known": [],
    },
    "china": {
        "features": ["soy_export_sales", "china_import_demand", "dalian_soy_close"],
        "known": ["china_holiday_golden_week", "seasonal_export_idx"],
    },
    "fx": {"features": ["brl_usd_close", "dxy_close", "us_treasury_10y"], "known": []},
    "fed": {
        "features": ["fed_funds_rate", "cpi_yoy", "pce_inflation"],
        "known": ["fomc_meeting_date", "scheduled_rate_decision"],
    },
    "tariff": {
        "features": ["geopolitical_risk_index", "trade_policy_uncertainty"],
        "known": ["tariff_implementation_date", "election_cycle_idx"],
    },
    "biofuel": {
        "features": ["rin_d6_price", "biodiesel_margin", "wti_close"],
        "known": ["epa_rvo_release_date"],
    },
    "energy": {
        "features": ["wti_close", "brent_close", "diesel_close"],
        "known": ["opec_meeting_schedule"],
    },
    "volatility": {
        "features": ["zl_iv_rank", "zl_hv_20d", "vix_close"],
        "known": ["usda_report_date"],
    },
}

HORIZONS = {"1w": 5, "1m": 22, "3m": 66, "6m": 132}

# 🛠️ 1.4 ARSENAL (LOCAL ONLY)
ALGO_CONFIG = {
    "Naive": {},
    "SeasonalNaive": {},
    "Theta": {},
    "RecursiveTabular": {
        "tabular_hyperparameters": {
            "XGB": {"booster": "gbtree", "grow_policy": "lossguide"},
            "CAT": {"depth": 6, "l2_leaf_reg": 3},
            "GBM": {},
        }
    },
    "DeepAR": {},
    "TemporalFusionTransformer": {},
    "TiDE": {},
    "PatchTST": {},
}


def train_agent(agent_name, config, df):
    print(f"\n[L0] 🧊 Processing Agent: {agent_name.upper()}")

    valid_known = [k for k in config["known"] if k in df.columns]

    for h_name, h_steps in HORIZONS.items():
        print(f"   🎯 Horizon: {h_name} ({h_steps} steps)")

        save_path = f"models/bucket_specialists/{agent_name}/{h_name}_{RUN_ID}"
        run_name = f"{agent_name}_{h_name}"

        # --- MLFLOW TRACKING BLOCK ---
        with mlflow.start_run(run_name=run_name):

            # 1. Log Configuration (API Observation)
            mlflow.log_param("agent", agent_name)
            mlflow.log_param("horizon", h_name)
            mlflow.log_param("preset", PRESET)
            mlflow.log_param("time_limit", TIME_LIMIT)
            mlflow.log_dict(
                ALGO_CONFIG, "algo_config.json"
            )  # Log full config as JSON artifact

            # 2. Tagging for Organization
            mlflow.set_tag("layer", "L0_Specialist")
            mlflow.set_tag("version", "V15.1")
            mlflow.set_tag("machine", "Mac_M_Series")

            # 3. Train
            predictor = TimeSeriesPredictor(
                target="zl_close",
                prediction_length=h_steps,
                path=save_path,
                eval_metric="MASE",
                known_covariates_names=valid_known,
            )

            predictor.fit(
                train_data=df,
                presets=PRESET,
                time_limit=TIME_LIMIT,
                hyperparameters=ALGO_CONFIG,
                excluded_model_types=["Chronos"],
                random_seed=42,
            )

            # 4. Log Results (Model Comparison)
            lb = predictor.leaderboard(silent=True)

            # Log the Winner's Score
            best_score = lb["score_val"].iloc[0]
            best_model = lb["model"].iloc[0]
            mlflow.log_metric("MASE", abs(best_score))
            mlflow.log_param("winning_model", best_model)

            # 5. Log Artifacts (Visuals)
            # Save Leaderboard as HTML for the Dashboard
            lb_html = f"lb_{agent_name}_{h_name}.html"
            lb.to_html(lb_html)
            mlflow.log_artifact(lb_html)

            # Cleanup local temp file
            if os.path.exists(lb_html):
                os.remove(lb_html)

            print(f"   ✅ LOCKED & LOGGED: {save_path}")


def main():
    # Initialize The Command Center
    mlflow.set_tracking_uri("file:./mlruns")  # Local storage
    mlflow.set_experiment("CBI_V15_Federation_L0")

    # ENABLE SYSTEM METRICS (CPU/RAM Monitoring)
    # This gives you the "Switches" view of your hardware health
    try:
        mlflow.enable_system_metrics_logging()
    except Exception as e:
        print(f"⚠️ System metrics logging unavailable (requires psutil): {e}")

    print(f"🚀 SYSTEM START: RUN_ID [{RUN_ID}] | MODE [{PRESET}]")

    data_path = "data/processed/training_matrix.csv"
    if not os.path.exists(data_path):
        print(f"❌ Missing Data: {data_path}")
        return

    print("⏳ Loading TimeSeriesDataFrame...")
    raw_df = pd.read_csv(data_path)

    train_data = TimeSeriesDataFrame.from_data_frame(
        raw_df, id_column="item_id", timestamp_column="timestamp"
    )

    print(f"✅ Loaded {len(train_data)} rows.")

    for agent, config in BUCKETS.items():
        train_agent(agent, config, train_data)


if __name__ == "__main__":
    main()
