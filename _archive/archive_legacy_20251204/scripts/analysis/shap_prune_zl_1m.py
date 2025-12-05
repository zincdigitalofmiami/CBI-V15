#!/usr/bin/env python3
"""
SHAP-based feature pruning for ZL 1m LightGBM baseline.

Process (institutional-style):
1. Load trained 1m model + validation split.
2. Compute SHAP values on validation set.
3. Rank features by absolute mean SHAP (global importance).
4. Keep features up to 99% cumulative SHAP gain.
5. Drop features contributing < 0.01% of total SHAP gain.
6. Cluster SHAP contributions and keep the top feature per cluster.
7. Emit final feature list to TrainingData/pruned_feature_list_1m.txt.
"""

import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from pathlib import Path
from scipy.cluster import hierarchy


DATA_DIR = Path("TrainingData/exports")
MODELS_DIR = Path("Models/local/baseline")
OUT_PATH = Path("TrainingData/pruned_feature_list_1m.txt")


def main() -> None:
    # 1) Load model
    model_path = MODELS_DIR / "zl_1m_lightgbm.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Model was saved with joblib including metadata; load Booster
    import joblib
    saved = joblib.load(model_path)
    model: lgb.Booster = saved["model"]

    # 2) Load validation split for 1m
    val_path = DATA_DIR / "zl_training_minimal_1m_val.parquet"
    if not val_path.exists():
        raise FileNotFoundError(f"Validation split not found: {val_path}")

    df_val = pd.read_parquet(val_path)

    target_col = "target_1m_price"
    # NOTE: 'price' is used as a feature in training, so we must NOT drop it here,
    # otherwise the feature shape will differ (58 vs 59) and LightGBM will error.
    drop_cols = {
        target_col,
        "price_current",
        "date",
        "symbol",
        "regime",
    }

    feature_cols = [
        c for c in df_val.columns
        if c not in drop_cols and not c.startswith("target_")
    ]
    if not feature_cols:
        raise RuntimeError("No feature columns found for SHAP pruning.")

    X_val = df_val[feature_cols].fillna(0.0)

    # 3) Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_val)

    # 4) Global importance
    shap_vals = np.array(shap_vals)
    if shap_vals.ndim == 2:
        # (n_samples, n_features)
        shap_abs_mean = np.abs(shap_vals).mean(axis=0)
    elif shap_vals.ndim == 3:
        # For multiclass, sum over classes: (n_classes, n_samples, n_features)
        shap_abs_mean = np.abs(shap_vals).mean(axis=(0, 1))
    else:
        raise RuntimeError(f"Unexpected SHAP shape: {shap_vals.shape}")

    imp = pd.DataFrame({
        "feature": feature_cols,
        "shap_gain": shap_abs_mean,
    }).sort_values("shap_gain", ascending=False)

    total_gain = imp["shap_gain"].sum()
    if total_gain <= 0:
        raise RuntimeError("Total SHAP gain is non-positive; cannot prune.")

    imp["cum"] = imp["shap_gain"].cumsum() / total_gain

    # 5) Hard cutoff #1: cumulative 99%
    keep_99 = set(imp.loc[imp["cum"] <= 0.99, "feature"])

    # 6) Hard cutoff #2: minimum 0.01% of total gain
    min_thresh = 0.0001 * total_gain
    keep_min = set(imp.loc[imp["shap_gain"] >= min_thresh, "feature"])

    keep_pre_cluster = [f for f in imp["feature"] if f in keep_99 and f in keep_min]
    if not keep_pre_cluster:
        raise RuntimeError("No features remain after SHAP thresholds; thresholds too strict.")

    # 7) Cluster redundancy on SHAP patterns for kept features
    kept_indices = [feature_cols.index(f) for f in keep_pre_cluster]
    shap_kept = shap_vals[:, kept_indices] if shap_vals.ndim == 2 else shap_vals.mean(axis=0)[:, kept_indices]

    # Corr over features
    corr = np.corrcoef(shap_kept.T)
    # Distance metric: 1 - |corr|
    dist = 1 - np.abs(corr)
    Z = hierarchy.linkage(dist, method="ward")
    clusters = hierarchy.fcluster(Z, t=0.7, criterion="distance")

    final_features = []
    for cluster_id in np.unique(clusters):
        idxs = [i for i, c in enumerate(clusters) if c == cluster_id]
        cluster_feats = [keep_pre_cluster[i] for i in idxs]
        # pick highest SHAP gain in this cluster
        best = max(
            cluster_feats,
            key=lambda name: imp.loc[imp["feature"] == name, "shap_gain"].iloc[0],
        )
        final_features.append(best)

    final_features = sorted(set(final_features))

    print(f"Initial features: {len(feature_cols)}")
    print(f"After SHAP 99% + threshold: {len(keep_pre_cluster)}")
    print(f"Final (cluster-pruned) features: {len(final_features)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(final_features))
    print(f"✅ Wrote pruned 1m feature list to {OUT_PATH}")


if __name__ == "__main__":
    main()
