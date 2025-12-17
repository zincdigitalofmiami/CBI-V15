# Ensemble Architecture (Historical – Pre-AutoGluon)

> **Important:** This document describes an earlier CatBoost/TFT-centric ensemble concept.  
> The **canonical V15.1 modeling stack** is now:
>
> - Big 8 bucket specialists → AutoGluon `TabularPredictor`
> - Core ZL forecaster → AutoGluon `TimeSeriesPredictor`
> - Meta + ensemble → AutoGluon stacking + `WeightedEnsemble_L2`
> - Monte Carlo → consumes final forecasts only
>
> Treat all direct CatBoost/TFT training and custom ensemble runner suggestions here as **historical design notes**, not implementation instructions. For current work, follow `docs/architecture/MASTER_PLAN.md` and `AGENTS.md`.

Purpose: outline a realistic, regime-aware ensemble for ZL that fits the current V15 constraints (SQL-first on DuckDB/MotherDuck, Mac-only training, Big 8 coverage, 1w/1m/3m/6m horizons).

## Ground Truth (Do Not Violate)

- Target: ZL (soybean oil) is the primary asset; horizons: 1w/1m/3m/6m.
- Features: Big 8 drivers are pre-computed in the SQL layer (DuckDB/MotherDuck). No ad-hoc one-off SQL outside the managed SQL definitions.
- Compute: training/inference is Mac-local; no cloud training/Vertex/BQML/AutoML. Storage/feature serving is local DuckDB/MotherDuck (not DuckDB/MotherDuck).
- Data quality: no synthetic/fake data; verify coverage before modeling (RINs, biodiesel, CFTC, Trump sentiment, etc.).

## Layered Model Plan

### Bucket Specialists (8 buckets, 4 horizons each)

Use bucket-specific feature subsets and quantile objectives. In V15.1 this is implemented via **AutoGluon `TabularPredictor`** per bucket; CatBoost/XGBoost/NNs run **inside AutoGluon**, not as stand-alone pipelines.

| Bucket     | Feature focus                                       | Recommended model family (inside AutoGluon)                                  | Status checks                                                                              |
| ---------- | --------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Crush      | ZL/ZS/ZM, board crush, oil share, BOHO              | Tree + tabular foundation models (CatBoost, LightGBM, XGBoost via AutoGluon) | Confirm crush spreads present in feature tables.                                           |
| China      | HG, exports, HG-ZS/ZL correlations                  | Same AutoGluon Tabular stack; deep models only via AutoGluon presets         | Verify China exports + HG series coverage.                                                 |
| FX         | DX, BRL (6L), AUD/CAD (6A/6C), DXY/Brl momentum/vol | AutoGluon Tabular stack                                                      | Ensure FX series are complete and aligned to trading days.                                 |
| Fed        | Fed funds, curve (T10Y2Y), rate momentum            | AutoGluon Tabular stack                                                      | Check FRED ingestion freshness.                                                            |
| Tariff     | Trump sentiment, tariff/trade policy signals        | AutoGluon Tabular stack (event features in SQL)                              | Requires ScrapeCreators/Trump ingest + tariff event flags; no data = no TFT.               |
| Biofuel    | RIN D4/D6, biodiesel prod, RFS volumes, BOHO        | AutoGluon Tabular stack                                                      | Verify EIA/RIN coverage and joins; no claims of “fully integrated” until checked.          |
| Energy     | CL, HO, RB, NG, CL-ZL correlation, crack/BOHO       | AutoGluon Tabular stack                                                      | Confirm petroleum series ingestion and correlations present.                               |
| Volatility | VIX, realized vol, NFCI/STLFSI4, vol regimes        | AutoGluon Tabular stack; any deep models only through AutoGluon              | Ensure realized vol features exist; add vol regime flags via AnoFox SQL macros if missing. |

### Ensemble Layer

- In V15.1, the primary ensemble is **AutoGluon `WeightedEnsemble_L2`** over the bucket specialists and meta-learner.
- Additional custom ensemble methods in this document (manual QRA, bespoke stacking code, etc.) are **historical ideas only** and should not be implemented outside AutoGluon.

### Regime Handling (keep simple and defensible)

- Use anchored walk-forward splits with regime labels, not hand-wavy per-presidency slices. Example (adjust with data availability):
  - Train: 2000–2016, Validate: 2017–2018 (trade-war onset)
  - Train: 2000–2019, Validate: 2020–2021 (COVID shock)
  - Train: 2000–2023, Validate: 2024–present (post-shock)
- Add regime indicators in the feature matrix (trade-war, pandemic, policy shock, high-vol). Do not create bespoke splits without data to justify them.
- Reweight buckets by validated performance per regime; log weights and rationale.

## Metrics and Validation

- Primary: pinball loss per quantile; coverage of P10/P90; calibration plots per bucket and horizon.
- Secondary: MAE/MAPE on P50; turnover of ensemble weights; stability of regime weights.
- Walk-forward only; no random splits. Keep validation windows contiguous and non-overlapping.

## Implementation Steps (grounded in current codebase)

1. Extend `src/training/baselines/catboost_zl.py` (and siblings) to support quantile training per bucket/horizon using bucket-specific feature lists.
2. Define bucket feature selectors in code/config (reuse existing SQL column prefixes; do not hardcode paths).
3. Build an ensemble runner in `src/ensemble/` (new) that:
   - loads bucket models,
   - produces quantile forecasts per horizon,
   - learns/loads weights,
   - emits calibrated quantiles.
4. Add regime flags to the feature matrix in the SQL layer (no ad-hoc pandas transforms).
5. Validation harness: pinball loss, coverage, and calibration plots saved under `reports/ensemble/`.

## Open Questions / To Verify

- Do we have fresh RIN/biodiesel series and Trump/tariff event flags in the feature tables? If not, ingest before modeling.
- Are HG, BRL, and curve series gap-free over 2000–present? If not, define backfill rules in AnoFox SQL macros.
- Do realized-vol features already exist, or do we need a AnoFox SQL macros macro for rolling vol/regime flags?

## What This Removes

- No claims that biofuels are “fully integrated” without evidence.
- No ICE softs (KC/SB/CC/CT) unless a licensed feed is added.
- No explosive model counts (e.g., “89 models”) until we have code and data to support them.
- No presidency-based hand-tuned splits; use data-driven regimes and walk-forward validation.
