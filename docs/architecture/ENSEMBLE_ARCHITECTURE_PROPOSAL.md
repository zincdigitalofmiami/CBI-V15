# Ensemble Architecture (V15-Aligned)

Purpose: outline a realistic, regime-aware ensemble for ZL that fits the current V15 constraints (SQL-first on DuckDB/MotherDuck, Mac-only training, Big 8 coverage, 1w/1m/3m/6m horizons).

## Ground Truth (Do Not Violate)
- Target: ZL (soybean oil) is the primary asset; horizons: 1w/1m/3m/6m.
- Features: Big 8 drivers are pre-computed in the SQL layer (DuckDB/MotherDuck). No ad-hoc one-off SQL outside the managed SQL definitions.
- Compute: training/inference is Mac-local; no cloud training/Vertex/BQML/AutoML. Storage/feature serving is local DuckDB/MotherDuck (not DuckDB/MotherDuck).
- Data quality: no synthetic/fake data; verify coverage before modeling (RINs, biodiesel, CFTC, Trump sentiment, etc.).

## Layered Model Plan

### Bucket Specialists (8 buckets, 4 horizons each)
Use bucket-specific feature subsets and quantile objectives. Start with CatBoost quantile baselines; only escalate to deep models for buckets that are demonstrably non-linear.

| Bucket | Feature focus | Recommended model | Status checks |
| --- | --- | --- | --- |
| Crush | ZL/ZS/ZM, board crush, oil share, BOHO | CatBoost quantile | Confirm crush spreads present in feature tables. |
| China | HG, exports, HG-ZS/ZL correlations | CatBoost quantile (start), TFT only if needed | Verify China exports + HG series coverage. |
| FX | DX, BRL (6L), AUD/CAD (6A/6C), DXY/Brl momentum/vol | CatBoost quantile | Ensure FX series are complete and aligned to trading days. |
| Fed | Fed funds, curve (T10Y2Y), rate momentum | CatBoost quantile | Check FRED ingestion freshness. |
| Tariff | Trump sentiment, tariff/trade policy signals | TFT optional; otherwise CatBoost + event features | Requires ScrapeCreators/Trump ingest + tariff event flags; no data = no TFT. |
| Biofuel | RIN D4/D6, biodiesel prod, RFS volumes, BOHO | CatBoost quantile | Verify EIA/RIN coverage and joins; no claims of “fully integrated” until checked. |
| Energy | CL, HO, RB, NG, CL-ZL correlation, crack/BOHO | CatBoost quantile | Confirm petroleum series ingestion and correlations present. |
| Volatility | VIX, realized vol, NFCI/STLFSI4, vol regimes | CatBoost quantile; TFT only if non-linear benefit is proven | Ensure realized vol features exist; add vol regime flags via AnoFox SQL macros if missing. |

### Ensemble Layer
- Baseline: weighted quantile averaging across buckets per horizon (weights learned on validation pinball loss).
- Optional: Monte Carlo sampling from bucket quantiles to produce P05/P10/P25/P50/P75/P90/P95, but only after weights are calibrated.
- Future: stacking meta-learner (ridge or gradient boosting) trained on out-of-fold bucket predictions.

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
1) Extend `src/training/baselines/catboost_zl.py` (and siblings) to support quantile training per bucket/horizon using bucket-specific feature lists.
2) Define bucket feature selectors in code/config (reuse existing SQL column prefixes; do not hardcode paths).
3) Build an ensemble runner in `src/ensemble/` (new) that:
   - loads bucket models,
   - produces quantile forecasts per horizon,
   - learns/loads weights,
   - emits calibrated quantiles.
4) Add regime flags to the feature matrix in the SQL layer (no ad-hoc pandas transforms).
5) Validation harness: pinball loss, coverage, and calibration plots saved under `reports/ensemble/`.

## Open Questions / To Verify
- Do we have fresh RIN/biodiesel series and Trump/tariff event flags in the feature tables? If not, ingest before modeling.
- Are HG, BRL, and curve series gap-free over 2000–present? If not, define backfill rules in AnoFox SQL macros.
- Do realized-vol features already exist, or do we need a AnoFox SQL macros macro for rolling vol/regime flags?

## What This Removes
- No claims that biofuels are “fully integrated” without evidence.
- No ICE softs (KC/SB/CC/CT) unless a licensed feed is added.
- No explosive model counts (e.g., “89 models”) until we have code and data to support them.
- No presidency-based hand-tuned splits; use data-driven regimes and walk-forward validation.
