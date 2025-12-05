# Sentiment & Regime Monitor (`/sentiment`)

## Purpose
Monitor the behavior of the Big‑8 drivers and associated regimes over time:
- Are crush, China, FX, Fed, tariffs, biofuel, energy, and vol aligned or diverging?
- Which regime are we in right now?

## Key Views

### 1. Big‑8 Heatmap
- Bucket (rows) vs time or lookback window (columns).
- Color = standardized score or recent change.

### 2. Regime Timeline
- Horizontal strip showing regime labels by date.

### 3. Bucket Detail
When selecting a bucket, show:
- Time series of the bucket score.
- Contribution to recent forecasts.
- Related news signals.

## Data Sources (MotherDuck)
- `features.daily_ml_matrix_zl_v15`
  - `*_bucket_score`, `*_neural_score`, `master_neural_score`.
- `reference.regime_calendar` or regime columns in features.
- `staging.news_signals_daily`.

## Big-8 Buckets

| Bucket | Description |
|--------|-------------|
| `crush` | Soybean crush margins and demand |
| `china` | China import demand and policy |
| `fx` | Currency effects on competitiveness |
| `fed` | Monetary policy and interest rates |
| `tariff` | Trade policy and tariff risk |
| `biofuel` | RFS, biodiesel, renewable diesel |
| `energy` | Crude oil and energy markets |
| `vol` | Market volatility and risk appetite |

## Regime Labels

Examples (actual labels defined in `reference.regime_calendar`):
- "Policy Shock + High Volatility"
- "China Demand Surge"
- "Supply Stress"
- "Normal Trading"

## Notes
- Heavy calculation (deltas, Z‑scores) should be done in SQL, not in React.
- This is primarily a quant / power user page.
- Use `lib/lightweight-charts/SentimentHeatmap.tsx` for visualization.
