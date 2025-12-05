# Dashboard / Procurement Command Center (`/`)

## Purpose
Primary cockpit for US Oil Solutions (Chris) to make procurement decisions:
- Should I BUY, WAIT, or MONITOR?
- What prices should I expect (1W / 1M / 3M / 6M)?
- What are the top drivers, in business language?

## Audience
- Chris (procurement, non-quant)
- Internal quant team (for quick checks)

## Key Views

### 1. ZL "Breathing" Chart
- 2–5 years historical ZL + 6M forecast.
- 1W, 1M, 3M, 6M forecast bands with distinct colors.
- BUY and RISK zones shaded from model output.

### 2. Chris's Four Factors
- **China Purchases / Cancellations**
- **Harvest Status** (US/Brazil/Argentina)
- **Biofuel Markets** (RFS, biodiesel, renewable diesel)
- **Palm/Substitute Oil Spread**
- Each card: current status + $$ impact per cwt.

### 3. Drivers Panel (Big‑8 → Chris Translation)
Internal features (VIX stress, tariff threat, industrial demand, etc.) rendered as simple labels:
- "Market Volatility"
- "China Demand"
- "Supply Pressure"
- "Trade War Risk"
- "Argentina Competition"
- "New Markets"

### 4. Forward Curve & Recommendation
- Forecast prices + bands for each horizon.
- BUY / WAIT / MONITOR banner with one‑sentence rationale.
- Confidence level shown clearly.

## Data Sources (MotherDuck)
- `forecasts.zl_v15_*`
- `training.daily_ml_matrix_zl_v15`
- `features.daily_ml_matrix_zl_v15`
- `reference.model_registry`
- Optional:
  - `staging.news_signals_daily`
  - `reference.regime_calendar`

## Translation Layer

### Big-8 → Chris Language

| Internal Feature | Chris Label |
|------------------|-------------|
| `feature_vix_stress` | Market Volatility |
| `feature_harvest_pace` | Supply Pressure |
| `feature_china_relations` | China Demand |
| `feature_tariff_threat` | Trade War Risk |
| `feature_argentina_competition` | Argentina Competition |
| `feature_industrial_demand` | New Markets |

## Non‑Goals
- No raw model jargon (SHAP, Big‑8, etc.) in the UI.
- No mock data in production. If data is missing, show explicit "No data" states.

## Technical Notes
- All reads go through `lib/md.ts` for MotherDuck HTTP API.
- Chart uses `lib/lightweight-charts/ZLForecastChart.tsx`.
- Action recommendation logic should come from `forecasts` table, not hard-coded.
