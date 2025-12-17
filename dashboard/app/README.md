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

- `forecasts.zl_predictions`
- `training.daily_ml_matrix_zl`
- `features.daily_ml_matrix_zl`
- `reference.model_registry`
- Optional:
  - `staging.news_signals_daily`
  - `reference.regime_calendar`

## Translation Layer

### Big-8 → Chris Language

| Internal Feature                | Chris Label           |
| ------------------------------- | --------------------- |
| `feature_vix_stress`            | Market Volatility     |
| `feature_harvest_pace`          | Supply Pressure       |
| `feature_china_relations`       | China Demand          |
| `feature_tariff_threat`         | Trade War Risk        |
| `feature_argentina_competition` | Argentina Competition |
| `feature_industrial_demand`     | New Markets           |

## Non‑Goals

- No raw model jargon (SHAP, Big‑8, etc.) in the UI.
- No mock data in production. If data is missing, show explicit "No data" states.

## Technical Notes

- All reads go through `lib/md.ts` for MotherDuck HTTP API.
- Chart uses `lib/lightweight-charts/ZLForecastChart.tsx`.
- Action recommendation logic should come from `forecasts` table, not hard-coded.

## Visual Design

### DashdarkX Theme

- **Background:** `rgb(0, 0, 0)` - pure black
- **Text:** White with `font-thin` (100) headers, `font-extralight` (200) body
- **Borders:** `border-zinc-800`
- **Cards:** `bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm`

### Chart Configuration

```ts
{
  layout: { background: { color: '#0a0e1a' }, textColor: '#9ca3af' },
  grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } }
}
```

### ZLForecastChart Props

```tsx
<ZLForecastChart
  historical={historicalData} // { time: number, value: number }[]
  forecasts={{
    "1W": { point: 45.2, lower: 44.1, upper: 46.3 },
    "1M": { point: 46.5, lower: 44.8, upper: 48.2 },
    "3M": { point: 48.1, lower: 45.2, upper: 51.0 },
    "6M": { point: 49.3, lower: 45.8, upper: 52.8 },
  }}
  buyZone={{ min: 44, max: 46 }}
  riskZone={{ min: 50, max: 55 }}
/>
```

## UI Components

| Component          | Purpose                                     |
| ------------------ | ------------------------------------------- |
| `ZLForecastChart`  | Historical + forecast with confidence bands |
| `TradingViewGauge` | Semicircle gauge with SHAP impact           |
| Driver cards       | Big-8 translated to Chris language          |
| Action banner      | BUY / WAIT / MONITOR with rationale         |
