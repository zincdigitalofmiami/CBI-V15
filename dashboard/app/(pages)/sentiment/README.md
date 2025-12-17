# Sentiment & Regime Monitor (`/sentiment`)

## Purpose

Monitor the behavior of the Big-8 drivers and associated regimes over time:

- Are crush, China, FX, Fed, tariffs, biofuel, energy, and vol aligned or diverging?
- Which regime are we in right now?
- How have regimes shifted historically?

## Audience

- Quant team - primary users
- Chris (Procurement) - simplified regime context
- **Power user page** - some quant terminology exposed

## Files

| File       | Purpose                                                   |
| ---------- | --------------------------------------------------------- |
| `page.tsx` | Main sentiment page component                             |
| `_files/`  | Page-specific components (empty - uses shared components) |

## Key Views

### 1. Big-8 Heatmap

- Bucket (rows) vs time or lookback window (columns)
- Color = standardized score or recent change
- Click to drill into bucket detail

### 2. Regime Timeline

- Horizontal strip showing regime labels by date
- Visual regime transitions over 1-5 year history
- Hover for regime characteristics

### 3. Bucket Detail Panel

When selecting a bucket, show:

- Time series of the bucket score (60-day rolling)
- Contribution to recent forecasts (feature importance)
- Related news signals from `staging.news_signals_daily`

## Data Sources (MotherDuck)

| Table                         | Purpose       | Key Columns                             |
| ----------------------------- | ------------- | --------------------------------------- |
| `features.daily_ml_matrix_zl` | Big-8 scores  | `*_bucket_score`, `master_neural_score` |
| `reference.regime_calendar`   | Regime labels | `date`, `regime_label`, `regime_id`     |
| `staging.news_signals_daily`  | News context  | `bucket`, `sentiment`, `intensity`      |

### SQL Example

```sql
SELECT
    as_of_date,
    crush_bucket_score,
    china_bucket_score,
    fx_bucket_score,
    fed_bucket_score,
    tariff_bucket_score,
    biofuel_bucket_score,
    energy_bucket_score,
    volatility_bucket_score
FROM features.daily_ml_matrix_zl
WHERE as_of_date >= CURRENT_DATE - INTERVAL '90 days'
ORDER BY as_of_date;
```

## Big-8 Buckets

| Bucket    | Score Column              | Description           | Key Inputs                   |
| --------- | ------------------------- | --------------------- | ---------------------------- |
| `crush`   | `crush_bucket_score`      | Soybean crush margins | ZL, ZS, ZM spreads           |
| `china`   | `china_bucket_score`      | China import demand   | HG copper, export sales      |
| `fx`      | `fx_bucket_score`         | Currency effects      | DX, BRL, CNY                 |
| `fed`     | `fed_bucket_score`        | Monetary policy       | Fed funds, yield curve       |
| `tariff`  | `tariff_bucket_score`     | Trade policy risk     | Trump sentiment, Section 301 |
| `biofuel` | `biofuel_bucket_score`    | RFS/biodiesel         | RIN prices, mandates         |
| `energy`  | `energy_bucket_score`     | Energy markets        | CL, HO, RB                   |
| `vol`     | `volatility_bucket_score` | Market volatility     | VIX, realized vol            |

## Regime Labels

Examples (defined in `reference.regime_calendar`):

| Regime                    | Characteristics                 | Typical Duration |
| ------------------------- | ------------------------------- | ---------------- |
| "Policy Shock + High Vol" | Tariff/fed spike, VIX > 25      | 1-4 weeks        |
| "China Demand Surge"      | China bucket > 70, import spike | 2-8 weeks        |
| "Supply Stress"           | Crush < 30, weather shock       | 4-12 weeks       |
| "Normal Trading"          | All buckets 40-60               | Weeks to months  |

## Business Logic Examples

### Regime Detection

```sql
CASE
    WHEN volatility_bucket_score > 70
     AND tariff_bucket_score > 60
    THEN 'Policy Shock + High Vol'

    WHEN china_bucket_score > 70
    THEN 'China Demand Surge'

    WHEN crush_bucket_score < 30
    THEN 'Supply Stress'

    ELSE 'Normal Trading'
END AS regime_label
```

### Divergence Alert

```
IF ABS(crush_bucket_score - energy_bucket_score) > 30
THEN alert = 'Crush-Energy Divergence'
AND investigate = TRUE
```

## UI Components

- `lib/lightweight-charts/SentimentHeatmap.tsx` - Big-8 heatmap
- Regime timeline (custom component)
- Bucket detail cards with sparklines
- News signal feed by bucket

## Notes

- Heavy calculation (deltas, Z-scores) should be done in SQL, not in React
- This is a **quant/power user page** - terminology is OK
- Regime labels should come from precomputed `reference.regime_calendar`
- Heatmap colors: Red (bearish) ← Neutral → Green (bullish)

## Visual Design

### DashdarkX Theme

- **Background:** `rgb(0, 0, 0)` - pure black
- **Category scores:** `font-extralight` with `text-zinc-400` labels
- **News items:** `border-zinc-800` with hover `bg-zinc-900/20`
- **All badges:** `font-extralight`

### SentimentHeatmap Props

```tsx
<SentimentHeatmap
  buckets={["crush", "china", "fx", "fed", "tariff", "biofuel", "energy", "vol"]}
  data={heatmapData} // 2D array of scores
  lookbacks={["1d", "5d", "20d"]}
/>
```

### Heatmap Color Scale

| Score  | Color | Meaning |
| ------ | ----- | ------- |
| 0-30   | Red   | Bearish |
| 30-70  | Gray  | Neutral |
| 70-100 | Green | Bullish |

### Regime Timeline

- Horizontal strip showing regime labels by date
- Click to expand regime characteristics
- Historical view: 1-5 years
