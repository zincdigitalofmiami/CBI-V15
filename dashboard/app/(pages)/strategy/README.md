# ZL Strategy & Procurement Plan (`/strategy`)

## Purpose
Provide a structured, model-driven procurement strategy:
- Recommended hedge ratios by horizon
- Suggested buy windows
- Scenario-based P&L distributions

## Audience
- Chris (Procurement) - primary user
- Quant team - strategy validation
- **Decision support page** - actionable recommendations

## Files

| File | Purpose |
|------|---------|
| `page.tsx` | Main strategy page component |
| `_files/` | Page-specific components (empty - uses shared components) |

## Key Views

### 1. Strategy Summary
- TSci reporter text: current recommendation and rationale
- BUY / WAIT / MONITOR banner with confidence level
- Key drivers behind recommendation

### 2. Hedge Ladder
- Visual representation of hedge volumes over time
- Uses `lib/lightweight-charts/HedgeLadder.tsx`
- Color-coded by urgency (green = covered, yellow = partial, red = exposed)

### 3. Scenario Panel
Predefined shocks and their effect on P&L:

| Scenario | Description | Typical Impact |
|----------|-------------|----------------|
| **China Soft** | Reduced import demand | -$0.02-0.05/lb |
| **Tariffs Escalate** | Section 301 expansion | -$0.03-0.08/lb |
| **Weather Shock** | US/Brazil/Argentina supply disruption | +$0.05-0.15/lb |
| **Fed Pivot** | Rate cut cycle begins | +$0.01-0.03/lb |

### 4. P&L Distribution
- Compare current strategy vs benchmarks
- Show percentile bands (P5, P25, P50, P75, P95)
- Monte Carlo simulation from L4 model

## Data Sources (MotherDuck)

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `forecasts.zl_predictions` | Point forecasts + bands | `price_1w`, `price_1m`, `ci_lower`, `ci_upper` |
| `training.daily_ml_matrix_zl` | Realized returns (backtests) | `target_ret_*` |
| `reference.model_registry` | Active model info | `model_id`, `metrics_json` |

### SQL Example
```sql
SELECT 
    horizon,
    forecast_price,
    ci_lower_90,
    ci_upper_90,
    recommendation,
    confidence
FROM forecasts.zl_predictions
WHERE as_of_date = CURRENT_DATE
ORDER BY horizon;
```

## Hedge Horizons

| Horizon | Trading Days | Description | Typical Volume |
|---------|--------------|-------------|----------------|
| 1W | 5 | Immediate needs | 100K lbs |
| 1M | 20 | Near-term procurement | 250K lbs |
| 3M | 60 | Quarterly planning | 500K lbs |
| 6M | 126 | Strategic positioning | 1M lbs |

## Business Logic Examples

### Recommendation Logic
```
IF forecast_direction = 'UP'
AND confidence > 0.7
AND volatility_bucket_score < 60
THEN recommendation = 'BUY'
AND rationale = 'Rising prices with stable volatility'
```

### Scenario Impact
```sql
-- China Soft scenario
-- Reduce china_bucket_score by 2 std devs
-- Rerun forecast with adjusted inputs
SELECT 
    forecast_price * (1 - 0.03) AS scenario_price,
    'China Soft: -3% price impact' AS scenario_label
FROM forecasts.zl_predictions
WHERE horizon = '1m';
```

### Hedge Coverage
```
IF current_hedge_pct < target_hedge_pct
AND recommendation = 'BUY'
THEN alert = 'UNDERHEDGED'
AND action = 'Increase coverage before price move'
```

## UI Components

- `lib/lightweight-charts/HedgeLadder.tsx` - Volume ladder visualization
- Strategy summary cards with action banners
- Scenario sliders for what-if analysis
- P&L distribution chart (histogram or violin plot)

## Notes
- All monetary values use Admin-configured base volume and currency
- Strategy text should come from AutoGluon + SQL outputs (TSci is removed)
- P&L calculations should be precomputed in SQL where possible
- Scenarios should be defined in SQL/orchestration, not hard-coded in React
- This is a **decision support page** - clarity over complexity

## Visual Design

### DashdarkX Theme
- **Background:** `rgb(0, 0, 0)` - pure black
- **Slider labels:** `font-extralight`
- **All badges:** `font-mono font-extralight`
- **Helper text:** `text-zinc-500 font-extralight`
- **Procurement guidance cards:** colored backgrounds with `font-extralight`

### HedgeLadder Props
```tsx
<HedgeLadder
  data={[
    { horizon: '1W', volume: 100000, expiry: '2025-12-12' },
    { horizon: '1M', volume: 250000, expiry: '2026-01-15' },
    { horizon: '3M', volume: 500000, expiry: '2026-03-15' },
    { horizon: '6M', volume: 1000000, expiry: '2026-06-15' },
  ]}
/>
```

### Urgency Color Coding
| Status | Color | Meaning |
|--------|-------|--------|
| Covered | Green | Hedge in place |
| Partial | Yellow | Needs attention |
| Exposed | Red | Urgent action needed |

### Scenario Sliders
- Interactive what-if analysis
- Real-time P&L recalculation
- Predefined shock scenarios (China Soft, Tariffs, Weather, Fed)
