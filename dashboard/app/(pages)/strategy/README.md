# ZL Strategy & Procurement Plan (`/strategy`)

## Purpose
Provide a structured, model-driven procurement strategy:
- Recommended hedge ratios by horizon.
- Suggested buy windows.
- Scenario-based P&L distributions.

## Key Views

### 1. Strategy Summary
- TSci reporter text: current recommendation and rationale.

### 2. Hedge Ladder
- Visual representation of hedge volumes over time.
- Uses `lib/lightweight-charts/HedgeLadder.tsx`.

### 3. Scenario Panel
Predefined shocks and their effect on P&L:
- **China Soft** - Reduced import demand
- **Tariffs Escalate** - Section 301 expansion
- **Weather Shock** - US/Brazil/Argentina supply disruption

### 4. P&L Distribution
- Compare current strategy vs benchmarks.
- Show percentile bands (P5, P25, P50, P75, P95).

## Data Sources (MotherDuck)
- `forecasts.zl_v15_*`
- `training.daily_ml_matrix_zl_v15` (for realized returns and backtests).
- `reference.model_registry`
- `tsci.runs` (for metrics and narratives).

## Hedge Horizons

| Horizon | Description | Typical Volume |
|---------|-------------|----------------|
| 1W | Immediate needs | 100K lbs |
| 1M | Near-term procurement | 250K lbs |
| 3M | Quarterly planning | 500K lbs |
| 6M | Strategic positioning | 1M lbs |

## Scenario Definitions

Scenarios are defined in SQL/TSci, not hard-coded in React:

```sql
-- Example: China Soft scenario
-- Reduce china_bucket_score by 2 std devs
-- Rerun forecast with adjusted inputs
```

## Notes
- All monetary values use Admin-configured baseline volume and currency.
- Strategy text should come from TSci reporter output (`tsci.runs.metrics_json`).
- P&L calculations should be precomputed in SQL where possible.
