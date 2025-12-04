# Concept Extraction: Legacy CBI-V15 → New Dashboard

**Purpose**: Extract ideas and concepts from legacy CBI-V15 to enhance our Next.js/DashdarkX/MotherDuck dashboard.

## Key Concepts from Legacy (Read-Only)

### Big 8 Drivers (Feature Importance)
From legacy README, these are the most important features:

1. **Crush Margin** (0.961 correlation - #1!)
2. **China Imports** (-0.813 correlation)
3. **Dollar Index** (-0.658)
4. **Fed Policy** (-0.656)
5. **Tariffs** (0.647)
6. **Biofuels** (-0.601)
7. **Crude Oil** (0.584)
8. **VIX** (0.398)

**How to Use**: Display these as gauges in the Forecasting tab with SHAP values showing their current impact.

### Legacy Pages Found
- `/admin` - Admin panel
- `/vegas-intel` - Analytics intelligence
- `/legislation` - Legislative tracking
- `/sentiment` - News sentiment
- `/strategy` - Trading strategy
- Main forecast page

**Reusable Components**:
- `Gauge.tsx` - Gauge component (concept only, we'll build our own)
- `TimeSeriesCharts.tsx` - Time series visualization
- `WeatherChoropleth.tsx` - Weather mapping

## Page Mapping (Legacy → New Dashboard)

### Our Current Tabs (to Enhance):

1. **Forecasting Tab**
   - **Add**: Big 8 Drivers as gauge panel
   - **Keep**: TradingView Lightweight Charts for price + bands

2. **Tariffs Tab**
   - **Concept**: "Tariffs" is #5 driver (0.647 correlation)

3. **Legislation Tab**
   - **Concept**: "Fed Policy" is #4 driver (-0.656 correlation)

4. **Relations Tab**
   - **Concept**: "China Imports" is #2 driver (-0.813 correlation)

5. **Sentiment Tab**
   - **Concept**: News sentiment feature from legacy

## Data Flow (AnoFox → MotherDuck → Vercel)

```
[AnoFox Local]
    ↓ Training (User's Machine)
    ↓ Forecasts + SHAP values
    ↓
[MotherDuck] (Cloud Database)
    ↓ Daily sync via Publisher script
    ↓
[Vercel Next.js Dashboard]
    ↓ Fetch from MotherDuck
    ↓
[User's Browser] (Real-time updates)
```

## What NOT to Copy
- ❌ BigQuery infrastructure (legacy)
- ❌ Python ingestion scripts (AnoFox handles this)
- ❌ Training scripts (AnoFox handles this)

## What TO Use
- ✅ Big 8 Drivers concept
- ✅ Page purposes and data relationships
- ✅ Gauge visualization approach
- ✅ SHAP explainability concept
