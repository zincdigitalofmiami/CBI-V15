# Vegas Intel / Sales Intelligence for Kevin (`/vegas-intel`)

## Purpose
Dedicated sales intelligence page for Kevin (Sales Director).

**Objective:**
Turn Glide customer data + historical volumes + Vegas event calendars + ZL forecast context into:
- Upsell targets
- Call lists
- Margin protection alerts

## Audience
- Kevin and the sales team.
- **NOT** a ZL trading or forecasting page.

## Core Concepts

### 1. Event-Driven Upsell Engine
- Pull major events (F1, CES, fights, conventions).
- Compute volume multipliers based on historical data.
- Match events to relevant customers (location/segment).
- Generate ranked list of "call now" opportunities.

### 2. Customer Relationship Dashboard
- Active customers from Glide.
- Relationship tier and `last_order_date`.
- At-risk flags (e.g., >14 days without order).
- Win-back suggestions.

### 3. Margin Protection Alerts
Use ZL price forecasts as context:
- If forecast spike before event → **"LOCK IN EARLY"**.
- If forecast softness → **"DELAY and preserve margin"**.

## Data Sources

### Primary (Glide + Warehouse)
- Glide App export (e.g. `glide_app.json` → `raw.glide_customers`).
- Historical volume:
  - `forecasting_data_warehouse.customer_service_history`
  - or equivalent `raw.customer_volume_history` in MotherDuck.

### Supplementary (Context)
- Vegas event calendars (web/event APIs).
- Visitor demographics (tourism data).

### Model-Driven (Context Only)
- `forecasts.zl_v15_*` (ZL forward price context).
- Internal margin model (ZL price × per-customer margin profile).

## Key Views

### Kevin's Upsell Targets
Table/list of accounts with:
- Customer name, segment, relationship tier.
- Recent volume, trend (up/down/flat).
- Upcoming relevant events (e.g., F1, CES).
- **Suggested action:**
  - "Call now – F1 weekend + prices rising"
  - "Win‑back – 24 days since last order, offer 5%"
  - "Standard check-in"

### Event Calendar
- Major Vegas events with dates.
- Historical volume multipliers per event type.
- Customer segments most affected.

### At-Risk Customers
- Customers with >14 days since last order.
- Declining volume trends.
- Suggested win-back offers.

## Business Logic Examples

### Upsell Timing
```
IF event_date - today < 14 days
AND forecast_trend = 'rising'
AND customer_segment matches event_type
THEN priority = 'HIGH' → "Call now"
```

### Win-Back
```
IF days_since_last_order > 21
AND historical_avg_volume > 50K
THEN suggest_offer = '5% discount'
```

### Margin Protection
```
IF forecast_spike before major_event
THEN alert = "LOCK IN EARLY to protect margin"
```

## Notes
- This page is **strictly separated** from ZL model diagnostics.
- Language and design must be sales-focused:
  - "Call MGM now", "Offer X%", "At-risk", "Win-back".
- No exposure of internal model complexity here.
- ZL forecasts are **context only**, not the primary driver.
