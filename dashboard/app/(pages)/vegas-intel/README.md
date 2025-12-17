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

## Files

| File       | Purpose                         |
| ---------- | ------------------------------- |
| `page.tsx` | Main Vegas Intel page component |
| `_files/`  | Page-specific components        |

---

## C1) Event Ingestion

### Sources

- **Ticketmaster** - Concert/show events
- **Eventbrite** - Convention/conference events
- **Vegas.com** - Casino/entertainment events
- **Proprietary convention calendars** - Trade shows, expos

### Schema (MotherDuck)

| Table                      | Purpose                  |
| -------------------------- | ------------------------ |
| `raw.events`               | Normalized event records |
| `raw.venues`               | Venue metadata           |
| `raw.casinos`              | Casino properties        |
| `raw.event_tickets`        | Optional ticket data     |
| `raw.hotel_occupancy`      | Occupancy/traffic data   |
| `raw.foot_traffic`         | Foot traffic metrics     |
| `raw.event_source_records` | Raw ingestion records    |
| `ops.ingestion_jobs`       | Job tracking             |
| `ops.ingestion_logs`       | Ingestion logs           |

### Ingestion Schedule

- **Nightly Cloud Run Job** pulls updates
- Writes to `ops.ingestion_jobs` and `ops.ingestion_logs`

---

## C2) Opportunity Model

### Event Weight Formula

```
event_weight(t) = attendance
                × venue_capacity_factor
                × recency_decay(t)
                × confidence
```

### Predicted Volume Delta

```
predicted_delta_lbs = event_weight
                    × cf_attendee_to_oil(cuisine, fryer_count, baseline)
                    × duration_days / 7
```

### Priority Ranking

```
priority = rank(
    predicted_delta_lbs,
    lead_days,
    is_current_customer,
    venue_tier,
    relationship_score
)
```

---

## C3) Outreach Generation

Button → `/outreach` builds subject/body using:

1. **Event facts** - Attendance, timing, headliner
2. **Restaurant baseline** - Fryer capacity, cuisine type
3. **Relationship memory** - Notes, last contact, tone

**CRITICAL:** Every draft is explainable and editable before send.

---

## Core Concepts

### 1. Event-Driven Upsell Engine

- Pull major events (F1, CES, fights, conventions)
- Compute volume multipliers based on historical data
- Match events to relevant customers (location/segment)
- Generate ranked list of "call now" opportunities

### 2. Customer Relationship Dashboard

- Active customers from Glide
- Relationship tier and `last_order_date`
- At-risk flags (e.g., >14 days without order)
- Win-back suggestions

### 3. Margin Protection Alerts

Use ZL price forecasts as context:

- If forecast spike before event → **"LOCK IN EARLY"**
- If forecast softness → **"DELAY and preserve margin"**

---

## Data Sources

### Primary (Glide + Warehouse)

| Source    | Table                         | Description              |
| --------- | ----------------------------- | ------------------------ |
| Glide App | `raw.glide_customers`         | Customer CRM data        |
| Warehouse | `raw.customer_volume_history` | Historical order volumes |

### Supplementary (Context)

| Source     | Table                 | Description           |
| ---------- | --------------------- | --------------------- |
| Event APIs | `raw.events`          | Vegas event calendars |
| Tourism    | `raw.hotel_occupancy` | Visitor demographics  |

### Model-Driven (Context Only)

| Source       | Table                      | Description                            |
| ------------ | -------------------------- | -------------------------------------- |
| Forecasts    | `forecasts.zl_predictions` | ZL forward price context               |
| Margin Model | Internal                   | ZL price × per-customer margin profile |

---

## Key Views

### Kevin's Upsell Targets

Table/list of accounts with:

- Customer name, segment, relationship tier
- Recent volume, trend (up/down/flat)
- Upcoming relevant events (e.g., F1, CES)
- **Suggested action:**
  - "Call now – F1 weekend + prices rising"
  - "Win‑back – 24 days since last order, offer 5%"
  - "Standard check-in"

### Event Calendar

- Major Vegas events with dates
- Historical volume multipliers per event type
- Customer segments most affected

### At-Risk Customers

- Customers with >14 days since last order
- Declining volume trends
- Suggested win-back offers

---

## Business Logic Examples

### Upsell Timing

```sql
IF event_date - today < 14 days
AND forecast_trend = 'rising'
AND customer_segment matches event_type
THEN priority = 'HIGH' → "Call now"
```

### Win-Back

```sql
IF days_since_last_order > 21
AND historical_avg_volume > 50K
THEN suggest_offer = '5% discount'
```

### Margin Protection

```sql
IF forecast_spike before major_event
THEN alert = "LOCK IN EARLY to protect margin"
```

---

## Notes

- This page is **strictly separated** from ZL model diagnostics
- Language and design must be sales-focused:
  - "Call MGM now", "Offer X%", "At-risk", "Win-back"
- No exposure of internal model complexity here
- ZL forecasts are **context only**, not the primary driver

## Visual Design

### DashdarkX Theme

- **Background:** `rgb(0, 0, 0)` - pure black
- **Event cards:** `border-zinc-800` with `bg-zinc-900/20` hover
- **Event details:** `text-zinc-400 font-extralight`
- **Email script:** `bg-black border-zinc-800` with monospace `font-extralight`
- **All text:** `font-thin` headers, `font-extralight` body

### Priority Color Coding

| Priority | Color  | Action             |
| -------- | ------ | ------------------ |
| HIGH     | Red    | Call NOW           |
| MEDIUM   | Yellow | Schedule this week |
| LOW      | Gray   | Standard check-in  |

### Customer Status Colors

| Status   | Color  | Indicator           |
| -------- | ------ | ------------------- |
| Active   | Green  | `bg-green-500` dot  |
| At-Risk  | Yellow | `bg-yellow-500` dot |
| Win-back | Red    | `bg-red-500` dot    |

### Event Calendar

- Card-based layout with event images
- Attendance badge with multiplier
- Date countdown (e.g., "3 days away")
- Customer match indicators

### Outreach Panel

- Editable email draft with preview
- Event facts highlighted
- Relationship context shown
- One-click copy or send
