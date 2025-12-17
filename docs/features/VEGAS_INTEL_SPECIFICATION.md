# V15 Vegas Intel - Complete Specification

## Overview

**V15 Vegas Intel** is a dedicated sales intelligence command center for US Oil Solutions that transforms event data, customer behavior, and demand patterns into actionable sales opportunities. This feature is completely separate from ZL commodity forecasting and focuses exclusively on restaurant demand prediction driven by Las Vegas events.

## Core Purpose

Sales intelligence tool that converts:

- Las Vegas event calendars
- Customer behavior patterns
- Historical volume data
- Restaurant capacity factors

Into prioritized sales actions and margin protection guidance.

## Primary Demand Drivers (Restaurant Oil Volume)

### 1. Event Size (Attendance)

- Larger events = More people in Vegas = More restaurant traffic
- Formula: `event_weight = attendance × venue_capacity_factor × confidence`

### 2. Geographic Proximity (Venue ↔ Restaurant Location)

- Events at nearby venues drive foot traffic to nearby restaurants
- Example: Concert at T-Mobile Arena → restaurants on The Strip get surge

### 3. Demographic Match (Event Type ↔ Restaurant Type)

Event demographic must align with restaurant's target customer:

| Event Type     | Target Restaurants                              | Rationale                              |
| -------------- | ----------------------------------------------- | -------------------------------------- |
| F1 Race        | Fine dining, upscale steakhouses                | High-end, international crowd          |
| CES Convention | Quick-service, casual dining, hotel restaurants | Tech professionals, business travelers |
| EDM Festival   | Nightclubs, late-night casual spots             | Young, party crowd                     |
| UFC Fight      | Sports bars, gastropubs                         | Sports fans, middle-upper income       |

### Demand Prediction Formula

```
predicted_delta_volume = event_attendance
                        × proximity_factor (closer venue = higher)
                        × demographic_match_score (0-1)
                        × restaurant_capacity_factor (fryers, kitchen size)
                        × event_duration_days
```

## UI Layout - Text Wireframe

### Navigation

```
> Dashboard
> Upsell Targets
> Event Calendar
> At-Risk Customers
> Outreach
> Margin Alerts
```

### Primary KPI Bar

```
| High-Priority Calls: 6 | At-Risk Accounts: 3 | Major Events This Week: 4 |
```

### Priority Levels

| Priority | Color  | Action             | Card Style                 |
| -------- | ------ | ------------------ | -------------------------- |
| HIGH     | Red    | Call NOW           | Red border, urgent styling |
| MEDIUM   | Yellow | Schedule this week | Yellow border              |
| LOW      | Gray   | Standard check-in  | Gray border                |

### Customer Status Indicators

| Status   | Color  | Indicator           |
| -------- | ------ | ------------------- |
| Active   | Green  | `bg-green-500` dot  |
| At-Risk  | Yellow | `bg-yellow-500` dot |
| Win-back | Red    | `bg-red-500` dot    |

## Key Screens

### 1. Upsell Targets (Home Screen)

High Priority Customer Cards:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Customer: MGM Grand (Tier A)                                                 │
│ Volume Trend: ↑ Rising                                                       │
│ Upcoming Event: CES (140,000 attendees) – 3 days out                         │
│ Urgency: HIGH — Call Now                                                     │
│ Action Buttons: [Generate Outreach] [View Profile]                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 2. Event Calendar

Event Cards:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Image: CES Logo / Vegas Convention Center                                    │
│ Event: CES 2026                                                              │
│ Attendance: 140,000 (Impact Multiplier: 2.4x)                                │
│ Starts In: 3 Days                                                            │
│ Customer Segments Affected: Buffets, Casual Dining, Resorts                  │
│ Button: [View Affected Customers]                                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3. At-Risk Customers

Win-Back Cards:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Customer: In-N-Out (Tier B)                                                  │
│ Last Order: 24 Days Ago                                                      │
│ Volume Trend: Sharp Decline                                                  │
│ Recommendation: Win-Back — Offer 5%                                          │
│ Button: [Generate Outreach]                                                  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4. Outreach Generator

**Left Column (Auto-Filled Inputs):**

```
Event: CES – 140,000 attendees
Customer: MGM Grand
Fryer Setup: 8 Fryers (High Burn)
Baseline: 325 lb/week
Notes: Prefers concise emails, responds mornings
```

**Right Column (Editable Email Draft):**

```
Subject: Getting Ahead of CES Demand

Body (editable):

Hey team —

CES kicks off in 3 days with an expected 140K attendees.

Based on your 8-fryer setup and typical demand pattern during large
conventions, you'll likely see elevated usage.

Let's get ahead of the surge and secure supply now.

Let me know your ideal drop time.

[Copy Email] [Send via CRM] [Regenerate Draft]
```

## Kevin's Daily 5-Minute Workflow

### Step 1: Open "Upsell Targets"

- View ranked call list
- Customers highlighted by event-driven demand bumps

### Step 2: Make First 2-3 Calls Immediately

- Actions: Call Now, Win-back, Standard Check-In

### Step 3: Open "At-Risk Customers"

- 14+ days without order
- Volume trending down
- Behavioral changes

### Step 4: Look at Margin Alerts

- **LOCK IN EARLY** (price spike expected)
- **DELAY** (soft pricing expected)

### Step 5: Use Outreach Generator

- Generate → Edit → Send
- Auto-includes: Event info, fryer setup, historical patterns, past notes

## Business Logic

### Upsell Timing

```
IF event_date - today < 14 days
AND customer_segment matches event_type
AND proximity_factor > threshold
THEN priority = 'HIGH' → "Call now"
```

### Win-Back Strategy

```
IF days_since_last_order > 21
AND historical_avg_volume > 50K
THEN suggest_offer = '5% discount'
```

### Demographic Matching

```
IF event_demographic matches restaurant_target_demo
AND venue_distance < 1 mile
AND event_size > minimum_threshold
THEN predicted_volume_increase = HIGH
```

## Data Architecture

### Event Ingestion (C1)

**Sources:**

- Ticketmaster — concerts, shows
- Eventbrite — conventions, conferences
- Vegas.com — entertainment, casino events
- Proprietary convention calendars — expos, trade shows

**Ingestion Schedule:** Nightly Cloud Run job updates all event tables and metadata.

### MotherDuck Schema

| Table                      | Purpose                        |
| -------------------------- | ------------------------------ |
| `raw.events`               | Normalized event records       |
| `raw.venues`               | Venue metadata                 |
| `raw.casinos`              | Casino listings                |
| `raw.event_tickets`        | Optional ticket metadata       |
| `raw.hotel_occupancy`      | Occupancy + tourism indicators |
| `raw.foot_traffic`         | Traffic patterns               |
| `raw.event_source_records` | Source traceability            |
| `ops.ingestion_jobs`       | Job tracking                   |
| `ops.ingestion_logs`       | Logging for ingestion events   |

### Opportunity Model (C2)

**Event Weight Formula:**

```
event_weight(t) = attendance × venue_capacity_factor × recency_decay(t) × confidence
```

**Predicted Volume Delta:**

```
predicted_delta_lbs = event_weight × cf_attendee_to_oil(cuisine, fryer_count, baseline) × duration_days / 7
```

**Priority Ranking:**

```
priority = rank(predicted_delta_lbs, lead_days, is_current_customer, venue_tier, relationship_score)
```

### Outreach Generation (C3)

When Kevin selects a customer and clicks "Generate Outreach," the system constructs a draft using:

- **Event Facts** — headliner, timing, attendance
- **Restaurant Baseline** — fryer count, cuisine, historical usage
- **Relationship Memory** — tone, notes, last contact

**CRITICAL:** Kevin always reviews and edits before sending. No automated sending.

## Core Intelligence Concepts

### 1. Event-Driven Upsell Engine

- Detect major events (CES, F1, fights, concerts)
- Quantify their likely impact on oil demand
- Match events to customer segments and geography
- Produce ranked "call now" opportunities

### 2. Customer Relationship Dashboard

- Daily view of customer activity
- Last order date + relationship tier
- At-risk and win-back signals
- Volume trend indicators

### 3. Margin Protection Alerts

ZL forecast context is simplified into:

- **LOCK IN EARLY** (spike expected)
- **DELAY** (soft pricing expected)

Sales does not see models—only the recommended timing action.

## Data Sources

### Primary Sources

| Source    | Table                         | Description       |
| --------- | ----------------------------- | ----------------- |
| Glide     | `raw.glide_customers`         | CRM data          |
| Warehouse | `raw.customer_volume_history` | Historical orders |

### Supplementary Sources

| Source       | Table                 | Description                |
| ------------ | --------------------- | -------------------------- |
| Event APIs   | `raw.events`          | All events                 |
| Tourism Data | `raw.hotel_occupancy` | Visitor demographic trends |

### Model Context Only

| Source       | Table                      | Description                       |
| ------------ | -------------------------- | --------------------------------- |
| Forecasts    | `forecasts.zl_predictions` | Forward guidance                  |
| Margin Model | Internal                   | Customer-specific margin exposure |

## Key Views

### Kevin's Upsell Targets

Displays:

- Customer name, segment, tier
- Volume trend
- Event impact badges
- Recommended action

**Examples:**

- "Call now – CES surge + rising prices"
- "Win-back – 24 days since last order, offer 5%"
- "Standard check-in"

### Event Calendar

Shows:

- Major Vegas events
- Attendance + multiplier
- Customer segments affected
- Days remaining countdown

### At-Risk Customers

- 14 days since last order
- Volume decline flags
- Recommended win-back strategy

## UI Implementation Notes

### Component Structure

- **Event cards:** Card-based layout with images, attendance badges, countdown
- **Priority indicators:** Color-coded borders based on urgency levels
- **Customer status:** Visual indicators for Active/At-Risk/Win-back states
- **Outreach generator:** Split-panel design with auto-filled inputs and editable draft

## Key Principles

### NO ZL Connection

- This is purely restaurant demand forecasting
- Event-driven volume predictions for sales targeting
- No commodity pricing, no soybean oil forecasting here

### Sales-Focused Language

- "Call NOW", "Win-back", "At-risk"
- Tactical cockpit design
- Fast, clear, directive, minimal cognitive load

### Editable Drafts

- All outreach is reviewable before sending
- Explainable and customizable
- No automated sending

## Implementation Notes

- **Audience:** Kevin (Sales Director) and sales team
- **Separation:** Strictly separated from ZL model diagnostics
- **Language:** Sales-focused, never analytic
- **ZL Context:** Forecasts are context only, not primary driver
- **No Model Exposure:** Internal model complexity never exposed

This specification defines V15 Vegas Intel as a complete sales intelligence platform that turns Las Vegas event data into revenue-generating opportunities through smart customer targeting and timing.
