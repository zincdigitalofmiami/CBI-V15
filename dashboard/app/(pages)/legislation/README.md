# Legislation & Policy Intel (`/legislation`)

## Purpose
Centralized view of policy and regulatory events that materially impact soybean oil:
- Tariffs and trade policy
- Farm Bill developments
- RFS and biofuel mandates
- Logistics and export regulations

## Key Views

### 1. US Policy Stream
- Cards by event (RFS, Farm Bill, EPA, USDA, CBP, DHS).
- Tag with "Biofuel", "Logistics", "Trade", etc.

### 2. China / Trade Policy Stream
- Section 301 moves, retaliation, WTO disputes.
- Mapped to `tariff` and `china` buckets.

### 3. Impact Timeline
- 30–90 day view of key events with bullish/bearish/neutral impact tags.

## Data Sources (MotherDuck)
- `raw.news_articles`
  - Filter by `source`, `tags`, and `bucket` (tariff/biofuel/fed/logistics).
- `raw.gdelt_events`
  - Trade and tariff-related events.
- `staging.news_topic_signals`
  - Aggregated sentiment and intensity per bucket.

## Event Tags

### Buckets
- `tariff` - Trade policy, Section 301, WTO
- `biofuel` - RFS, biodiesel, renewable diesel mandates
- `fed` - Monetary policy, interest rates
- `logistics` - Export regulations, CBP, DHS

### Impact
- `bullish` - Positive for ZL prices
- `bearish` - Negative for ZL prices
- `neutral` - Informational only

## Notes
- This page is explanatory and contextual; it does not issue BUY/WAIT/MONITOR signals.
- Modeling impact mapping (event → $$) should happen in SQL or TSci, not in the React component.
- Link directly to source documents (Federal Register, ProFarmer, etc.) where available.
