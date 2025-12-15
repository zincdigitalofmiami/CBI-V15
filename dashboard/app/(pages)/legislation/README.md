# Legislation & Policy Intel (`/legislation`)

## Purpose
Centralized view of policy and regulatory events that materially impact soybean oil:
- Tariffs and trade policy
- Farm Bill developments
- RFS and biofuel mandates
- Logistics and export regulations

## Audience
- Chris (Procurement) - policy context for timing decisions
- Internal analysts - tracking regulatory pipeline
- **NOT** a quantitative or model page

## Files

| File | Purpose |
|------|---------|
| `page.tsx` | Main legislation page component |
| `_files/` | Page-specific components (empty - uses shared components) |

## Key Views

### 1. US Policy Stream
- Cards by event (RFS, Farm Bill, EPA, USDA, CBP, DHS)
- Tag with "Biofuel", "Logistics", "Trade", etc.
- Links to source documents (Federal Register, etc.)

### 2. China / Trade Policy Stream
- Section 301 moves, retaliation, WTO disputes
- Mapped to `tariff` and `china` buckets
- Historical impact correlation shown

### 3. Impact Timeline
- 30–90 day view of key events with bullish/bearish/neutral impact tags
- Color-coded by impact direction
- Hover for detailed context

## Data Sources (MotherDuck)

| Table | Purpose | Update Frequency |
|-------|---------|------------------|
| `raw.news_articles` | Source articles by bucket | Every 15 min |
| `raw.gdelt_events` | Trade/tariff events | Daily |
| `staging.news_topic_signals` | Aggregated sentiment/intensity | Hourly |

### SQL Example
```sql
SELECT 
    date,
    title,
    source,
    bucket,
    impact_tag,
    summary
FROM raw.news_articles
WHERE bucket IN ('tariff', 'biofuel', 'fed', 'logistics')
  AND date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY date DESC;
```

## Event Tags

### Buckets
| Bucket | Description | Sources |
|--------|-------------|---------|
| `tariff` | Trade policy, Section 301, WTO | USTR, CBP, Federal Register |
| `biofuel` | RFS, biodiesel, renewable diesel mandates | EPA, EIA, ProFarmer |
| `fed` | Monetary policy, interest rates | Fed, FOMC, FRED |
| `logistics` | Export regulations, CBP, DHS | CBP, DHS, USDA |

### Impact Tags
| Tag | Meaning | Color |
|-----|---------|-------|
| `bullish` | Positive for ZL prices | Green |
| `bearish` | Negative for ZL prices | Red |
| `neutral` | Informational only | Gray |

## Business Logic Examples

### Policy Impact Scoring
```
IF policy_type = 'RFS_MANDATE'
AND direction = 'INCREASE'
THEN impact = 'bullish'
AND expected_effect = '+$0.02-0.05/lb'
```

### Tariff Alert
```
IF country = 'China'
AND policy_type = 'Section 301'
AND change = 'ESCALATION'
THEN impact = 'bearish'
AND alert_level = 'HIGH'
```

## UI Components

- Policy cards with source badges
- Timeline view with date markers
- Filter by bucket, impact, date range
- Expandable detail panels

## Notes
- This page is **explanatory and contextual** - does NOT issue BUY/WAIT/MONITOR signals
- Impact mapping (event → $$) happens in SQL or AutoGluon, not in React
- Link directly to source documents where available
- Heavy calculation should be precomputed in `staging.news_topic_signals`

## Visual Design

### DashdarkX Theme
- **Background:** `rgb(0, 0, 0)` - pure black
- **Text:** White with `font-thin` headers, `font-extralight` body
- **Policy cards:** `border-zinc-800` with hover `bg-zinc-900/20`
- **Event titles:** `font-extralight`
- **Deadline text:** `text-zinc-400 font-extralight`

### Color Coding
| Impact | Color | CSS |
|--------|-------|-----|
| Bullish | Green | `text-green-400` |
| Bearish | Red | `text-red-400` |
| Neutral | Gray | `text-zinc-400` |

### Timeline Component
- Horizontal strip with date markers
- Color-coded impact dots
- Hover for detail expansion
