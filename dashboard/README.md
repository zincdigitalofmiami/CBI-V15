# CBI-V15 Dashboard

**ZL Soybean Oil Procurement Intelligence Platform**

---

## Overview

The CBI-V15 Dashboard is a Next.js application that provides procurement intelligence for soybean oil (ZL) trading and sales. It combines quantitative forecasting, policy intelligence, and sales CRM data to support decision-making across procurement, sales, and operations.

## Quick Start

```bash
cd dashboard
npm install
npm run dev
```

Visit `http://localhost:3000`

## Architecture

### Tech Stack
- **Framework:** Next.js 16 (App Router)
- **Database:** MotherDuck (DuckDB cloud) via HTTP API
- **Charts:** Lightweight Charts v5
- **Styling:** Tailwind CSS v4
- **Deployment:** Vercel

### Data Flow
```
MotherDuck (cbi_v15) 
  ↓ HTTP API
lib/md.ts (query layer)
  ↓
API Routes (/api/*)
  ↓
Page Components
  ↓
Lightweight Charts
```

## Pages

| Route | Purpose | Audience |
|-------|---------|----------|
| `/` | Dashboard / Procurement Command Center | Chris (Procurement) |
| `/legislation` | Policy & Tariff Intel | Policy analysts |
| `/sentiment` | Big-8 Sentiment Monitor | Quant team |
| `/strategy` | ZL Strategy & Procurement Plan | Procurement + Quant |
| `/vegas-intel` | Sales Intelligence for Kevin | Sales team |
| `/admin` | Business Configuration | Business admins |
| `/quant-admin` | TSci + AnoFox Internal | Quant team (off-menu) |

### Page Details

#### Dashboard (`/`)
- ZL "Breathing" Chart (2-5yr historical + 6M forecast)
- Chris's Four Factors (China, Harvest, Biofuel, Palm Spread)
- Drivers Panel (Big-8 translated to business language)
- Forward Curve & BUY/WAIT/MONITOR recommendation

#### Legislation (`/legislation`)
- US Policy Stream (RFS, Farm Bill, EPA, USDA)
- China/Trade Policy Stream (Section 301, WTO)
- Impact Timeline (30-90 day view with bullish/bearish tags)

#### Sentiment (`/sentiment`)
- Big-8 Heatmap (bucket × time)
- Regime Timeline (historical regime labels)
- Bucket Detail Panel (time series, feature importance, news)

#### Strategy (`/strategy`)
- Strategy Summary (TSci recommendation + rationale)
- Hedge Ladder (volume by horizon, color-coded urgency)
- Scenario Panel (China Soft, Tariffs Escalate, Weather Shock, Fed Pivot)
- P&L Distribution (percentile bands from Monte Carlo)

#### Vegas Intel (`/vegas-intel`)
- **C1) Event Ingestion** - Ticketmaster, Eventbrite, Vegas.com → `raw.events`
- **C2) Opportunity Model** - `event_weight × cf_attendee_to_oil × duration/7`
- **C3) Outreach Generation** - Button → `/outreach` with event facts + relationship memory
- Kevin's Upsell Targets, Event Calendar, At-Risk Customers

#### Admin (`/admin`)
- App Settings (base volume, currency, contract)
- Risk Thresholds (green/yellow/red per score)
- Visibility Toggles (show/hide dashboard tiles)

#### Quant Admin (`/quant-admin`)
- Pipeline Health (ingestion lag, error counts)
- Matrix Status (row counts, gaps, coverage)
- Model Registry (champion vs challenger, metrics)
- TSci Runs (status, logs, narratives)

## Charting Package

Reusable chart components built on [Lightweight Charts v5](https://tradingview.github.io/lightweight-charts/).

📁 **[lib/lightweight-charts/](./lib/lightweight-charts/README.md)**

### Components

| Component | Purpose | Used On |
|-----------|---------|---------|
| `ZLForecastChart.tsx` | Historical + forecast with confidence bands | Dashboard `/` |
| `SentimentHeatmap.tsx` | Big-8 bucket heatmap over time | Sentiment `/sentiment` |
| `HedgeLadder.tsx` | Volume ladder by horizon | Strategy `/strategy` |

### Dark Theme Configuration
```ts
{
  layout: { background: { color: '#0a0e1a' }, textColor: '#9ca3af' },
  grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } }
}
```

### Usage Pattern
```tsx
'use client';

import { useEffect, useRef } from 'react';

export function Chart({ data }) {
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!chartRef.current) return;
    
    // Dynamic import to avoid SSR issues
    import('lightweight-charts').then(({ createChart }) => {
      const chart = createChart(chartRef.current!, { /* options */ });
      const lineSeries = chart.addLineSeries({ color: '#ffffff', lineWidth: 2 });
      lineSeries.setData(data);
      
      return () => chart.remove(); // Cleanup on unmount
    });
  }, [data]);

  return <div ref={chartRef} className="h-full w-full" />;
}
```

### Best Practices
- **Dynamic imports** - Avoid SSR issues with `import('lightweight-charts')`
- **Cleanup** - Always `chart.remove()` on unmount
- **Data format** - Unix timestamps in seconds: `Math.floor(Date.now() / 1000)`
- **Performance** - Limit to ~2000 points, use `setData()` for initial, `update()` for incremental

### Additional Chart Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `TradingViewGauge.tsx` | `app/components/` | Semicircle gauge with SHAP impact |
| `Gauge.tsx` | `app/components/charts/` | Strong sell → Strong buy gauge |
| `TimeSeriesCharts.tsx` | `app/components/charts/` | Nivo-based yield curves, fundamentals |

## Database Schema

### MotherDuck (`cbi_v15`)

| Schema | Purpose | Owner |
|--------|---------|-------|
| `raw` | Immutable source data | AnoFox |
| `staging` | Transformations | AnoFox |
| `features` | Feature matrices | AnoFox |
| `training` | ML training data | AnoFox |
| `forecasts` | Model predictions | AnoFox |
| `reference` | Catalogs & registries | AnoFox |
| `ops` | Pipeline health | AnoFox |
| `tsci` | Agent jobs & QA | TSci |

**Key Tables:**
- `features.daily_ml_matrix_zl` - Big-8 scores + features
- `training.daily_ml_matrix_zl` - Training data with targets
- `forecasts.zl_predictions` - Point forecasts + bands by horizon
- `reference.model_registry` - Active models
- `tsci.runs` - TSci experiment runs

See [Database Audit Report](../../docs/database_audit_report.md) for current state.

## Environment Variables

```bash
# MotherDuck
MOTHERDUCK_TOKEN=<your_token>
MOTHERDUCK_READ_SCALING_TOKEN=<optional>

# Vercel (auto-populated)
VERCEL_OIDC_TOKEN=<auto>
```

## Development

### Running Locally
```bash
npm run dev
```

### Building
```bash
npm run build
```

### Linting
```bash
npm run lint
```

## Deployment

### Vercel (Production)

The dashboard is deployed to Vercel and linked to the `cbi-v15` project.

```bash
# Deploy to production
vercel --prod

# Check deployment status
vercel ls
```

**Production URL:** https://cbi-v15-7qj13f6ef-zincdigitalofmiamis-projects.vercel.app

See [VERCEL_CONNECTION.md](./VERCEL_CONNECTION.md) for troubleshooting.

## Design Principles

### Visual Design (DashdarkX Theme)

**Pure Black & Ultra-Thin Aesthetic:**
- Background: `rgb(0, 0, 0)` - pure black, no gradients
- Text: White (`rgb(255, 255, 255)`) 
- Font weight: `100` (ultra-thin) for headers, `200` for emphasis
- Borders: `border-zinc-800` for subtle definition
- Glass panels: `bg-zinc-950/40 border border-zinc-800 backdrop-blur-sm`

**Typography:**
- Headers: `font-thin` with increased letter spacing
- Body: `font-extralight`
- Numbers: `font-mono font-extralight`

**TradingView-Inspired:**
- Scrollbar: 6px width, pure black track, white thumb (15% opacity)
- Charts: Dark grid (`#1f2937`), white lines
- Gauges: Semicircle style with value indicator

### For Chris (Procurement)
- **No quant jargon** - Translate Big-8 to business language
- **Clear actions** - BUY / WAIT / MONITOR with confidence
- **Dollar impacts** - Show $/cwt for every driver

### For Kevin (Sales)
- **Event-driven** - Match Vegas events to customer opportunities
- **Call lists** - Ranked upsell targets with timing
- **Margin protection** - Use ZL forecasts as context only

### For Quant Team
- **Full transparency** - Expose pipeline health, model metrics
- **No simplification** - Show raw SHAP, features, QA checks
- **Experiment tracking** - TSci runs and narratives

## Key Files

| File | Purpose |
|------|---------|
| `lib/md.ts` | MotherDuck query layer (HTTP API) |
| `lib/motherduck.ts` | WASM client (not used in prod) |
| `lib/lightweight-charts/` | Chart component package |
| `app/api/*/route.ts` | API routes for data fetching |
| `app/components/` | Shared UI components |
| `app/components/TradingViewGauge.tsx` | TradingView-style semicircle gauge |
| `app/components/charts/Gauge.tsx` | Strong sell → Strong buy gauge |
| `app/components/charts/TimeSeriesCharts.tsx` | Nivo-based yield curves |
| `next.config.ts` | Next.js config (COOP/COEP headers) |
| `globals.css` | Pure black theme, ultra-thin fonts |
| `tailwind.config.ts` | Color palette (zinc-800 borders, black backgrounds) |

## Big-8 Buckets Reference

| Bucket | Score Column | Description | Key Inputs |
|--------|--------------|-------------|------------|
| `crush` | `crush_bucket_score` | Soybean crush margins | ZL, ZS, ZM spreads |
| `china` | `china_bucket_score` | China import demand | HG copper, export sales |
| `fx` | `fx_bucket_score` | Currency effects | DX, BRL, CNY |
| `fed` | `fed_bucket_score` | Monetary policy | Fed funds, yield curve |
| `tariff` | `tariff_bucket_score` | Trade policy risk | Trump sentiment, Section 301 |
| `biofuel` | `biofuel_bucket_score` | RFS/biodiesel | RIN prices, mandates |
| `energy` | `energy_bucket_score` | Energy markets | CL, HO, RB |
| `vol` | `volatility_bucket_score` | Market volatility | VIX, realized vol |

## Hedge Horizons

| Horizon | Trading Days | Description | Typical Volume |
|---------|--------------|-------------|----------------|
| 1W | 5 | Immediate needs | 100K lbs |
| 1M | 20 | Near-term procurement | 250K lbs |
| 3M | 60 | Quarterly planning | 500K lbs |
| 6M | 126 | Strategic positioning | 1M lbs |

## Troubleshooting

### MotherDuck Connection Issues
- Check `MOTHERDUCK_TOKEN` is set
- Verify token hasn't expired
- See [VERCEL_CONNECTION.md](./VERCEL_CONNECTION.md)

### Build Failures
- DuckDB native binary doesn't work on Vercel (use HTTP API)
- Check Next.js version is 16.0.7+
- Verify all dependencies installed

### Empty Data
- Check [Database Audit Report](../../docs/database_audit_report.md)
- Most raw tables are empty - ingestion pipeline needs to run
- Only `features` and `training` have sample data (22 rows)

## Contributing

1. Create feature branch
2. Make changes
3. Test locally with `npm run dev`
4. Build with `npm run build`
5. Deploy preview with `vercel`
6. Merge to `main` for production deploy

## License

Proprietary - US Oil Solutions / ZL Intelligence
