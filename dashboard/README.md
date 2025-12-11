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
- **Database:** MotherDuck (DuckDB cloud)
- **API Routes:** Native DuckDB with `md:` connection (Vercel compatible)
- **Client Components:** WASM client available (browser-only)
- **Charts:** Lightweight Charts v5
- **Styling:** Tailwind CSS v4
- **Deployment:** Vercel

### Data Flow
```
MotherDuck (cbi_v15) 
  ↓ Native DuckDB (md: protocol)
API Routes (/api/*) → lib/md.ts → DuckDB connection
  ↓
Page Components (fetch from API routes)
  ↓
Lightweight Charts
```

### MotherDuck Connection Architecture

The dashboard uses a **hybrid approach** based on execution context:

**1. API Routes (Server-Side) - Native DuckDB**
- **File**: `lib/md.ts`
- **Method**: Native DuckDB Node.js client with `md:` connection string
- **Connection**: `md:cbi_v15?motherduck_token={token}`
- **Why**: Works in Vercel serverless, full DuckDB SQL support
- **Used by**: All `/api/*` routes
- **Status**: ✅ **Primary method for API routes (Vercel compatible)**

**2. Client Components (Browser) - WASM Client**
- **File**: `lib/motherduck.ts`
- **Method**: `@motherduck/wasm-client` (WASM)
- **Why**: Full DuckDB functionality in browser, better performance
- **Used by**: Client components that need direct database access
- **Status**: ✅ **Available for client-side use**

**Configuration:**
- Environment variable `MOTHERDUCK_TOKEN` required (for both methods)
- Database name defaults to `cbi_v15` (configurable via `MOTHERDUCK_DB`)
- COOP/COEP headers in `next.config.ts` enable WASM support for client-side
- `serverExternalPackages: ["duckdb"]` in `next.config.ts` for native DuckDB

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
# MotherDuck (REQUIRED)
MOTHERDUCK_TOKEN=<your_motherduck_token>
MOTHERDUCK_DB=cbi_v15  # Optional, defaults to cbi_v15

# Vercel (auto-populated)
VERCEL_OIDC_TOKEN=<auto>
```

### Setting Up MotherDuck Token

1. **Get Token:**
   - Log in to [MotherDuck UI](https://app.motherduck.com/)
   - Go to Settings → Create Token
   - Choose "Read/Write" token type
   - Copy the token

2. **Local Development:**
   ```bash
   # Add to dashboard/.env.local
   MOTHERDUCK_TOKEN=your_token_here
   MOTHERDUCK_DB=cbi_v15
   ```

3. **Vercel Production:**
   ```bash
   cd dashboard
   vercel env add MOTHERDUCK_TOKEN
   # Paste token when prompted
   vercel env add MOTHERDUCK_DB
   # Enter: cbi_v15
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
| `lib/md.ts` | Native DuckDB connection (for API routes/server-side) |
| `lib/motherduck.ts` | WASM client (for client components/browser) |
| `lib/lightweight-charts/` | Chart component package |
| `app/api/*/route.ts` | API routes for data fetching |
| `app/components/` | Shared UI components |
| `app/components/TradingViewGauge.tsx` | TradingView-style semicircle gauge |
| `app/components/charts/Gauge.tsx` | Strong sell → Strong buy gauge |
| `app/components/charts/TimeSeriesCharts.tsx` | Nivo-based yield curves |
| `next.config.ts` | Next.js config (COOP/COEP headers for WASM) |
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

**Native DuckDB (API Routes):**
```bash
# Check token is set
echo $MOTHERDUCK_TOKEN

# Verify in Vercel
vercel env ls
```

**Common Errors:**
- `MOTHERDUCK_TOKEN is not defined` → Set environment variable
- `Catalog Error: Table does not exist` → Table/schema doesn't exist yet (expected for new tables)
- `Connection failed` → Check token validity or network connectivity

**WASM Client (Client Components):**
- `Worker is not defined` → Expected in server-side, WASM only works in browser
- `Failed to initialize MotherDuck WASM client` → Check token validity
- Use WASM client only in `'use client'` components, not in API routes

**Debug Steps:**
1. Check `MOTHERDUCK_TOKEN` is set in environment
2. Verify token hasn't expired (check MotherDuck dashboard)
3. For API routes: Use native DuckDB (`lib/md.ts`) - works in Vercel serverless
4. For client components: Use WASM client (`lib/motherduck.ts`) - requires browser environment
5. Test connection: Visit `/api/test-wasm` endpoint
6. See [VERCEL_CONNECTION.md](./VERCEL_CONNECTION.md) for detailed troubleshooting

### Build Failures
- **DuckDB native binary doesn't work on Vercel** → Using WASM client (correct approach)
- Check Next.js version is 16.0.7+
- Verify all dependencies installed: `npm install`
- Ensure `@motherduck/wasm-client` is in `package.json`

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
