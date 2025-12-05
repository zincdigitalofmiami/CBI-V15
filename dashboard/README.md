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

| Route | Purpose | Audience | README |
|-------|---------|----------|--------|
| `/` | Dashboard / Procurement Command Center | Chris (Procurement) | [README](./app/README.md) |
| `/legislation` | Policy & Tariff Intel | Policy analysts | [README](./app/legislation/README.md) |
| `/sentiment` | Big-8 Sentiment Monitor | Quant team | [README](./app/sentiment/README.md) |
| `/strategy` | ZL Strategy & Procurement Plan | Procurement + Quant | [README](./app/strategy/README.md) |
| `/vegas-intel` | Sales Intelligence for Kevin | Sales team | [README](./app/vegas-intel/README.md) |
| `/admin` | Business Configuration | Business admins | [README](./app/admin/README.md) |
| `/quant-admin` | TSci + AnoFox Internal | Quant team (off-menu) | [README](./app/quant-admin/README.md) |

## Charting Package

Reusable chart components built on Lightweight Charts:

📁 **[lib/lightweight-charts/](./lib/lightweight-charts/README.md)**

- `ZLForecastChart.tsx` - Historical + forecast with bands
- `SentimentHeatmap.tsx` - Big-8 bucket heatmap
- `HedgeLadder.tsx` - Volume ladder by horizon
- Usage examples and API docs in package README

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
- `features.daily_ml_matrix_zl_v15` - Big-8 scores + features
- `training.daily_ml_matrix_zl_v15` - Training data with targets
- `forecasts.zl_v15_*` - Point forecasts + bands by horizon
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
| `next.config.ts` | Next.js config (COOP/COEP headers) |

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
