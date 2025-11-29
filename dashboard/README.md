# CBI-V15 Dashboard

Fresh Next.js 15 dashboard for CBI-V15 project.

## Local Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Environment Variables

For Vercel deployment, add:
- `GCP_PROJECT_ID=cbi-v15`
- Service account credentials (JSON or split fields)

## Data Sources

- **Project**: `cbi-v15` (BigQuery)
- **Forecasts**: `forecasts.zl_predictions_1m/1w/3m/6m`
- **Training**: `training.daily_ml_matrix`
- **Raw Data**: `raw.databento_futures_ohlcv_1d`, `raw.fred_economic`, etc.

## Deploy to Vercel

```bash
vercel --prod
```

Or connect GitHub repo to Vercel for automatic deployments.
