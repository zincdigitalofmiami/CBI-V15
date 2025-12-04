# Deployment Checklist

## ✅ Completed

### 1. Vercel Deployment
- [x] Platform deployed to Vercel
- [x] URL: https://web-6fzk3365k-zincdigitalofmiamis-projects.vercel.app
- [x] All 6 pages built successfully:
  - Dashboard
  - Sentiment
  - Strategy
  - Legislation
  - Vegas Intel
  - Admin

### 2. Documentation Created
- [x] Databento Integration Guide
- [x] Python backfill script (`databento_to_motherduck.py`)
- [x] MotherDuck schema design

## 🔄 Next Steps

### 3. Set Vercel Environment Variables

Run these commands:

```bash
cd /Users/zincdigital/.gemini/antigravity/workspaces/zl-intelligence/web

# Databento API Key
vercel env add DATABENTO_API_KEY
# When prompted, paste: db-8uKak7BPpJejVjqxtJ4xnh9sGWYHE

# MotherDuck Token
vercel env add MOTHERDUCK_TOKEN
# When prompted, paste your MotherDuck token

#FRED API Key (optional - for later)
vercel env add FRED_API_KEY
```

### 4. MotherDuck Setup

1. Go to https://motherduck.com
2. Sign up / log in
3. Copy your token
4. Run the backfill script on AnoFox:

```bash
# Install dependencies
pip install databento duckdb

# Set environment variable
export MOTHERDUCK_TOKEN=your_token_here

# Run backfill
python scripts/databento_to_motherduck.py
```

### 5. Create Next.js API Route for MotherDuck

Location: `web/app/api/motherd uck/route.ts`

```typescript
import { NextResponse } from 'next/server';
import duckdb from 'duckdb';

const MOTHERDUCK_TOKEN = process.env.MOTHERDUCK_TOKEN;

export async function GET() {
    try {
        const db = new duckdb.Database(':memory:');
        const conn = db.connect(`md:usoil_intelligence?motherduck_token=${MOTHERDUCK_TOKEN}`);
        
        const result = await conn.all(`
            SELECT date, close, volume
            FROM zl_futures_ohlcv
            WHERE date >= CURRENT_DATE - INTERVAL '1 year'
            ORDER BY date ASC
        `);
        
        return NextResponse.json({ success: true, data: result });
    } catch (error) {
        return NextResponse.json({ success: false, error: error.message }, { status: 500 });
    }
}
```

### 6. Update Dashboard to Use MotherDuck

Replace the data fetch in `Dashboard /page.tsx`:

```typescript
const response = await fetch('/api/motherduck');
const { data } = await response.json();
setChartData(data);
```

## 📋 Final Deployment Steps

1. ✅ Deploy to Vercel
2. ⏳ Add environment variables
3. ⏳ Run Databento backfill
4. ⏳ Create MotherDuck API route
5. ⏳ Update Dashboard to query MotherDuck
6. ⏳ Redeploy to Vercel
7. ⏳ Verify real data is displaying

## 🔐 Credentials Summary

| Service | Key | Location |
|---------|-----|----------|
| Databento | `db-8uKak7BPpJejVjqxtJ4xnh9sGWYHE` | Vercel env |
| MotherDuck | (user's token) | Vercel env |
| Vercel | https://web-6fzk... | Live |
