# ProFarmer Credentials Setup

**CRITICAL:** ProFarmer is the most important news source. All URLs must be scraped daily.

---

## Required Credentials

**Add to `.env.local` and Vercel:**

```bash
PROFARMER_USERNAME=your_username_here
PROFARMER_PASSWORD=your_password_here
```

---

## ProFarmer URLs Being Scraped

### CRITICAL Priority (Must Run Daily)

**Daily Editions:**
- `/news/first-thing-today` - Pre-market edition
- `/news/ahead-of-the-open` - Pre-market edition
- `/news/after-the-bell` - Post-market edition

**Commodity Pages:**
- `/markets/soybeans` - ZS coverage
- `/markets/soybean-oil` - ZL coverage (PRIMARY TARGET)
- `/markets/soybean-meal` - ZM coverage

### HIGH Priority

**News Sections:**
- `/news/agriculture-news` - General ag news
- `/news/markets` - Market updates
- `/news/policy` - Policy changes
- `/news/weather` - Weather impacts

**Analysis:**
- `/analysis/grains` - Grain market analysis
- `/markets/crude-oil` - Energy correlation

**Weather:**
- `/weather/forecast` - Weather forecasts
- `/weather/crop-conditions` - Crop condition reports

### MEDIUM Priority

**Newsletters:**
- `/newsletters` - All newsletters
- `/newsletters/weekly-outlook` - Weekly outlook

**Analysis:**
- `/analysis/livestock` - Livestock markets
- `/analysis/energy` - Energy markets

**Commodities:**
- `/markets/corn` - Corn markets
- `/markets/wheat` - Wheat markets

---

## Scraping Schedule

**3x Daily (Covers All Market Cycles):**
- **6 AM UTC (1 AM ET)** - Pre-market: First Thing Today, Ahead of the Open
- **12 PM UTC (7 AM ET)** - Intraday: Agriculture News, Market News, Weather
- **6 PM UTC (1 PM ET)** - Post-market: After the Bell, Analysis

---

## Trigger.dev Job

**File:** `trigger/profarmer_all_urls.ts`

**Features:**
- Anchor browser automation (handles JavaScript)
- Authenticated login
- 500-word content limit per article
- Priority-based scraping
- Error handling per URL

**Trigger Manually:**
```bash
# All CRITICAL + HIGH priority URLs
npx trigger.dev@latest trigger profarmer-all-urls

# Only CRITICAL URLs
npx trigger.dev@latest trigger profarmer-all-urls --payload '{"priorities": ["CRITICAL"]}'
```

---

## Verification

**After adding credentials, test:**

```bash
# Set environment
export PROFARMER_USERNAME="your_username"
export PROFARMER_PASSWORD="your_password"
export ANCHOR_API_KEY="your_anchor_key"

# Test scraper
npx trigger.dev@latest dev

# In another terminal
npx trigger.dev@latest trigger profarmer-all-urls
```

**Check MotherDuck:**
```sql
SELECT 
    edition_type,
    priority,
    COUNT(*) as article_count,
    MAX(published_at) as latest_article
FROM raw.bucket_news
WHERE source = 'ProFarmer'
GROUP BY edition_type, priority
ORDER BY priority, edition_type;
```

---

## Troubleshooting

### Login Fails
- Verify credentials are correct
- Check if ProFarmer changed login page structure
- Inspect Anchor logs for errors

### No Articles Found
- ProFarmer may have changed HTML structure
- Update selectors in `profarmer_all_urls.ts`
- Check if URL paths changed

### Rate Limiting
- ProFarmer may rate limit aggressive scraping
- Add delays between URLs if needed
- Contact ProFarmer support if blocked

---

## Cost Considerations

**Trigger.dev Limits:**
- Free tier: 10 concurrent runs
- Hobby tier: 25 concurrent runs
- Pro tier: 100+ concurrent runs

**Current Usage:**
- 3 runs per day (6 AM, 12 PM, 6 PM UTC)
- ~30 URLs per run
- ~20 articles per URL
- **Total: ~1,800 articles/day**

**Recommendation:** Start with Hobby tier (\$20/month) for 25 concurrent runs.

---

## Next Steps

1. **Add credentials to `.env.local`**
2. **Add credentials to Vercel environment variables**
3. **Test scraper locally**
4. **Deploy to Trigger.dev**
5. **Monitor first 24 hours**
6. **Adjust priorities/schedules as needed**

---

**Last Updated:** December 9, 2024

