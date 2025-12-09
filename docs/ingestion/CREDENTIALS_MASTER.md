# CBI-V15 Credentials Master List

**Last Updated:** December 9, 2024

---

## ✅ CREDENTIALS ADDED

### Anchor Browser Automation
```bash
ANCHOR_API_KEY=sk-d22742b80f7f01b306fd39a2aac5d131
```
**Status:** ✅ Added to `.env.local`

### FRED API
```bash
FRED_API_KEY=dc195c8658c46ee1df83bcd4fd8a690b
```
**Status:** ✅ Added to `.env.local`

### NOAA Weather
```bash
NOAA_TOKEN=rxoLrCxYOlQyWvVjbBGRlMMhIRElWKZi
```
**Status:** ✅ Added to `.env.local`

### ScrapeCreators API
```bash
SCRAPECREATORS_API_KEY=B1TOgQvMVSV6TDglqB8lJ2cirqi2
```
**Status:** ✅ Added to `.env.local`

### Trigger.dev
```bash
TRIGGER_SECRET_KEY=tr_dev_5cabtqdvsHwK8L9sQqRi
```
**Status:** ✅ Added to `.env.local`

---

## ⚠️ CREDENTIALS NEEDED

### ProFarmer (CRITICAL)
```bash
PROFARMER_USERNAME=?
PROFARMER_PASSWORD=?
```
**Status:** ❌ NOT PROVIDED YET
**Priority:** CRITICAL - ProFarmer is the most important news source
**Action:** Check Apple Keychain for credentials

### OpenAI (CRITICAL)
```bash
OPENAI_API_KEY=?
```
**Status:** ❌ NOT PROVIDED YET
**Priority:** CRITICAL - Required for AI Agents
**Action:** Check Apple Keychain for credentials

### TradingEconomics (HIGH)
```bash
TRADINGECONOMICS_USERNAME=?
TRADINGECONOMICS_PASSWORD=?
```
**Status:** ❌ NOT PROVIDED YET
**Priority:** HIGH - "GOLD MINE" for commodity data
**Action:** Check Apple Keychain for credentials

### Databento (HIGH)
```bash
DATABENTO_API_KEY=?
```
**Status:** ❌ NOT PROVIDED YET
**Priority:** HIGH - Required for futures price data
**Action:** Check Apple Keychain for credentials

### EIA (MEDIUM)
```bash
EIA_API_KEY=?
```
**Status:** ❌ NOT PROVIDED YET
**Priority:** MEDIUM - Energy data
**Action:** Check Apple Keychain for credentials

### MotherDuck (CRITICAL)
```bash
MOTHERDUCK_TOKEN=?
MOTHERDUCK_DB=cbi_v15
```
**Status:** ❌ NOT PROVIDED YET
**Priority:** CRITICAL - Required for all data storage
**Action:** Check Apple Keychain for credentials

---

## Apple Keychain Instructions

**To find credentials in Apple Keychain:**

1. Open **Keychain Access** app (Cmd+Space → "Keychain Access")
2. Search for:
   - "ProFarmer"
   - "TradingEconomics"
   - "OpenAI"
   - "Databento"
   - "EIA"
   - "MotherDuck"
3. Double-click credential → Check "Show password"
4. Copy to `.env.local`

---

## Adding Credentials

### Local Development (`.env.local`)
```bash
cd /Volumes/Satechi\ Hub/CBI-V15
nano .env.local

# Add credentials
PROFARMER_USERNAME=your_username
PROFARMER_PASSWORD=your_password
# ... etc
```

### Vercel Production
```bash
# Add each credential
vercel env add PROFARMER_USERNAME
vercel env add PROFARMER_PASSWORD
vercel env add OPENAI_API_KEY
vercel env add TRADINGECONOMICS_USERNAME
vercel env add TRADINGECONOMICS_PASSWORD
vercel env add DATABENTO_API_KEY
vercel env add EIA_API_KEY
vercel env add MOTHERDUCK_TOKEN
```

---

## Verification

**After adding credentials, test:**

```bash
# Test FRED
curl "https://api.stlouisfed.org/fred/series?series_id=DGS10&api_key=$FRED_API_KEY&file_type=json"

# Test NOAA
curl "https://www.ncdc.noaa.gov/cdo-web/api/v2/datasets" -H "token: $NOAA_TOKEN"

# Test Anchor (requires running Trigger.dev)
npx trigger.dev@latest dev
npx trigger.dev@latest trigger profarmer-all-urls
```

---

## Priority Order

1. **CRITICAL (Deploy Blockers):**
   - MotherDuck (all data storage)
   - ProFarmer (primary news source)
   - OpenAI (AI Agents)

2. **HIGH (Core Features):**
   - Databento (futures prices)
   - TradingEconomics (commodity data)

3. **MEDIUM (Nice to Have):**
   - EIA (energy data)

---

## Next Steps

1. ✅ Check Apple Keychain for all credentials
2. ✅ Add credentials to `.env.local`
3. ✅ Add credentials to Vercel
4. ✅ Test each API connection
5. ✅ Deploy to Trigger.dev

---

**Last Updated:** December 9, 2024

