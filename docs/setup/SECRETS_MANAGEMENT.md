# CBI-V15 Secrets Management Guide

**Last Updated:** December 9, 2024

---

## Overview

CBI-V15 uses a **three-tier secrets management system**:

1. **macOS Keychain** - Secure storage for local Python scripts
2. **.env.local** - Environment variables for Trigger.dev local development
3. **Vercel** - Production environment variables for deployed jobs

---

## Quick Start

### Option 1: Pull from Apple Keychain (Recommended)

If you already have credentials stored in Apple Keychain:

```bash
cd /Volumes/Satechi\ Hub/CBI-V15
./scripts/setup/pull_from_keychain.sh
```

This will:
- Search macOS Keychain for all required keys
- Populate `.env.local` automatically
- Create a backup of existing `.env.local`

### Option 2: Manual Entry

If credentials are NOT in Keychain:

```bash
cd /Volumes/Satechi\ Hub/CBI-V15
./scripts/setup/store_api_keys.sh
```

Choose option 4 (All three) to store in:
- macOS Keychain
- .env.local
- Vercel

---

## Required Credentials

### CRITICAL (Deploy Blockers)

| Key | Service | Where to Find |
|-----|---------|---------------|
| `TRIGGER_SECRET_KEY` | Trigger.dev | Already set: `tr_dev_5cabtqdvsHwK8L9sQqRi` |
| `MOTHERDUCK_TOKEN` | MotherDuck | Apple Keychain or MotherDuck dashboard |
| `OPENAI_API_KEY` | OpenAI | Apple Keychain or OpenAI dashboard |
| `ANCHOR_API_KEY` | Anchor | Already set: `sk-d22742b80f7f01b306fd39a2aac5d131` |

### HIGH Priority

| Key | Service | Where to Find |
|-----|---------|---------------|
| `PROFARMER_USERNAME` | ProFarmer | Apple Keychain |
| `PROFARMER_PASSWORD` | ProFarmer | Apple Keychain |
| `TRADINGECONOMICS_USERNAME` | TradingEconomics | Apple Keychain |
| `TRADINGECONOMICS_PASSWORD` | TradingEconomics | Apple Keychain |
| `DATABENTO_API_KEY` | Databento | Apple Keychain or Databento dashboard |
| `FRED_API_KEY` | FRED | Already set: `dc195c8658c46ee1df83bcd4fd8a690b` |

### MEDIUM Priority

| Key | Service | Where to Find |
|-----|---------|---------------|
| `EIA_API_KEY` | EIA | Apple Keychain or EIA dashboard |
| `NOAA_API_TOKEN` | NOAA | Already set: `rxoLrCxYOlQyWvVjbBGRlMMhIRElWKZi` |
| `SCRAPECREATORS_API_KEY` | ScrapeCreators | Already set: `B1TOgQvMVSV6TDglqB8lJ2cirqi2` |

---

## Finding Credentials in Apple Keychain

### Step 1: Open Keychain Access

```bash
open -a "Keychain Access"
```

Or: `Cmd+Space` → type "Keychain Access"

### Step 2: Search for Service

Search for:
- "ProFarmer"
- "TradingEconomics"
- "OpenAI"
- "Databento"
- "MotherDuck"
- "EIA"

### Step 3: Copy Password

1. Double-click the credential
2. Check "Show password"
3. Enter your Mac password
4. Copy the password

### Step 4: Store in .env.local

```bash
# Option A: Use the pull script
./scripts/setup/pull_from_keychain.sh

# Option B: Manual edit
nano .env.local
# Paste the value
```

---

## Verification

### Check All Locations

```bash
./scripts/setup/verify_api_keys.sh
```

This will show:
- ✅ Keys found in Keychain
- ✅ Keys set in .env.local
- ✅ Keys deployed to Vercel
- ⚠️  Missing keys

### Test Individual Keys

```bash
# Test FRED API
curl "https://api.stlouisfed.org/fred/series?series_id=DGS10&api_key=$FRED_API_KEY&file_type=json"

# Test NOAA API
curl "https://www.ncdc.noaa.gov/cdo-web/api/v2/datasets" -H "token: $NOAA_TOKEN"

# Test Trigger.dev (requires dev server)
npx trigger.dev@latest dev
```

---

## Deployment to Vercel

### Install Vercel CLI

```bash
npm i -g vercel
```

### Login to Vercel

```bash
vercel login
```

### Add Environment Variables

```bash
# Option A: Use the store script
./scripts/setup/store_api_keys.sh
# Choose option 3 (Vercel only) or 4 (All three)

# Option B: Manual add
vercel env add PROFARMER_USERNAME
vercel env add PROFARMER_PASSWORD
vercel env add OPENAI_API_KEY
# ... etc
```

### Verify Vercel Deployment

```bash
vercel env ls
```

---

## Troubleshooting

### "Key not found in Keychain"

**Solution:**
1. Open Keychain Access app
2. Search for the service name
3. If not found, check other keychains (System, iCloud)
4. If still not found, retrieve from service dashboard

### ".env.local file not found"

**Solution:**
```bash
cd /Volumes/Satechi\ Hub/CBI-V15
touch .env.local
./scripts/setup/pull_from_keychain.sh
```

### "Vercel CLI not installed"

**Solution:**
```bash
npm i -g vercel
vercel login
```

### "Permission denied" when running scripts

**Solution:**
```bash
chmod +x /Volumes/Satechi\ Hub/CBI-V15/scripts/setup/*.sh
```

---

## Security Best Practices

1. **Never commit .env.local to Git**
   - Already in `.gitignore`
   - Verify: `git status` should NOT show `.env.local`

2. **Use Keychain for long-term storage**
   - More secure than plain text files
   - Encrypted by macOS

3. **Rotate credentials regularly**
   - Update in all three locations
   - Test after rotation

4. **Backup .env.local**
   - Scripts automatically create backups
   - Format: `.env.local.backup.YYYYMMDD_HHMMSS`

---

## Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `pull_from_keychain.sh` | Pull from Keychain → .env.local | `./scripts/setup/pull_from_keychain.sh` |
| `store_api_keys.sh` | Store in Keychain/.env/Vercel | `./scripts/setup/store_api_keys.sh` |
| `verify_api_keys.sh` | Verify all locations | `./scripts/setup/verify_api_keys.sh` |

---

## Next Steps

1. ✅ Pull credentials from Keychain
2. ✅ Verify all keys are set
3. ✅ Deploy to Vercel
4. ✅ Test Trigger.dev jobs

```bash
# Complete workflow
./scripts/setup/pull_from_keychain.sh
./scripts/setup/verify_api_keys.sh
npx trigger.dev@latest dev
```

---

**Last Updated:** December 9, 2024

