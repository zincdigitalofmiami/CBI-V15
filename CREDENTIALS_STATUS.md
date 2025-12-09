# API Credentials Status

**Last Updated:** December 7, 2024  
**Checked:** macOS Keychain + documentation

---

## ✅ Active & Verified

| Service | Key Name | Status | Source |
|---------|----------|--------|--------|
| **MotherDuck** | `MOTHERDUCK_TOKEN` | ✅ In Keychain | zinc@zincdigital.co |
| **Databento** | `DATABENTO_API_KEY` | ✅ In Keychain | `db-8uKak7BPpJe...` |
| **OpenAI** | `OPENAI_API_KEY` | ✅ In Keychain | `sk-svcacct-5ZfGx...` (service account) |
| **FRED** | `FRED_API_KEY` | ✅ Documented | `dc195c8658c46ee...` |
| **NOAA** | `NOAA_API_TOKEN` | ✅ Documented | `rxoLrCxYOlQyWvV...` |
| **ScrapeCreators** | `SCRAPECREATORS_API_KEY` | ✅ Documented | `B1TOgQvMVSV6TDg...` |
| **Anchor** | `ANCHOR_API_KEY` | ✅ Documented | `sk-d22742b80f7f...` |
| **Trigger.dev** | `TRIGGER_SECRET_KEY` | ✅ Documented | `tr_dev_5cabtqdv...` |
| **ProFarmer** | `PROFARMER_USERNAME`, `PROFARMER_PASSWORD` | ✅ In Keychain | chris@usoilsolutions.com / *Usoil12025 |

**Total Active:** 9 services with working credentials

---

## ⚠️ Placeholders Created (No Values Yet)

| Service | Key Name | Priority | Next Action |
|---------|----------|----------|-------------|
| **TradingEconomics** | `TRADINGECONOMICS_API_KEY` | HIGH | Register at tradingeconomics.com/analytics/api |
| **EIA** | `EIA_API_KEY` | MEDIUM | Register at eia.gov/opendata/register.php (FREE) |
| **MotherDuck Read Scaling** | `MOTHERDUCK_READ_SCALING_TOKEN` | LOW | Optional, for read replicas |

---

## 📊 Credential Storage Locations

### macOS Keychain
**Location:** Login keychain  
**Keys stored:**
- MOTHERDUCK_TOKEN
- DATABENTO_API_KEY  
- OPENAI_API_KEY
- PROFARMER_USERNAME / PROFARMER_PASSWORD (from screenshot)

**Access:**
```bash
security find-generic-password -s "DATABENTO_API_KEY" -w
```

### .env File (Local Development)
**Location:** `/Volumes/Satechi Hub/CBI-V15/.env` (gitignored)  
**Template:** `config/env-templates/env.template`

**Should contain:**
- All keys from Keychain
- Plus FRED, NOAA, SCRAPECREATORS, ANCHOR, TRIGGER (from docs)

### Vercel (Production Dashboard)
**Location:** Vercel project environment variables  
**Keys needed:**
- MOTHERDUCK_TOKEN
- DATABENTO_API_KEY (optional, for API routes)

### Legacy: `.cbi-v15.zsh`
**Location:** `/Volumes/Satechi Hub/CBI-V15/_archive/.cbi-v15.zsh`  
**Status:** Moved to archive (contained exposed secrets)

⚠️ **IMPORTANT:** This file used `MOTHERDUCK_DATABASE` (wrong) instead of `MOTHERDUCK_DB` (correct)

---

## 🔧 Quick Setup

### Pull All Keys from Keychain
```bash
./scripts/setup/pull_from_keychain.sh
```

This creates `.env` with all available keys.

### Verify Keys
```bash
python scripts/ops/test_connections.py
```

Tests:
- MotherDuck connection
- Databento API
- OpenAI API
- Keychain access

---

## 💰 Cost Summary

| Service | Type | Monthly Cost | Status |
|---------|------|--------------|--------|
| MotherDuck | Cloud DB | ~$10-50 | ✅ Active |
| Databento | Market Data | ~$50-500 | ✅ Active |
| FRED | Economic Data | **FREE** | ✅ Active |
| EIA | Energy Data | **FREE** | ⚠️ Need key |
| NOAA | Weather | **FREE** | ✅ Active |
| ScrapeCreators | News API | ~$50-200 | ✅ Active |
| **ProFarmer** | **News** | **~$500** | ✅ Active |
| **TradingEconomics** | **Data** | **~$200** | ⚠️ Need key |
| Anchor | Automation | ~$20 | ✅ Active |
| Trigger.dev | Orchestration | Free tier | ✅ Active |
| OpenAI | LLM | ~$20-100 | ✅ Active |

**Total Monthly (if all active):** ~$850-1,650

---

## 🎯 Priority Actions

### Immediate (Today)
1. ✅ Verify DATABENTO_API_KEY works
2. ✅ Verify OPENAI_API_KEY works
3. ✅ Verify MOTHERDUCK_TOKEN works
4. ⚠️ Get EIA_API_KEY (free, 5 min signup)

### This Week
5. ⚠️ Decide on TradingEconomics subscription ($200/mo)
6. ✅ Confirm ProFarmer credentials work

### Optional
7. Research MOTHERDUCK_READ_SCALING_TOKEN (if needed for scaling)

---

## 🔐 Security Notes

### Exposed Secrets (From Earlier Today)
- ⚠️ Old MotherDuck token was in `.cbi-v15.zsh` (now archived)
- ⚠️ Old OpenAI key was in `.cbi-v15.zsh` (now archived)
- ✅ New credentials are in Keychain (secure)

### Best Practices
1. ✅ Use `.env` (gitignored) for local development
2. ✅ Use Keychain for persistent storage
3. ✅ Use Vercel env vars for production
4. ✅ Never commit `.env` or credentials to git
5. ⚠️ Rotate tokens if exposed in screenshots or logs

---

**For setup instructions, see:** `docs/setup/API_KEYS_SETUP.md`  
**For credential management, see:** `scripts/setup/store_api_keys.sh`

