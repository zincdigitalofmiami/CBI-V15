# 🌐 Web Scraping Targets - Master List

**CBI-V15 Intelligence Platform**  
**Last Updated:** December 7, 2024

---

## 🎯 Overview

This document contains ALL web sources we scrape/monitor for ZL (Soybean Oil) forecasting. These are organized by priority and bucket type.

---

## 🔴 TIER 1: Critical Sources (MANDATORY)

### Institutional Alpha Feed

#### 1. **The Jacobsen** (PAID SUBSCRIPTION REQUIRED) ⚠️
- **URL:** https://thejacobsen.com
- **Focus:** Biofuels, Vegetable Oils, RIN prices, D4 RINs
- **Bucket:** `biofuel_policy`
- **Priority:** CRITICAL
- **Why:** Real-time RIN prices = direct ZL price driver
- **Data:** Daily physical price bulletin

#### 2. **Oil World (ISTA Mielke)** (PAID SUBSCRIPTION)
- **URL:** https://www.oilworld.biz
- **Focus:** Global vegetable oil supply & demand balances
- **Bucket:** `market_fundamentals`
- **Priority:** CRITICAL
- **Why:** Gold standard for global veg oil intelligence

#### 3. **NOPA (National Oilseed Processors Association)**
- **URL:** https://www.nopa.org
- **Focus:** Monthly Crush Report (crush volume, oil stocks)
- **Bucket:** `market_fundamentals`
- **Priority:** CRITICAL
- **Schedule:** 15th of each month
- **Why:** U.S. soybean crush = direct ZL supply

#### 4. **AgriCensus (Fastmarkets)**
- **URL:** https://www.agricensus.com
- **Focus:** Cash premiums, tender results, export taxes
- **Bucket:** `market_intelligence`
- **Priority:** HIGH
- **Why:** Real-time global ag trade intelligence

---

## 🟡 TIER 2: High-Priority Sources

### News & Market Intelligence

#### 5. **Reuters Commodities**
- **URL:** https://www.reuters.com/markets/commodities
- **Focus:** Policy leaks, geopolitical news, trade flows
- **Bucket:** `market_intelligence`
- **Keywords:** Soybeans, soybean oil, biofuels, China trade

#### 6. **Bloomberg Green/Commodities**
- **URL:** https://www.bloomberg.com/green
- **URL (Alt):** https://www.bloomberg.com/markets/commodities
- **Focus:** Currency wars, trade flow shifts, fund flows
- **Bucket:** `market_intelligence`

#### 7. **DTN / The Progressive Farmer**
- **URL:** https://www.dtnpf.com/agriculture/web/ag/home
- **Focus:** Weather, logistics, basis, barge rates
- **Bucket:** `market_fundamentals`

#### 8. **Soybean & Corn Advisor (Michael Cordonnier)**
- **URL:** https://www.soybeansandcorn.com
- **Focus:** South American crop conditions (Brazil, Argentina)
- **Bucket:** `crop_conditions`
- **Priority:** HIGH
- **Why:** Early Brazil/Argentina yield signals

### Government Data & Policy

#### 9. **USDA FAS (Foreign Agricultural Service)**
- **URL:** https://apps.fas.usda.gov/esrquery/
- **URL (Reports):** https://apps.fas.usda.gov/newgainapi/
- **Focus:** Export sales, GAIN reports
- **Bucket:** `market_fundamentals`
- **Schedule:** Weekly export sales (Thursdays)

#### 10. **EPA (Environmental Protection Agency)**
- **URL:** https://www.epa.gov
- **Focus:** Notices of Rulemaking, RFS updates, RVO
- **Bucket:** `biofuel_policy`
- **Keywords:** RVO, SRE, Renewable Fuel Standard

#### 11. **Federal Register API**
- **URL:** https://www.federalregister.gov/api/v1/documents.json
- **Focus:** EPA rules, USDA regulations, trade policy
- **Bucket:** `policy_risk`
- **Why:** Advance notice of regulatory changes

#### 12. **Clean Fuels Alliance America**
- **URL:** https://cleanfuels.org
- **Focus:** Lobbying, EPA letters, biodiesel industry advocacy
- **Bucket:** `biofuel_policy`

---

## 🟢 TIER 3: Medium-Priority Sources

### Agricultural Trade Publications

#### 13. **AgWeb Soybeans**
- **URL:** https://www.agweb.com/news/crops/soybeans
- **Focus:** Soybean market news, prices, analysis
- **Bucket:** `market_intelligence`

#### 14. **Farm Progress Soybeans**
- **URL:** https://www.farmprogress.com/soybeans
- **Focus:** Soybean production, markets
- **Bucket:** `market_intelligence`

#### 15. **Agriculture.com Markets**
- **URL:** https://www.agriculture.com/markets-commodities
- **Focus:** Commodity prices, market news
- **Bucket:** `market_intelligence`

#### 16. **Agrimoney Grains/Oilseeds**
- **URL:** https://www.agrimoney.com/news/grains-oilseeds/
- **Focus:** Global grains & oilseeds news
- **Bucket:** `market_intelligence`

#### 17. **Agrimoney China**
- **URL:** https://www.agrimoney.com/news/china/
- **Focus:** China agriculture & trade news
- **Bucket:** `china_demand`

#### 18. **World Grain**
- **URL:** https://www.world-grain.com/
- **Focus:** Global grain industry news
- **Bucket:** `market_intelligence`

#### 19. **Farm Policy News (University of Illinois)**
- **URL:** https://farmpolicynews.illinois.edu
- **Focus:** Farm Bill, crop insurance, subsidies
- **Bucket:** `policy_risk`

#### 20. **Farmdoc Daily (University of Illinois)**
- **URL:** https://farmdocdaily.illinois.edu
- **Focus:** Agricultural economics analysis
- **Bucket:** `market_intelligence`

### South American Sources

#### 21. **Conab (Brazil)**
- **URL:** https://www.conab.gov.br/ultimas-noticias
- **Focus:** Brazilian crop reports, yield estimates
- **Bucket:** `crop_conditions`
- **Why:** Compare against USDA estimates

#### 22. **ABIOVE (Brazilian Oilseed Processors)**
- **URL:** https://abiove.org.br/en/statistics/
- **Focus:** Brazilian soy crush, exports
- **Bucket:** `market_fundamentals`

### Immigration & Labor (Farm Labor Impact)

#### 23. **ICE News Releases**
- **URL:** https://www.ice.gov/news/releases
- **Focus:** Immigration enforcement, deportations
- **Bucket:** `tariffs_trade` (farm labor proxy)

#### 24. **DHS News Releases**
- **URL:** https://www.dhs.gov/news-releases
- **Focus:** Immigration policy
- **Bucket:** `policy_risk`

#### 25. **CBP Newsroom**
- **URL:** https://www.cbp.gov/newsroom
- **Focus:** Border/customs enforcement
- **Bucket:** `policy_risk`

### State Agriculture Departments

#### 26. **Florida Dept of Agriculture (FDACS)**
- **URL:** https://www.fdacs.gov/
- **Focus:** Florida agriculture, labor, inspections
- **Bucket:** `policy_risk`

#### 27. **Texas Agriculture**
- **URL:** https://www.texasagriculture.gov/
- **Focus:** Texas agriculture, trade
- **Bucket:** `policy_risk`

#### 28. **California Farm Bureau**
- **URL:** https://www.cfbf.com/news/
- **Focus:** California ag policy, labor
- **Bucket:** `policy_risk`

---

## 🔵 TIER 4: Think Tanks & Research

### Trade Policy & Analysis

#### 29. **Peterson Institute (PIIE) - Trade War Charts**
- **URL:** https://www.piie.com/research/piie-charts/us-china-trade-war-tariffs-date-chart
- **Focus:** U.S.-China tariff tracking
- **Bucket:** `tariffs_trade`
- **Why:** Quantified tariff timeline

#### 30. **CSIS Trade War Monitor**
- **URL:** https://www.csis.org/programs/scholl-chair-international-business/trade-war-monitor
- **Focus:** Trade war analysis
- **Bucket:** `tariffs_trade`

#### 31. **US-China Business Council**
- **URL:** https://www.uschina.org/
- **Focus:** U.S.-China business relations
- **Bucket:** `china_demand`

#### 32. **Heritage Foundation - Agriculture**
- **URL:** https://www.heritage.org/agriculture
- **Focus:** Conservative ag policy
- **Bucket:** `policy_risk`

#### 33. **America First Policy Institute**
- **URL:** https://americafirstpolicy.com/
- **Focus:** Trump 2.0 policy signals
- **Bucket:** `tariffs_trade`

#### 34. **Tax Foundation - Trade**
- **URL:** https://taxfoundation.org/research/all/federal/trade/
- **Focus:** Tariff economics
- **Bucket:** `tariffs_trade`

#### 35. **AEI - Trade Policy**
- **URL:** https://www.aei.org/tag/trade-policy/
- **Focus:** Trade policy analysis
- **Bucket:** `tariffs_trade`

### Labor & Immigration Research

#### 36. **Farm Labor Organizing Committee**
- **URL:** https://www.farmlabororganizing.org/
- **Focus:** Farm worker organizing, labor rights
- **Bucket:** `policy_risk`

#### 37. **United Farm Workers (UFW)**
- **URL:** https://ufw.org/
- **Focus:** Farm labor advocacy
- **Bucket:** `policy_risk`

#### 38. **Western Growers Association**
- **URL:** https://www.wga.com/
- **Focus:** West Coast ag, labor
- **Bucket:** `policy_risk`

#### 39. **Migration Policy Institute**
- **URL:** https://www.migrationpolicy.org/
- **Focus:** Immigration policy research
- **Bucket:** `policy_risk`

#### 40. **American Immigration Council**
- **URL:** https://immigrationimpact.com/
- **Focus:** Immigration policy impact
- **Bucket:** `policy_risk`

#### 41. **SPLC Immigrant Justice**
- **URL:** https://www.splcenter.org/issues/immigrant-justice
- **Focus:** Immigrant worker rights
- **Bucket:** `policy_risk`

---

## 🟣 TIER 5: Weather & Climate

### South American Weather

#### 42. **INMET (Brazil National Institute of Meteorology)**
- **API:** https://apitempo.inmet.gov.br/estacao/{start}/{end}/{station_id}
- **Token Portal:** https://apitempo.inmet.gov.br/token
- **Stations:** https://apitempo.inmet.gov.br/estacoes
- **Focus:** Brazilian weather stations (Mato Grosso, Goiás, Paraná, etc.)
- **Bucket:** `weather`

#### 43. **SMN Argentina (Servicio Meteorológico Nacional)**
- **API:** https://ssl.smn.gob.ar/dpd/descarga_opendata.php?file=observaciones/datohorario{station_id}.txt
- **Focus:** Argentina weather (Buenos Aires, Córdoba, Santa Fe, Entre Ríos)
- **Bucket:** `weather`

### U.S. & Global Weather

#### 44. **NOAA CDO API**
- **URL:** https://www.ncei.noaa.gov/cdo-web/api/v2/data
- **Token Portal:** https://www.ncdc.noaa.gov/cdo-web/token
- **Focus:** U.S. Corn Belt weather
- **Bucket:** `weather`

#### 45. **Copernicus CDS API**
- **URL:** https://cds.climate.copernicus.eu/api
- **Focus:** Global climate data
- **Bucket:** `weather`

#### 46. **Meteomatics API**
- **URL:** https://api.meteomatics.com
- **Focus:** High-resolution weather forecasts
- **Bucket:** `weather`

---

## 🟤 TIER 6: Macro & Central Banks

### Central Banks & Economic Data

#### 47. **Federal Reserve Speeches**
- **URL:** https://www.federalreserve.gov/newsevents/speech/
- **Focus:** Fed policy signals
- **Bucket:** `market_intelligence`

#### 48. **US Treasury Fiscal Data API**
- **URL:** https://api.fiscaldata.treasury.gov/services/api/v1/
- **Focus:** Government finances
- **Bucket:** `market_intelligence`

#### 49. **BLS Public API**
- **URL:** https://api.bls.gov/publicAPI/v2/
- **Focus:** Employment, inflation
- **Bucket:** `market_intelligence`

#### 50. **ECB Statistical Data Warehouse REST**
- **URL:** https://sdw-wsrest.ecb.europa.eu/service/
- **Focus:** Eurozone data
- **Bucket:** `market_intelligence`

#### 51. **BCB (Banco Central do Brasil)**
- **URL:** https://www3.bcb.gov.br/sgspub/
- **Focus:** Brazilian monetary policy, currency
- **Bucket:** `market_intelligence`

#### 52. **PBOC (People's Bank of China)**
- **URL:** http://www.pbc.gov.cn/en/
- **Focus:** Chinese monetary policy
- **Bucket:** `china_demand`

---

## 🟠 TIER 7: Social Media Influencers (Twitter/X)

### Key People to Monitor

#### 53. **Karen Braun** (@kannbwx)
- **Platform:** Twitter/X
- **Role:** Global Ag Columnist (Reuters)
- **Focus:** USDA data anomalies, yield discrepancies
- **Bucket:** `social_influencers`

#### 54. **Arlan Suderman** (@ArlanFF101)
- **Platform:** Twitter/X
- **Role:** Chief Commodities Economist (StoneX)
- **Focus:** Institutional positioning, China demand flows
- **Bucket:** `social_influencers`

#### 55. **Scott Irwin** (@ScottIrwinUIUC)
- **Platform:** Twitter/X
- **Role:** Ag Economist (University of Illinois)
- **Focus:** RFS, biofuel policy, EPA math
- **Bucket:** `social_influencers`

#### 56. **Dr. Michael Cordonnier** (@SoybeanCorn)
- **Platform:** Twitter/X
- **Role:** Agronomist
- **Focus:** South American crop health, Mato Grosso weather
- **Bucket:** `social_influencers`

#### 57. **Javier Blas** (@JavierBlas)
- **Platform:** Twitter/X
- **Role:** Energy/Commodities Opinion (Bloomberg)
- **Focus:** Energy/Ags intersection, Food vs Fuel
- **Bucket:** `social_influencers`

---

## 🔶 TIER 8: Corporate Signals

### Public Companies to Monitor

#### 58. **ADM (Archer Daniels Midland)**
- **Ticker:** ADM
- **Source:** Earnings calls, investor relations
- **Focus:** Crushing margins, capacity utilization
- **Bucket:** `corporate_signals`

#### 59. **Bunge Global**
- **Ticker:** BG
- **Source:** Earnings calls, investor relations
- **Focus:** South America exposure, biofuels, crushing
- **Bucket:** `corporate_signals`

---

## 📋 Scraping Implementation Status

| Source | Status | Method | Frequency |
|--------|--------|--------|-----------|
| **The Jacobsen** | ⚠️ Pending | Manual/Paid | Daily |
| **Oil World** | ⚠️ Pending | Manual/Paid | Weekly |
| **NOPA** | ⚠️ Pending | Web scrape | Monthly |
| **AgriCensus** | ⚠️ Pending | Paid API | Daily |
| **Reuters** | ✅ Active | ScrapeCreators | Real-time |
| **Bloomberg** | ⚠️ Limited | ScrapeCreators | Real-time |
| **DTN** | ⚠️ Pending | Web scrape | Daily |
| **Cordonnier** | ⚠️ Pending | Email/Web | Daily |
| **USDA FAS** | ⚠️ Pending | API | Weekly |
| **EPA** | ⚠️ Pending | Federal Register API | Daily |
| **AgWeb** | ⚠️ Pending | RSS/Scrape | Daily |
| **Conab** | ⚠️ Pending | Web scrape | Monthly |
| **Twitter/X Influencers** | ⚠️ Pending | Twitter API | Real-time |
| **ADM/Bunge** | ⚠️ Pending | Edgar/IR scrape | Quarterly |

---

## 🔧 ScrapeCreators Integration

**Currently Active via ScrapeCreators API:**
- Reuters Commodities
- Bloomberg (limited)
- AgWeb
- Farm Progress
- Agriculture.com
- General news aggregation

**Buckets:**
1. `biofuel_policy` - RFS, biodiesel mandates
2. `china_demand` - Import signals, trade data
3. `tariffs_trade` - Trade policy, tariffs
4. `trump_truth_social` - Trump Truth Social posts

---

## 🎯 Priority Implementation Order

### Phase 1 (Immediate - High Alpha)
1. ✅ ScrapeCreators (Active)
2. ⚠️ The Jacobsen (Paid - RIN prices)
3. ⚠️ NOPA (Monthly crush)
4. ⚠️ USDA FAS Export Sales (API)

### Phase 2 (Next - Fundamental Data)
5. ⚠️ Conab (Brazil crops)
6. ⚠️ EPA Federal Register (Policy)
7. ⚠️ Michael Cordonnier (South America)
8. ⚠️ Twitter/X Influencers (Social)

### Phase 3 (Later - Enrichment)
9. ⚠️ AgriCensus (Paid)
10. ⚠️ Oil World (Paid)
11. ⚠️ Corporate earnings (ADM/Bunge)
12. ⚠️ Weather APIs (INMET/SMN)

---

## 📊 Bucket Mapping

| Bucket | Sources Count | Priority |
|--------|---------------|----------|
| `biofuel_policy` | 8 sources | CRITICAL |
| `market_fundamentals` | 12 sources | CRITICAL |
| `market_intelligence` | 15 sources | HIGH |
| `china_demand` | 6 sources | HIGH |
| `tariffs_trade` | 10 sources | HIGH |
| `crop_conditions` | 5 sources | MEDIUM |
| `policy_risk` | 14 sources | MEDIUM |
| `weather` | 6 sources | MEDIUM |
| `social_influencers` | 5 sources | MEDIUM |
| `corporate_signals` | 2 sources | LOW |

**Total Sources: 59**

---

## 🔑 API Keys Needed

| Service | Env Var | Status | Cost |
|---------|---------|--------|------|
| ScrapeCreators | `SCRAPECREATORS_API_KEY` | ✅ Active | Subscription |
| The Jacobsen | `JACOBSEN_USERNAME`/`PASSWORD` | ⚠️ Needed | ~$500/month |
| Oil World | Manual access | ⚠️ Needed | ~$1,000/year |
| USDA FAS | Public API | ✅ Free | Free |
| Federal Register | Public API | ✅ Free | Free |
| NOAA CDO | `NOAA_API_KEY` | ⚠️ Needed | Free |
| INMET | `INMET_TOKEN` | ⚠️ Needed | Free |
| Twitter/X | `TWITTER_BEARER_TOKEN` | ⚠️ Needed | $100/month (Basic) |
| Meteomatics | `METEOMATICS_KEY` | ⚠️ Optional | Paid |

---

## 📝 Notes

1. **Paid Sources:** The Jacobsen and Oil World require subscriptions but are CRITICAL for institutional-grade ZL forecasting.
2. **Rate Limits:** Respect rate limits on all public APIs.
3. **Legal:** Ensure all scraping complies with Terms of Service.
4. **Storage:** All scraped data goes to `raw.news_articles` or bucket-specific tables in MotherDuck.
5. **Frequency:** High-priority sources should be checked daily or real-time.

---

**For implementation details, see:**
- `src/ingestion/scrape_creator/` - Active ScrapeCreators integration
- `config/ingestion/sources.yaml` - Ingestion configuration
- `database/definitions/01_raw/` - Raw table schemas

---

**Last Updated:** December 7, 2024  
**Total Sources:** 59  
**Active Sources:** 4 (ScrapeCreators buckets)  
**Pending Sources:** 55

