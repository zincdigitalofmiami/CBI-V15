# 📊 CBI-V15 Data Sources - Feature Engineering Reference

⚠️ **DEPRECATED - See `DATA_LINKS_MASTER.md` for canonical data links and API endpoints.**

This file is kept for feature engineering prefix reference only. All data source URLs, API endpoints, Trigger jobs, and ingestion scripts are documented in **`DATA_LINKS_MASTER.md`**.

---

## Feature Engineering Prefixes

**All features must use these prefixes:**

| Prefix           | Source                   | Example                                        |
| ---------------- | ------------------------ | ---------------------------------------------- |
| `fred_*`         | FRED economic data       | `fred_FEDFUNDS`, `fred_DGS10`                  |
| `fx_*`           | FX pair derived features | `fx_EURUSD_ret_1d`, `fx_USDJPY_volatility_21d` |
| `databento_*`    | Futures OHLCV + tick     | `databento_ZL_close`, `databento_CL_volume`    |
| `weather_*`      | NOAA regional weather    | `weather_brazil_mato_grosso_precip`            |
| `eia_*`          | Biofuel & petroleum      | `eia_RIN_D4`, `eia_BIODIESEL_PROD`             |
| `usda_*`         | WASDE, export, crop      | `usda_export_soybeans_weekly`                  |
| `cftc_*`         | COT data                 | `cftc_ZL_net_noncomm`                          |
| `scrc_*`         | Sentiment & news         | `scrc_biofuel_policy_sentiment`                |
| `policy_trump_*` | Trump/Truth Social       | `policy_trump_tariff_mentions`                 |
| `basis_*`        | Price basis spreads      | `basis_brazil_fob_paranagua`                   |

---

## Available Symbols & Series

**For complete data source details, URLs, API endpoints, Trigger jobs, and ingestion scripts, see:** `DATA_LINKS_MASTER.md`

### Futures Symbols (38 total)

- **10 FX Futures:** 6E, 6J, 6B, 6C, 6A, 6N, 6M, 6L, 6S, DX
- **24 Commodity Futures:** ZL, ZS, ZM, ZC, ZW, ZO, ZR, HE, LE, GF, FCPO, CL, HO, RB, NG, HG, GC, SI, PL, PA
- **4 Treasuries:** ZF, ZN, ZB

### Macro Indicators (24 FRED series)

- Interest Rates & Yields: FEDFUNDS, DGS1MO-DGS30
- Yield Spreads: T10Y2Y, T10Y3M, TEDRATE
- Financial Conditions: NFCI, STLFSI4
- Economic Indicators: UNRATE, CPIAUCSL, GDP, PAYEMS
- Market Indicators: VIXCLS, DTWEXBGS, DTWEXAFEGS, DTWEXEMEGS

### Other Data Sources

- **14 Weather Regions** (NOAA, INMET, SMN)
- **8 News Buckets** (ScrapeCreators)
- **USDA Series** (WASDE, export sales, crop progress)
- **EIA Biofuels** (RIN prices, biodiesel production)
- **CFTC COT** (all futures positioning)

---

**For complete data source information, see:** `DATA_LINKS_MASTER.md`
