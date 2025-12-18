# CFTC Commitment of Traders (COT) Data Ingestion

## 📊 Overview

Ingests weekly Commitment of Traders (COT) reports from the CFTC for all 38 futures symbols.

**Data source:** https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm  
**Frequency:** Weekly (released Friday 3:30 PM ET, data as of prior Tuesday)  
**Historical data:** Available back to 2006 (Disaggregated reports)
**Canonical script:** `src/ingestion/cftc/ingest_cot.py`

---

## 🎯 What is COT Data?

The CFTC publishes weekly reports showing the positions of different trader categories in futures markets:

### **Disaggregated Report (Commodities):**

- **Producer/Merchant/Processor/User** - Commercial hedgers
- **Swap Dealers** - Banks and dealers
- **Managed Money** - Hedge funds, CTAs (speculators)
- **Other Reportable** - Other large traders
- **Non-Reportable** - Small traders

### **Traders in Financial Futures (TFF) Report (FX & Treasuries):**

- **Dealer/Intermediary** - Banks and dealers
- **Asset Manager/Institutional** - Pension funds, endowments
- **Leveraged Funds** - Hedge funds, CTAs (speculators)
- **Other Reportable** - Other large traders
- **Non-Reportable** - Small traders

---

## 🚀 Usage

### **Quick Start (Recent Data):**

```bash
# Download last 5 years
python src/ingestion/cftc/ingest_cot.py --start-year 2020 --end-year 2025
```

### **Backfill All Historical Data:**

```bash
# Download all data from 2006 to present
python src/ingestion/cftc/ingest_cot.py --backfill
```

### **Custom Date Range:**

```bash
# Download specific years
python src/ingestion/cftc/ingest_cot.py --start-year 2015 --end-year 2020
```

---

## 📋 Command Line Options

| Option         | Description                                 | Default      |
| -------------- | ------------------------------------------- | ------------ |
| `--start-year` | Start year for data download                | 2020         |
| `--end-year`   | End year for data download                  | Current year |
| `--backfill`   | Download all historical data (2006-present) | False        |

---

## 📊 Data Tables

### **raw.cftc_cot_disaggregated**

Commodity futures (ZL, ZS, ZM, ZC, CL, HO, HG, GC, etc.)

**Columns:**

- `report_date` - Report date (Tuesday)
- `symbol` - Our symbol (ZL, ZS, etc.)
- `open_interest` - Total open interest
- `managed_money_long/short/net` - Hedge fund positions
- `managed_money_net_pct_oi` - Net as % of open interest
- `prod_merc_long/short/net` - Commercial hedger positions
- `swap_long/short/net` - Swap dealer positions
- `other_rept_long/short/net` - Other reportable positions
- `nonrept_long/short/net` - Small trader positions

### **raw.cftc_cot_tff**

FX and Treasury futures (6E, 6J, 6B, ZF, ZN, ZB, etc.)

**Columns:**

- `report_date` - Report date (Tuesday)
- `symbol` - Our symbol (6E, 6J, etc.)
- `open_interest` - Total open interest
- `lev_money_long/short/net` - Leveraged fund positions
- `lev_money_net_pct_oi` - Net as % of open interest
- `asset_mgr_long/short/net` - Asset manager positions
- `dealer_long/short/net` - Dealer positions
- `other_rept_long/short/net` - Other reportable positions
- `nonrept_long/short/net` - Small trader positions

---

## 🔍 Example Queries

### **Get Latest COT Data for ZL:**

```sql
SELECT *
FROM raw.cftc_cot_disaggregated
WHERE symbol = 'ZL'
ORDER BY report_date DESC
LIMIT 10;
```

### **Get Managed Money Net Positions (All Commodities):**

```sql
SELECT
    symbol,
    report_date,
    managed_money_net,
    managed_money_net_pct_oi
FROM raw.cftc_cot_disaggregated
WHERE report_date = (SELECT MAX(report_date) FROM raw.cftc_cot_disaggregated)
ORDER BY managed_money_net_pct_oi DESC;
```

### **Week-over-Week Change in Positions:**

```sql
WITH current_week AS (
    SELECT * FROM raw.cftc_cot_disaggregated
    WHERE report_date = (SELECT MAX(report_date) FROM raw.cftc_cot_disaggregated)
),
prior_week AS (
    SELECT * FROM raw.cftc_cot_disaggregated
    WHERE report_date = (
        SELECT MAX(report_date)
        FROM raw.cftc_cot_disaggregated
        WHERE report_date < (SELECT MAX(report_date) FROM raw.cftc_cot_disaggregated)
    )
)
SELECT
    c.symbol,
    c.managed_money_net AS current_net,
    p.managed_money_net AS prior_net,
    c.managed_money_net - p.managed_money_net AS net_change
FROM current_week c
JOIN prior_week p ON c.symbol = p.symbol
ORDER BY ABS(net_change) DESC;
```

---

## 📈 Feature Engineering

Use COT data to create features:

```sql
-- Managed money net position as % of open interest
cftc_ZL_managed_money_net_pct_oi

-- Week-over-week change in managed money positions
cftc_ZL_managed_money_net_change_1w

-- 4-week moving average of managed money positions
cftc_ZL_managed_money_net_sma_4w

-- Extreme positioning (> 2 std dev from mean)
cftc_ZL_managed_money_extreme_long
cftc_ZL_managed_money_extreme_short
```

---

## ⚠️ Important Notes

### **Symbol Mapping:**

CFTC uses different contract names than our symbols. The script automatically maps:

- "SOYBEAN OIL" → ZL
- "SOYBEANS" → ZS
- "CRUDE OIL, LIGHT SWEET" → CL
- "EURO FX" → 6E
- etc.

### **Data Lag:**

- COT data is released every Friday at 3:30 PM ET
- Data is as of the prior Tuesday (3-day lag)
- Example: Friday Dec 8 release contains Tuesday Dec 5 data

### **Missing Symbols:**

Some symbols may not appear in COT reports if:

- Fewer than 20 reportable traders
- Contract is too new
- Contract is not traded on CME/NYMEX/COMEX

---

## 🔄 Automation

### **Weekly Cron Job:**

```bash
# Run every Friday at 4:00 PM ET (after 3:30 PM release)
0 16 * * 5 cd /path/to/CBI-V15 && python src/ingestion/cftc/ingest_cot.py --start-year 2024 --end-year 2025
```

---

## ✅ Verification

After ingestion, verify data:

```bash
# Check row counts
duckdb md:cbi-v15 -c "SELECT COUNT(*) FROM raw.cftc_cot_disaggregated"
duckdb md:cbi-v15 -c "SELECT COUNT(*) FROM raw.cftc_cot_tff"

# Check latest date
duckdb md:cbi-v15 -c "SELECT MAX(report_date) FROM raw.cftc_cot_disaggregated"

# Check symbols
duckdb md:cbi-v15 -c "SELECT DISTINCT symbol FROM raw.cftc_cot_disaggregated ORDER BY symbol"
```

---

## 📚 Resources

- [CFTC COT Reports](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)
- [Disaggregated Explanatory Notes](https://www.cftc.gov/MarketReports/CommitmentsofTraders/ExplanatoryNotes/index.htm)
- [Historical Compressed Data](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm)
