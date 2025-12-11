# ✅ CFTC COT Ingestion Pipeline

## 🎉 What Was Built

A complete CFTC Commitment of Traders (COT) data ingestion pipeline for all 38 futures symbols.

---

## 📦 Deliverables

### **1. Python Ingestion Script**
**File:** `trigger/CFTC/Scripts/ingest_cot.py`

**Features:**
- ✅ Downloads weekly COT reports from CFTC
- ✅ Processes Disaggregated reports (commodities)
- ✅ Processes TFF reports (FX & Treasuries)
- ✅ Maps CFTC contract names to our symbols
- ✅ Calculates net positions and % of open interest
- ✅ Supports backfill (2006-present)
- ✅ Inserts into DuckDB/MotherDuck

**Usage:**
```bash
# Quick start (last 5 years)
python trigger/CFTC/Scripts/ingest_cot.py --start-year 2020

# Backfill all historical data
python trigger/CFTC/Scripts/ingest_cot.py --backfill
```

---

### **2. Database Tables**
**File:** `database/definitions/01_raw/cftc_cot.sql`

**Tables created:**

#### **raw.cftc_cot_disaggregated** (Commodities)
- Report date, symbol, open interest
- Producer/Merchant positions (commercial hedgers)
- Swap Dealer positions
- Managed Money positions (hedge funds, CTAs)
- Other Reportable positions
- Non-Reportable positions (small traders)
- Net positions and % of open interest

#### **raw.cftc_cot_tff** (FX & Treasuries)
- Report date, symbol, open interest
- Dealer/Intermediary positions
- Asset Manager positions (institutional)
- Leveraged Funds positions (hedge funds, CTAs)
- Other Reportable positions
- Non-Reportable positions
- Net positions and % of open interest

---

## 📊 Data Coverage

### **Symbols Covered (38 total):**

**Commodities (24):**
- Agricultural: ZL, ZS, ZM, ZC, ZW, ZO, ZR, OJ, HE, LE, GF, FCPO
- Energy: CL, HO, RB, NG, UL
- Metals: HG, GC, SI, PL, PA, AL

**FX Futures (10):**
- 6E, 6J, 6B, 6C, 6A, 6N, 6M, 6L, 6S, DX

**Treasuries (4):**
- ZF, ZN, ZB, TY

---

## 🔍 What COT Data Tells You

### **Managed Money (Speculators):**
- **Net long** = Bullish positioning
- **Net short** = Bearish positioning
- **Extreme positioning** = Potential reversal signal

### **Producer/Merchant (Hedgers):**
- **Net short** = Normal (producers hedge by selling)
- **Extreme short** = High production expectations
- **Net long** = Unusual (potential supply concerns)

### **Positioning as % of Open Interest:**
- **> 20%** = Strong positioning
- **> 30%** = Extreme positioning
- **> 40%** = Very extreme (potential reversal)

---

## 📈 Feature Engineering Examples

### **Basic Features:**
```sql
-- Managed money net position as % of OI
cftc_ZL_managed_money_net_pct_oi

-- Producer/merchant net position
cftc_ZL_prod_merc_net

-- Open interest
cftc_ZL_open_interest
```

### **Derived Features:**
```sql
-- Week-over-week change
cftc_ZL_managed_money_net_change_1w

-- 4-week moving average
cftc_ZL_managed_money_net_sma_4w

-- Z-score (extreme positioning)
cftc_ZL_managed_money_net_zscore

-- Extreme long/short flags
cftc_ZL_extreme_long (zscore > 2)
cftc_ZL_extreme_short (zscore < -2)
```

---

## 🚀 Quick Start

### **1. Create Tables:**
```bash
python scripts/setup_database.py --both --force
```

### **2. Ingest Data:**
```bash
python src/ingestion/cftc/ingest_cot.py --backfill
```

### **3. Verify:**
```bash
duckdb md:cbi-v15 -c "SELECT COUNT(*) FROM raw.cftc_cot_disaggregated"
```

---

## 📅 Data Release Schedule

**CFTC releases COT data:**
- **Every Friday at 3:30 PM ET**
- **Data as of prior Tuesday** (3-day lag)

---

## 🎯 Trigger Job + Python Runner

- **Python runner (canonical ingestion):** `trigger/CFTC/Scripts/ingest_cot.py`
- **Trigger.dev job:** `CFTC/Scripts/cftc_cot_reports.ts` (❗ planned, not yet created)

---

**Last Updated:** December 10, 2025
