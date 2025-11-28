# News Bucket System - Deep Review & Integration

**Date**: November 28, 2025  
**Status**: 🔍 **DEEP REVIEW** - Analyzing what we need vs. don't need  
**Purpose**: Integrate 3-way segmentation with existing Trump/ZL engine

---

## 🎯 Current State Analysis

### What We Have:

1. ✅ **Trump Policy Intelligence** (`staging.trump_policy_intelligence`)
   - ScrapeCreators Truth Social feed
   - Policy events extraction
   - ZL impact scores

2. ✅ **News Buckets** (planned):
   - `raw.scrapecreators_news_buckets`
   - `staging.news_bucketed`
   - `staging.sentiment_buckets`
   - `features.sentiment_features_daily`

3. ✅ **Regime System**:
   - `reference.regime_calendar`
   - `reference.regime_weights`
   - `features.regime_indicators_daily`

4. ✅ **Neural Features**:
   - `features.neural_signals_daily`
   - `features.neural_master_score`

---

## 🔍 What We NEED vs. DON'T NEED

### ✅ NEED: 3-Way Segmentation System

**Why**: Prevents brittleness, enables proper feature engineering

**Structure**:
1. **Thematic buckets** (7 buckets) - What the news is about
2. **Time-horizon buckets** (3 buckets) - How long it matters
3. **Impact/sentiment buckets** (Direction × Intensity) - How it moves ZL

---

### ✅ NEED: Trump-Specific Bucket Integration

**Why**: 
- Existing Trump predictor/ZL effects engine REQUIRES news bucket input
- Legislative page needs news flow to show context around events
- Regime system needs Trump news sentiment to modulate weights

**Integration Points**:
1. **Trump Predictor Engine** → Uses Trump news bucket features
2. **ZL Effects Engine** → Uses Trump news bucket sentiment
3. **Legislative Page** → Shows news flow around discrete events
4. **Regime System** → Modulates regime weights based on Trump news sentiment

---

### ⚠️ DON'T NEED: Redundant Buckets

**Analysis**:
- We already have `trump_policy_intelligence` table
- We don't need a SEPARATE Trump bucket IF we can tag Trump items within thematic buckets
- **Solution**: Add `is_trump_related` flag + `policy_axis` to existing news buckets

---

## 📊 Proposed News Bucket Schema

### Core Table: `raw.scrapecreators_news_buckets`

```sql
CREATE TABLE `raw.scrapecreators_news_buckets` (
  date DATE,
  article_id STRING,
  
  -- Thematic Bucket (PRIMARY - forced)
  theme_primary STRING,  -- One of: SUPPLY_WEATHER, DEMAND_BIOFUELS, TRADE_GEO, MACRO_FX, LOGISTICS, POSITIONING, IDIOSYNCRATIC
  
  -- Trump-Specific Tags (if theme_primary = TRADE_GEO or related)
  is_trump_related BOOL,
  policy_axis STRING,  -- If is_trump_related: TRADE_CHINA, TRADE_TARIFFS, BIOFUELS_RFS, EPA_REGS, AGRICULTURE_SUBSIDY, GEOPOLITICS_SOY_ROUTE
  
  -- Time-Horizon Bucket
  horizon STRING,  -- One of: FLASH, TACTICAL, STRUCTURAL
  
  -- Impact & Sentiment (ZL-specific)
  zl_sentiment STRING,  -- BULLISH_ZL, BEARISH_ZL, NEUTRAL
  impact_magnitude STRING,  -- HIGH, MEDIUM, LOW
  
  -- Raw Content
  headline STRING,
  content STRING,
  source STRING,
  source_trust_score FLOAT64,
  
  -- Metadata
  created_at TIMESTAMP
)
PARTITION BY DATE(date)
CLUSTER BY theme_primary, is_trump_related;
```

---

## 🔄 Integration with Trump/ZL Engine

### How News Buckets Feed Trump Predictor:

**Input Features** (from news buckets):
1. `news_trump_trade_china_net_7d` - Net sentiment (bullish - bearish) about Trump/China/trade
2. `news_trump_biofuels_net_7d` - Net sentiment about Trump/RFS/biofuels
3. `news_trump_tariffs_net_30d` - 30-day structural trade stance
4. `trump_zl_bull_score_7d` - Weighted sum of (impact × bullish_zl) over Trump stories
5. `trump_zl_bear_score_7d` - Weighted sum of (impact × bearish_zl) over Trump stories
6. `trump_zl_net_score_7d` - Bull score - Bear score

**Derived Features** (for `daily_ml_matrix`):
- `policy_trump_trade_china_net_7d`
- `policy_trump_trade_china_net_30d`
- `policy_trump_biofuels_net_7d`
- `policy_trump_biofuels_net_30d`
- `policy_trump_zl_net_7d`
- `policy_trump_zl_net_30d`

**Regime Modulation**:
- `regime_trump_anticipation_weight` - Adjusted by `news_trump_trade_china_net_30d`
- `regime_trump_second_term_weight` - Adjusted by `news_trump_biofuels_net_30d`

---

## 📋 Feature Mapping: News → ML Features

### For Baselines (LightGBM):

**Coarse Aggregates** (3-5 rolling sentiment scores per theme):
- `news_supply_tact_net_7d` - Supply-side tactical news (7-day net)
- `news_biofuel_tact_net_7d` - Biofuel tactical news (7-day net)
- `news_trade_struct_net_30d` - Trade structural news (30-day net)
- `news_macro_risk_net_7d` - Macro risk-on/off (7-day net)
- `news_logistics_tact_net_7d` - Logistics tactical news (7-day net)
- `news_zl_pulse_7d` - Overall ZL pulse (Red/Yellow/Green score)

**Trump-Specific** (6-10 features):
- `policy_trump_trade_china_net_7d`
- `policy_trump_trade_china_net_30d`
- `policy_trump_biofuels_net_7d`
- `policy_trump_biofuels_net_30d`
- `policy_trump_zl_net_7d`
- `policy_trump_zl_net_30d`

**Total**: ~12-15 news features for baselines ✅

---

### For Advanced Models (TFT/LSTM):

**Segmented Time-Series** (per-day counts/intensity per bucket):
- Daily counts per theme × horizon × sentiment
- Daily intensity scores per theme × horizon × sentiment
- Daily Trump-specific features (per policy_axis)

**Total**: ~50-100 features for advanced models ✅

---

## 🎯 What We DON'T Need (Avoid Bloat)

### ❌ DON'T NEED:

1. **Separate Trump Bucket Table**
   - ✅ **Solution**: Use `is_trump_related` flag + `policy_axis` in existing news buckets

2. **Duplicate Sentiment Calculation**
   - ✅ **Solution**: Calculate sentiment once at ingestion, store in `staging.sentiment_buckets`

3. **Separate Legislative Events Table** (if it's just news)
   - ✅ **Solution**: Tag legislative events with `horizon = STRUCTURAL` + `impact_magnitude = HIGH`

4. **Redundant Regime Flags**
   - ✅ **Solution**: Derive regime flags from news buckets + existing regime calendar

---

## ✅ What We NEED (Critical)

### 1. Enhanced News Bucket Schema ✅

**Add to `raw.scrapecreators_news_buckets`**:
- `is_trump_related` BOOL
- `policy_axis` STRING (if Trump-related)
- `horizon` STRING (FLASH, TACTICAL, STRUCTURAL)
- `zl_sentiment` STRING (BULLISH_ZL, BEARISH_ZL, NEUTRAL)
- `impact_magnitude` STRING (HIGH, MEDIUM, LOW)

---

### 2. Trump-Specific Feature Engineering ✅

**Create `features.trump_news_features_daily`**:
- Aggregates Trump news bucket data into ML-ready features
- Calculates net sentiment per policy axis
- Calculates ZL impact scores
- Feeds into `daily_ml_matrix`

---

### 3. Integration with Legislative Page ✅

**Dashboard Features**:
- Show `news_trump_trade_china_net_7d` before/after discrete events
- Show `trump_zl_net_score_7d` and 30d context
- Mini time-series chart behind event dots
- ZL directional tag (Bull/Neutral/Bear) with probability bar

---

### 4. Regime Weight Modulation ✅

**Update `reference.regime_weights`**:
- Modulate `trump_anticipation_2024` weight by `news_trump_trade_china_net_30d`
- Modulate `trump_second_term` weight by `news_trump_biofuels_net_30d`
- Adjust trade-war-style regime weight based on Trump news sentiment

---

## 📊 Final Feature Count

### News Features for Baselines:
- **Thematic Aggregates**: 5 features (supply, biofuel, trade, macro, logistics)
- **Trump-Specific**: 6 features (trade_china 7d/30d, biofuels 7d/30d, zl_net 7d/30d)
- **Overall Pulse**: 1 feature (zl_pulse_7d)

**Total**: **12 features** ✅ (lean, not bloated)

---

### News Features for Advanced Models:
- **Segmented Time-Series**: ~50-100 features (per-day counts/intensity)
- **Trump-Specific**: ~20 features (per policy_axis, per horizon)

**Total**: **70-120 features** ✅ (can expand for TFT/LSTM)

---

## ✅ Integration Checklist

### Before BigQuery Setup:

- [ ] ✅ Review news bucket schema (add Trump tags)
- [ ] ✅ Create `features.trump_news_features_daily` table
- [ ] ✅ Map news buckets to Trump/ZL engine inputs
- [ ] ✅ Document Legislative page integration
- [ ] ✅ Update regime weight modulation logic

### After BigQuery Setup:

- [ ] ✅ Test Trump news bucket ingestion
- [ ] ✅ Verify Trump feature calculation
- [ ] ✅ Test Legislative page integration
- [ ] ✅ Validate regime weight modulation

---

## 🎯 Summary

### What We NEED:
1. ✅ Enhanced news bucket schema with Trump tags
2. ✅ Trump-specific feature engineering
3. ✅ Integration with Legislative page
4. ✅ Regime weight modulation

### What We DON'T NEED:
1. ❌ Separate Trump bucket table (use flags)
2. ❌ Duplicate sentiment calculation
3. ❌ Redundant regime flags
4. ❌ Bloat (keep lean: 12 features for baselines)

---

**Status**: ✅ **REVIEW COMPLETE** - Ready to integrate

**Recommendation**: ✅ **PROCEED** with enhanced schema + Trump integration

---

**Last Updated**: November 28, 2025

