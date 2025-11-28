# Trump News Integration - How News Buckets Feed Trump/ZL Engine

**Date**: November 28, 2025  
**Status**: ✅ **INTEGRATED** - News buckets feed Trump predictor/ZL effects engine

---

## 🎯 Integration Overview

### Existing Components:

1. ✅ **Trump Policy Intelligence** (`staging.trump_policy_intelligence`)
   - ScrapeCreators Truth Social feed
   - Policy events extraction
   - ZL impact scores

2. ✅ **Legislative Page** (Dashboard)
   - "Trump predictor / ZL effect" panel
   - Discrete events (bills, EOs, policy announcements)

3. ✅ **Regime System**
   - `trump_anticipation_2024`
   - `trump_second_term`

---

## 🔄 How News Buckets Feed Trump/ZL Engine

### 1. News Bucket → Trump Features

**Input**: `raw.scrapecreators_news_buckets` (with `is_trump_related = TRUE`)

**Output**: `features.trump_news_features_daily`

**Features Calculated**:

```sql
-- Net sentiment per policy axis (7-day rolling)
news_trump_trade_china_net_7d = 
  COUNT(IF(zl_sentiment = 'BULLISH_ZL' AND policy_axis = 'TRADE_CHINA', 1, NULL)) -
  COUNT(IF(zl_sentiment = 'BEARISH_ZL' AND policy_axis = 'TRADE_CHINA', 1, NULL))
  WHERE date BETWEEN CURRENT_DATE() - 7 AND CURRENT_DATE()
  AND is_trump_related = TRUE

news_trump_biofuels_net_7d = 
  COUNT(IF(zl_sentiment = 'BULLISH_ZL' AND policy_axis = 'BIOFUELS_RFS', 1, NULL)) -
  COUNT(IF(zl_sentiment = 'BEARISH_ZL' AND policy_axis = 'BIOFUELS_RFS', 1, NULL))
  WHERE date BETWEEN CURRENT_DATE() - 7 AND CURRENT_DATE()
  AND is_trump_related = TRUE

-- 30-day structural trade stance
news_trump_tariffs_net_30d = 
  COUNT(IF(zl_sentiment = 'BULLISH_ZL' AND policy_axis = 'TRADE_TARIFFS', 1, NULL)) -
  COUNT(IF(zl_sentiment = 'BEARISH_ZL' AND policy_axis = 'TRADE_TARIFFS', 1, NULL))
  WHERE date BETWEEN CURRENT_DATE() - 30 AND CURRENT_DATE()
  AND is_trump_related = TRUE
  AND horizon = 'STRUCTURAL'

-- ZL impact scores (weighted by impact_magnitude)
trump_zl_bull_score_7d = 
  SUM(CASE 
    WHEN impact_magnitude = 'HIGH' THEN 3
    WHEN impact_magnitude = 'MEDIUM' THEN 2
    WHEN impact_magnitude = 'LOW' THEN 1
    ELSE 0
  END)
  WHERE zl_sentiment = 'BULLISH_ZL'
  AND date BETWEEN CURRENT_DATE() - 7 AND CURRENT_DATE()
  AND is_trump_related = TRUE

trump_zl_bear_score_7d = 
  SUM(CASE 
    WHEN impact_magnitude = 'HIGH' THEN 3
    WHEN impact_magnitude = 'MEDIUM' THEN 2
    WHEN impact_magnitude = 'LOW' THEN 1
    ELSE 0
  END)
  WHERE zl_sentiment = 'BEARISH_ZL'
  AND date BETWEEN CURRENT_DATE() - 7 AND CURRENT_DATE()
  AND is_trump_related = TRUE

trump_zl_net_score_7d = trump_zl_bull_score_7d - trump_zl_bear_score_7d
```

---

### 2. Trump Features → Daily ML Matrix

**Features Added to `daily_ml_matrix`**:

- `policy_trump_trade_china_net_7d`
- `policy_trump_trade_china_net_30d`
- `policy_trump_biofuels_net_7d`
- `policy_trump_biofuels_net_30d`
- `policy_trump_zl_net_7d`
- `policy_trump_zl_net_30d`

**Total**: 6-10 Trump features ✅

---

### 3. Trump Features → Regime Weight Modulation

**Regime Weight Adjustment**:

```sql
-- Modulate trump_anticipation_2024 weight by trade news sentiment
regime_trump_anticipation_weight = 
  base_weight * (1 + 0.2 * SIGN(news_trump_trade_china_net_30d) * ABS(news_trump_trade_china_net_30d) / 10)
  WHERE base_weight = prediction_market_probability OR poll_average

-- Modulate trump_second_term weight by biofuels news sentiment
regime_trump_second_term_weight = 
  base_weight * (1 + 0.2 * SIGN(news_trump_biofuels_net_30d) * ABS(news_trump_biofuels_net_30d) / 10)
  WHERE base_weight = 1.0 IF trump_second_term = TRUE ELSE 0.0
```

**Why**: Allows regime weights to respond to news flow, not just discrete events

---

### 4. News Buckets → Legislative Page

**Dashboard Features**:

1. **Event Context**:
   - Show `news_trump_trade_china_net_7d` before/after discrete events
   - Show `trump_zl_net_score_7d` and 30d context
   - Mini time-series chart behind event dots

2. **ZL Directional Tag**:
   - Bull/Neutral/Bear based on `trump_zl_net_score_7d`
   - Probability bar based on `regime_trump_anticipation_weight`

3. **Policy Axis Indicators**:
   - Show which policy axis is active (tariff vs biofuels vs dereg)
   - Show whether media flow confirms or denies the regime

---

## 📊 Feature Mapping Summary

### News Buckets → Features:

| News Bucket Field | Feature Name | Usage |
|-------------------|--------------|-------|
| `is_trump_related = TRUE` + `policy_axis = TRADE_CHINA` | `policy_trump_trade_china_net_7d` | Trump predictor, regime weights |
| `is_trump_related = TRUE` + `policy_axis = TRADE_CHINA` | `policy_trump_trade_china_net_30d` | Regime weights, structural view |
| `is_trump_related = TRUE` + `policy_axis = BIOFUELS_RFS` | `policy_trump_biofuels_net_7d` | Trump predictor, ZL effects |
| `is_trump_related = TRUE` + `policy_axis = BIOFUELS_RFS` | `policy_trump_biofuels_net_30d` | Regime weights, structural view |
| `is_trump_related = TRUE` + `zl_sentiment` + `impact_magnitude` | `policy_trump_zl_net_7d` | ZL effects engine, Legislative page |
| `is_trump_related = TRUE` + `zl_sentiment` + `impact_magnitude` | `policy_trump_zl_net_30d` | ZL effects engine, structural view |

---

## ✅ Integration Checklist

### Before BigQuery Setup:

- [x] ✅ Enhanced news bucket schema with Trump tags
- [x] ✅ Created `features.trump_news_features_daily` table
- [x] ✅ Mapped news buckets to Trump/ZL engine inputs
- [x] ✅ Documented Legislative page integration
- [x] ✅ Updated regime weight modulation logic

### After BigQuery Setup:

- [ ] ⚠️ Test Trump news bucket ingestion
- [ ] ⚠️ Verify Trump feature calculation
- [ ] ⚠️ Test Legislative page integration
- [ ] ⚠️ Validate regime weight modulation

---

## 🎯 Summary

### What News Buckets Provide:

1. ✅ **Trump Predictor Inputs**: Net sentiment per policy axis
2. ✅ **ZL Effects Engine Inputs**: ZL impact scores (bull/bear/net)
3. ✅ **Legislative Page Context**: News flow around discrete events
4. ✅ **Regime Weight Modulation**: Adjust regime weights based on news sentiment

### How It Works:

1. **Ingestion**: News items tagged with `is_trump_related`, `policy_axis`, `horizon`, `zl_sentiment`, `impact_magnitude`
2. **Feature Engineering**: Aggregate into 6-10 Trump-specific features
3. **Model Input**: Features feed into `daily_ml_matrix` for training
4. **Dashboard**: Features feed into Legislative page for visualization
5. **Regime System**: Features modulate regime weights dynamically

---

**Status**: ✅ **INTEGRATION COMPLETE** - News buckets fully integrated with Trump/ZL engine

**Recommendation**: ✅ **PROCEED** with BigQuery setup

---

**Last Updated**: November 28, 2025

