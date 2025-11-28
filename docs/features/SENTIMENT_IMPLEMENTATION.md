# Sentiment Implementation - Complete Guide

**Date**: November 28, 2025  
**Status**: ✅ **IMPLEMENTATION GUIDE** - How sentiment flows through the system

---

## 🎯 Sentiment Flow Overview

### Pipeline:

```
News Ingestion (ScrapeCreators API)
    ↓
Bucket Segmentation (Python - AT INGESTION)
    ↓
Sentiment Calculation (FinBERT - Python)
    ↓
Raw Table: scrapecreators_news_buckets
    ↓
Staging Table: sentiment_buckets
    ↓
Features Table: sentiment_features_daily
    ↓
Daily ML Matrix: sentiment features joined
```

---

## 📊 Part 1: Sentiment Calculation Methods

### Method 1: FinBERT (Primary) ✅

**What**: Financial BERT model for sentiment analysis

**Implementation**: Python script at ingestion

**Code Structure**:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load FinBERT model
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def calculate_sentiment_finbert(text: str, bucket_type: str) -> dict:
    """
    Calculate sentiment using FinBERT
    
    Args:
        text: News article headline/content
        bucket_type: Bucket type (biofuel, China, tariffs, etc.)
    
    Returns:
        {
            'sentiment': 'BULLISH_ZL' | 'BEARISH_ZL' | 'NEUTRAL',
            'confidence': float (0-1),
            'raw_score': float (FinBERT output)
        }
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
    
    # FinBERT outputs: ['positive', 'negative', 'neutral']
    positive_score = probs[0][0].item()
    negative_score = probs[0][1].item()
    neutral_score = probs[0][2].item()
    
    # Map to ZL-specific sentiment
    # Positive news → BULLISH_ZL (if supportive of ZL)
    # Negative news → BEARISH_ZL (if negative for ZL)
    # Neutral → NEUTRAL
    
    # Bucket-specific mapping
    if bucket_type == 'biofuel_policy':
        # Positive biofuel news → BULLISH_ZL (more demand)
        # Negative biofuel news → BEARISH_ZL (less demand)
        if positive_score > 0.5:
            return {
                'sentiment': 'BULLISH_ZL',
                'confidence': positive_score,
                'raw_score': positive_score
            }
        elif negative_score > 0.5:
            return {
                'sentiment': 'BEARISH_ZL',
                'confidence': negative_score,
                'raw_score': negative_score
            }
        else:
            return {
                'sentiment': 'NEUTRAL',
                'confidence': neutral_score,
                'raw_score': neutral_score
            }
    
    elif bucket_type == 'china_demand':
        # ✅ CORRECTED: US is net exporter, China is primary buyer
        # Positive China buying news → BULLISH_ZL (drains US stocks = higher price)
        # Negative China cancellation news → BEARISH_ZL (stocks build up = lower price)
        # Economic Reality: More China imports = drain US ending stocks = HIGHER prices
        if positive_score > 0.5:
            return {
                'sentiment': 'BULLISH_ZL',  # ✅ CORRECTED
                'confidence': positive_score,
                'raw_score': positive_score
            }
        elif negative_score > 0.5:
            return {
                'sentiment': 'BEARISH_ZL',  # ✅ CORRECTED
                'confidence': negative_score,
                'raw_score': negative_score
            }
        else:
            return {
                'sentiment': 'NEUTRAL',
                'confidence': neutral_score,
                'raw_score': neutral_score
            }
    
    elif bucket_type == 'tariffs_trade_policy':
        # ✅ CORRECTED: Tariffs are context-dependent (not always bullish)
        # Economic Reality: 2018 Trade War caused ZS to crash from $10.50 to $8.00
        # Nuance: Tariffs on US exports (retaliation) = BEARISH (demand destruction)
        #         Tariffs on Chinese UCO imports = BULLISH (protects US biofuel demand)
        text_lower = text.lower()
        
        # Tariffs on Chinese UCO/Biodiesel imports = BULLISH (protects US demand)
        if any(kw in text_lower for kw in ['uco', 'used cooking oil', 'biodiesel import', 'chinese import']):
            if positive_score > 0.5:
                return {
                    'sentiment': 'BULLISH_ZL',  # ✅ CORRECTED
                    'confidence': positive_score,
                    'raw_score': positive_score
                }
            else:
                return {
                    'sentiment': 'BEARISH_ZL',
                    'confidence': negative_score,
                    'raw_score': negative_score
                }
        
        # Tariffs on US exports (retaliation) = BEARISH (demand destruction)
        elif any(kw in text_lower for kw in ['us export', 'retaliation', 'trade war', 'china tariff']):
            if positive_score > 0.5:
                return {
                    'sentiment': 'BEARISH_ZL',  # ✅ CORRECTED (demand destruction)
                    'confidence': positive_score,
                    'raw_score': positive_score
                }
            else:
                return {
                    'sentiment': 'BULLISH_ZL',  # Removing tariffs = bullish
                    'confidence': negative_score,
                    'raw_score': negative_score
                }
        
        # Default: BEARISH for soy complex (conservative)
        else:
            return {
                'sentiment': 'BEARISH_ZL',  # ✅ DEFAULT TO BEARISH
                'confidence': max(positive_score, negative_score),
                'raw_score': max(positive_score, negative_score)
            }
    
    # Default: use FinBERT output directly
    else:
        if positive_score > 0.5:
            return {'sentiment': 'BULLISH_ZL', 'confidence': positive_score, 'raw_score': positive_score}
        elif negative_score > 0.5:
            return {'sentiment': 'BEARISH_ZL', 'confidence': negative_score, 'raw_score': negative_score}
        else:
            return {'sentiment': 'NEUTRAL', 'confidence': neutral_score, 'raw_score': neutral_score}
```

**Why FinBERT**: 
- ✅ Trained on financial news
- ✅ Better than generic sentiment models for finance
- ✅ Industry standard (ProsusAI)

---

### Method 2: Bucket-Specific Models (Advanced) ⚠️

**What**: Fine-tuned FinBERT models per bucket

**Implementation**: Optional enhancement

**Code Structure**:
```python
# Load bucket-specific models
biofuel_model = AutoModelForSequenceClassification.from_pretrained("models/finbert_biofuel")
china_model = AutoModelForSequenceClassification.from_pretrained("models/finbert_china")
tariff_model = AutoModelForSequenceClassification.from_pretrained("models/finbert_tariff")

def calculate_sentiment_bucket_specific(text: str, bucket_type: str) -> dict:
    """Use bucket-specific fine-tuned model"""
    if bucket_type == 'biofuel_policy':
        model = biofuel_model
    elif bucket_type == 'china_demand':
        model = china_model
    elif bucket_type == 'tariffs_trade_policy':
        model = tariff_model
    else:
        model = default_finbert_model
    
    # Same prediction logic as above
    ...
```

**Status**: ⚠️ **OPTIONAL** - Can add later if needed

---

## 📊 Part 2: Sentiment Segmentation (AT INGESTION)

### Step 1: Bucket Assignment ✅

**When**: IMMEDIATELY at ingestion (Python script)

**Code**:
```python
def segment_news_at_ingestion(news_item):
    """Segment news into buckets IMMEDIATELY at ingestion"""
    
    buckets = {
        'biofuel_policy': {
            'keywords': ['EPA', 'RFS', 'biodiesel', 'RIN', 'renewable fuel'],
            'sentiment_model': 'finbert_biofuel'
        },
        'china_demand': {
            'keywords': ['China', 'import', 'export', 'trade', 'soybean'],
            'sentiment_model': 'finbert_china'
        },
        'tariffs_trade_policy': {
            'keywords': ['tariff', 'trade war', 'USTR', 'Section 301'],
            'sentiment_model': 'finbert_tariff'
        },
        'supply_weather': {
            'keywords': ['drought', 'flood', 'harvest', 'yield', 'USDA'],
            'sentiment_model': 'finbert_default'
        },
        'demand_biofuels': {
            'keywords': ['biodiesel', 'renewable diesel', 'RFS', 'RVO'],
            'sentiment_model': 'finbert_biofuel'
        },
        'trade_geo': {
            'keywords': ['tariff', 'sanction', 'export', 'quota', 'trade'],
            'sentiment_model': 'finbert_tariff'
        },
        'macro_fx': {
            'keywords': ['Fed', 'CPI', 'dollar', 'DXY', 'rates'],
            'sentiment_model': 'finbert_default'
        },
        'logistics': {
            'keywords': ['Panama Canal', 'Mississippi', 'port', 'freight'],
            'sentiment_model': 'finbert_default'
        },
        'positioning': {
            'keywords': ['CFTC', 'COT', 'positioning', 'speculator'],
            'sentiment_model': 'finbert_default'
        },
        'idiosyncratic': {
            'keywords': ['ADM', 'Bunge', 'Wilmar', 'fire', 'outage'],
            'sentiment_model': 'finbert_default'
        }
    }
    
    # Assign to bucket(s) - can be multiple
    assigned_buckets = []
    for bucket_name, bucket_config in buckets.items():
        if any(kw in news_item['content'].lower() for kw in bucket_config['keywords']):
            assigned_buckets.append({
                'bucket': bucket_name,
                'confidence': calculate_confidence(news_item, bucket_config),
                'sentiment': calculate_sentiment_finbert(news_item['content'], bucket_name)
            })
    
    return assigned_buckets
```

---

### Step 2: Sentiment Calculation ✅

**When**: IMMEDIATELY after bucket assignment

**Code**:
```python
def calculate_sentiment_for_bucket(news_item, bucket_type):
    """Calculate sentiment for a specific bucket"""
    
    # Get FinBERT sentiment
    finbert_result = calculate_sentiment_finbert(news_item['content'], bucket_type)
    
    # Map to ZL-specific sentiment
    zl_sentiment = map_to_zl_sentiment(finbert_result, bucket_type)
    
    # Calculate impact magnitude
    impact_magnitude = calculate_impact_magnitude(news_item, bucket_type)
    
    return {
        'zl_sentiment': zl_sentiment,  # BULLISH_ZL, BEARISH_ZL, NEUTRAL
        'confidence': finbert_result['confidence'],
        'raw_score': finbert_result['raw_score'],
        'impact_magnitude': impact_magnitude  # HIGH, MEDIUM, LOW
    }
```

---

### Step 3: Impact Magnitude Calculation ✅

**Code**:
```python
def calculate_impact_magnitude(news_item, bucket_type):
    """Calculate impact magnitude based on source, keywords, etc."""
    
    # High impact indicators
    high_impact_keywords = ['policy', 'announcement', 'decision', 'WASDE', 'tariff', 'sanction']
    high_impact_sources = ['government', 'USDA', 'EPA', 'USTR', 'White House']
    
    # Medium impact indicators
    medium_impact_keywords = ['report', 'update', 'forecast', 'estimate']
    medium_impact_sources = ['major_news', 'Reuters', 'Bloomberg']
    
    # Check keywords
    if any(kw in news_item['content'].lower() for kw in high_impact_keywords):
        return 'HIGH'
    elif any(kw in news_item['content'].lower() for kw in medium_impact_keywords):
        return 'MEDIUM'
    elif news_item['source'] in high_impact_sources:
        return 'HIGH'
    elif news_item['source'] in medium_impact_sources:
        return 'MEDIUM'
    else:
        return 'LOW'
```

---

## 📊 Part 3: Sentiment Storage

### Raw Table: `raw.scrapecreators_news_buckets`

**Schema**:
```sql
CREATE TABLE `raw.scrapecreators_news_buckets` (
  date DATE,
  article_id STRING,
  
  -- Thematic Bucket (PRIMARY)
  theme_primary STRING,  -- SUPPLY_WEATHER, DEMAND_BIOFUELS, TRADE_GEO, etc.
  
  -- Trump-Specific Tags
  is_trump_related BOOL,
  policy_axis STRING,  -- If Trump-related
  
  -- Time-Horizon Bucket
  horizon STRING,  -- FLASH, TACTICAL, STRUCTURAL
  
  -- Impact & Sentiment (ZL-specific)
  zl_sentiment STRING,  -- BULLISH_ZL, BEARISH_ZL, NEUTRAL
  impact_magnitude STRING,  -- HIGH, MEDIUM, LOW
  sentiment_confidence FLOAT64,  -- FinBERT confidence (0-1)
  sentiment_raw_score FLOAT64,  -- FinBERT raw score
  
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

### Staging Table: `staging.sentiment_buckets`

**Schema**:
```sql
CREATE TABLE `staging.sentiment_buckets` (
  date DATE,
  bucket_type STRING,  -- theme_primary
  
  -- Aggregated Sentiment Scores
  bullish_count INT64,
  bearish_count INT64,
  neutral_count INT64,
  total_count INT64,
  
  -- Net Sentiment
  net_sentiment INT64,  -- bullish_count - bearish_count
  
  -- Weighted Sentiment (by impact_magnitude)
  bullish_weighted_score FLOAT64,  -- SUM(impact_weight WHERE bullish)
  bearish_weighted_score FLOAT64,  -- SUM(impact_weight WHERE bearish)
  net_weighted_score FLOAT64,  -- bullish_weighted_score - bearish_weighted_score
  
  -- Average Confidence
  avg_confidence FLOAT64,
  
  -- Volume Normalization
  normalized_sentiment FLOAT64  -- Normalized by rolling average volume
)
PARTITION BY DATE(date)
CLUSTER BY bucket_type;
```

**Calculation**:
```sql
SELECT 
  date,
  theme_primary AS bucket_type,
  
  -- Counts
  COUNT(IF(zl_sentiment = 'BULLISH_ZL', 1, NULL)) AS bullish_count,
  COUNT(IF(zl_sentiment = 'BEARISH_ZL', 1, NULL)) AS bearish_count,
  COUNT(IF(zl_sentiment = 'NEUTRAL', 1, NULL)) AS neutral_count,
  COUNT(*) AS total_count,
  
  -- Net Sentiment
  COUNT(IF(zl_sentiment = 'BULLISH_ZL', 1, NULL)) - 
  COUNT(IF(zl_sentiment = 'BEARISH_ZL', 1, NULL)) AS net_sentiment,
  
  -- Weighted Scores
  SUM(CASE 
    WHEN zl_sentiment = 'BULLISH_ZL' THEN 
      CASE 
        WHEN impact_magnitude = 'HIGH' THEN 3
        WHEN impact_magnitude = 'MEDIUM' THEN 2
        WHEN impact_magnitude = 'LOW' THEN 1
        ELSE 0
      END
    ELSE 0
  END) AS bullish_weighted_score,
  
  SUM(CASE 
    WHEN zl_sentiment = 'BEARISH_ZL' THEN 
      CASE 
        WHEN impact_magnitude = 'HIGH' THEN 3
        WHEN impact_magnitude = 'MEDIUM' THEN 2
        WHEN impact_magnitude = 'LOW' THEN 1
        ELSE 0
      END
    ELSE 0
  END) AS bearish_weighted_score,
  
  -- Net Weighted Score
  SUM(CASE 
    WHEN zl_sentiment = 'BULLISH_ZL' THEN 
      CASE 
        WHEN impact_magnitude = 'HIGH' THEN 3
        WHEN impact_magnitude = 'MEDIUM' THEN 2
        WHEN impact_magnitude = 'LOW' THEN 1
        ELSE 0
      END
    WHEN zl_sentiment = 'BEARISH_ZL' THEN 
      CASE 
        WHEN impact_magnitude = 'HIGH' THEN -3
        WHEN impact_magnitude = 'MEDIUM' THEN -2
        WHEN impact_magnitude = 'LOW' THEN -1
        ELSE 0
      END
    ELSE 0
  END) AS net_weighted_score,
  
  -- Average Confidence
  AVG(sentiment_confidence) AS avg_confidence
  
FROM `raw.scrapecreators_news_buckets`
GROUP BY date, theme_primary
```

---

### Features Table: `features.sentiment_features_daily`

**Schema**:
```sql
CREATE TABLE `features.sentiment_features_daily` (
  date DATE,
  
  -- Thematic Aggregates (7-day rolling)
  news_supply_tact_net_7d FLOAT64,  -- Supply-side tactical news (7-day net)
  news_biofuel_tact_net_7d FLOAT64,  -- Biofuel tactical news (7-day net)
  news_trade_struct_net_30d FLOAT64,  -- Trade structural news (30-day net)
  news_macro_risk_net_7d FLOAT64,  -- Macro risk-on/off (7-day net)
  news_logistics_tact_net_7d FLOAT64,  -- Logistics tactical news (7-day net)
  
  -- Overall Pulse
  news_zl_pulse_7d STRING,  -- Red/Yellow/Green summary
  
  -- Trump-Specific (from trump_news_features_daily)
  policy_trump_trade_china_net_7d FLOAT64,
  policy_trump_trade_china_net_30d FLOAT64,
  policy_trump_biofuels_net_7d FLOAT64,
  policy_trump_biofuels_net_30d FLOAT64,
  policy_trump_zl_net_7d FLOAT64,
  policy_trump_zl_net_30d FLOAT64
)
PARTITION BY DATE(date);
```

**Calculation**:
```sql
WITH rolling_sentiment AS (
  SELECT 
    date,
    
    -- 7-day rolling net sentiment per theme
    SUM(IF(bucket_type = 'SUPPLY_WEATHER' AND horizon = 'TACTICAL', net_weighted_score, 0)) 
      OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS news_supply_tact_net_7d,
    
    SUM(IF(bucket_type = 'DEMAND_BIOFUELS' AND horizon = 'TACTICAL', net_weighted_score, 0)) 
      OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS news_biofuel_tact_net_7d,
    
    SUM(IF(bucket_type = 'TRADE_GEO' AND horizon = 'STRUCTURAL', net_weighted_score, 0)) 
      OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS news_trade_struct_net_30d,
    
    SUM(IF(bucket_type = 'MACRO_FX' AND horizon = 'TACTICAL', net_weighted_score, 0)) 
      OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS news_macro_risk_net_7d,
    
    SUM(IF(bucket_type = 'LOGISTICS' AND horizon = 'TACTICAL', net_weighted_score, 0)) 
      OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS news_logistics_tact_net_7d,
    
    -- Overall pulse (Red/Yellow/Green)
    CASE
      WHEN SUM(net_weighted_score) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) > 5 THEN 'GREEN'
      WHEN SUM(net_weighted_score) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) < -5 THEN 'RED'
      ELSE 'YELLOW'
    END AS news_zl_pulse_7d
    
  FROM `staging.sentiment_buckets`
)
SELECT 
  r.*,
  t.policy_trump_trade_china_net_7d,
  t.policy_trump_trade_china_net_30d,
  t.policy_trump_biofuels_net_7d,
  t.policy_trump_biofuels_net_30d,
  t.policy_trump_zl_net_7d,
  t.policy_trump_zl_net_30d
FROM rolling_sentiment r
LEFT JOIN `features.trump_news_features_daily` t ON r.date = t.date
```

---

## 📊 Part 4: Sentiment Features for ML

### For Baselines (LightGBM): 14 Features ✅

1. `news_supply_tact_net_7d` - Supply-side tactical news
2. `news_biofuel_tact_net_7d` - Biofuel tactical news
3. `news_trade_struct_net_30d` - Trade structural news
4. `news_macro_risk_net_7d` - Macro risk-on/off
5. `news_logistics_tact_net_7d` - Logistics tactical news
6. `news_zl_pulse_7d` - Overall pulse (encoded as numeric)
7. `news_sentiment_change_1d` - ✅ NEW: Sentiment velocity (change from previous day)
8. `news_sentiment_velocity_7d` - ✅ NEW: Rate of change over 7 days
9. `policy_trump_trade_china_net_7d` - Trump trade China (7d)
10. `policy_trump_trade_china_net_30d` - Trump trade China (30d)
11. `policy_trump_biofuels_net_7d` - Trump biofuels (7d)
12. `policy_trump_biofuels_net_30d` - Trump biofuels (30d)
13. `policy_trump_zl_net_7d` - Trump ZL net (7d)
14. `policy_trump_zl_net_30d` - Trump ZL net (30d)

---

## ✅ Summary

### Sentiment Flow:

1. **Ingestion**: News items pulled from ScrapeCreators API
2. **Bucket Assignment**: Segment into thematic buckets (Python)
3. **Sentiment Calculation**: FinBERT calculates sentiment (Python)
4. **ZL Mapping**: Map FinBERT output to ZL-specific sentiment (Python)
5. **Impact Calculation**: Calculate impact magnitude (Python)
6. **Raw Storage**: Store in `raw.scrapecreators_news_buckets`
7. **Staging Aggregation**: Aggregate by bucket in `staging.sentiment_buckets`
8. **Feature Engineering**: Create rolling features in `features.sentiment_features_daily`
9. **ML Matrix**: Join sentiment features into `daily_ml_matrix`

### Key Points:

- ✅ **Segmentation at Ingestion**: Prevents drift
- ✅ **FinBERT**: Industry-standard financial sentiment model
- ✅ **ZL-Specific Mapping**: ✅ **CORRECTED** - China buying = BULLISH (not BEARISH)
- ✅ **Tariff Logic**: ✅ **CORRECTED** - Context-dependent (not always bullish)
- ✅ **Impact Weighting**: HIGH=3, MEDIUM=2, LOW=1
- ✅ **Volume Normalization**: Prevents volume spikes from dominating
- ✅ **Rolling Windows**: 7d (tactical), 30d (structural)
- ✅ **Sentiment Velocity**: ✅ **NEW** - Detects novelty (change from previous day)
- ✅ **Zero-Shot Classification**: ✅ **RECOMMENDED** - Use BART-large-mnli for bucketing

---

**Last Updated**: November 28, 2025

