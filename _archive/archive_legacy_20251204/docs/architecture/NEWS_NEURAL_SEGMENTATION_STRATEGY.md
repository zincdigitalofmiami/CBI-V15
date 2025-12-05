# News/Neural Segmentation Strategy - Prevent Brittleness & Drift

**Date**: November 28, 2025  
**Purpose**: Proper segmentation at ingestion to reduce model brittleness and drift  
**Status**: ✅ **CRITICAL** - Must implement before ingestion

---

## 🎯 Problem Statement

### Why Segmentation Matters

**Brittleness**: Model breaks when news patterns change (2018 trade war ≠ 2024 trade war)  
**Drift**: Model performance degrades over time (semantic drift, source drift, volume drift)

**Root Causes**:
1. **Temporal Drift**: News patterns change over time
2. **Source Drift**: News sources evolve (new outlets, social media changes)
3. **Semantic Drift**: Language evolves (same words mean different things)
4. **Volume Drift**: News volume spikes (crisis periods dominate)

---

## ✅ Solution: Segmentation at Ingestion

### Strategy 1: Bucket Segmentation (CRITICAL)

#### At Ingestion (Python Script):

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
        }
    }
    
    # Assign to bucket(s) - can be multiple
    assigned_buckets = []
    for bucket_name, bucket_config in buckets.items():
        if any(kw in news_item['content'].lower() for kw in bucket_config['keywords']):
            assigned_buckets.append({
                'bucket': bucket_name,
                'confidence': calculate_confidence(news_item, bucket_config),
                'sentiment': calculate_sentiment(news_item, bucket_config['sentiment_model'])
            })
    
    return assigned_buckets
```

**Why**: Prevents cross-contamination, allows bucket-specific modeling

---

### Strategy 2: Temporal Segmentation (CRITICAL)

#### At Ingestion:

```python
def tag_temporal_markers(news_item):
    """Tag news with temporal markers to prevent temporal drift"""
    
    date = news_item['date']
    
    # Regime periods
    regime_periods = {
        'trump_2018': (DATE(2018, 1, 1), DATE(2020, 12, 31)),
        'trump_2024': (DATE(2024, 1, 1), DATE(2025, 12, 31)),
        'normal': (DATE(2010, 1, 1), DATE(2017, 12, 31)),
        'crisis': [(DATE(2020, 3, 1), DATE(2020, 6, 30))]  # COVID
    }
    
    # Determine regime
    regime = determine_regime(date, regime_periods)
    
    # Date buckets (pre/post major events)
    date_buckets = {
        'pre_trade_war': date < DATE(2018, 3, 1),
        'trade_war': DATE(2018, 3, 1) <= date <= DATE(2020, 1, 15),
        'post_trade_war': date > DATE(2020, 1, 15)
    }
    
    return {
        'regime': regime,
        'date_buckets': date_buckets
    }
```

**Why**: Allows regime-specific feature engineering, reduces temporal drift

---

### Strategy 3: Source Segmentation (CRITICAL)

#### At Ingestion:

```python
def tag_source_metadata(news_item):
    """Tag news with source metadata and trust scores"""
    
    source_configs = {
        'government': {'trust_score': 0.95, 'decay_factor': 0.99},
        'major_news': {'trust_score': 0.85, 'decay_factor': 0.98},
        'social_media': {'trust_score': 0.60, 'decay_factor': 0.95},
        'unknown': {'trust_score': 0.50, 'decay_factor': 0.90}
    }
    
    source_type = classify_source(news_item['source'])
    config = source_configs.get(source_type, source_configs['unknown'])
    
    # Calculate age-adjusted trust
    age_days = (datetime.now() - news_item['date']).days
    trust_score = config['trust_score'] * (config['decay_factor'] ** age_days)
    
    return {
        'source_type': source_type,
        'trust_score': trust_score,
        'decay_factor': config['decay_factor']
    }
```

**Why**: Prevents low-quality sources from polluting features

---

### Strategy 4: Volume Normalization (CRITICAL)

#### At Ingestion:

```python
def normalize_sentiment_by_volume(bucket_sentiment, bucket_volume):
    """Normalize sentiment by volume to prevent volume spikes from dominating"""
    
    # Calculate rolling average volume (30-day window)
    avg_volume = calculate_rolling_average(bucket_volume, window=30)
    
    # Normalize sentiment
    if avg_volume > 0:
        normalized_sentiment = bucket_sentiment * (avg_volume / bucket_volume)
    else:
        normalized_sentiment = bucket_sentiment
    
    return normalized_sentiment
```

**Why**: Prevents volume spikes from causing feature drift

---

## 🧠 Neural Segmentation Strategy

### Problem: Neural Features Can Drift

**Issue**: Neural composite signals (dollar_neural_score, fed_neural_score) can drift if underlying drivers change.

**Solution**: Segment by Driver Layer

### Layer Architecture:

```
Layer 3 (Deep Drivers):
  - Rate differentials
  - Employment data
  - Processing capacity

Layer 2 (Neural Scores):
  - dollar_neural_score
  - fed_neural_score
  - crush_neural_score

Layer 1 (Master Score):
  - Master neural score
```

### Segmentation at Feature Engineering:

```python
def calculate_neural_signals_layer2(date):
    """Calculate Layer 2 neural signals (segmented by driver)"""
    
    # Layer 3 inputs (deep drivers)
    layer3_dollar = get_layer3_drivers('dollar', date)
    layer3_fed = get_layer3_drivers('fed', date)
    layer3_crush = get_layer3_drivers('crush', date)
    
    # Layer 2 outputs (neural scores)
    dollar_neural_score = neural_model_dollar.predict(layer3_dollar)
    fed_neural_score = neural_model_fed.predict(layer3_fed)
    crush_neural_score = neural_model_crush.predict(layer3_crush)
    
    return {
        'dollar_neural_score': dollar_neural_score,
        'fed_neural_score': fed_neural_score,
        'crush_neural_score': crush_neural_score,
        'layer3_dollar': layer3_dollar,  # Store for drift detection
        'layer3_fed': layer3_fed,
        'layer3_crush': layer3_crush
    }
```

**Why**: Allows layer-specific drift detection, prevents cross-layer contamination

---

## 📊 Implementation Checklist

### Before Ingestion Scripts:

- [ ] ✅ Implement bucket segmentation at ingestion
- [ ] ✅ Implement temporal segmentation (regime tagging)
- [ ] ✅ Implement source segmentation (trust scoring)
- [ ] ✅ Implement volume normalization
- [ ] ✅ Store segmented data in separate tables

### Before Feature Engineering:

- [ ] ✅ Implement neural layer segmentation
- [ ] ✅ Store each layer separately
- [ ] ✅ Calculate layer-specific weights
- [ ] ✅ Implement drift detection per layer

### Before Training:

- [ ] ✅ Verify segmentation is working
- [ ] ✅ Validate regime weighting
- [ ] ✅ Test drift detection
- [ ] ✅ Monitor feature stability

---

## ✅ Summary

### Segmentation Strategy:

1. ✅ **Bucket Segmentation**: Segment news into buckets at ingestion
2. ✅ **Temporal Segmentation**: Tag with regime/date buckets
3. ✅ **Source Segmentation**: Tag with source trust scores
4. ✅ **Volume Normalization**: Normalize sentiment by volume
5. ✅ **Neural Layer Segmentation**: Store each layer separately

### Tables Required:

1. ✅ `raw.scrapecreators_news_buckets` - Segmented at ingestion
2. ✅ `staging.news_bucketed` - Aggregated by date, bucket
3. ✅ `staging.sentiment_buckets` - Sentiment scores by bucket
4. ✅ `features.sentiment_features_daily` - ML-ready features
5. ✅ `features.neural_signals_daily` - Layer 2 scores
6. ✅ `features.neural_master_score` - Layer 1 score

---

**Last Updated**: November 28, 2025

