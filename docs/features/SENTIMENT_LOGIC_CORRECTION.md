# Sentiment Logic Correction - Critical Fixes

**Date**: November 28, 2025  
**Status**: ⚠️ **CRITICAL CORRECTIONS** - Third agent validation  
**Source**: US Oil Solutions validation feedback

---

## 🚨 Critical Logic Errors Identified

### Error 1: China Import Logic is INVERTED ❌

**My Original Logic** (WRONG):
```python
elif bucket_type == 'china_demand':
    # Positive China import news → BEARISH_ZL (more supply = lower price)
    if positive_score > 0.5:
        return {'sentiment': 'BEARISH_ZL'}  # ❌ WRONG
```

**Economic Reality**:
- US is a **net exporter** of soy complex
- China is the **primary buyer**
- **Mechanism**: More China imports = drain US ending stocks = **HIGHER prices**

**Corrected Logic** (RIGHT):
```python
elif bucket_type == 'china_demand':
    # Positive China buying news → BULLISH_ZL (drains US stocks = higher price)
    # Negative China cancellation news → BEARISH_ZL (stocks build up)
    if positive_score > 0.5:
        return {'sentiment': 'BULLISH_ZL'}  # ✅ CORRECT
    elif negative_score > 0.5:
        return {'sentiment': 'BEARISH_ZL'}  # ✅ CORRECT
```

**Impact**: ⚠️ **CRITICAL** - Would lose money on first major export announcement

---

### Error 2: Tariff Logic is Too Simplistic ❌

**My Original Logic** (WRONG):
```python
elif bucket_type == 'tariffs_trade_policy':
    # Positive tariff news → BULLISH_ZL (trade war = volatility)
    if positive_score > 0.5:
        return {'sentiment': 'BULLISH_ZL'}  # ❌ TOO SIMPLE
```

**Economic Reality**:
- 2018 Trade War: ZS crashed from $10.50 to $8.00 (China stopped buying)
- ZL follows ZS (crushing slows down)
- **Nuance**: Tariffs are double-edged:
  - Tariffs on **US exports** (retaliation) = **BEARISH** (demand destruction)
  - Tariffs on **Chinese UCO imports** = **BULLISH** (protects US biofuel demand)

**Corrected Logic** (RIGHT):
```python
elif bucket_type == 'tariffs_trade_policy':
    # Need to detect TARGET of tariff
    text_lower = text.lower()
    
    # Tariffs on Chinese UCO/Biodiesel imports = BULLISH (protects US demand)
    if any(kw in text_lower for kw in ['uco', 'used cooking oil', 'biodiesel import', 'chinese import']):
        if positive_score > 0.5:
            return {'sentiment': 'BULLISH_ZL'}  # ✅ CORRECT
        else:
            return {'sentiment': 'BEARISH_ZL'}
    
    # Tariffs on US exports (retaliation) = BEARISH (demand destruction)
    elif any(kw in text_lower for kw in ['us export', 'retaliation', 'trade war', 'china tariff']):
        if positive_score > 0.5:
            return {'sentiment': 'BEARISH_ZL'}  # ✅ CORRECT (demand destruction)
        else:
            return {'sentiment': 'BULLISH_ZL'}  # Removing tariffs = bullish
    
    # Default: BEARISH for soy complex (conservative)
    else:
        return {'sentiment': 'BEARISH_ZL'}  # ✅ DEFAULT TO BEARISH
```

**Impact**: ⚠️ **HIGH** - Would misclassify trade war events

---

## ✅ Corrected ZL Mapping Table

### Approved Logic (Revised Mapping)

| Bucket | FinBERT Sentiment | Correct ZL Mapping | Rationale |
|--------|------------------|-------------------|-----------|
| **B1: Biofuel (EPA)** | Positive | **BULLISH_ZL** | Higher mandates = Higher demand |
| **B1: Biofuel (EPA)** | Negative | **BEARISH_ZL** | RVO delays or SRE waivers = Crash |
| **B2: Supply (USDA)** | Positive (High Yield) | **BEARISH_ZL** | "Good weather" = More supply = Lower Price |
| **B2: Supply (USDA)** | Negative (Drought) | **BULLISH_ZL** | Supply destruction = Higher Price |
| **B3: Trade (China)** | Positive (Buying) | **BULLISH_ZL** ✅ **FIXED** | Exports drain stocks = Higher Price |
| **B3: Trade (China)** | Negative (Cancellation) | **BEARISH_ZL** | Stocks build up = Lower Price |
| **B7: Trump (Tariff)** | Positive (Protectionism) | **CONTEXT DEPENDENT** ✅ **FIXED** | Default BEARISH unless "UCO" or "Biodiesel" mentioned |

---

## ✅ Architecture Validation

### Approved Components:

1. ✅ **Ingestion**: ScrapeCreators is solid
   - **Note**: Ensure capturing **Full Body**, not just headline
   - FinBERT needs ~64 tokens of context to detect nuance

2. ✅ **Weighting (3/2/1)**: Approved
   - Standard "Signal Strength" approach

3. ✅ **Features (Net 7d/30d)**: Approved
   - Net calculation (Bull_Sum - Bear_Sum) correctly captures tug-of-war

---

## ⚠️ Missing Piece: Novelty/Velocity Feature

### Current Status: Missing ❌

**Why it matters**:
- If FinBERT says "Bearish" for 20th day in a row → market doesn't care (priced in)
- If FinBERT says "Bearish" after 30 days of quiet → market crashes

**Solution**: Add "Sentiment Velocity" feature

```python
# Simple velocity feature
news_sentiment_change_1d = current_score - lag(current_score, 1)

# Or more sophisticated: Cosine Similarity
# Compare today's news vector vs. last 30 days
# High similarity = "more of the same" = lower impact
# Low similarity = "novel" = higher impact
```

**Action**: Add to `features.sentiment_features_daily`:
- `news_sentiment_change_1d` - Change from previous day
- `news_sentiment_velocity_7d` - Rate of change over 7 days

---

## ✅ Zero-Shot Classification Approach

### Recommended: Use BART-large-mnli for Bucketing

**Why Better Than Retraining**:
- ✅ No need to label 5,000+ rows
- ✅ Adaptable (change labels, no retraining)
- ✅ Production-ready immediately

**Implementation**:
```python
from transformers import pipeline

# Zero-Shot Classifier for Bucketing
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Custom Bucket Labels
CANDIDATE_LABELS = [
    "Biofuel Policy and EPA Regulations",
    "Soybean Supply and Harvest Yields",
    "China Trade and Export Demand",
    "Macro Economy and Interest Rates",
    "Logistics and Transportation"
]

# FinBERT for Sentiment
sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert")
```

---

## 🔧 Corrected Python Implementation

### Updated `calculate_sentiment_finbert` Function:

```python
def calculate_sentiment_finbert(text: str, bucket_type: str) -> dict:
    """
    Calculate sentiment using FinBERT with CORRECTED ZL mapping
    
    CRITICAL FIXES:
    1. China buying = BULLISH (not BEARISH)
    2. Tariffs = Context-dependent (not always bullish)
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
    
    text_lower = text.lower()
    
    # Bucket-specific mapping (CORRECTED)
    if bucket_type == 'biofuel_policy':
        # Positive biofuel news → BULLISH_ZL (more demand)
        if positive_score > 0.5:
            return {'sentiment': 'BULLISH_ZL', 'confidence': positive_score, 'raw_score': positive_score}
        elif negative_score > 0.5:
            return {'sentiment': 'BEARISH_ZL', 'confidence': negative_score, 'raw_score': negative_score}
        else:
            return {'sentiment': 'NEUTRAL', 'confidence': neutral_score, 'raw_score': neutral_score}
    
    elif bucket_type == 'supply_weather':
        # Positive supply news (bumper crop) → BEARISH_ZL (more supply)
        # Negative supply news (drought) → BULLISH_ZL (supply destruction)
        if positive_score > 0.5:
            return {'sentiment': 'BEARISH_ZL', 'confidence': positive_score, 'raw_score': positive_score}
        elif negative_score > 0.5:
            return {'sentiment': 'BULLISH_ZL', 'confidence': negative_score, 'raw_score': negative_score}
        else:
            return {'sentiment': 'NEUTRAL', 'confidence': neutral_score, 'raw_score': neutral_score}
    
    elif bucket_type == 'china_demand':
        # ✅ CORRECTED: Positive China buying → BULLISH_ZL (drains US stocks)
        # Negative China cancellation → BEARISH_ZL (stocks build up)
        if positive_score > 0.5:
            return {'sentiment': 'BULLISH_ZL', 'confidence': positive_score, 'raw_score': positive_score}
        elif negative_score > 0.5:
            return {'sentiment': 'BEARISH_ZL', 'confidence': negative_score, 'raw_score': negative_score}
        else:
            return {'sentiment': 'NEUTRAL', 'confidence': neutral_score, 'raw_score': neutral_score}
    
    elif bucket_type == 'tariffs_trade_policy':
        # ✅ CORRECTED: Context-dependent tariff logic
        # Tariffs on Chinese UCO/Biodiesel imports = BULLISH (protects US demand)
        if any(kw in text_lower for kw in ['uco', 'used cooking oil', 'biodiesel import', 'chinese import']):
            if positive_score > 0.5:
                return {'sentiment': 'BULLISH_ZL', 'confidence': positive_score, 'raw_score': positive_score}
            else:
                return {'sentiment': 'BEARISH_ZL', 'confidence': negative_score, 'raw_score': negative_score}
        
        # Tariffs on US exports (retaliation) = BEARISH (demand destruction)
        elif any(kw in text_lower for kw in ['us export', 'retaliation', 'trade war', 'china tariff']):
            if positive_score > 0.5:
                return {'sentiment': 'BEARISH_ZL', 'confidence': positive_score, 'raw_score': positive_score}
            else:
                return {'sentiment': 'BULLISH_ZL', 'confidence': negative_score, 'raw_score': negative_score}
        
        # Default: BEARISH for soy complex (conservative)
        else:
            return {'sentiment': 'BEARISH_ZL', 'confidence': max(positive_score, negative_score), 'raw_score': max(positive_score, negative_score)}
    
    # Default: use FinBERT output directly
    else:
        if positive_score > 0.5:
            return {'sentiment': 'BULLISH_ZL', 'confidence': positive_score, 'raw_score': positive_score}
        elif negative_score > 0.5:
            return {'sentiment': 'BEARISH_ZL', 'confidence': negative_score, 'raw_score': negative_score}
        else:
            return {'sentiment': 'NEUTRAL', 'confidence': neutral_score, 'raw_score': neutral_score}
```

---

## ✅ Final Verdict

### Readiness Score: 85% → 95% (After Fixes)

**Blockers Fixed**:
- ✅ China logic inverted (FIXED)
- ✅ Tariff logic too simplistic (FIXED)

**Remaining Actions**:
- ⚠️ Add sentiment velocity feature
- ⚠️ Implement zero-shot classification for bucketing
- ⚠️ Ensure full body text capture (not just headline)

**Status**: ✅ **READY TO BUILD** `features.sentiment_features_daily` after logic fixes applied

---

**Last Updated**: November 28, 2025

