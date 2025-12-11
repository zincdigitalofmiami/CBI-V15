#!/usr/bin/env python3
"""
Sentiment Calculator for News Buckets
Uses FinBERT for sentiment + Zero-Shot Classification for bucketing
CORRECTED LOGIC: China buying = BULLISH, Tariffs = Context-dependent
"""

from transformers import pipeline
import torch
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models (run once, reuse)
classifier = None
sentiment_analyzer = None

def initialize_models():
    """Initialize models (run once at startup)"""
    global classifier, sentiment_analyzer
    
    if classifier is None:
        logger.info("Loading Zero-Shot Classifier (BART-large-mnli)...")
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    if sentiment_analyzer is None:
        logger.info("Loading FinBERT...")
        sentiment_analyzer = pipeline("text-classification", model="ProsusAI/finbert")
    
    logger.info("Models loaded successfully")

# Custom Bucket Labels (Zero-Shot Classification)
CANDIDATE_LABELS = [
    "Biofuel Policy and EPA Regulations",
    "Soybean Supply and Harvest Yields",
    "China Trade and Export Demand",
    "Macro Economy and Interest Rates",
    "Logistics and Transportation",
    "Trade Policy and Tariffs",
    "Positioning and Market Microstructure",
    "Idiosyncratic Corporate News"
]

def calculate_sentiment_finbert(text: str, bucket_type: str) -> Dict[str, any]:
    """
    Calculate sentiment using FinBERT with CORRECTED ZL mapping
    
    CRITICAL FIXES:
    1. China buying = BULLISH (not BEARISH) - US is net exporter
    2. Tariffs = Context-dependent (not always bullish) - 2018 trade war crashed ZS
    
    Args:
        text: News article headline/content (use FULL BODY, not just headline)
        bucket_type: Bucket type (biofuel, China, tariffs, etc.)
    
    Returns:
        {
            'sentiment': 'BULLISH_ZL' | 'BEARISH_ZL' | 'NEUTRAL',
            'confidence': float (0-1),
            'raw_score': float (FinBERT output)
        }
    """
    global sentiment_analyzer
    
    if sentiment_analyzer is None:
        initialize_models()
    
    # Ensure we have enough context (FinBERT needs ~64 tokens)
    if len(text.split()) < 10:
        logger.warning(f"Text too short ({len(text.split())} words), may need more context")
    
    # Predict sentiment
    sent_result = sentiment_analyzer(text, truncation=True, max_length=512)[0]
    sent_label = sent_result['label']  # 'positive', 'negative', 'neutral'
    sent_score = sent_result['score']
    
    # Map FinBERT output to scores
    if sent_label == 'positive':
        positive_score = sent_score
        negative_score = 0.0
        neutral_score = 0.0
    elif sent_label == 'negative':
        positive_score = 0.0
        negative_score = sent_score
        neutral_score = 0.0
    else:  # neutral
        positive_score = 0.0
        negative_score = 0.0
        neutral_score = sent_score
    
    text_lower = text.lower()
    
    # Bucket-specific mapping (CORRECTED LOGIC)
    if 'biofuel' in bucket_type.lower() or 'epa' in bucket_type.lower():
        # Positive biofuel news → BULLISH_ZL (more demand)
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
    
    elif 'supply' in bucket_type.lower() or 'weather' in bucket_type.lower() or 'harvest' in bucket_type.lower():
        # Positive supply news (bumper crop) → BEARISH_ZL (more supply)
        # Negative supply news (drought) → BULLISH_ZL (supply destruction)
        if positive_score > 0.5:
            return {
                'sentiment': 'BEARISH_ZL',
                'confidence': positive_score,
                'raw_score': positive_score
            }
        elif negative_score > 0.5:
            return {
                'sentiment': 'BULLISH_ZL',
                'confidence': negative_score,
                'raw_score': negative_score
            }
        else:
            return {
                'sentiment': 'NEUTRAL',
                'confidence': neutral_score,
                'raw_score': neutral_score
            }
    
    elif 'china' in bucket_type.lower() or 'trade' in bucket_type.lower():
        # ✅ CORRECTED: Positive China buying → BULLISH_ZL (drains US stocks)
        # Negative China cancellation → BEARISH_ZL (stocks build up)
        # Economic Reality: US is net exporter, China is primary buyer
        # More China imports = drain US ending stocks = HIGHER prices
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
    
    elif 'tariff' in bucket_type.lower() or 'trade policy' in bucket_type.lower():
        # ✅ CORRECTED: Tariffs are context-dependent (not always bullish)
        # Economic Reality: 2018 Trade War caused ZS to crash from $10.50 to $8.00
        # Nuance: Tariffs on US exports (retaliation) = BEARISH (demand destruction)
        #         Tariffs on Chinese UCO imports = BULLISH (protects US biofuel demand)
        
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


def segment_news_zero_shot(text: str) -> Dict[str, any]:
    """
    Segment news into buckets using Zero-Shot Classification
    
    Uses BART-large-mnli to classify text into custom buckets
    without needing to retrain a model
    
    Args:
        text: News article headline/content (use FULL BODY)
    
    Returns:
        {
            'bucket': str (primary bucket),
            'confidence': float (0-1),
            'all_scores': dict (all bucket scores)
        }
    """
    global classifier
    
    if classifier is None:
        initialize_models()
    
    # Classify into buckets
    result = classifier(text, CANDIDATE_LABELS, multi_label=False)
    
    return {
        'bucket': result['labels'][0],
        'confidence': result['scores'][0],
        'all_scores': dict(zip(result['labels'], result['scores']))
    }


def process_news_item(text: str, headline: Optional[str] = None) -> Dict[str, any]:
    """
    Complete pipeline: Bucket Segmentation → Sentiment Calculation → ZL Mapping
    
    Args:
        text: Full article body (REQUIRED - FinBERT needs ~64 tokens)
        headline: Optional headline (for reference)
    
    Returns:
        {
            'bucket': str,
            'bucket_confidence': float,
            'sentiment': str (BULLISH_ZL, BEARISH_ZL, NEUTRAL),
            'sentiment_confidence': float,
            'zl_signal': int (1=Bullish, -1=Bearish, 0=Neutral)
        }
    """
    # Use full body text (not just headline)
    # FinBERT needs ~64 tokens of context to detect nuance
    if len(text.split()) < 10:
        logger.warning("Text too short, may need more context for accurate sentiment")
    
    # Step 1: Bucket Segmentation (Zero-Shot)
    bucket_result = segment_news_zero_shot(text)
    primary_bucket = bucket_result['bucket']
    bucket_conf = bucket_result['confidence']
    
    # Step 2: Sentiment Calculation (FinBERT)
    sentiment_result = calculate_sentiment_finbert(text, primary_bucket)
    
    # Step 3: ZL Signal Mapping
    zl_signal = 1 if sentiment_result['sentiment'] == 'BULLISH_ZL' else \
                -1 if sentiment_result['sentiment'] == 'BEARISH_ZL' else 0
    
    return {
        'bucket': primary_bucket,
        'bucket_confidence': bucket_conf,
        'sentiment': sentiment_result['sentiment'],
        'sentiment_confidence': sentiment_result['confidence'],
        'zl_signal': zl_signal,
        'raw_sentiment_score': sentiment_result['raw_score']
    }


if __name__ == "__main__":
    # Test run
    initialize_models()
    
    sample_news = [
        "EPA is considering delaying the RVO mandate announcements due to pressure.",
        "USDA reports record soybean yields in Illinois, beating expectations.",
        "Sinograin purchases 10 cargoes of US Soybeans for immediate delivery.",
        "China announces 60% tariff on US soybean exports in retaliation.",
        "US imposes tariffs on Chinese UCO imports to protect biodiesel industry."
    ]
    
    print("-" * 60)
    for item in sample_news:
        result = process_news_item(item)
        print(f"NEWS: {item[:50]}...")
        print(f"  -> BUCKET: {result['bucket']} ({result['bucket_confidence']:.3f})")
        print(f"  -> SENTIMENT: {result['sentiment']} ({result['sentiment_confidence']:.3f})")
        print(f"  -> ZL SIGNAL: {result['zl_signal']} (1=Bull, -1=Bear, 0=Neutral)")
        print("-" * 60)

