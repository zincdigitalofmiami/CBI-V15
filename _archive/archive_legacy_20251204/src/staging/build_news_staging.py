"""
News/Sentiment Staging Builder - Python-First Architecture
Builds staging.news_bucketed and staging.sentiment_buckets from raw data.

Fixes:
- Aligns theme_primary → theme (was bucket_type)
- Computes both avg_sentiment_score AND avg_confidence
- Validates enum fields (sentiment, impact)
- Avoids selecting headline/content (cost control)
"""

import os
from typing import List

import pandas as pd
from google.cloud import bigquery

PROJECT_ID = os.getenv("GCP_PROJECT", "cbi-v15")

# Validation enums
VALID_SENTIMENTS = ['BULLISH_ZL', 'BEARISH_ZL', 'NEUTRAL']
VALID_IMPACTS = ['HIGH', 'MEDIUM', 'LOW']


def load_news_raw() -> pd.DataFrame:
    """
    Load raw news data WITHOUT large text columns (cost control).
    """
    client = bigquery.Client(project=PROJECT_ID)
    
    # CRITICAL: Do NOT select headline/content to avoid scan costs
    query = f"""
    SELECT
        date,
        article_id,
        theme_primary,
        is_trump_related,
        policy_axis,
        horizon,
        zl_sentiment,
        impact_magnitude,
        sentiment_confidence,
        sentiment_raw_score,
        source,
        source_trust_score
    FROM `{PROJECT_ID}.raw.scrapecreators_news_buckets`
    WHERE date >= '2010-01-01'
    """
    
    print("[build_news_staging] Loading raw news (without TEXT columns)...")
    df = client.query(query).to_dataframe()
    print(f"  -> Loaded {len(df):,} articles")
    return df


def validate_news_raw(df: pd.DataFrame):
    """Validate raw news data against schemas."""
    required = ['date', 'article_id', 'theme_primary', 'zl_sentiment', 'impact_magnitude']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"[validate_news_raw] Missing required columns: {missing}")
    
    # Validate enums
    invalid_sentiments = ~df['zl_sentiment'].isin(VALID_SENTIMENTS)
    if invalid_sentiments.any():
        bad_vals = df.loc[invalid_sentiments, 'zl_sentiment'].unique()
        raise RuntimeError(f"[validate_news_raw] Invalid zl_sentiment values: {bad_vals}")
    
    invalid_impacts = ~df['impact_magnitude'].isin(VALID_IMPACTS)
    if invalid_impacts.any():
        bad_vals = df.loc[invalid_impacts, 'impact_magnitude'].unique()
        raise RuntimeError(f"[validate_news_raw] Invalid impact_magnitude values: {bad_vals}")
    
    print("[validate_news_raw] ✅ Validation passed")


def build_news_bucketed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate news by date + theme.
    
    Schema: date, theme, article_count, avg_sentiment_score, avg_confidence,
            high/medium/low_impact_count, bullish/bearish/neutral_count
    """
    print("[build_news_bucketed] Aggregating by date + theme...")
    
    # Rename theme_primary -> theme for clarity
    df = df.rename(columns={'theme_primary': 'theme'})
    
    grouped = df.groupby(['date', 'theme']).agg({
        'article_id': 'count',
        'sentiment_raw_score': 'mean',
        'sentiment_confidence': 'mean'
    }).reset_index()
    
    grouped.columns = ['date', 'theme', 'article_count', 'avg_sentiment_score', 'avg_confidence']
    
    # Count by impact magnitude (HIGH / MEDIUM / LOW)
    impact_counts = (
        df.groupby(['date', 'theme', 'impact_magnitude'])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )
    # Ensure all impact columns exist
    for imp in VALID_IMPACTS:
        if imp not in impact_counts.columns:
            impact_counts[imp] = 0

    impact_counts = impact_counts.rename(columns={
        'HIGH': 'high_impact_count',
        'MEDIUM': 'medium_impact_count',
        'LOW': 'low_impact_count',
    })
    impact_counts = impact_counts[
        ['date', 'theme', 'high_impact_count', 'medium_impact_count', 'low_impact_count']
    ]
    
    # Count by sentiment (BULLISH_ZL / BEARISH_ZL / NEUTRAL)
    sentiment_counts = (
        df.groupby(['date', 'theme', 'zl_sentiment'])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )
    # Ensure all sentiment columns exist
    for sent in VALID_SENTIMENTS:
        if sent not in sentiment_counts.columns:
            sentiment_counts[sent] = 0

    sentiment_counts = sentiment_counts.rename(columns={
        'BULLISH_ZL': 'bullish_count',
        'BEARISH_ZL': 'bearish_count',
        'NEUTRAL': 'neutral_count',
    })
    sentiment_counts = sentiment_counts[
        ['date', 'theme', 'bullish_count', 'bearish_count', 'neutral_count']
    ]
    
    # Merge all
    result = grouped.merge(impact_counts, on=['date', 'theme'], how='left')
    result = result.merge(sentiment_counts, on=['date', 'theme'], how='left')
    
    # Fill NaN counts with 0
    count_cols = ['high_impact_count', 'medium_impact_count', 'low_impact_count',
                  'bullish_count', 'bearish_count', 'neutral_count']
    for col in count_cols:
        result[col] = result[col].fillna(0).astype('int64')
    
    print(f"[build_news_bucketed] ✅ Aggregated to {len(result):,} rows")
    return result


def build_sentiment_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build sentiment metrics by date + theme.
    
    Schema: date, theme, bullish/bearish/neutral_count, net_sentiment, 
            sentiment_ratio, weighted_score
    """
    print("[build_sentiment_buckets] Computing sentiment metrics...")
    
    df = df.rename(columns={'theme_primary': 'theme'})
    
    # Count by sentiment
    sentiment_counts = df.groupby(['date', 'theme', 'zl_sentiment']).size().unstack(fill_value=0).reset_index()
    
    # Ensure all sentiment columns exist
    for sent in ['BULLISH_ZL', 'BEARISH_ZL', 'NEUTRAL']:
        if sent not in sentiment_counts.columns:
            sentiment_counts[sent] = 0
    
    sentiment_counts = sentiment_counts.rename(columns={
        'BULLISH_ZL': 'bullish_count',
        'BEARISH_ZL': 'bearish_count',
        'NEUTRAL': 'neutral_count'
    })
    
    # Net sentiment
    sentiment_counts['net_sentiment'] = (
        sentiment_counts['bullish_count'] - sentiment_counts['bearish_count']
    )
    
    # Sentiment ratio
    total_directional = sentiment_counts['bullish_count'] + sentiment_counts['bearish_count']
    sentiment_counts['sentiment_ratio'] = sentiment_counts['bullish_count'] / total_directional.replace(0, 1)
    
    # Weighted score (sentiment * confidence)
    weighted = df.copy()
    weighted['weighted'] = weighted['sentiment_raw_score'] * weighted['sentiment_confidence']
    weighted_scores = weighted.groupby(['date', 'theme'])['weighted'].mean().reset_index()
    weighted_scores.columns = ['date', 'theme', 'weighted_score']
    
    result = sentiment_counts.merge(weighted_scores, on=['date', 'theme'], how='left')
    
    print(f"[build_sentiment_buckets] ✅ Computed metrics for {len(result):,} rows")
    return result


def load_to_bigquery(df: pd.DataFrame, table_id: str, partition_field: str = 'date'):
    """Load DataFrame to BigQuery with partitioning."""
    client = bigquery.Client(project=PROJECT_ID)
    
    # Ensure date is proper type
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    job_config = bigquery.LoadJobConfig(
        write_disposition='WRITE_TRUNCATE',
        time_partitioning=bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.MONTH,
            field=partition_field,
        ),
        clustering_fields=['theme'] if 'theme' in df.columns else None,
    )
    
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    
    print(f"[load_to_bigquery] ✅ Loaded {job.output_rows:,} rows to {table_id}")


def main():
    """Build news/sentiment staging tables."""
    
    # Load and validate
    df_raw = load_news_raw()
    validate_news_raw(df_raw)
    
    # Build staging tables
    news_bucketed = build_news_bucketed(df_raw)
    sentiment_buckets = build_sentiment_buckets(df_raw)
    
    # Load to BigQuery
    load_to_bigquery(news_bucketed, f"{PROJECT_ID}.staging.news_bucketed")
    load_to_bigquery(sentiment_buckets, f"{PROJECT_ID}.staging.sentiment_buckets")
    
    print("\n✅ News/sentiment staging complete")
    print(f"   news_bucketed: {len(news_bucketed):,} rows")
    print(f"   sentiment_buckets: {len(sentiment_buckets):,} rows")


if __name__ == "__main__":
    main()

