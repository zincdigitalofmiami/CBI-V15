#!/usr/bin/env python3
"""
Pandera Validation Schema for Training Data
Prevents logic inversions and data quality issues
"""

from typing import Optional

import pandas as pd
import pandera as pa
from pandera import Check, Column

# ============================================================================
# Training Data Schema (Prevents Logic Inversions)
# ============================================================================

training_data_schema = pa.DataFrameSchema(
    {
        # Meta Columns
        "date": Column(pd.Timestamp, nullable=False),
        "symbol": Column(str, nullable=False),
        # Targets (Price Levels)
        "target_1w_price": Column(
            float,
            nullable=True,
            checks=[Check(lambda x: x > 0, error="Target price must be positive")],
        ),
        "target_1m_price": Column(
            float,
            nullable=True,
            checks=[Check(lambda x: x > 0, error="Target price must be positive")],
        ),
        "target_3m_price": Column(
            float,
            nullable=True,
            checks=[Check(lambda x: x > 0, error="Target price must be positive")],
        ),
        "target_6m_price": Column(
            float,
            nullable=True,
            checks=[Check(lambda x: x > 0, error="Target price must be positive")],
        ),
        # Fundamental Spreads (Must be economically valid)
        "board_crush": Column(
            float,
            nullable=True,
            checks=[
                Check(
                    lambda x: x > -100, error="Crush margin too negative (likely error)"
                )
            ],
        ),
        "oil_share": Column(
            float,
            nullable=True,
            checks=[
                Check(lambda x: (x >= 0) & (x <= 1), error="Oil share must be [0, 1]")
            ],
        ),
        # Sentiment Features (CRITICAL: Logic Validation)
        "news_china_sentiment_net_7d": Column(
            float,
            nullable=True,
            checks=[
                # CRITICAL: China buying must be positively correlated with ZL returns
                # This check validates the logic isn't inverted
                Check(
                    lambda df: (
                        df.corr(df["zl_return"]) > -0.1
                        if "zl_return" in df.columns
                        else True
                    ),
                    error="CRITICAL: China sentiment negatively correlated with ZL returns. Logic inverted?",
                )
            ],
        ),
        "news_biofuel_sentiment_net_7d": Column(
            float,
            nullable=True,
            checks=[
                # Biofuel sentiment must be positively correlated with ZL returns
                Check(
                    lambda df: (
                        df.corr(df["zl_return"]) > -0.1
                        if "zl_return" in df.columns
                        else True
                    ),
                    error="CRITICAL: Biofuel sentiment negatively correlated. Logic inverted?",
                )
            ],
        ),
        # Technical Indicators (Must be within reasonable bounds)
        "rsi_14": Column(
            float,
            nullable=True,
            checks=[
                Check(lambda x: (x >= 0) & (x <= 100), error="RSI must be [0, 100]")
            ],
        ),
        "bb_pct_b": Column(
            float,
            nullable=True,
            checks=[Check(lambda x: x >= -1, error="Bollinger %B too negative")],
        ),
        # Volatility (Must be positive)
        "volatility_21d": Column(
            float,
            nullable=True,
            checks=[Check(lambda x: x >= 0, error="Volatility must be non-negative")],
        ),
        # Correlations (Must be [-1, 1])
        "corr_zl_brl_60d": Column(
            float,
            nullable=True,
            checks=[
                Check(
                    lambda x: (x >= -1) & (x <= 1), error="Correlation must be [-1, 1]"
                )
            ],
        ),
        # Regime Weights (Must be positive)
        "regime_weight": Column(
            float,
            nullable=True,
            checks=[Check(lambda x: x > 0, error="Regime weight must be positive")],
        ),
    },
    strict=False,
)  # Allow extra columns


# ============================================================================
# Feature Validation (Post-Feature Engineering)
# ============================================================================


def validate_feature_logic(
    df: pd.DataFrame, feature_name: str, expected_correlation: float = 0.1
) -> bool:
    """
    Validate that a feature has the expected correlation with ZL returns

    Args:
        df: DataFrame with feature and 'zl_return' column
        feature_name: Name of feature to validate
        expected_correlation: Minimum expected correlation (default 0.1)

    Returns:
        True if validation passes, raises SchemaError if fails
    """
    if "zl_return" not in df.columns:
        return True  # Can't validate without returns

    correlation = df[feature_name].corr(df["zl_return"])

    if correlation < expected_correlation:
        raise pa.errors.SchemaError(
            f"CRITICAL: {feature_name} has correlation {correlation:.3f} with ZL returns. "
            f"Expected at least {expected_correlation:.3f}. Logic may be inverted."
        )

    return True


# ============================================================================
# Sentiment Feature Validation (CRITICAL)
# ============================================================================

sentiment_feature_schema = pa.DataFrameSchema(
    {
        # China Sentiment (CRITICAL: Must be positively correlated)
        "news_china_sentiment_net_7d": Column(
            float,
            nullable=True,
            checks=[
                Check(
                    lambda df: validate_feature_logic(
                        df, "news_china_sentiment_net_7d", expected_correlation=0.1
                    ),
                    error="China sentiment logic inverted",
                )
            ],
        ),
        # Biofuel Sentiment (CRITICAL: Must be positively correlated)
        "news_biofuel_sentiment_net_7d": Column(
            float,
            nullable=True,
            checks=[
                Check(
                    lambda df: validate_feature_logic(
                        df, "news_biofuel_sentiment_net_7d", expected_correlation=0.1
                    ),
                    error="Biofuel sentiment logic inverted",
                )
            ],
        ),
        # Supply Sentiment (CRITICAL: Must be NEGATIVELY correlated - more supply = lower price)
        "news_supply_sentiment_net_7d": Column(
            float,
            nullable=True,
            checks=[
                Check(
                    lambda df: (
                        df["news_supply_sentiment_net_7d"].corr(df["zl_return"]) < 0.1
                        if "zl_return" in df.columns
                        else True
                    ),
                    error="Supply sentiment should be negatively correlated (more supply = lower price)",
                )
            ],
        ),
    },
    strict=False,
)


# ============================================================================
# Usage Example
# ============================================================================


def validate_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate training data before model training

    Args:
        df: Training DataFrame

    Returns:
        Validated DataFrame

    Raises:
        SchemaError: If validation fails
    """
    try:
        validated_df = training_data_schema(df)
        logger.info("✅ Training data validation passed")
        return validated_df
    except pa.errors.SchemaError as e:
        logger.error(f"❌ Training data validation failed: {e}")
        raise


if __name__ == "__main__":
    # Test validation
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create test DataFrame
    test_df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=100),
            "symbol": "ZL",
            "target_1w_price": [50.0] * 100,
            "board_crush": [0.5] * 100,
            "oil_share": [0.45] * 100,
            "news_china_sentiment_net_7d": [1.0] * 100,  # Positive sentiment
            "zl_return": [0.01] * 100,  # Positive returns (should correlate positively)
            "rsi_14": [50.0] * 100,
            "volatility_21d": [0.15] * 100,
        }
    )

    try:
        validated = validate_training_data(test_df)
        logger.info("✅ Validation test passed")
    except Exception as e:
        logger.error(f"❌ Validation test failed: {e}")
