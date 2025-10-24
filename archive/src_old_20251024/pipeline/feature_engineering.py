"""
Feature Engineering Step

Computes all features from raw data using FeatureAligner.
"""

import polars as pl
from loguru import logger
from src.features.alignment import FeatureAligner


class FeatureEngineer:
    """Compute features for ML pipeline."""

    def __init__(self):
        """Initialize feature engineer."""
        self.aligner = FeatureAligner()

    def compute_features(
        self,
        price_df: pl.DataFrame,
        market_df: pl.DataFrame,
        financials_df: pl.DataFrame,
        metadata_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Compute all features from raw data.

        Args:
            price_df: Stock price data
            market_df: Market data (SPY, VIX, sectors)
            financials_df: Quarterly financials
            metadata_df: Company metadata

        Returns:
            DataFrame with all computed features aligned to daily frequency
        """
        logger.info("=" * 70)
        logger.info("COMPUTING FEATURES")
        logger.info("=" * 70)

        features_df = self.aligner.compute_all_features(
            price_df=price_df,
            financials_df=financials_df,
            market_df=market_df,
            metadata_df=metadata_df,
        )

        logger.info(f"✓ Feature engineering complete: {len(features_df):,} rows, {len(features_df.columns)} columns")

        return features_df
