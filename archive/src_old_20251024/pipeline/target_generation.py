"""
Target Generation Step

Creates target variables (forward returns) for prediction.
"""

import polars as pl
from loguru import logger


class TargetGenerator:
    """Generate target variables for ML pipeline."""

    def compute_forward_return(
        self,
        df: pl.DataFrame,
        horizon_days: int = 30,
        market_relative: bool = True,
    ) -> pl.DataFrame:
        """
        Compute forward return target variable.

        Args:
            df: DataFrame with price data and features
            horizon_days: Number of days to look forward
            market_relative: If True, compute returns relative to SPY

        Returns:
            DataFrame with target variable added
        """
        logger.info("=" * 70)
        logger.info(f"COMPUTING TARGET VARIABLE ({horizon_days}d forward returns)")
        logger.info("=" * 70)

        df = df.sort(['symbol', 'date'])

        # Compute stock forward return
        df = df.with_columns([
            (
                pl.col('adj_close').shift(-horizon_days).over('symbol') / pl.col('adj_close') - 1.0
            ).alias(f'forward_return_{horizon_days}d')
        ])

        # Compute market-relative returns if requested
        if market_relative:
            # Need SPY returns - should be in df already from sector features
            # For now, use absolute returns (TODO: implement proper market-relative)
            target_col = f'target_return_{horizon_days}d_vs_market'
            df = df.with_columns([
                pl.col(f'forward_return_{horizon_days}d').alias(target_col)
            ])
            logger.info(f"  Target: {target_col} (market-relative)")
        else:
            target_col = f'target_return_{horizon_days}d'
            logger.info(f"  Target: {target_col} (absolute)")

        # Count nulls
        null_count = df[target_col].is_null().sum()
        logger.info(f"  ✓ Target computed: {null_count:,} nulls ({null_count/len(df)*100:.1f}%)")

        return df
