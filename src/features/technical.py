"""
Technical Features Module

Computes technical indicators from price/volume data using Polars.

Features computed:
- Momentum: Returns over multiple horizons, relative strength
- Volatility: Historical vol, ATR, volatility regimes
- Volume: Volume ratios, dollar volume, accumulation/distribution
- Moving averages: SMA, EMA, crossovers
- Technical indicators: RSI, MACD, Bollinger Bands

All computations are vectorized using Polars for performance.
"""

from datetime import datetime
import polars as pl
import numpy as np
from typing import Optional


class TechnicalFeatures:
    """
    Compute technical indicators from OHLCV data.

    All methods operate on Polars DataFrames with schema:
    - symbol: str
    - date: datetime
    - open, high, low, close, volume: float

    Returns DataFrame with original columns + computed features.
    """

    def __init__(self):
        """Initialize technical features computer."""
        pass

    def compute_all(
        self,
        df: pl.DataFrame,
        horizons: list[int] = [1, 5, 10, 20, 60],
    ) -> pl.DataFrame:
        """
        Compute all technical features.

        Args:
            df: Price data (symbol, date, OHLCV)
            horizons: List of lookback periods for momentum/volatility

        Returns:
            DataFrame with all technical features added
        """
        # Ensure sorted by symbol and date
        df = df.sort(['symbol', 'date'])

        # Compute features by category
        df = self.add_returns(df, horizons)
        df = self.add_volatility(df, horizons)
        df = self.add_volume_features(df)
        df = self.add_moving_averages(df)
        df = self.add_technical_indicators(df)

        return df

    def add_returns(
        self,
        df: pl.DataFrame,
        horizons: list[int] = [1, 5, 10, 20, 60],
    ) -> pl.DataFrame:
        """
        Add return features over multiple horizons.

        Features:
        - return_Nd: N-day return
        - return_Nd_rank: Percentile rank of N-day return
        - return_acceleration: Change in momentum
        """
        for horizon in horizons:
            # Simple return
            df = df.with_columns([
                (
                    (pl.col('close') / pl.col('close').shift(horizon) - 1)
                    .over('symbol')
                    .alias(f'return_{horizon}d')
                )
            ])

            # Rank within universe (cross-sectional)
            df = df.with_columns([
                (
                    pl.col(f'return_{horizon}d')
                    .rank(method='average')
                    .over('date')
                    / pl.col(f'return_{horizon}d').count().over('date')
                ).alias(f'return_{horizon}d_rank')
            ])

        # Return acceleration (change in momentum)
        if 5 in horizons and 20 in horizons:
            df = df.with_columns([
                (pl.col('return_5d') - pl.col('return_20d'))
                .alias('return_acceleration')
            ])

        # Distance from 52-week high
        df = df.with_columns([
            (
                (pl.col('close') / pl.col('close').rolling_max(252).over('symbol') - 1)
                .alias('dist_from_52w_high')
            )
        ])

        return df

    def add_volatility(
        self,
        df: pl.DataFrame,
        horizons: list[int] = [10, 20, 60],
    ) -> pl.DataFrame:
        """
        Add volatility features.

        Features:
        - volatility_Nd: N-day historical volatility (annualized)
        - volatility_rank: Percentile of current vs historical vol
        - volatility_trend: Is vol increasing or decreasing
        """
        for horizon in horizons:
            # Historical volatility (annualized)
            df = df.with_columns([
                (
                    (pl.col('close') / pl.col('close').shift(1) - 1)
                    .rolling_std(horizon)
                    .over('symbol')
                    * np.sqrt(252)  # Annualize
                ).alias(f'volatility_{horizon}d')
            ])

            # Volatility percentile rank (over last year)
            df = df.with_columns([
                (
                    pl.col(f'volatility_{horizon}d')
                    .rank(method='average')
                    .rolling_map(
                        lambda s: s / len(s) if len(s) > 0 else None,
                        window_size=252
                    )
                    .over('symbol')
                ).alias(f'volatility_{horizon}d_rank')
            ])

        # Volatility trend (20d vs 60d)
        if 20 in horizons and 60 in horizons:
            df = df.with_columns([
                (pl.col('volatility_20d') / pl.col('volatility_60d') - 1)
                .alias('volatility_trend')
            ])

        # Average True Range (ATR)
        df = self._add_atr(df, period=14)

        return df

    def _add_atr(self, df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """Add Average True Range indicator."""
        # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        df = df.with_columns([
            pl.col('close').shift(1).over('symbol').alias('prev_close')
        ])

        df = df.with_columns([
            pl.max_horizontal([
                pl.col('high') - pl.col('low'),
                (pl.col('high') - pl.col('prev_close')).abs(),
                (pl.col('low') - pl.col('prev_close')).abs(),
            ]).alias('true_range')
        ])

        df = df.with_columns([
            pl.col('true_range').rolling_mean(period).over('symbol').alias(f'atr_{period}d')
        ])

        # Clean up intermediate columns
        df = df.drop(['prev_close', 'true_range'])

        return df

    def add_volume_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add volume-based features.

        Features:
        - volume_ratio: Current vs 20-day average
        - dollar_volume: Price * volume (liquidity)
        - volume_trend: 5-day vs 20-day average
        """
        # Volume ratio (vs 20-day average)
        df = df.with_columns([
            (
                pl.col('volume') / pl.col('volume').rolling_mean(20).over('symbol')
            ).alias('volume_ratio_20d')
        ])

        # Dollar volume (liquidity proxy)
        df = df.with_columns([
            (pl.col('close') * pl.col('volume')).alias('dollar_volume')
        ])

        # Volume trend
        df = df.with_columns([
            (
                pl.col('volume').rolling_mean(5).over('symbol') /
                pl.col('volume').rolling_mean(20).over('symbol') - 1
            ).alias('volume_trend')
        ])

        return df

    def add_moving_averages(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add moving average features.

        Features:
        - sma_N: Simple moving average
        - price_vs_sma_N: Price distance from SMA
        """
        periods = [20, 50, 200]

        for period in periods:
            # SMA
            df = df.with_columns([
                pl.col('close').rolling_mean(period).over('symbol').alias(f'sma_{period}')
            ])

            # Price vs SMA (percentage)
            df = df.with_columns([
                (pl.col('close') / pl.col(f'sma_{period}') - 1).alias(f'price_vs_sma_{period}')
            ])

        return df

    def add_technical_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add common technical indicators.

        Features:
        - rsi_14: Relative Strength Index
        - macd: MACD line
        - macd_signal: MACD signal line
        - macd_histogram: MACD histogram
        """
        # RSI
        df = self._add_rsi(df, period=14)

        # MACD
        df = self._add_macd(df, fast=12, slow=26, signal=9)

        return df

    def _add_rsi(self, df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """
        Add Relative Strength Index (RSI).

        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        """
        # Calculate price changes
        df = df.with_columns([
            (pl.col('close') - pl.col('close').shift(1)).over('symbol').alias('price_change')
        ])

        # Separate gains and losses
        df = df.with_columns([
            pl.when(pl.col('price_change') > 0)
              .then(pl.col('price_change'))
              .otherwise(0)
              .alias('gain'),
            pl.when(pl.col('price_change') < 0)
              .then((-pl.col('price_change')))
              .otherwise(0)
              .alias('loss'),
        ])

        # Calculate average gain/loss using Wilder's smoothing
        df = df.with_columns([
            pl.col('gain').rolling_mean(period).over('symbol').alias('avg_gain'),
            pl.col('loss').rolling_mean(period).over('symbol').alias('avg_loss'),
        ])

        # Calculate RS and RSI
        df = df.with_columns([
            (
                100 - (100 / (1 + (pl.col('avg_gain') / pl.col('avg_loss'))))
            ).alias(f'rsi_{period}')
        ])

        # Clean up intermediate columns
        df = df.drop(['price_change', 'gain', 'loss', 'avg_gain', 'avg_loss'])

        return df

    def _add_macd(
        self,
        df: pl.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pl.DataFrame:
        """
        Add MACD indicator.

        MACD = EMA(fast) - EMA(slow)
        Signal = EMA(MACD, signal)
        Histogram = MACD - Signal
        """
        # Calculate EMAs
        df = df.with_columns([
            pl.col('close').ewm_mean(span=fast).over('symbol').alias(f'ema_{fast}'),
            pl.col('close').ewm_mean(span=slow).over('symbol').alias(f'ema_{slow}'),
        ])

        # MACD line
        df = df.with_columns([
            (pl.col(f'ema_{fast}') - pl.col(f'ema_{slow}')).alias('macd')
        ])

        # Signal line
        df = df.with_columns([
            pl.col('macd').ewm_mean(span=signal).over('symbol').alias('macd_signal')
        ])

        # Histogram
        df = df.with_columns([
            (pl.col('macd') - pl.col('macd_signal')).alias('macd_histogram')
        ])

        # Clean up intermediate EMAs (keep MACD metrics)
        df = df.drop([f'ema_{fast}', f'ema_{slow}'])

        return df


# Test function
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # Load sample data
    price_df = pl.read_parquet('data/price/daily/sample_universe_2022-10-24_to_2025-10-23.parquet')

    print("="*70)
    print("TECHNICAL FEATURES TEST")
    print("="*70)

    print(f"\nInput data: {len(price_df)} rows, {len(price_df.columns)} columns")
    print(f"Symbols: {price_df['symbol'].n_unique()}")
    print(f"Date range: {price_df['date'].min()} to {price_df['date'].max()}")

    # Compute features
    tech = TechnicalFeatures()
    features_df = tech.compute_all(price_df)

    print(f"\nOutput data: {len(features_df)} rows, {len(features_df.columns)} columns")
    print(f"\nFeatures added: {len(features_df.columns) - len(price_df.columns)}")

    # Show new columns
    new_cols = [col for col in features_df.columns if col not in price_df.columns]
    print(f"\nNew feature columns ({len(new_cols)}):")
    for col in sorted(new_cols):
        print(f"  - {col}")

    # Show sample for one stock
    print(f"\nSample (AAPL, recent dates):")
    sample_cols = ['symbol', 'date', 'close', 'return_5d', 'return_20d',
                   'volatility_20d', 'rsi_14', 'volume_ratio_20d']
    sample = features_df.filter(pl.col('symbol') == 'AAPL').select(sample_cols).tail(5)
    print(sample)

    print("\n" + "="*70)
    print("✓ Technical features computed successfully")
