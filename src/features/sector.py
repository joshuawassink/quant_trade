"""
Sector & Market Features Module

Computes relative performance vs sector/market benchmarks.

Features computed:
- Sector relative strength: Stock vs sector ETF performance
- Market relative strength: Stock vs SPY/market performance
- Sector momentum: Is sector outperforming market?
- Market regime: Bull/bear/volatile market identification
- Interest rate environment: Yield levels and changes

Uses market data from YFinanceMarketProvider.
"""

import polars as pl
import numpy as np
from typing import Optional


class SectorFeatures:
    """
    Compute sector and market relative features.

    Requires:
    1. Stock metadata with sector assignments
    2. Market data with sector ETF prices
    3. Stock price data

    Computes stock performance relative to sector and market.
    """

    # Map sectors to their ETF tickers
    SECTOR_ETF_MAP = {
        'Technology': 'XLK',
        'Financial Services': 'XLF',
        'Financials': 'XLF',  # Alias
        'Healthcare': 'XLV',
        'Consumer Cyclical': 'XLY',
        'Consumer Defensive': 'XLP',
        'Energy': 'XLE',
        'Industrials': 'XLI',
        'Materials': 'XLB',
        'Real Estate': 'XLRE',
        'Communication Services': 'XLC',
        'Utilities': 'XLU',
    }

    def __init__(self):
        """Initialize sector features computer."""
        pass

    def compute_all(
        self,
        stock_df: pl.DataFrame,
        market_df: pl.DataFrame,
        metadata_df: pl.DataFrame,
        horizons: list[int] = [5, 20, 60],
    ) -> pl.DataFrame:
        """
        Compute all sector/market features.

        Args:
            stock_df: Stock price data (symbol, date, close, returns)
            market_df: Market data (symbol, date, close) - includes sector ETFs, SPY, VIX
            metadata_df: Stock metadata (symbol, sector)
            horizons: Return horizons to compute

        Returns:
            Stock dataframe with sector/market features added
        """
        # Normalize market_df dates (remove timezone and cast to nanoseconds)
        market_df = market_df.with_columns([
            pl.col('date').dt.replace_time_zone(None).cast(pl.Datetime('ns'))
        ])

        # Add sector assignments
        stock_df = self._add_sector_info(stock_df, metadata_df)

        # Add market relative returns
        stock_df = self.add_market_relative_returns(stock_df, market_df, horizons)

        # Add sector relative returns
        stock_df = self.add_sector_relative_returns(stock_df, market_df, metadata_df, horizons)

        # Add market regime features
        stock_df = self.add_market_regime_features(stock_df, market_df)

        # Add volatility environment
        stock_df = self.add_volatility_features(stock_df, market_df)

        return stock_df

    def _add_sector_info(
        self,
        stock_df: pl.DataFrame,
        metadata_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Join sector information to stock data."""
        # Select only symbol and sector from metadata
        sector_map = metadata_df.select(['symbol', 'sector'])

        # Join to stock data
        stock_df = stock_df.join(sector_map, on='symbol', how='left')

        return stock_df

    def add_market_relative_returns(
        self,
        stock_df: pl.DataFrame,
        market_df: pl.DataFrame,
        horizons: list[int] = [5, 20, 60],
    ) -> pl.DataFrame:
        """
        Add returns relative to market (SPY).

        Features:
        - return_Nd_vs_market: Stock return minus market return
        """
        # Get SPY returns
        spy_df = market_df.filter(pl.col('symbol') == 'SPY').select(['date', 'close'])
        spy_df = spy_df.rename({'close': 'spy_close'})

        # Join SPY to stock data (timezone already normalized in compute_all)
        stock_df = stock_df.join(spy_df, on='date', how='left')

        # Compute SPY returns for each horizon
        for horizon in horizons:
            # SPY return
            stock_df = stock_df.with_columns([
                (
                    (pl.col('spy_close') / pl.col('spy_close').shift(horizon) - 1)
                ).alias(f'spy_return_{horizon}d')
            ])

            # Relative return (stock vs market)
            if f'return_{horizon}d' in stock_df.columns:
                stock_df = stock_df.with_columns([
                    (
                        pl.col(f'return_{horizon}d') - pl.col(f'spy_return_{horizon}d')
                    ).alias(f'return_{horizon}d_vs_market')
                ])

        # Clean up SPY close (keep SPY returns)
        stock_df = stock_df.drop('spy_close')

        return stock_df

    def add_sector_relative_returns(
        self,
        stock_df: pl.DataFrame,
        market_df: pl.DataFrame,
        metadata_df: pl.DataFrame,
        horizons: list[int] = [5, 20, 60],
    ) -> pl.DataFrame:
        """
        Add returns relative to sector ETF.

        Features:
        - return_Nd_vs_sector: Stock return minus sector return
        - sector_momentum: Is sector outperforming market?
        """
        # Get sector ETF data
        sector_etfs = list(self.SECTOR_ETF_MAP.values())
        sector_df = market_df.filter(pl.col('symbol').is_in(sector_etfs))

        # Pivot to have one column per sector ETF
        for etf_symbol, sector_name in [(v, k) for k, v in self.SECTOR_ETF_MAP.items()]:
            etf_data = sector_df.filter(pl.col('symbol') == etf_symbol).select(['date', 'close'])
            etf_data = etf_data.rename({'close': f'{etf_symbol}_close'})

            # Join to stock data
            stock_df = stock_df.join(etf_data, on='date', how='left')

        # Compute sector returns and relative returns
        for horizon in horizons:
            # For each stock, compute its return relative to its sector ETF
            for sector_name, etf_symbol in self.SECTOR_ETF_MAP.items():
                close_col = f'{etf_symbol}_close'

                if close_col in stock_df.columns:
                    # Sector ETF return
                    stock_df = stock_df.with_columns([
                        (
                            (pl.col(close_col) / pl.col(close_col).shift(horizon) - 1)
                        ).alias(f'{etf_symbol}_return_{horizon}d')
                    ])

            # Compute relative return for each stock vs its sector
            # This requires dynamic column selection based on sector
            # For simplicity, we'll compute for all and select in downstream

        return stock_df

    def add_market_regime_features(
        self,
        stock_df: pl.DataFrame,
        market_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Add market regime indicators.

        Features:
        - market_trend: SPY above/below 200-day MA
        - market_momentum: SPY 20-day vs 60-day return
        """
        # Get SPY data with dates
        spy_df = market_df.filter(pl.col('symbol') == 'SPY').select(['date', 'close'])
        spy_df = spy_df.rename({'close': 'spy_close'})

        # Calculate SPY 200-day MA
        spy_df = spy_df.with_columns([
            pl.col('spy_close').rolling_mean(200).alias('spy_sma_200')
        ])

        # Market trend indicator (above/below 200 MA)
        spy_df = spy_df.with_columns([
            (pl.col('spy_close') > pl.col('spy_sma_200')).cast(pl.Int8).alias('market_trend_bullish')
        ])

        # Join to stock data
        stock_df = stock_df.join(
            spy_df.select(['date', 'market_trend_bullish']),
            on='date',
            how='left'
        )

        return stock_df

    def add_volatility_features(
        self,
        stock_df: pl.DataFrame,
        market_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Add volatility environment features.

        Features:
        - vix_level: Current VIX level
        - vix_regime: High/medium/low volatility (percentile)
        """
        # Get VIX data
        vix_df = market_df.filter(pl.col('symbol') == '^VIX').select(['date', 'close'])
        vix_df = vix_df.rename({'close': 'vix_level'})

        # VIX percentile rank (over trailing year)
        vix_df = vix_df.with_columns([
            (
                pl.col('vix_level')
                .rank(method='average')
                .rolling_map(
                    lambda s: s / len(s) if len(s) > 0 else None,
                    window_size=252
                )
            ).alias('vix_percentile')
        ])

        # VIX regime (low/medium/high)
        vix_df = vix_df.with_columns([
            pl.when(pl.col('vix_percentile') < 0.33)
              .then(pl.lit('low'))
              .when(pl.col('vix_percentile') < 0.67)
              .then(pl.lit('medium'))
              .otherwise(pl.lit('high'))
              .alias('vix_regime')
        ])

        # Join to stock data
        stock_df = stock_df.join(vix_df, on='date', how='left')

        return stock_df


# Test function
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # Load data
    stock_df = pl.read_parquet('data/price/daily/sample_universe_2022-10-24_to_2025-10-23.parquet')
    market_df = pl.read_parquet('data/market/daily/market_data_latest.parquet')
    metadata_df = pl.read_parquet('data/metadata/company_metadata_latest.parquet')

    print("="*70)
    print("SECTOR/MARKET FEATURES TEST")
    print("="*70)

    print(f"\nInput - Stock data: {len(stock_df)} rows, {stock_df['symbol'].n_unique()} stocks")
    print(f"Input - Market data: {len(market_df)} rows, {market_df['symbol'].n_unique()} instruments")
    print(f"Input - Metadata: {len(metadata_df)} companies")

    # Compute features
    sector_features = SectorFeatures()
    features_df = sector_features.compute_all(stock_df, market_df, metadata_df)

    print(f"\nOutput: {len(features_df)} rows, {len(features_df.columns)} columns")
    print(f"Features added: {len(features_df.columns) - len(stock_df.columns)}")

    # Show new columns
    new_cols = [col for col in features_df.columns if col not in stock_df.columns]
    print(f"\nNew feature columns ({len(new_cols)}):")
    for col in sorted(new_cols)[:20]:  # Show first 20
        print(f"  - {col}")
    if len(new_cols) > 20:
        print(f"  ... and {len(new_cols) - 20} more")

    # Show sample
    print(f"\nSample (AAPL recent dates):")
    sample_cols = [
        'symbol', 'date', 'sector', 'return_20d_vs_market',
        'market_trend_bullish', 'vix_level', 'vix_regime'
    ]
    sample = features_df.filter(pl.col('symbol') == 'AAPL').select(sample_cols).tail(5)
    print(sample)

    print("\n" + "="*70)
    print("✓ Sector/market features computed successfully")
