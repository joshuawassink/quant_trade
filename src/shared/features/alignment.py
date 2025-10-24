"""
Feature alignment utilities.

Merges technical, fundamental, and sector features into a single dataset
ready for model training.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import polars as pl
from loguru import logger

from src.features.technical import TechnicalFeatures
from src.features.fundamental import FundamentalFeatures
from src.features.sector import SectorFeatures


class FeatureAligner:
    """Align and merge all feature sets into training-ready dataset."""

    def __init__(self):
        """Initialize feature alignment."""
        self.tech_features = TechnicalFeatures()
        self.fund_features = FundamentalFeatures()
        self.sector_features = SectorFeatures()

    def compute_all_features(
        self,
        price_df: pl.DataFrame,
        financials_df: pl.DataFrame,
        market_df: pl.DataFrame,
        metadata_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Compute all features and merge into single dataset.

        Args:
            price_df: Daily price data (symbol, date, OHLCV)
            financials_df: Quarterly financials (symbol, date, metrics)
            market_df: Market data (symbol, date, close) - SPY, VIX, sector ETFs
            metadata_df: Company metadata (symbol, sector)

        Returns:
            DataFrame with all features aligned to daily frequency

        Process:
            1. Compute technical features (daily frequency)
            2. Compute fundamental features (quarterly → forward-fill to daily)
            3. Compute sector/market features (daily frequency)
            4. Merge all features on (symbol, date)
        """
        logger.info("Computing all features...")

        # 1. Technical features (already daily)
        logger.info("  Computing technical features...")
        tech_df = self.tech_features.compute_all(price_df)
        logger.info(f"    ✓ {len([c for c in tech_df.columns if c not in price_df.columns])} technical features")

        # 2. Fundamental features (quarterly → daily via forward-fill)
        logger.info("  Computing fundamental features...")
        fund_df = self.fund_features.compute_all(financials_df)
        logger.info(f"    ✓ {len([c for c in fund_df.columns if c not in financials_df.columns])} fundamental features")

        # 3. Sector/market features (daily)
        logger.info("  Computing sector/market features...")
        sector_df = self.sector_features.compute_all(tech_df, market_df, metadata_df)
        logger.info(f"    ✓ {len([c for c in sector_df.columns if c not in tech_df.columns])} sector/market features")

        # 4. Merge fundamental features to daily frequency
        logger.info("  Aligning fundamental features to daily frequency...")
        aligned_df = self._align_fundamentals_to_daily(sector_df, fund_df)

        logger.info(f"✓ Feature alignment complete: {len(aligned_df)} rows, {len(aligned_df.columns)} columns")
        return aligned_df

    def _align_fundamentals_to_daily(
        self,
        daily_df: pl.DataFrame,
        quarterly_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Align quarterly fundamental features to daily frequency.

        Strategy:
            - Forward-fill quarterly metrics to daily
            - Fundamental metrics stay constant until next earnings report

        Args:
            daily_df: Daily data (from technical + sector features)
            quarterly_df: Quarterly fundamental features (with quarter_end_date)

        Returns:
            Daily DataFrame with fundamental features added
        """
        # Rename quarter_end_date to date for joining
        quarterly_df = quarterly_df.rename({'quarter_end_date': 'date'})

        # Cast date to pl.Date to match daily data
        quarterly_df = quarterly_df.with_columns([
            pl.col('date').cast(pl.Date)
        ])

        # Get fundamental feature columns (exclude base columns)
        base_cols = ['symbol', 'date', 'fetch_date', 'fiscal_quarter', 'fiscal_year']
        fund_feature_cols = [col for col in quarterly_df.columns if col not in base_cols]

        logger.debug(f"    Forward-filling {len(fund_feature_cols)} fundamental features")

        # Join quarterly fundamentals to daily data
        # Use asof join to get the most recent quarterly value for each daily date
        aligned_df = daily_df.join_asof(
            quarterly_df.select(['symbol', 'date'] + fund_feature_cols),
            on='date',
            by='symbol',
            strategy='backward'  # Use most recent past quarter
        )

        return aligned_df

    def compute_target_variable(
        self,
        df: pl.DataFrame,
        market_df: pl.DataFrame,
        horizon: int = 30,
        relative_to_market: bool = True,
    ) -> pl.DataFrame:
        """
        Compute target variable for prediction.

        Target: Forward-looking N-day return (default 30 days)
        Can be absolute or market-relative (vs SPY)

        Args:
            df: DataFrame with features (must include 'close')
            market_df: Market data including SPY
            horizon: Forward-looking period in days (default 30)
            relative_to_market: If True, compute return relative to SPY

        Returns:
            DataFrame with target variable added

        Target Columns:
            - target_return_Nd: N-day forward return
            - target_return_Nd_vs_market: N-day forward return vs SPY (if relative_to_market=True)
        """
        logger.info(f"Computing target variable (horizon={horizon}d, relative={relative_to_market})...")

        # Compute forward returns (shift backwards because we're looking forward)
        df = df.with_columns([
            (
                (pl.col('close').shift(-horizon) / pl.col('close') - 1)
                .over('symbol')
                .alias(f'target_return_{horizon}d')
            )
        ])

        # Compute market-relative target if requested
        if relative_to_market:
            # Get SPY data and compute forward returns
            spy_df = market_df.filter(pl.col('symbol') == 'SPY').select(['date', 'close'])
            spy_df = spy_df.rename({'close': 'spy_close'})

            # Compute SPY forward return
            spy_df = spy_df.with_columns([
                (
                    (pl.col('spy_close').shift(-horizon) / pl.col('spy_close') - 1)
                ).alias(f'spy_return_{horizon}d_forward')
            ])

            # Join SPY returns to stock data
            df = df.join(spy_df.select(['date', f'spy_return_{horizon}d_forward']), on='date', how='left')

            # Compute relative return (stock vs market)
            df = df.with_columns([
                (
                    pl.col(f'target_return_{horizon}d') - pl.col(f'spy_return_{horizon}d_forward')
                ).alias(f'target_return_{horizon}d_vs_market')
            ])

            # Drop temporary column
            df = df.drop(f'spy_return_{horizon}d_forward')

        # Count nulls (last N days won't have target)
        null_count = df[f'target_return_{horizon}d'].null_count()
        logger.info(f"  ✓ Target computed: {null_count} nulls (expected ~{horizon * df['symbol'].n_unique()})")

        return df

    def create_training_dataset(
        self,
        price_df: pl.DataFrame,
        financials_df: pl.DataFrame,
        market_df: pl.DataFrame,
        metadata_df: pl.DataFrame,
        target_horizon: int = 30,
        relative_target: bool = True,
    ) -> pl.DataFrame:
        """
        Create complete training dataset with all features and target.

        Args:
            price_df: Daily price data
            financials_df: Quarterly financials
            market_df: Market data (SPY, VIX, sector ETFs)
            metadata_df: Company metadata
            target_horizon: Forward-looking horizon for target (days)
            relative_target: If True, use market-relative returns

        Returns:
            Training-ready DataFrame with:
                - All features (technical, fundamental, sector/market)
                - Target variable
                - No nulls in features (rows with nulls dropped)
        """
        logger.info("="*70)
        logger.info("CREATING TRAINING DATASET")
        logger.info("="*70)

        # 1. Compute all features
        df = self.compute_all_features(price_df, financials_df, market_df, metadata_df)

        # 2. Compute target variable
        df = self.compute_target_variable(df, market_df, horizon=target_horizon, relative_to_market=relative_target)

        # 3. Identify feature columns (exclude metadata and target)
        exclude_cols = [
            'symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume',
            'dividends', 'stock_splits', 'sector', 'period_ending', 'fiscal_quarter', 'fiscal_year'
        ]
        target_col = f'target_return_{target_horizon}d_vs_market' if relative_target else f'target_return_{target_horizon}d'

        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols
            and not col.startswith('target_')
            and col != target_col
        ]

        logger.info(f"\nFeature columns identified: {len(feature_cols)}")

        # 4. Drop rows with nulls in features or target
        rows_before = len(df)
        df = df.drop_nulls(subset=feature_cols + [target_col])
        rows_after = len(df)
        rows_dropped = rows_before - rows_after

        logger.info(f"\nNull handling:")
        logger.info(f"  Rows before: {rows_before:,}")
        logger.info(f"  Rows after:  {rows_after:,}")
        logger.info(f"  Dropped:     {rows_dropped:,} ({rows_dropped/rows_before*100:.1f}%)")

        # 5. Summary statistics
        logger.info(f"\nDataset summary:")
        logger.info(f"  Symbols: {df['symbol'].n_unique()}")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  Features: {len(feature_cols)}")
        logger.info(f"  Target: {target_col}")
        logger.info(f"  Training rows: {len(df):,}")

        # Show target distribution
        target_stats = df.select([
            pl.col(target_col).mean().alias('mean'),
            pl.col(target_col).std().alias('std'),
            pl.col(target_col).min().alias('min'),
            pl.col(target_col).quantile(0.25).alias('p25'),
            pl.col(target_col).median().alias('median'),
            pl.col(target_col).quantile(0.75).alias('p75'),
            pl.col(target_col).max().alias('max'),
        ])

        logger.info(f"\nTarget variable distribution ({target_col}):")
        logger.info(f"  Mean:   {target_stats['mean'][0]:.4f}")
        logger.info(f"  Std:    {target_stats['std'][0]:.4f}")
        logger.info(f"  Min:    {target_stats['min'][0]:.4f}")
        logger.info(f"  25th:   {target_stats['p25'][0]:.4f}")
        logger.info(f"  Median: {target_stats['median'][0]:.4f}")
        logger.info(f"  75th:   {target_stats['p75'][0]:.4f}")
        logger.info(f"  Max:    {target_stats['max'][0]:.4f}")

        logger.info("\n" + "="*70)
        logger.info("✓ TRAINING DATASET READY")
        logger.info("="*70)

        return df


# Standalone test
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    # Configure logging
    from loguru import logger
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    # Load data
    logger.info("Loading data...")
    price_df = pl.read_parquet('data/price/daily/sample_universe_2022-10-24_to_2025-10-23.parquet')
    financials_df = pl.read_parquet('data/financials/quarterly_financials_latest.parquet')
    market_df = pl.read_parquet('data/market/daily/market_data_latest.parquet')
    metadata_df = pl.read_parquet('data/metadata/company_metadata_latest.parquet')

    logger.info(f"  Price data: {len(price_df):,} rows")
    logger.info(f"  Financials: {len(financials_df):,} quarters")
    logger.info(f"  Market data: {len(market_df):,} rows")
    logger.info(f"  Metadata: {len(metadata_df):,} companies")

    # Create training dataset
    aligner = FeatureAligner()
    train_df = aligner.create_training_dataset(
        price_df=price_df,
        financials_df=financials_df,
        market_df=market_df,
        metadata_df=metadata_df,
        target_horizon=30,
        relative_target=True,
    )

    # Show sample
    logger.info("\nSample of training data (AAPL recent):")
    sample_cols = ['symbol', 'date', 'return_20d', 'rsi_14', 'roe', 'vix_regime', 'target_return_30d_vs_market']
    sample = train_df.filter(pl.col('symbol') == 'AAPL').select(sample_cols).tail(10)
    print(sample)

    # Save training dataset
    output_path = 'data/training/training_data_30d_latest.parquet'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    train_df.write_parquet(output_path)
    logger.info(f"\n✓ Training dataset saved to {output_path}")
    logger.info(f"  File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
