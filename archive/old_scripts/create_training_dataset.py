"""
Create training dataset from raw data.

Loads all data sources, computes features, aligns to daily frequency,
and creates training-ready dataset with target variables.
"""

import sys
from pathlib import Path
from datetime import datetime
import polars as pl

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.shared.features.alignment import FeatureAligner
from src.shared.config.universe import get_universe
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")


def load_price_data(symbols: list[str]) -> pl.DataFrame:
    """Load price data for all symbols."""
    logger.info(f"Loading price data for {len(symbols)} symbols...")

    price_dir = project_root / "data" / "price" / "daily"

    dfs = []
    for symbol in symbols:
        file_path = price_dir / f"{symbol}.parquet"
        if file_path.exists():
            df = pl.read_parquet(file_path)
            dfs.append(df)
        else:
            logger.warning(f"  Missing price data for {symbol}")

    combined = pl.concat(dfs)
    logger.info(f"  ✓ Loaded {len(combined):,} rows for {combined['symbol'].n_unique()} symbols")
    return combined


def load_market_data() -> pl.DataFrame:
    """Load market data (SPY, VIX, sector ETFs)."""
    logger.info("Loading market data...")

    market_file = project_root / "data" / "market" / "daily" / "market_data_latest.parquet"

    if not market_file.exists():
        logger.error(f"Market data not found: {market_file}")
        raise FileNotFoundError(f"Market data missing: {market_file}")

    df = pl.read_parquet(market_file)
    logger.info(f"  ✓ Loaded {len(df):,} rows for {df['symbol'].n_unique()} market symbols")
    return df


def load_financials() -> pl.DataFrame:
    """Load quarterly financial data."""
    logger.info("Loading financial data...")

    fin_file = project_root / "data" / "financials" / "quarterly_financials_latest.parquet"

    if not fin_file.exists():
        logger.error(f"Financials not found: {fin_file}")
        raise FileNotFoundError(f"Financials missing: {fin_file}")

    df = pl.read_parquet(fin_file)
    logger.info(f"  ✓ Loaded {len(df):,} quarterly records for {df['symbol'].n_unique()} symbols")
    return df


def load_metadata() -> pl.DataFrame:
    """Load company metadata."""
    logger.info("Loading metadata...")

    meta_file = project_root / "data" / "metadata" / "company_metadata_latest.parquet"

    if not meta_file.exists():
        logger.error(f"Metadata not found: {meta_file}")
        raise FileNotFoundError(f"Metadata missing: {meta_file}")

    df = pl.read_parquet(meta_file)
    logger.info(f"  ✓ Loaded metadata for {len(df)} companies")
    return df


def compute_target_variable(df: pl.DataFrame, horizon_days: int = 30) -> pl.DataFrame:
    """
    Compute target variable: future returns vs market.

    Args:
        df: DataFrame with price data
        horizon_days: Prediction horizon in days

    Returns:
        DataFrame with target_return_Nd_vs_market column
    """
    logger.info(f"Computing target variable (forward {horizon_days}d returns vs SPY)...")

    # Need SPY for market returns - should already be in df from sector features
    # For now, compute simple forward returns
    # TODO: Implement proper market-relative returns

    df = df.sort(['symbol', 'date'])

    # Compute forward return
    df = df.with_columns([
        (
            pl.col('adj_close').shift(-horizon_days).over('symbol') / pl.col('adj_close') - 1.0
        ).alias(f'target_return_{horizon_days}d_vs_market')
    ])

    logger.info(f"  ✓ Target variable computed")
    return df


def filter_and_save(df: pl.DataFrame, horizon_days: int = 30) -> pl.DataFrame:
    """
    Filter dataset and save to parquet.

    Strategy:
    1. Drop rows where target is null (can't train without label)
    2. Drop rows where critical price/volume features are null
    3. Keep rows with some null fundamentals (will be handled by model)

    Args:
        df: Full feature dataset
        horizon_days: Prediction horizon (for filename)

    Returns:
        Filtered dataset
    """
    logger.info("Filtering and saving dataset...")

    # Count nulls before dropping
    initial_rows = len(df)

    # STEP 1: Drop rows with null target (required)
    df = df.filter(pl.col(f'target_return_{horizon_days}d_vs_market').is_not_null())
    logger.info(f"  After target filter: {len(df):,} rows ({len(df)/initial_rows*100:.1f}%)")

    # STEP 2: Drop rows where critical features are null
    # These are features that should ALWAYS be available from price data
    critical_features = [
        'adj_close', 'volume', 'return_5d', 'return_20d',
        'volatility_5d', 'rsi_14', 'sma_20', 'sma_50',
        'spy_return_5d', 'vix_level'
    ]

    # Only keep critical features that actually exist
    critical_features = [f for f in critical_features if f in df.columns]

    for feature in critical_features:
        before = len(df)
        df = df.filter(pl.col(feature).is_not_null())
        dropped = before - len(df)
        if dropped > 0:
            logger.info(f"    Dropped {dropped:,} rows with null {feature}")

    logger.info(f"  After critical feature filter: {len(df):,} rows ({len(df)/initial_rows*100:.1f}%)")

    # STEP 3: For fundamental features, we accept nulls
    # The model will handle them via imputation or by ignoring them
    # This is a MUCH more lenient approach than drop_nulls()

    logger.info(f"  Final dataset: {len(df):,} rows ({len(df)/initial_rows*100:.1f}% retained)")

    # Count nulls in final dataset
    null_counts = {}
    for col in df.columns:
        null_count = df[col].is_null().sum()
        if null_count > 0:
            null_counts[col] = null_count

    if null_counts:
        logger.info(f"  Remaining nulls in {len(null_counts)} columns (will be imputed during training)")
        # Show top 5 columns with most nulls
        sorted_nulls = sorted(null_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for col, count in sorted_nulls:
            logger.info(f"    {col}: {count:,} nulls ({count/len(df)*100:.1f}%)")

    # Save dataset
    output_dir = project_root / "data" / "training"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = output_dir / f"training_data_{horizon_days}d_{timestamp}.parquet"
    latest_file = output_dir / f"training_data_{horizon_days}d_latest.parquet"

    df.write_parquet(output_file)
    df.write_parquet(latest_file)

    logger.info(f"  ✓ Saved to {output_file}")
    logger.info(f"  ✓ Also saved as {latest_file}")

    # Print summary stats
    print("\n" + "="*70)
    print("TRAINING DATASET SUMMARY")
    print("="*70)
    print(f"Total rows:        {len(df):,}")
    print(f"Symbols:           {df['symbol'].n_unique()}")
    print(f"Date range:        {df['date'].min()} to {df['date'].max()}")
    print(f"Features:          {len([c for c in df.columns if c not in ['symbol', 'date', f'target_return_{horizon_days}d_vs_market']])}")
    print(f"Target variable:   target_return_{horizon_days}d_vs_market")
    print(f"Nulls remaining:   {len(null_counts)} columns have nulls")
    print(f"File size:         {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print("="*70)

    return df


def main():
    """Main execution."""
    logger.info("="*70)
    logger.info("CREATING TRAINING DATASET")
    logger.info("="*70)

    # Get production universe
    universe = get_universe('production')
    logger.info(f"Universe: {len(universe)} stocks\n")

    # Load all data sources
    price_df = load_price_data(universe)
    market_df = load_market_data()
    financials_df = load_financials()
    metadata_df = load_metadata()

    # Compute all features
    logger.info("\nComputing features...")
    aligner = FeatureAligner()
    features_df = aligner.compute_all_features(
        price_df=price_df,
        financials_df=financials_df,
        market_df=market_df,
        metadata_df=metadata_df
    )

    # Compute target variable
    features_df = compute_target_variable(features_df, horizon_days=30)

    # Filter and save
    training_df = filter_and_save(features_df, horizon_days=30)

    logger.info("\n" + "="*70)
    logger.info("✓ TRAINING DATASET CREATED SUCCESSFULLY")
    logger.info("="*70)

    return training_df


if __name__ == "__main__":
    df = main()
