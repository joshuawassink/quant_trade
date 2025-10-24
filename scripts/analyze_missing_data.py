"""
Analyze missing data patterns in training dataset.

Identifies which features cause the most data loss so we can
develop a smart missing data strategy.
"""

import sys
from pathlib import Path
import polars as pl

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.shared.features.alignment import FeatureAligner
from src.shared.config.universe import get_universe
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")


def load_all_data():
    """Load all data sources."""
    logger.info("Loading data...")

    # Load price data
    universe = get_universe('production')
    price_dir = project_root / "data" / "price" / "daily"
    price_dfs = []
    for symbol in universe[:50]:  # Sample first 50 for speed
        file_path = price_dir / f"{symbol}.parquet"
        if file_path.exists():
            price_dfs.append(pl.read_parquet(file_path))
    price_df = pl.concat(price_dfs)

    # Load other data
    market_df = pl.read_parquet(project_root / "data" / "market" / "daily" / "market_data_latest.parquet")
    financials_df = pl.read_parquet(project_root / "data" / "financials" / "quarterly_financials_latest.parquet")
    metadata_df = pl.read_parquet(project_root / "data" / "metadata" / "company_metadata_latest.parquet")

    logger.info(f"  Price: {len(price_df):,} rows, {price_df['symbol'].n_unique()} symbols")
    logger.info(f"  Market: {len(market_df):,} rows")
    logger.info(f"  Financials: {len(financials_df):,} rows")
    logger.info(f"  Metadata: {len(metadata_df)} rows")

    return price_df, market_df, financials_df, metadata_df


def analyze_null_patterns(df: pl.DataFrame):
    """Analyze null patterns in feature dataset."""
    logger.info("\n" + "="*70)
    logger.info("NULL ANALYSIS BY FEATURE")
    logger.info("="*70)

    total_rows = len(df)

    # Calculate null percentage per column
    null_stats = []
    for col in df.columns:
        null_count = df[col].is_null().sum()
        null_pct = (null_count / total_rows) * 100
        null_stats.append({
            'feature': col,
            'null_count': null_count,
            'null_pct': null_pct,
            'valid_count': total_rows - null_count
        })

    # Convert to DataFrame and sort by null percentage
    null_df = pl.DataFrame(null_stats).sort('null_pct', descending=True)

    # Print summary
    print("\n🔴 HIGH NULL FEATURES (>50% missing):")
    high_null = null_df.filter(pl.col('null_pct') > 50)
    if len(high_null) > 0:
        for row in high_null.iter_rows(named=True):
            print(f"  {row['feature']:40s}: {row['null_pct']:5.1f}% null ({row['valid_count']:,} valid rows)")
    else:
        print("  None!")

    print("\n🟡 MODERATE NULL FEATURES (20-50% missing):")
    mod_null = null_df.filter((pl.col('null_pct') >= 20) & (pl.col('null_pct') <= 50))
    if len(mod_null) > 0:
        for row in mod_null.iter_rows(named=True):
            print(f"  {row['feature']:40s}: {row['null_pct']:5.1f}% null ({row['valid_count']:,} valid rows)")
    else:
        print("  None!")

    print("\n🟢 LOW NULL FEATURES (<20% missing):")
    low_null_count = len(null_df.filter(pl.col('null_pct') < 20))
    print(f"  {low_null_count} features with <20% nulls")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total features: {len(null_df)}")
    print(f"Features with >80% nulls: {len(null_df.filter(pl.col('null_pct') > 80))}")
    print(f"Features with >50% nulls: {len(null_df.filter(pl.col('null_pct') > 50))}")
    print(f"Features with <20% nulls: {low_null_count}")
    print(f"\nTotal rows in dataset: {total_rows:,}")

    # Calculate how many rows would remain after dropping nulls
    print("\nROW RETENTION AFTER DROP_NULLS:")
    rows_no_nulls = df.drop_nulls().height
    print(f"  Rows with NO nulls: {rows_no_nulls:,} ({rows_no_nulls/total_rows*100:.1f}%)")
    print(f"  Rows lost: {total_rows - rows_no_nulls:,} ({(total_rows-rows_no_nulls)/total_rows*100:.1f}%)")

    return null_df


def analyze_by_date(df: pl.DataFrame):
    """Analyze null patterns by date."""
    logger.info("\n" + "="*70)
    logger.info("NULL ANALYSIS BY DATE")
    logger.info("="*70)

    # Count nulls per date
    by_date = df.group_by('date').agg([
        pl.len().alias('total_rows'),
        pl.all().is_null().sum().sum().alias('total_nulls'),
    ]).sort('date')

    # Calculate first date with <10% nulls
    by_date = by_date.with_columns([
        (pl.col('total_nulls') / (pl.col('total_rows') * len(df.columns)) * 100).alias('null_pct')
    ])

    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nFirst 10 dates (showing null %):")
    print(by_date.head(10).select(['date', 'total_rows', 'null_pct']))

    print(f"\nLast 10 dates (showing null %):")
    print(by_date.tail(10).select(['date', 'total_rows', 'null_pct']))

    # Find first date with <10% nulls
    good_dates = by_date.filter(pl.col('null_pct') < 10)
    if len(good_dates) > 0:
        first_good_date = good_dates['date'].min()
        print(f"\n✓ First date with <10% nulls: {first_good_date}")
        print(f"  This explains why training data starts around this date!")


def main():
    """Main execution."""
    print("="*70)
    print("MISSING DATA ANALYSIS")
    print("="*70)
    print()

    # Load data
    price_df, market_df, financials_df, metadata_df = load_all_data()

    # Compute features
    logger.info("\nComputing features...")
    aligner = FeatureAligner()
    features_df = aligner.compute_all_features(
        price_df=price_df,
        financials_df=financials_df,
        market_df=market_df,
        metadata_df=metadata_df
    )

    # Analyze null patterns
    null_df = analyze_null_patterns(features_df)

    # Analyze by date
    analyze_by_date(features_df)

    # Save null analysis
    output_file = project_root / "data" / "analysis" / "null_analysis.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    null_df.write_parquet(output_file)
    logger.info(f"\n✓ Null analysis saved to {output_file}")

    return null_df


if __name__ == "__main__":
    null_df = main()
