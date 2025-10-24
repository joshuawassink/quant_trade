"""
Data Filtering Step

Filters dataset to ensure training readiness.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
import polars as pl
from loguru import logger


class DataFilter:
    """Filter dataset for ML pipeline."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize data filter.

        Args:
            output_dir: Directory to save filtered data. Defaults to project_root/data/training
        """
        if output_dir is None:
            self.output_dir = Path(__file__).parent.parent.parent / "data" / "training"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def filter_for_training(
        self,
        df: pl.DataFrame,
        target_col: str,
        critical_features: Optional[list[str]] = None,
    ) -> pl.DataFrame:
        """
        Filter dataset for training readiness.

        Strategy:
        1. Drop rows where target is null (can't train without label)
        2. Drop rows where critical features are null
        3. Keep rows with some null fundamentals (handled by preprocessing)

        Args:
            df: Full feature dataset
            target_col: Name of target column
            critical_features: Features that must not be null. If None, uses defaults.

        Returns:
            Filtered dataset
        """
        logger.info("=" * 70)
        logger.info("FILTERING DATASET FOR TRAINING")
        logger.info("=" * 70)

        initial_rows = len(df)

        # STEP 1: Drop rows with null target
        df = df.filter(pl.col(target_col).is_not_null())
        logger.info(f"After target filter: {len(df):,} rows ({len(df)/initial_rows*100:.1f}%)")

        # STEP 2: Drop rows with null critical features
        if critical_features is None:
            critical_features = [
                'adj_close', 'volume', 'return_5d', 'return_20d',
                'volatility_5d', 'rsi_14', 'sma_20', 'sma_50',
                'spy_return_5d', 'vix_level'
            ]

        # Only keep features that exist
        critical_features = [f for f in critical_features if f in df.columns]

        for feature in critical_features:
            before = len(df)
            df = df.filter(pl.col(feature).is_not_null())
            dropped = before - len(df)
            if dropped > 0:
                logger.info(f"  Dropped {dropped:,} rows with null {feature}")

        logger.info(f"After critical feature filter: {len(df):,} rows ({len(df)/initial_rows*100:.1f}%)")

        # STEP 3: Accept nulls in fundamental features (model will handle via imputation)
        logger.info(f"Final dataset: {len(df):,} rows ({len(df)/initial_rows*100:.1f}% retained)")

        # Count remaining nulls
        null_counts = {}
        for col in df.columns:
            null_count = df[col].is_null().sum()
            if null_count > 0:
                null_counts[col] = null_count

        if null_counts:
            logger.info(f"Remaining nulls in {len(null_counts)} columns (will be imputed)")
            sorted_nulls = sorted(null_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for col, count in sorted_nulls:
                logger.info(f"  {col}: {count:,} nulls ({count/len(df)*100:.1f}%)")

        return df

    def save_training_data(
        self,
        df: pl.DataFrame,
        horizon_days: int,
        version: Optional[str] = None,
    ) -> Path:
        """
        Save filtered training dataset.

        Args:
            df: Filtered dataset
            horizon_days: Prediction horizon (for filename)
            version: Optional version string. If None, uses timestamp.

        Returns:
            Path to saved file
        """
        logger.info("=" * 70)
        logger.info("SAVING TRAINING DATASET")
        logger.info("=" * 70)

        if version is None:
            version = datetime.now().strftime("%Y%m%d")

        output_file = self.output_dir / f"training_data_{horizon_days}d_{version}.parquet"
        latest_file = self.output_dir / f"training_data_{horizon_days}d_latest.parquet"

        df.write_parquet(output_file)
        df.write_parquet(latest_file)

        logger.info(f"✓ Saved to {output_file}")
        logger.info(f"✓ Also saved as {latest_file}")

        # Print summary
        target_col = f'target_return_{horizon_days}d_vs_market'
        feature_cols = [c for c in df.columns if c not in ['symbol', 'date', target_col]]

        print("\n" + "=" * 70)
        print("TRAINING DATASET SUMMARY")
        print("=" * 70)
        print(f"Total rows:        {len(df):,}")
        print(f"Symbols:           {df['symbol'].n_unique()}")
        print(f"Date range:        {df['date'].min()} to {df['date'].max()}")
        print(f"Features:          {len(feature_cols)}")
        print(f"Target:            {target_col}")
        print(f"File size:         {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        print("=" * 70 + "\n")

        return output_file
