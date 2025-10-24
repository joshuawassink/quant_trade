"""
Data Splitting Step

Split data into train and test sets with time-series awareness.
"""

from pathlib import Path
from typing import Optional, Tuple
import polars as pl
from loguru import logger


class DataSplitter:
    """Split data into train/test sets for ML pipeline."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize data splitter.

        Args:
            output_dir: Directory to save split data. Defaults to project_root/data/training
        """
        if output_dir is None:
            self.output_dir = Path(__file__).parent.parent.parent / "data" / "training"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def time_series_split(
        self,
        df: pl.DataFrame,
        test_size: float = 0.2,
        date_col: str = 'date',
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Split data into train/test using time-series split.

        IMPORTANT: For time-series, we don't shuffle. We split by date to prevent
        lookahead bias. Last `test_size` portion becomes test set.

        Args:
            df: Full dataset
            test_size: Fraction for test set (0.2 = last 20%)
            date_col: Name of date column

        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info("=" * 70)
        logger.info("TIME-SERIES TRAIN/TEST SPLIT")
        logger.info("=" * 70)

        # Sort by date
        df = df.sort(date_col)

        # Calculate split point
        n_samples = len(df)
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test

        # Split
        train_df = df.head(n_train)
        test_df = df.tail(n_test)

        # Get date ranges
        train_start = train_df[date_col].min()
        train_end = train_df[date_col].max()
        test_start = test_df[date_col].min()
        test_end = test_df[date_col].max()

        logger.info(f"Total samples:     {n_samples:,}")
        logger.info(f"Train samples:     {n_train:,} ({(1-test_size)*100:.1f}%)")
        logger.info(f"Test samples:      {n_test:,} ({test_size*100:.1f}%)")
        logger.info(f"Train date range:  {train_start} to {train_end}")
        logger.info(f"Test date range:   {test_start} to {test_end}")

        # Check for overlap (should be none)
        if train_end >= test_start:
            logger.warning("⚠ Train and test periods overlap! This may cause data leakage.")

        logger.info("✓ Split complete")

        return train_df, test_df

    def save_splits(
        self,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        horizon_days: int,
        version: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """
        Save train and test splits to parquet.

        Args:
            train_df: Training data
            test_df: Test data
            horizon_days: Prediction horizon (for filename)
            version: Optional version string. If None, uses 'latest'

        Returns:
            Tuple of (train_path, test_path)
        """
        logger.info("Saving train/test splits...")

        if version is None:
            version = 'latest'

        train_path = self.output_dir / f"train_{horizon_days}d_{version}.parquet"
        test_path = self.output_dir / f"test_{horizon_days}d_{version}.parquet"

        train_df.write_parquet(train_path)
        test_df.write_parquet(test_path)

        logger.info(f"✓ Train saved to {train_path}")
        logger.info(f"✓ Test saved to {test_path}")

        # Print summary
        print("\n" + "=" * 70)
        print("TRAIN/TEST SPLIT SUMMARY")
        print("=" * 70)
        print(f"Train: {len(train_df):,} samples")
        print(f"Test:  {len(test_df):,} samples")
        print(f"Train dates: {train_df['date'].min()} to {train_df['date'].max()}")
        print(f"Test dates:  {test_df['date'].min()} to {test_df['date'].max()}")
        print("=" * 70 + "\n")

        return train_path, test_path

    def split_and_save(
        self,
        df: pl.DataFrame,
        horizon_days: int,
        test_size: float = 0.2,
        version: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """
        Convenience method: split and save in one step.

        Args:
            df: Full dataset
            horizon_days: Prediction horizon
            test_size: Test set fraction
            version: Optional version string

        Returns:
            Tuple of (train_path, test_path)
        """
        train_df, test_df = self.time_series_split(df, test_size=test_size)
        train_path, test_path = self.save_splits(train_df, test_df, horizon_days, version)
        return train_path, test_path
