"""
Model Prediction Step

Generate predictions on new data (test set or production).
This is separate from evaluation to follow best practices:
- Train on train set
- Predict on test set
- Evaluate predictions

This allows predictions to be saved and reused for different analyses.
"""

from pathlib import Path
from typing import Optional
import polars as pl
import numpy as np
import joblib
from loguru import logger

from src.shared.models.preprocessing import FeaturePreprocessor


class ModelPredictor:
    """Generate predictions from trained model."""

    def __init__(self, model_dir: Path):
        """
        Initialize predictor.

        Args:
            model_dir: Directory containing trained model artifacts
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.preprocessor = None
        self.feature_names = None

    def load_model(self, model_filename: Optional[str] = None):
        """
        Load trained model and preprocessor.

        Args:
            model_filename: Name of model file. If None, auto-detects.
        """
        logger.info("Loading model artifacts...")

        # Auto-detect model file if not specified
        if model_filename is None:
            # Look for common model filenames
            for pattern in ['*_model.joblib', 'model.joblib']:
                model_files = list(self.model_dir.glob(pattern))
                if model_files:
                    model_filename = model_files[0].name
                    break

            if model_filename is None:
                raise FileNotFoundError(f"No model file found in {self.model_dir}")

        # Load model
        model_file = self.model_dir / model_filename
        self.model = joblib.load(model_file)
        logger.info(f"  ✓ Loaded model from {model_file}")

        # Load preprocessor
        preprocessor_file = self.model_dir / "preprocessor.joblib"
        self.preprocessor = FeaturePreprocessor.load(preprocessor_file)
        logger.info(f"  ✓ Loaded preprocessor from {preprocessor_file}")

        # Load feature names
        features_file = self.model_dir / "feature_names.txt"
        with open(features_file, 'r') as f:
            self.feature_names = [line.strip() for line in f if line.strip()]
        logger.info(f"  ✓ Loaded {len(self.feature_names)} feature names")

    def load_data(self, data_path: str):
        """
        Load data for prediction.

        Args:
            data_path: Path to data parquet file
        """
        logger.info(f"Loading data from {data_path}...")
        self.data_df = pl.read_parquet(data_path)
        logger.info(f"  ✓ Loaded {len(self.data_df):,} rows")

    def prepare_features(self):
        """
        Prepare features for prediction.

        Returns:
            X array, metadata DataFrame
        """
        logger.info("Preparing features...")

        # Metadata columns to exclude
        metadata_cols = [
            'symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume',
            'dividends', 'stock_splits', 'sector'
        ]
        categorical_cols = ['vix_regime']

        feature_cols = [
            col for col in self.data_df.columns
            if col not in metadata_cols
            and col not in categorical_cols
            and not col.startswith('target_')
            and not col.startswith('forward_return_')
            and not col.startswith('XL')
            and col != 'spy_close'
        ]

        # Keep only features in training
        feature_cols = [col for col in feature_cols if col in self.feature_names]

        X = self.data_df.select(feature_cols).to_numpy()

        # Metadata for tracking predictions
        metadata_cols_present = [col for col in ['symbol', 'date', 'sector'] if col in self.data_df.columns]
        metadata_df = self.data_df.select(metadata_cols_present)

        logger.info(f"  ✓ X shape: {X.shape}")

        return X, metadata_df

    def predict(
        self,
        data_path: str,
        output_path: Optional[Path] = None,
    ) -> pl.DataFrame:
        """
        Generate predictions on data.

        Args:
            data_path: Path to data for prediction
            output_path: Optional path to save predictions. If None, uses temp location.

        Returns:
            DataFrame with predictions
        """
        logger.info("=" * 70)
        logger.info("GENERATING PREDICTIONS")
        logger.info("=" * 70)

        # Load data
        self.load_data(data_path)

        # Prepare features
        X, metadata_df = self.prepare_features()

        # Preprocess
        logger.info("Preprocessing features...")
        X_processed = self.preprocessor.transform(X)
        logger.info(f"  ✓ Preprocessed shape: {X_processed.shape}")

        # Predict
        logger.info("Making predictions...")
        y_pred = self.model.predict(X_processed)
        logger.info(f"  ✓ Generated {len(y_pred):,} predictions")

        # Create predictions DataFrame
        predictions_df = metadata_df.with_columns([
            pl.lit(y_pred).alias('predicted_return')
        ])

        # Add actual returns if available (for test set)
        target_cols = [col for col in self.data_df.columns if col.startswith('target_return_')]
        if target_cols:
            target_col = target_cols[0]
            predictions_df = predictions_df.with_columns([
                self.data_df[target_col].alias('actual_return')
            ])
            logger.info(f"  ✓ Included actual returns from {target_col}")

        # Save predictions
        if output_path is None:
            output_path = self.model_dir / "predictions_latest.parquet"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.write_parquet(output_path)
        logger.info(f"✓ Predictions saved to {output_path}")

        # Print summary
        print("\n" + "=" * 70)
        print("PREDICTION SUMMARY")
        print("=" * 70)
        print(f"Samples:           {len(predictions_df):,}")
        print(f"Date range:        {predictions_df['date'].min()} to {predictions_df['date'].max()}")
        print(f"Unique symbols:    {predictions_df['symbol'].n_unique()}")
        print(f"Pred mean:         {y_pred.mean():+.4f}")
        print(f"Pred std:          {y_pred.std():.4f}")
        if 'actual_return' in predictions_df.columns:
            actual = predictions_df['actual_return'].to_numpy()
            print(f"Actual mean:       {actual.mean():+.4f}")
            print(f"Actual std:        {actual.std():.4f}")
        print("=" * 70 + "\n")

        return predictions_df

    def predict_batch(
        self,
        data_paths: list[str],
        output_dir: Path,
    ) -> list[Path]:
        """
        Generate predictions on multiple datasets.

        Args:
            data_paths: List of paths to data files
            output_dir: Directory to save all predictions

        Returns:
            List of paths to saved prediction files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = []

        for i, data_path in enumerate(data_paths, 1):
            logger.info(f"\nProcessing dataset {i}/{len(data_paths)}: {data_path}")

            # Create output path
            data_name = Path(data_path).stem
            output_path = output_dir / f"predictions_{data_name}.parquet"

            # Predict
            self.predict(data_path, output_path)

            output_paths.append(output_path)

        logger.info(f"\n✓ Generated predictions for {len(data_paths)} datasets")

        return output_paths
