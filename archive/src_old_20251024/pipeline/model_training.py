"""
Model Training Step

Trains ML models with cross-validation.
"""

from pathlib import Path
from typing import Optional, Any, Dict, List
import polars as pl
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import joblib
from loguru import logger

from src.models.preprocessing import FeaturePreprocessor


class ModelTrainer:
    """Train ML models with cross-validation."""

    def __init__(
        self,
        model_type: str = 'ridge',
        model_params: Optional[Dict[str, Any]] = None,
        preprocessing_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize model trainer.

        Args:
            model_type: Type of model to train ('ridge', 'quantile', etc.)
            model_params: Parameters for model initialization
            preprocessing_params: Parameters for FeaturePreprocessor
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.preprocessing_params = preprocessing_params or {
            'imputation_strategy': 'median',
            'scaling_method': 'standard',
            'clip_outliers': True,
            'outlier_percentiles': (0.1, 99.9),
            'filter_null_features': True,
            'max_null_pct': 70.0,
        }

    def prepare_features_and_target(
        self,
        df: pl.DataFrame,
        target_col: str,
    ) -> tuple[np.ndarray, np.ndarray, list[str], pl.DataFrame]:
        """
        Prepare X, y, feature names, and metadata from DataFrame.

        Args:
            df: Training DataFrame
            target_col: Name of target column

        Returns:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            metadata: DataFrame with symbol, date for tracking
        """
        logger.info("Preparing features and target...")

        # Metadata columns to exclude from features
        metadata_cols = [
            'symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume',
            'dividends', 'stock_splits', 'sector'
        ]
        categorical_cols = ['vix_regime']

        # Get feature columns
        feature_cols = [
            col for col in df.columns
            if col not in metadata_cols
            and col not in categorical_cols
            and not col.startswith('target_')
            and not col.startswith('forward_return_')
            and not col.startswith('XL')  # Exclude ETF close prices
            and col != 'spy_close'
            and col != target_col
        ]

        logger.info(f"  Feature columns: {len(feature_cols)}")
        logger.info(f"  Target: {target_col}")

        # Convert to numpy
        X = df.select(feature_cols).to_numpy()
        y = df[target_col].to_numpy()

        # Clean infinity and extreme values
        X = np.where(np.isinf(X), np.nan, X)

        # Clip extreme values
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            valid_data = col_data[~np.isnan(col_data)]
            if len(valid_data) > 0:
                upper = np.percentile(valid_data, 99.9)
                lower = np.percentile(valid_data, 0.1)
                X[:, col_idx] = np.clip(col_data, lower, upper)

        # Metadata for tracking
        metadata = df.select(['symbol', 'date', target_col])

        logger.info(f"  ✓ X shape: {X.shape}")
        logger.info(f"  ✓ y shape: {y.shape}")

        return X, y, feature_cols, metadata

    def _create_model(self):
        """Create model instance based on model_type."""
        if self.model_type == 'ridge':
            from sklearn.linear_model import Ridge
            alpha = self.model_params.get('alpha', 1.0)
            return Ridge(alpha=alpha)
        elif self.model_type == 'quantile':
            from sklearn.linear_model import QuantileRegressor
            quantile = self.model_params.get('quantile', 0.5)
            alpha = self.model_params.get('alpha', 1.0)
            solver = self.model_params.get('solver', 'highs')
            return QuantileRegressor(quantile=quantile, alpha=alpha, solver=solver)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train_with_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        metadata_df: pl.DataFrame,
        n_splits: int = 5,
    ) -> tuple[List[Dict], Dict[str, float]]:
        """
        Train with time-series cross-validation.

        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            metadata_df: DataFrame with date for time ordering
            n_splits: Number of CV splits

        Returns:
            models: List of trained models (one per fold)
            cv_scores: Dictionary with CV metrics
        """
        logger.info("=" * 70)
        logger.info(f"TRAINING {self.model_type.upper()} WITH TIME-SERIES CV")
        logger.info("=" * 70)

        # Sort by date for time-series split
        dates = metadata_df['date'].to_numpy()
        sort_idx = np.argsort(dates)

        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)

        models = []
        train_scores = []
        val_scores = []
        train_mses = []
        val_mses = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sorted), 1):
            logger.info(f"\nFold {fold}/{n_splits}")
            logger.info(f"  Train: {len(train_idx)} samples")
            logger.info(f"  Val:   {len(val_idx)} samples")

            X_train, X_val = X_sorted[train_idx], X_sorted[val_idx]
            y_train, y_val = y_sorted[train_idx], y_sorted[val_idx]

            # Create and fit preprocessing pipeline
            preprocessor = FeaturePreprocessor(**self.preprocessing_params)

            X_train_processed = preprocessor.fit_transform(X_train, feature_names=feature_names)
            X_val_processed = preprocessor.transform(X_val)

            logger.info(f"  Features after preprocessing: {X_train_processed.shape[1]}/{X_train.shape[1]}")

            # Train model
            model = self._create_model()
            model.fit(X_train_processed, y_train)

            # Evaluate
            y_train_pred = model.predict(X_train_processed)
            y_val_pred = model.predict(X_val_processed)

            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            val_mse = mean_squared_error(y_val, y_val_pred)

            train_scores.append(train_r2)
            val_scores.append(val_r2)
            train_mses.append(train_mse)
            val_mses.append(val_mse)

            logger.info(f"  Train R²: {train_r2:.4f}, MSE: {train_mse:.6f}")
            logger.info(f"  Val   R²: {val_r2:.4f}, MSE: {val_mse:.6f}")

            # Store model and preprocessor
            models.append({'model': model, 'preprocessor': preprocessor})

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("CROSS-VALIDATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Train R² - Mean: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
        logger.info(f"Val   R² - Mean: {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
        logger.info(f"Train MSE - Mean: {np.mean(train_mses):.6f} ± {np.std(train_mses):.6f}")
        logger.info(f"Val   MSE - Mean: {np.mean(val_mses):.6f} ± {np.std(val_mses):.6f}")

        cv_scores = {
            'train_r2_mean': np.mean(train_scores),
            'train_r2_std': np.std(train_scores),
            'val_r2_mean': np.mean(val_scores),
            'val_r2_std': np.std(val_scores),
            'train_mse_mean': np.mean(train_mses),
            'train_mse_std': np.std(train_mses),
            'val_mse_mean': np.mean(val_mses),
            'val_mse_std': np.std(val_mses),
        }

        return models, cv_scores

    def train_final_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
    ) -> tuple[Any, FeaturePreprocessor]:
        """
        Train final model on all data.

        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names

        Returns:
            model: Trained model
            preprocessor: Fitted preprocessor
        """
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING FINAL MODEL (all data)")
        logger.info("=" * 70)

        # Create and fit preprocessing pipeline
        preprocessor = FeaturePreprocessor(**self.preprocessing_params)
        X_processed = preprocessor.fit_transform(X, feature_names=feature_names)

        logger.info(f"  Features after preprocessing: {X_processed.shape[1]}/{X.shape[1]}")

        # Train
        model = self._create_model()
        model.fit(X_processed, y)

        # Evaluate on training data (for reference)
        y_pred = model.predict(X_processed)
        train_r2 = r2_score(y, y_pred)
        train_mse = mean_squared_error(y, y_pred)

        logger.info(f"  Train R²: {train_r2:.4f}")
        logger.info(f"  Train MSE: {train_mse:.6f}")
        logger.info(f"  Train RMSE: {np.sqrt(train_mse):.4f} ({np.sqrt(train_mse)*100:.2f}%)")

        return model, preprocessor

    def save_model(
        self,
        model: Any,
        preprocessor: FeaturePreprocessor,
        feature_names: list[str],
        output_dir: Path,
        model_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Save trained model and artifacts.

        Args:
            model: Trained model
            preprocessor: Fitted preprocessor
            feature_names: List of feature names
            output_dir: Directory to save artifacts
            model_info: Optional metadata about the model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_file = output_dir / f'{self.model_type}_model.joblib'
        joblib.dump(model, model_file)
        logger.info(f"\n✓ Model saved to {model_file}")

        # Save preprocessor
        preprocessor_file = output_dir / 'preprocessor.joblib'
        preprocessor.save(preprocessor_file)
        logger.info(f"✓ Preprocessor saved to {preprocessor_file}")

        # Save feature names
        features_file = output_dir / 'feature_names.txt'
        with open(features_file, 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        logger.info(f"✓ Feature names saved to {features_file} ({len(feature_names)} features)")

        # Save model info
        info_file = output_dir / 'model_info.txt'
        with open(info_file, 'w') as f:
            f.write(f"{self.model_type.upper()} MODEL\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model type: {self.model_type}\n")
            f.write(f"Parameters: {self.model_params}\n")
            f.write(f"Features: {len(feature_names)}\n")

            if model_info:
                f.write("\nAdditional Info:\n")
                for key, value in model_info.items():
                    f.write(f"  {key}: {value}\n")

        logger.info(f"✓ Model info saved to {info_file}")
