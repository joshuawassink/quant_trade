"""
Train model with target transformation to address skewness.

Transforms for skewed targets that can handle negative values:
1. Signed log: sign(x) * log(1 + |x|)
2. Yeo-Johnson: Power transform that handles negatives
3. Rank-based: Convert to percentile ranks (0-1)
4. Winsorize: Clip extreme values

Usage:
    python scripts/train_with_target_transform.py --transform signed_log
    python scripts/train_with_target_transform.py --transform yeo_johnson
    python scripts/train_with_target_transform.py --transform rank
"""

import sys
from pathlib import Path
import argparse

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from loguru import logger
import joblib

from src.shared.models.preprocessing import FeaturePreprocessor

logger.remove()
logger.add(sys.stdout, level="INFO")


class TargetTransformer:
    """Transform target variable to reduce skewness."""

    def __init__(self, method='signed_log'):
        """
        Initialize target transformer.

        Args:
            method: Transformation method
                - 'signed_log': sign(x) * log(1 + |x|)
                - 'yeo_johnson': Yeo-Johnson power transform
                - 'rank': Percentile ranks
                - 'winsorize': Clip to percentiles
                - 'none': No transformation
        """
        self.method = method
        self.power_transformer = None
        self.fitted = False

    def fit(self, y):
        """Fit transformer on target values."""
        if self.method == 'yeo_johnson':
            self.power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
            self.power_transformer.fit(y.reshape(-1, 1))

        self.fitted = True
        return self

    def transform(self, y):
        """Transform target values."""
        if not self.fitted and self.method != 'none':
            raise ValueError("Transformer must be fitted before transform")

        if self.method == 'signed_log':
            # sign(x) * log(1 + |x|)
            # Handles negative values, reduces skewness
            return np.sign(y) * np.log1p(np.abs(y))

        elif self.method == 'yeo_johnson':
            # Power transform that automatically handles negatives
            return self.power_transformer.transform(y.reshape(-1, 1)).ravel()

        elif self.method == 'rank':
            # Convert to percentile ranks (0-1)
            # Very robust but loses magnitude information
            return stats.rankdata(y) / len(y)

        elif self.method == 'winsorize':
            # Clip to 1st and 99th percentiles
            lower = np.percentile(y, 1)
            upper = np.percentile(y, 99)
            return np.clip(y, lower, upper)

        elif self.method == 'none':
            return y

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def inverse_transform(self, y_transformed):
        """Inverse transform predictions back to original scale."""
        if self.method == 'signed_log':
            # Inverse: sign(x) * (exp(|x|) - 1)
            return np.sign(y_transformed) * (np.exp(np.abs(y_transformed)) - 1)

        elif self.method == 'yeo_johnson':
            return self.power_transformer.inverse_transform(y_transformed.reshape(-1, 1)).ravel()

        elif self.method == 'rank':
            # Can't perfectly invert ranks, use linear approximation
            logger.warning("Rank transform can't be inverted perfectly")
            return y_transformed  # Return as-is

        elif self.method == 'winsorize':
            return y_transformed  # Already in original scale

        elif self.method == 'none':
            return y_transformed

        else:
            raise ValueError(f"Unknown method: {self.method}")


def load_training_data():
    """Load training dataset."""
    logger.info("Loading training data...")
    df = pl.read_parquet('data/training/training_data_30d_latest.parquet')
    logger.info(f"  ✓ Loaded {len(df):,} rows")
    return df


def prepare_features_and_target(df):
    """Prepare X and y."""
    logger.info("Preparing features and target...")

    metadata_cols = [
        'symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume',
        'dividends', 'stock_splits', 'sector'
    ]
    categorical_cols = ['vix_regime']

    feature_cols = [
        col for col in df.columns
        if col not in metadata_cols
        and col not in categorical_cols
        and not col.startswith('target_')
        and not col.startswith('XL')
        and col != 'spy_close'
    ]

    X = df.select(feature_cols).to_numpy()
    y = df['target_return_30d_vs_market'].to_numpy()
    metadata = df.select(['symbol', 'date', 'target_return_30d_vs_market'])

    logger.info(f"  ✓ X: {X.shape}, y: {y.shape}")

    return X, y, feature_cols, metadata


def analyze_transformation(y_original, y_transformed, method):
    """Analyze effect of transformation on distribution."""
    logger.info(f"\nTransformation Analysis ({method}):")
    logger.info("="*70)

    # Original statistics
    orig_skew = stats.skew(y_original)
    orig_kurt = stats.kurtosis(y_original)

    # Transformed statistics
    trans_skew = stats.skew(y_transformed)
    trans_kurt = stats.kurtosis(y_transformed)

    logger.info(f"Original - Skew: {orig_skew:.4f}, Kurtosis: {orig_kurt:.4f}")
    logger.info(f"Transformed - Skew: {trans_skew:.4f}, Kurtosis: {trans_kurt:.4f}")
    logger.info(f"  → Skew reduced by: {abs(trans_skew) - abs(orig_skew):.4f}")
    logger.info(f"  → Kurtosis reduced by: {trans_kurt - orig_kurt:.4f}")

    if abs(trans_skew) < abs(orig_skew):
        logger.info(f"  ✓ Skewness improved!")
    else:
        logger.warning(f"  ⚠ Skewness increased (transformation may not help)")


def train_with_cv(X, y, feature_names, metadata, transform_method, n_splits=5):
    """Train with time-series CV and target transformation."""
    logger.info("="*70)
    logger.info(f"TRAINING WITH TARGET TRANSFORM: {transform_method}")
    logger.info("="*70)

    dates = metadata['date'].to_numpy()
    sort_idx = np.argsort(dates)

    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    models = []
    val_scores = []
    val_maes = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sorted), 1):
        logger.info(f"\nFold {fold}/{n_splits}")

        X_train, X_val = X_sorted[train_idx], X_sorted[val_idx]
        y_train, y_val = y_sorted[train_idx], y_sorted[val_idx]

        # Transform target
        target_transformer = TargetTransformer(method=transform_method)
        target_transformer.fit(y_train)
        y_train_transformed = target_transformer.transform(y_train)

        if fold == 1:
            # Analyze transformation effect (only once)
            analyze_transformation(y_train, y_train_transformed, transform_method)

        # Preprocess features
        preprocessor = FeaturePreprocessor(
            imputation_strategy='median',
            scaling_method='standard',
            clip_outliers=True,
            outlier_percentiles=(0.5, 99.5),
            filter_null_features=True,
            max_null_pct=70.0,
        )

        X_train_processed = preprocessor.fit_transform(X_train, feature_names=feature_names)
        X_val_processed = preprocessor.transform(X_val)

        logger.info(f"  Features: {X_train_processed.shape[1]}")

        # Train on transformed target
        model = Ridge(alpha=1.0)
        model.fit(X_train_processed, y_train_transformed)

        # Predict in transformed space
        y_val_pred_transformed = model.predict(X_val_processed)

        # Inverse transform predictions back to original scale
        y_val_pred = target_transformer.inverse_transform(y_val_pred_transformed)

        # Evaluate in original scale
        val_r2 = r2_score(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)

        val_scores.append(val_r2)
        val_maes.append(val_mae)

        logger.info(f"  Val R²: {val_r2:.4f}, MAE: {val_mae:.4f}")

        models.append({
            'model': model,
            'preprocessor': preprocessor,
            'target_transformer': target_transformer
        })

    logger.info("\n" + "="*70)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("="*70)
    logger.info(f"Val R²:  {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
    logger.info(f"Val MAE: {np.mean(val_maes):.4f} ± {np.std(val_maes):.4f}")

    cv_scores = {
        'val_r2_mean': np.mean(val_scores),
        'val_r2_std': np.std(val_scores),
        'val_mae_mean': np.mean(val_maes),
        'val_mae_std': np.std(val_maes),
    }

    return models, cv_scores


def train_final_model(X, y, feature_names, transform_method):
    """Train final model on all data."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING FINAL MODEL")
    logger.info("="*70)

    # Transform target
    target_transformer = TargetTransformer(method=transform_method)
    target_transformer.fit(y)
    y_transformed = target_transformer.transform(y)

    # Preprocess features
    preprocessor = FeaturePreprocessor(
        imputation_strategy='median',
        scaling_method='standard',
        clip_outliers=True,
        outlier_percentiles=(0.5, 99.5),
        filter_null_features=True,
        max_null_pct=70.0,
    )

    X_processed = preprocessor.fit_transform(X, feature_names=feature_names)

    # Train
    model = Ridge(alpha=1.0)
    model.fit(X_processed, y_transformed)

    # Evaluate
    y_pred_transformed = model.predict(X_processed)
    y_pred = target_transformer.inverse_transform(y_pred_transformed)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    logger.info(f"  Train R²: {r2:.4f}")
    logger.info(f"  Train MAE: {mae:.4f}")

    return model, preprocessor, target_transformer


def save_model(model, preprocessor, target_transformer, feature_names, transform_method):
    """Save model artifacts."""
    output_dir = Path(f'models/transformed_{transform_method}')
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_dir / 'ridge_model.joblib')
    preprocessor.save(output_dir / 'preprocessor.joblib')
    joblib.dump(target_transformer, output_dir / 'target_transformer.joblib')

    with open(output_dir / 'feature_names.txt', 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")

    with open(output_dir / 'model_info.txt', 'w') as f:
        f.write("RIDGE WITH TARGET TRANSFORMATION\n")
        f.write("="*50 + "\n\n")
        f.write(f"Transformation: {transform_method}\n")
        f.write(f"Features: {len(feature_names)}\n")
        f.write(f"\nHow it works:\n")
        f.write(f"  1. Transform target to reduce skewness\n")
        f.write(f"  2. Train Ridge on transformed target\n")
        f.write(f"  3. Inverse transform predictions\n")

    logger.info(f"\n✓ Model saved to {output_dir}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--transform', type=str, default='signed_log',
                       choices=['signed_log', 'yeo_johnson', 'rank', 'winsorize', 'none'],
                       help='Target transformation method')
    args = parser.parse_args()

    # Load data
    df = load_training_data()
    X, y, feature_names, metadata = prepare_features_and_target(df)

    # Train with CV
    cv_models, cv_scores = train_with_cv(X, y, feature_names, metadata, args.transform)

    # Train final model
    final_model, final_preprocessor, final_transformer = train_final_model(
        X, y, feature_names, args.transform
    )

    # Save
    save_model(final_model, final_preprocessor, final_transformer, feature_names, args.transform)

    logger.info("\n" + "="*70)
    logger.info("✓ TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"  Transform: {args.transform}")
    logger.info(f"  Val R²: {cv_scores['val_r2_mean']:.4f}")
    logger.info(f"  Val MAE: {cv_scores['val_mae_mean']:.4f}")

    print("\nCOMPARE TO BASELINE:")
    print("  Baseline Ridge - R²: 0.065, MAE: 0.0865")
    print(f"  This model    - R²: {cv_scores['val_r2_mean']:.4f}, MAE: {cv_scores['val_mae_mean']:.4f}")

    print("\nNEXT STEPS:")
    print(f"  1. Evaluate: python scripts/evaluate_model.py --model-dir models/transformed_{args.transform}")
    print("  2. Check if error skew is reduced")
    print("  3. Try other transformations if needed")


if __name__ == "__main__":
    main()
