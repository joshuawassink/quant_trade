"""
Train Quantile Regression Model (addresses negative error skew)

Instead of predicting the mean (Ridge), predict the 60th-75th percentile.
This helps capture positive moves better without being penalized by MSE loss.

Usage:
    python scripts/train_quantile_model.py --quantile 0.60
    python scripts/train_quantile_model.py --quantile 0.75
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loguru import logger
import joblib

from src.shared.models.preprocessing import FeaturePreprocessor

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")


def load_training_data(path: str = 'data/training/training_data_30d_latest.parquet'):
    """Load training dataset."""
    logger.info(f"Loading training data from {path}...")
    df = pl.read_parquet(path)
    logger.info(f"  ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def prepare_features_and_target(df: pl.DataFrame, target_col: str = 'target_return_30d_vs_market'):
    """
    Prepare X (features) and y (target) for training.
    """
    logger.info("Preparing features and target...")

    # Metadata columns to exclude from features
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
        and col != target_col
    ]

    logger.info(f"  Feature columns: {len(feature_cols)}")
    logger.info(f"  Target: {target_col}")

    # Convert to numpy
    X = df.select(feature_cols).to_numpy()
    y = df[target_col].to_numpy()

    # Metadata for tracking
    metadata = df.select(['symbol', 'date', target_col])

    logger.info(f"  ✓ X shape: {X.shape}")
    logger.info(f"  ✓ y shape: {y.shape}")

    return X, y, feature_cols, metadata


def train_with_time_series_cv(X, y, feature_names, metadata_df, quantile=0.60, n_splits=5, alpha=1.0):
    """
    Train Quantile regression with time-series cross-validation.

    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        metadata_df: Polars DataFrame with date and symbol
        quantile: Which quantile to predict (0.6 = 60th percentile)
        n_splits: Number of CV splits
        alpha: L1 regularization strength

    Returns:
        models: List of trained models
        cv_scores: Dictionary with CV performance metrics
    """
    logger.info("="*70)
    logger.info(f"TRAINING QUANTILE REGRESSION (q={quantile})")
    logger.info("="*70)
    logger.info(f"This predicts the {quantile*100:.0f}th percentile instead of the mean")
    logger.info(f"→ Helps capture positive moves (addresses negative error skew)")
    logger.info("")

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
    train_maes = []
    val_maes = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sorted), 1):
        logger.info(f"\nFold {fold}/{n_splits}")
        logger.info(f"  Train: {len(train_idx)} samples")
        logger.info(f"  Val:   {len(val_idx)} samples")

        X_train, X_val = X_sorted[train_idx], X_sorted[val_idx]
        y_train, y_val = y_sorted[train_idx], y_sorted[val_idx]

        # Create and fit preprocessing pipeline
        preprocessor = FeaturePreprocessor(
            imputation_strategy='median',
            scaling_method='standard',
            clip_outliers=True,
            outlier_percentiles=(0.5, 99.5),  # More asymmetric - keep positive outliers
            filter_null_features=True,
            max_null_pct=70.0,
        )

        # Fit on training data only
        X_train_processed = preprocessor.fit_transform(X_train, feature_names=feature_names)
        X_val_processed = preprocessor.transform(X_val)

        logger.info(f"  Features after preprocessing: {X_train_processed.shape[1]}/{X_train.shape[1]}")

        # Train Quantile regression
        model = QuantileRegressor(
            quantile=quantile,
            alpha=alpha,
            solver='highs'  # Fast solver
        )
        model.fit(X_train_processed, y_train)

        # Evaluate
        y_train_pred = model.predict(X_train_processed)
        y_val_pred = model.predict(X_val_processed)

        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)

        train_scores.append(train_r2)
        val_scores.append(val_r2)
        train_maes.append(train_mae)
        val_maes.append(val_mae)

        logger.info(f"  Train R²: {train_r2:.4f}, MAE: {train_mae:.4f}")
        logger.info(f"  Val   R²: {val_r2:.4f}, MAE: {val_mae:.4f}")

        # Store model and preprocessor
        models.append({'model': model, 'preprocessor': preprocessor})

    # Summary
    logger.info("\n" + "="*70)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("="*70)
    logger.info(f"Train R² - Mean: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
    logger.info(f"Val   R² - Mean: {np.mean(val_scores):.4f} ± {np.std(val_scores):.4f}")
    logger.info(f"Train MAE - Mean: {np.mean(train_maes):.4f} ± {np.std(train_maes):.4f}")
    logger.info(f"Val   MAE - Mean: {np.mean(val_maes):.4f} ± {np.std(val_maes):.4f}")

    cv_scores = {
        'train_r2_mean': np.mean(train_scores),
        'train_r2_std': np.std(train_scores),
        'val_r2_mean': np.mean(val_scores),
        'val_r2_std': np.std(val_scores),
        'train_mae_mean': np.mean(train_maes),
        'train_mae_std': np.std(train_maes),
        'val_mae_mean': np.mean(val_maes),
        'val_mae_std': np.std(val_maes),
    }

    return models, cv_scores


def train_final_model(X, y, feature_names, quantile=0.60, alpha=1.0):
    """
    Train final quantile model on all data.
    """
    logger.info("\n" + "="*70)
    logger.info(f"TRAINING FINAL MODEL (q={quantile})")
    logger.info("="*70)

    # Create and fit preprocessing pipeline
    preprocessor = FeaturePreprocessor(
        imputation_strategy='median',
        scaling_method='standard',
        clip_outliers=True,
        outlier_percentiles=(0.5, 99.5),  # Asymmetric
        filter_null_features=True,
        max_null_pct=70.0,
    )

    X_processed = preprocessor.fit_transform(X, feature_names=feature_names)

    logger.info(f"  Features after preprocessing: {X_processed.shape[1]}/{X.shape[1]}")

    # Train
    model = QuantileRegressor(
        quantile=quantile,
        alpha=alpha,
        solver='highs'
    )
    model.fit(X_processed, y)

    # Evaluate on training data (for reference)
    y_pred = model.predict(X_processed)
    train_r2 = r2_score(y, y_pred)
    train_mae = mean_absolute_error(y, y_pred)
    train_rmse = np.sqrt(mean_squared_error(y, y_pred))

    logger.info(f"  Train R²: {train_r2:.4f}")
    logger.info(f"  Train MAE: {train_mae:.4f} ({train_mae*100:.2f}%)")
    logger.info(f"  Train RMSE: {train_rmse:.4f} ({train_rmse*100:.2f}%)")

    return model, preprocessor


def save_model(model, preprocessor, feature_names, quantile, alpha, output_dir='models/quantile'):
    """
    Save trained model, preprocessor, and metadata.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_file = output_path / f'quantile_model_q{int(quantile*100)}.joblib'
    joblib.dump(model, model_file)
    logger.info(f"\n✓ Model saved to {model_file}")

    # Save preprocessor
    preprocessor_file = output_path / f'preprocessor_q{int(quantile*100)}.joblib'
    preprocessor.save(preprocessor_file)
    logger.info(f"✓ Preprocessor saved to {preprocessor_file}")

    # Save feature names
    features_file = output_path / 'feature_names.txt'
    with open(features_file, 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    logger.info(f"✓ Feature names saved to {features_file} ({len(feature_names)} features)")

    # Save model info
    info_file = output_path / 'model_info.txt'
    with open(info_file, 'w') as f:
        f.write("QUANTILE REGRESSION MODEL\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model type: Quantile Regression\n")
        f.write(f"Quantile: {quantile} ({quantile*100:.0f}th percentile)\n")
        f.write(f"Alpha (L1 regularization): {alpha}\n")
        f.write(f"Features: {len(feature_names)}\n")
        f.write(f"Target: 30-day market-relative returns\n")
        f.write(f"\nAdvantage over Ridge:\n")
        f.write(f"  - Predicts {quantile*100:.0f}th percentile instead of mean\n")
        f.write(f"  - Better captures upside moves\n")
        f.write(f"  - Addresses negative error skew\n")
        f.write(f"  - Robust to outliers\n")
    logger.info(f"✓ Model info saved to {info_file}")


def main():
    """Train quantile regression model."""
    parser = argparse.ArgumentParser(description="Train quantile regression model")
    parser.add_argument('--quantile', type=float, default=0.60,
                       help='Quantile to predict (default: 0.60)')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='L1 regularization strength (default: 1.0)')
    args = parser.parse_args()

    # 1. Load data
    df = load_training_data()

    # 2. Prepare features and target
    X, y, feature_names, metadata = prepare_features_and_target(df)

    # 3. Train with cross-validation
    cv_models, cv_scores = train_with_time_series_cv(
        X, y, feature_names, metadata,
        quantile=args.quantile,
        alpha=args.alpha,
        n_splits=5
    )

    # 4. Train final model on all data
    final_model, final_preprocessor = train_final_model(
        X, y, feature_names,
        quantile=args.quantile,
        alpha=args.alpha
    )

    # 5. Save model
    save_model(final_model, final_preprocessor, feature_names, args.quantile, args.alpha)

    logger.info("\n" + "="*70)
    logger.info("✓ QUANTILE MODEL TRAINING COMPLETE")
    logger.info("="*70)

    # Print key metrics
    logger.info(f"\nKey Metrics (q={args.quantile}):")
    logger.info(f"  Cross-Val R²: {cv_scores['val_r2_mean']:.4f} ± {cv_scores['val_r2_std']:.4f}")
    logger.info(f"  Cross-Val MAE: {cv_scores['val_mae_mean']:.4f}")
    logger.info(f"  Features: {len(feature_names)}")
    logger.info(f"  Training samples: {len(X)}")

    print("\nNEXT STEPS:")
    print("  1. Compare to baseline Ridge model")
    print(f"  2. Run: python scripts/evaluate_model.py --model-dir models/quantile")
    print("  3. Check if error skew is reduced (target: <-0.5)")


if __name__ == "__main__":
    main()
