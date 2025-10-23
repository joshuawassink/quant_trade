"""
Train baseline Ridge regression model.

Simple linear model to establish performance baseline for 30-day return prediction.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from loguru import logger
import joblib

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

    Args:
        df: Training DataFrame
        target_col: Name of target column

    Returns:
        X: Feature matrix (numpy array)
        y: Target vector (numpy array)
        feature_names: List of feature column names
        metadata: DataFrame with symbol, date for tracking
    """
    logger.info("Preparing features and target...")

    # Metadata columns to exclude from features
    metadata_cols = [
        'symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume',
        'dividends', 'stock_splits', 'sector'
    ]

    # Get feature columns (exclude metadata, SPY columns, ETF closes, categorical, target)
    categorical_cols = ['vix_regime']  # Categorical features to exclude

    feature_cols = [
        col for col in df.columns
        if col not in metadata_cols
        and col not in categorical_cols
        and not col.startswith('target_')
        and not col.startswith('XL')  # Exclude ETF close prices (keep returns)
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


def train_with_time_series_cv(X, y, metadata_df, n_splits=5):
    """
    Train Ridge regression with time-series cross-validation.

    Args:
        X: Feature matrix
        y: Target vector
        metadata_df: Polars DataFrame with date and symbol for time ordering
        n_splits: Number of CV splits

    Returns:
        models: List of trained models (one per fold)
        cv_scores: Dictionary with CV performance metrics
    """
    logger.info("="*70)
    logger.info("TRAINING WITH TIME-SERIES CROSS-VALIDATION")
    logger.info("="*70)

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

        # Standardize features (fit on train only!)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train Ridge regression
        model = Ridge(alpha=1.0)  # L2 regularization
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)

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

        # Store model and scaler
        models.append({'model': model, 'scaler': scaler})

    # Summary
    logger.info("\n" + "="*70)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("="*70)
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


def train_final_model(X, y):
    """
    Train final model on all data.

    Args:
        X: Feature matrix
        y: Target vector

    Returns:
        model: Trained Ridge model
        scaler: Fitted StandardScaler
    """
    logger.info("\n" + "="*70)
    logger.info("TRAINING FINAL MODEL (all data)")
    logger.info("="*70)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    # Evaluate on training data (for reference)
    y_pred = model.predict(X_scaled)
    train_r2 = r2_score(y, y_pred)
    train_mse = mean_squared_error(y, y_pred)

    logger.info(f"  Train R²: {train_r2:.4f}")
    logger.info(f"  Train MSE: {train_mse:.6f}")
    logger.info(f"  Train RMSE: {np.sqrt(train_mse):.4f} ({np.sqrt(train_mse)*100:.2f}%)")

    return model, scaler


def save_model(model, scaler, feature_names, output_dir='models/baseline'):
    """
    Save trained model, scaler, and metadata.

    Args:
        model: Trained model
        scaler: Fitted scaler
        feature_names: List of feature column names
        output_dir: Directory to save model artifacts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model
    model_file = output_path / 'ridge_model.joblib'
    joblib.dump(model, model_file)
    logger.info(f"\n✓ Model saved to {model_file}")

    # Save scaler
    scaler_file = output_path / 'scaler.joblib'
    joblib.dump(scaler, scaler_file)
    logger.info(f"✓ Scaler saved to {scaler_file}")

    # Save feature names
    features_file = output_path / 'feature_names.txt'
    with open(features_file, 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    logger.info(f"✓ Feature names saved to {features_file} ({len(feature_names)} features)")

    # Save model info
    info_file = output_path / 'model_info.txt'
    with open(info_file, 'w') as f:
        f.write("BASELINE RIDGE REGRESSION MODEL\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model type: Ridge Regression\n")
        f.write(f"Alpha (L2 regularization): 1.0\n")
        f.write(f"Features: {len(feature_names)}\n")
        f.write(f"Target: 30-day market-relative returns\n")
        f.write(f"\nFeature categories:\n")
        f.write(f"  - Technical indicators\n")
        f.write(f"  - Fundamental metrics\n")
        f.write(f"  - Sector/market features\n")
    logger.info(f"✓ Model info saved to {info_file}")


def main():
    """Train baseline Ridge regression model."""

    # 1. Load data
    df = load_training_data()

    # 2. Prepare features and target
    X, y, feature_names, metadata = prepare_features_and_target(df)

    # 3. Train with cross-validation
    cv_models, cv_scores = train_with_time_series_cv(X, y, metadata, n_splits=5)

    # 4. Train final model on all data
    final_model, final_scaler = train_final_model(X, y)

    # 5. Save model
    save_model(final_model, final_scaler, feature_names)

    logger.info("\n" + "="*70)
    logger.info("✓ BASELINE MODEL TRAINING COMPLETE")
    logger.info("="*70)

    # Print key metrics
    logger.info(f"\nKey Metrics:")
    logger.info(f"  Cross-Val R²: {cv_scores['val_r2_mean']:.4f} ± {cv_scores['val_r2_std']:.4f}")
    logger.info(f"  Cross-Val MSE: {cv_scores['val_mse_mean']:.6f}")
    logger.info(f"  Features: {len(feature_names)}")
    logger.info(f"  Training samples: {len(X)}")


if __name__ == "__main__":
    main()
