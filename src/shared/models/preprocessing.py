"""
Feature Preprocessing Pipeline

Flexible sklearn-based preprocessing pipeline that handles:
- Missing value imputation (median, mean, constant)
- Outlier clipping
- Feature scaling (standard, minmax, robust)
- One-hot encoding for categorical features
- Feature selection based on null thresholds

The pipeline is designed to work with time-series cross-validation,
ensuring no data leakage by fitting only on training data.

Works with both numpy arrays and polars DataFrames.
"""

import numpy as np
import polars as pl
from typing import Optional, List, Dict, Literal, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from loguru import logger
import joblib
from pathlib import Path


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Clip outliers to specified percentiles.

    This prevents extreme values from dominating the model,
    especially important for financial ratios that can have inf values.
    """

    def __init__(self, lower_percentile: float = 0.1, upper_percentile: float = 99.9):
        """
        Initialize outlier clipper.

        Args:
            lower_percentile: Lower percentile for clipping (default 0.1)
            upper_percentile: Upper percentile for clipping (default 99.9)
        """
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        """
        Learn clipping bounds from training data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target (ignored)

        Returns:
            self
        """
        # Replace inf with nan for percentile calculation
        X_clean = np.where(np.isinf(X), np.nan, X)

        # Calculate bounds per feature
        self.lower_bounds_ = np.nanpercentile(X_clean, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.nanpercentile(X_clean, self.upper_percentile, axis=0)

        return self

    def transform(self, X):
        """
        Clip outliers to learned bounds.

        Args:
            X: Feature matrix

        Returns:
            Clipped feature matrix
        """
        if self.lower_bounds_ is None or self.upper_bounds_ is None:
            raise ValueError("OutlierClipper must be fitted before transform")

        # Replace inf with nan first
        X_clean = np.where(np.isinf(X), np.nan, X)

        # Clip to bounds
        return np.clip(X_clean, self.lower_bounds_, self.upper_bounds_)


class NullFeatureFilter(BaseEstimator, TransformerMixin):
    """
    Filter out features that have too many null values.

    This is useful for removing features that won't be useful
    due to lack of data, especially in early time periods.
    """

    def __init__(self, max_null_pct: float = 70.0):
        """
        Initialize null feature filter.

        Args:
            max_null_pct: Maximum allowed null percentage (default 70%)
        """
        self.max_null_pct = max_null_pct
        self.keep_features_ = None
        self.feature_null_pcts_ = None

    def fit(self, X, y=None):
        """
        Identify features to keep based on null percentage.

        Args:
            X: Feature matrix
            y: Target (ignored)

        Returns:
            self
        """
        # Calculate null percentage per feature
        null_counts = np.isnan(X).sum(axis=0)
        null_pcts = (null_counts / X.shape[0]) * 100

        # Keep features below threshold
        self.keep_features_ = null_pcts <= self.max_null_pct
        self.feature_null_pcts_ = null_pcts

        n_filtered = (~self.keep_features_).sum()
        if n_filtered > 0:
            logger.info(f"  Filtered {n_filtered} features with >{self.max_null_pct}% nulls")

        return self

    def transform(self, X):
        """
        Keep only features below null threshold.

        Args:
            X: Feature matrix

        Returns:
            Filtered feature matrix
        """
        if self.keep_features_ is None:
            raise ValueError("NullFeatureFilter must be fitted before transform")

        return X[:, self.keep_features_]

    def get_feature_names_out(self, input_features=None):
        """
        Get names of features that passed the filter.

        Args:
            input_features: Original feature names

        Returns:
            List of kept feature names
        """
        if input_features is None:
            return None

        return [feat for feat, keep in zip(input_features, self.keep_features_) if keep]


class FeaturePreprocessor:
    """
    Main preprocessing pipeline builder.

    Creates sklearn pipeline with:
    1. Null feature filtering
    2. Outlier clipping
    3. Missing value imputation
    4. Feature scaling
    """

    def __init__(
        self,
        imputation_strategy: Literal['median', 'mean', 'constant'] = 'median',
        scaling_method: Literal['standard', 'minmax', 'robust'] = 'standard',
        clip_outliers: bool = True,
        outlier_percentiles: tuple = (0.1, 99.9),
        filter_null_features: bool = True,
        max_null_pct: float = 70.0,
    ):
        """
        Initialize feature preprocessor.

        Args:
            imputation_strategy: How to impute missing values ('median', 'mean', 'constant')
            scaling_method: How to scale features ('standard', 'minmax', 'robust')
            clip_outliers: Whether to clip outliers
            outlier_percentiles: (lower, upper) percentiles for clipping
            filter_null_features: Whether to filter features with too many nulls
            max_null_pct: Maximum allowed null percentage for features
        """
        self.imputation_strategy = imputation_strategy
        self.scaling_method = scaling_method
        self.clip_outliers = clip_outliers
        self.outlier_percentiles = outlier_percentiles
        self.filter_null_features = filter_null_features
        self.max_null_pct = max_null_pct
        self.pipeline_ = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None

    def build_pipeline(self) -> Pipeline:
        """
        Build sklearn pipeline.

        Returns:
            Configured sklearn Pipeline
        """
        steps = []

        # Step 1: Filter features with too many nulls
        if self.filter_null_features:
            steps.append(('null_filter', NullFeatureFilter(max_null_pct=self.max_null_pct)))

        # Step 2: Clip outliers
        if self.clip_outliers:
            steps.append(('outlier_clipper', OutlierClipper(
                lower_percentile=self.outlier_percentiles[0],
                upper_percentile=self.outlier_percentiles[1]
            )))

        # Step 3: Impute missing values
        imputer = SimpleImputer(
            strategy=self.imputation_strategy,
            fill_value=0 if self.imputation_strategy == 'constant' else None
        )
        steps.append(('imputer', imputer))

        # Step 4: Scale features
        if self.scaling_method == 'standard':
            scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")

        steps.append(('scaler', scaler))

        return Pipeline(steps)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, feature_names: Optional[List[str]] = None):
        """
        Fit preprocessing pipeline.

        Args:
            X: Feature matrix
            y: Target (optional, not used)
            feature_names: List of feature names (optional)

        Returns:
            self
        """
        self.pipeline_ = self.build_pipeline()
        self.feature_names_in_ = feature_names

        logger.info(f"Fitting preprocessing pipeline...")
        logger.info(f"  Input features: {X.shape[1]}")
        logger.info(f"  Imputation: {self.imputation_strategy}")
        logger.info(f"  Scaling: {self.scaling_method}")
        logger.info(f"  Outlier clipping: {self.clip_outliers}")
        logger.info(f"  Null feature filter: {self.filter_null_features} (max {self.max_null_pct}%)")

        self.pipeline_.fit(X, y)

        # Track output feature names
        if feature_names is not None and self.filter_null_features:
            null_filter = self.pipeline_.named_steps['null_filter']
            self.feature_names_out_ = null_filter.get_feature_names_out(feature_names)
        else:
            self.feature_names_out_ = feature_names

        if self.feature_names_out_ is not None:
            logger.info(f"  Output features: {len(self.feature_names_out_)}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted pipeline.

        Args:
            X: Feature matrix

        Returns:
            Transformed feature matrix
        """
        if self.pipeline_ is None:
            raise ValueError("FeaturePreprocessor must be fitted before transform")

        return self.pipeline_.transform(X)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            X: Feature matrix
            y: Target (optional)
            feature_names: Feature names (optional)

        Returns:
            Transformed feature matrix
        """
        return self.fit(X, y, feature_names).transform(X)

    def get_feature_names_out(self) -> Optional[List[str]]:
        """
        Get output feature names.

        Returns:
            List of feature names after preprocessing
        """
        return self.feature_names_out_

    def get_params(self) -> Dict:
        """
        Get preprocessing parameters.

        Returns:
            Dictionary of parameters
        """
        return {
            'imputation_strategy': self.imputation_strategy,
            'scaling_method': self.scaling_method,
            'clip_outliers': self.clip_outliers,
            'outlier_percentiles': self.outlier_percentiles,
            'filter_null_features': self.filter_null_features,
            'max_null_pct': self.max_null_pct,
        }

    def save(self, file_path: Union[str, Path]) -> Path:
        """
        Save the fitted preprocessing pipeline to disk using joblib.

        Args:
            file_path: Path where to save the preprocessor

        Returns:
            Complete file path

        Raises:
            ValueError: If the pipeline hasn't been fitted yet
        """
        if self.pipeline_ is None:
            raise ValueError(
                "Cannot save unfitted preprocessor. Call fit() or fit_transform() first."
            )

        # Ensure path is Path object
        file_path = Path(file_path)

        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the entire FeaturePreprocessor instance using joblib
        joblib.dump(self, file_path)

        logger.info(f"✓ Preprocessor saved to {file_path}")

        return file_path

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'FeaturePreprocessor':
        """
        Load a saved preprocessor from disk.

        Args:
            file_path: Path to the saved preprocessor file

        Returns:
            FeaturePreprocessor instance

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {file_path}")

        # Load the FeaturePreprocessor instance
        preprocessor = joblib.load(file_path)

        # Validate it's the correct type
        if not isinstance(preprocessor, cls):
            raise ValueError(
                f"Loaded object is not a {cls.__name__} instance. "
                f"Got {type(preprocessor).__name__} instead."
            )

        logger.info(f"✓ Preprocessor loaded from {file_path}")

        return preprocessor


# Example usage
if __name__ == "__main__":
    # Create sample data with nulls, inf, and outliers
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)

    # Add nulls
    null_mask = np.random.rand(n_samples, n_features) < 0.2
    X[null_mask] = np.nan

    # Add some inf values
    X[0:10, 0] = np.inf
    X[0:10, 1] = -np.inf

    # Add outliers
    X[50:60, 2] = X[50:60, 2] * 100

    print("="*70)
    print("FEATURE PREPROCESSING EXAMPLE")
    print("="*70)
    print(f"\nInput data:")
    print(f"  Shape: {X.shape}")
    print(f"  Nulls: {np.isnan(X).sum()} ({np.isnan(X).sum() / X.size * 100:.1f}%)")
    print(f"  Infs: {np.isinf(X).sum()}")

    # Create preprocessor
    preprocessor = FeaturePreprocessor(
        imputation_strategy='median',
        scaling_method='standard',
        clip_outliers=True,
        filter_null_features=False,
    )

    # Fit and transform
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_transformed = preprocessor.fit_transform(X, feature_names=feature_names)

    print(f"\nOutput data:")
    print(f"  Shape: {X_transformed.shape}")
    print(f"  Nulls: {np.isnan(X_transformed).sum()}")
    print(f"  Infs: {np.isinf(X_transformed).sum()}")
    print(f"  Mean: {X_transformed.mean(axis=0).mean():.4f}")
    print(f"  Std: {X_transformed.std(axis=0).mean():.4f}")

    print("\n" + "="*70)
    print("✓ Preprocessing pipeline created successfully")
