"""
Model Registry and Configurations

Central registry of all available models with their default parameters,
hyperparameter grids, and preprocessing configurations.

This allows workflows to be config-driven rather than hardcoded.

Usage:
    from src.shared.config.models import get_model, MODEL_REGISTRY

    # Get model class and params
    model_class, params = get_model('ridge')
    model = model_class(**params)

    # Get full config including preprocessing
    config = MODEL_REGISTRY['ranking_sgd']
"""

from typing import Dict, Any, Tuple, Type
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator

# Import model classes
from src.shared.models.ranking_sgd import RankingSGDRegressor

# XGBoost models (optional dependency)
try:
    from src.shared.models.ranking_xgb import RankingXGBRegressor, LambdaRankXGBRegressor
    HAS_XGBOOST_MODELS = True
except Exception:
    # XGBoost not installed - models won't be available
    RankingXGBRegressor = None
    LambdaRankXGBRegressor = None
    HAS_XGBOOST_MODELS = False


# ============================================================================
# Model Registry
# ============================================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ========================================================================
    # Linear Models
    # ========================================================================
    'ridge': {
        'name': 'Ridge Regression',
        'class': Ridge,
        'default_params': {
            'alpha': 1.0,
            'fit_intercept': True,
            'max_iter': None,
            'tol': 0.0001,
            'solver': 'auto',
        },
        'param_grid': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        'preprocessing': {
            'imputation_strategy': 'median',
            'scaling_method': 'standard',
            'clip_outliers': True,
            'outlier_percentiles': (0.1, 99.9),
            'filter_null_features': True,
            'max_null_pct': 70.0,
        },
        'supports_sample_weight': True,
        'supports_cv': True,
    },

    # ========================================================================
    # Ranking Models - SGD
    # ========================================================================
    'ranking_sgd': {
        'name': 'Ranking SGD Regressor',
        'class': RankingSGDRegressor,
        'default_params': {
            'rank_weight': 0.5,      # Balance MSE and ranking
            'mse_weight': 1.0,
            'alpha': 0.0001,         # Regularization
            'penalty': 'l2',         # Ridge-style
            'learning_rate': 'optimal',
            'max_iter': 1000,
            'tol': 1e-3,
            'batch_size': 32,
            'random_state': 42,
            'verbose': False,
        },
        'param_grid': {
            'rank_weight': [0.0, 0.3, 0.5, 0.7, 1.0],
            'alpha': [1e-5, 1e-4, 1e-3],
        },
        'preprocessing': {
            'imputation_strategy': 'median',
            'scaling_method': 'standard',  # Critical for SGD!
            'clip_outliers': True,
            'outlier_percentiles': (0.1, 99.9),
            'filter_null_features': True,
            'max_null_pct': 70.0,
        },
        'supports_sample_weight': False,
        'supports_cv': True,
        'notes': 'Feature scaling is critical for SGD performance',
    },

    'ranking_sgd_pure': {
        'name': 'Pure Ranking SGD (rank_weight=1.0)',
        'class': RankingSGDRegressor,
        'default_params': {
            'rank_weight': 1.0,      # Pure ranking loss
            'mse_weight': 0.0,       # No MSE term
            'alpha': 0.0001,
            'penalty': 'l2',
            'learning_rate': 'optimal',
            'max_iter': 1000,
            'random_state': 42,
        },
        'param_grid': {
            'alpha': [1e-5, 1e-4, 1e-3],
        },
        'preprocessing': {
            'imputation_strategy': 'median',
            'scaling_method': 'standard',
            'clip_outliers': True,
            'outlier_percentiles': (0.1, 99.9),
            'filter_null_features': True,
            'max_null_pct': 70.0,
        },
        'supports_sample_weight': False,
        'supports_cv': True,
    },

    # ========================================================================
    # Ranking Models - XGBoost
    # ========================================================================
    'ranking_xgb': {
        'name': 'Ranking XGBoost',
        'class': RankingXGBRegressor,
        'default_params': {
            'rank_weight': 0.5,      # Balance MSE and ranking
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.0,        # L1 regularization
            'reg_lambda': 1.0,       # L2 regularization
            'early_stopping_rounds': 10,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',   # Fast histogram method
            'verbose': False,
        },
        'param_grid': {
            'rank_weight': [0.0, 0.3, 0.5, 0.7, 1.0],
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.05, 0.1],
        },
        'preprocessing': {
            'imputation_strategy': 'median',
            'scaling_method': 'none',  # XGBoost doesn't need scaling
            'clip_outliers': True,
            'outlier_percentiles': (0.1, 99.9),
            'filter_null_features': True,
            'max_null_pct': 70.0,
        },
        'supports_sample_weight': False,
        'supports_cv': True,
        'supports_early_stopping': True,
        'notes': 'XGBoost handles missing values natively, scaling not needed',
    },

    'xgboost': {
        'name': 'Standard XGBoost (MSE)',
        'class': RankingXGBRegressor,
        'default_params': {
            'rank_weight': 0.0,      # Pure MSE (standard XGBoost)
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 1.0,
            'early_stopping_rounds': 10,
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
        },
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.05, 0.1],
        },
        'preprocessing': {
            'imputation_strategy': 'median',
            'scaling_method': 'none',
            'clip_outliers': True,
            'outlier_percentiles': (0.1, 99.9),
            'filter_null_features': True,
            'max_null_pct': 70.0,
        },
        'supports_sample_weight': False,
        'supports_cv': True,
        'supports_early_stopping': True,
    },

    'lambdarank_xgb': {
        'name': 'LambdaRank XGBoost',
        'class': LambdaRankXGBRegressor,
        'default_params': {
            'objective': 'rank:ndcg',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
        },
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
        },
        'preprocessing': {
            'imputation_strategy': 'median',
            'scaling_method': 'none',
            'clip_outliers': True,
            'outlier_percentiles': (0.1, 99.9),
            'filter_null_features': True,
            'max_null_pct': 70.0,
        },
        'supports_sample_weight': False,
        'supports_cv': False,  # LambdaRank requires groups
        'notes': 'Requires group information for ranking (dates in our case)',
    },
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_model(
    model_type: str,
    custom_params: Dict[str, Any] = None
) -> Tuple[Type[BaseEstimator], Dict[str, Any]]:
    """
    Get model class and parameters from registry.

    Args:
        model_type: Model type key (e.g., 'ridge', 'ranking_sgd')
        custom_params: Optional parameter overrides

    Returns:
        (model_class, params) tuple

    Examples:
        >>> model_class, params = get_model('ridge')
        >>> model = model_class(**params)

        >>> # Override default params
        >>> model_class, params = get_model('ridge', {'alpha': 10.0})
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    config = MODEL_REGISTRY[model_type]
    model_class = config['class']
    params = config['default_params'].copy()

    # Override with custom params
    if custom_params:
        params.update(custom_params)

    return model_class, params


def get_preprocessing_config(model_type: str) -> Dict[str, Any]:
    """
    Get preprocessing configuration for a model.

    Args:
        model_type: Model type key

    Returns:
        Preprocessing config dict
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")

    return MODEL_REGISTRY[model_type]['preprocessing'].copy()


def get_param_grid(model_type: str) -> Dict[str, Any]:
    """
    Get hyperparameter grid for cross-validation.

    Args:
        model_type: Model type key

    Returns:
        Parameter grid dict
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")

    return MODEL_REGISTRY[model_type]['param_grid'].copy()


def list_available_models() -> list[str]:
    """List all available model types."""
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_type: str) -> str:
    """
    Get formatted information about a model.

    Args:
        model_type: Model type key

    Returns:
        Formatted string with model information
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")

    config = MODEL_REGISTRY[model_type]

    info = [
        f"Model: {config['name']}",
        f"Class: {config['class'].__name__}",
        f"Default Params: {config['default_params']}",
        f"Supports CV: {config['supports_cv']}",
    ]

    if 'notes' in config:
        info.append(f"Notes: {config['notes']}")

    return '\n'.join(info)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("MODEL REGISTRY")
    print("=" * 70)

    print("\nAvailable models:")
    for model_type in list_available_models():
        config = MODEL_REGISTRY[model_type]
        print(f"  - {model_type:20s} : {config['name']}")

    print("\n" + "=" * 70)
    print("Example: Ridge Regression")
    print("=" * 70)
    print(get_model_info('ridge'))

    print("\n" + "=" * 70)
    print("Example: Get Model")
    print("=" * 70)
    model_class, params = get_model('ridge', {'alpha': 10.0})
    print(f"Class: {model_class}")
    print(f"Params: {params}")