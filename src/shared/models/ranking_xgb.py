"""
XGBoost with Custom Ranking Objective

Implements XGBoost models optimized for rank-ordering accuracy using custom
objective functions and evaluation metrics.

Key Features:
    - Custom ranking objectives (rank MSE/MAE)
    - Built-in LambdaRank support (rank:ndcg, rank:map)
    - Flexible loss combinations
    - GPU acceleration support

Models:
    - RankingXGBRegressor: XGBoost with combined MSE + ranking loss
    - LambdaRankXGBRegressor: XGBoost with LambdaRank objective

Usage:
    from src.shared.models.ranking_xgb import RankingXGBRegressor

    model = RankingXGBRegressor(
        rank_weight=0.5,
        n_estimators=100
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
"""

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from loguru import logger

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed. Install with: pip install xgboost")


class RankingXGBRegressor(BaseEstimator, RegressorMixin):
    """
    XGBoost with combined MSE and ranking objective.

    Uses XGBoost's custom objective API to combine standard regression loss
    with ranking-aware penalty.

    Parameters
    ----------
    rank_weight : float, default=0.5
        Weight for ranking term. Higher = more focus on ranking.
        - 0.0 = pure MSE (standard XGBoost)
        - 0.5 = balanced
        - 1.0 = equal weight to MSE and ranking

    n_estimators : int, default=100
        Number of boosting rounds.

    max_depth : int, default=6
        Maximum tree depth.

    learning_rate : float, default=0.1
        Boosting learning rate (eta).

    min_child_weight : int, default=1
        Minimum sum of instance weight needed in a child.

    subsample : float, default=0.8
        Subsample ratio of the training instances.

    colsample_bytree : float, default=0.8
        Subsample ratio of columns when constructing each tree.

    reg_alpha : float, default=0.0
        L1 regularization on weights.

    reg_lambda : float, default=1.0
        L2 regularization on weights.

    early_stopping_rounds : int, default=10
        Activates early stopping. Validation metric must improve within
        early_stopping_rounds to continue training.

    eval_metric : str or list, default='rmse'
        Evaluation metric(s) for validation.
        - 'rmse': Root mean squared error
        - 'mae': Mean absolute error
        - Custom: Can pass custom ranking metrics

    random_state : int, default=None
        Random seed for reproducibility.

    n_jobs : int, default=-1
        Number of parallel threads. -1 uses all cores.

    tree_method : str, default='auto'
        Tree construction algorithm:
        - 'auto': Use heuristic
        - 'exact': Exact greedy algorithm
        - 'approx': Approximate greedy algorithm
        - 'hist': Faster histogram optimized
        - 'gpu_hist': GPU acceleration

    verbose : bool, default=False
        Whether to print progress messages.

    Attributes
    ----------
    model_ : xgboost.Booster
        The underlying XGBoost model.

    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from src.shared.models.ranking_xgb import RankingXGBRegressor
    >>> model = RankingXGBRegressor(rank_weight=0.5, n_estimators=100)
    >>> model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    >>> y_pred = model.predict(X_test)

    Notes
    -----
    Requires xgboost to be installed:
        pip install xgboost

    For GPU acceleration:
        pip install xgboost[gpu]
        # Then use tree_method='gpu_hist'
    """

    def __init__(
        self,
        rank_weight: float = 0.5,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 10,
        eval_metric: str = 'rmse',
        random_state: int = None,
        n_jobs: int = -1,
        tree_method: str = 'auto',
        verbose: bool = False,
    ):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")

        self.rank_weight = rank_weight
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.tree_method = tree_method
        self.verbose = verbose

    def _custom_objective(self, y_pred: np.ndarray, dtrain: xgb.DMatrix):
        """
        Custom objective combining MSE and rank loss.

        Parameters
        ----------
        y_pred : array-like
            Current predictions.
        dtrain : xgb.DMatrix
            Training data.

        Returns
        -------
        grad : array-like
            Gradient of loss w.r.t. predictions.
        hess : array-like
            Hessian of loss w.r.t. predictions (approximated).
        """
        y_true = dtrain.get_label()
        n = len(y_true)

        # MSE gradient and hessian
        mse_grad = 2.0 * (y_pred - y_true) / n
        mse_hess = 2.0 * np.ones_like(y_true) / n

        # Ranking gradient (approximate)
        if self.rank_weight > 0:
            # Get current ranks
            true_ranks = stats.rankdata(y_true, method='average')
            pred_ranks = stats.rankdata(y_pred, method='average')
            rank_errors = pred_ranks - true_ranks

            # Approximate ranking gradient
            # This is a simplification; true rank gradient requires pairwise comparisons
            rank_grad = 2.0 * rank_errors / n
            rank_hess = 2.0 * np.ones_like(y_true) / n
        else:
            rank_grad = np.zeros_like(y_true)
            rank_hess = np.zeros_like(y_true)

        # Combine
        grad = mse_grad + self.rank_weight * rank_grad
        hess = mse_hess + self.rank_weight * rank_hess

        return grad, hess

    def _spearman_eval(self, y_pred: np.ndarray, dtrain: xgb.DMatrix):
        """
        Custom evaluation metric: Spearman correlation.

        Parameters
        ----------
        y_pred : array-like
            Predictions.
        dtrain : xgb.DMatrix
            Data with labels.

        Returns
        -------
        name : str
            Metric name.
        score : float
            Spearman correlation (higher is better).
        """
        y_true = dtrain.get_label()
        corr, _ = stats.spearmanr(y_true, y_pred)
        # XGBoost minimizes metrics, so return negative for maximization
        return 'spearman', -corr  # Negative because XGBoost minimizes

    def fit(self, X, y, eval_set=None, verbose=None):
        """
        Fit the XGBoost ranking model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        eval_set : list of (X, y) tuples, default=None
            List of validation sets for early stopping.
        verbose : bool, default=None
            Override class verbose setting.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        self.n_features_in_ = X.shape[1]

        # Use class verbose if not overridden
        if verbose is None:
            verbose = self.verbose

        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)

        # Prepare validation sets
        evals = [(dtrain, 'train')]
        if eval_set is not None:
            for i, (X_val, y_val) in enumerate(eval_set):
                dval = xgb.DMatrix(X_val, label=y_val)
                evals.append((dval, f'val_{i}'))

        # XGBoost parameters
        params = {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'tree_method': self.tree_method,
            'seed': self.random_state,
            'nthread': self.n_jobs if self.n_jobs > 0 else None,
        }

        # Train
        if verbose:
            logger.info(f"Training XGBoost with rank_weight={self.rank_weight}...")

        self.model_ = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            obj=self._custom_objective if self.rank_weight > 0 else None,
            custom_metric=self._spearman_eval,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
            verbose_eval=verbose,
        )

        if verbose:
            logger.info("✓ XGBoost training complete")

        return self

    def predict(self, X):
        """
        Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ['model_'])
        X = check_array(X, accept_sparse=False)

        dtest = xgb.DMatrix(X)
        return self.model_.predict(dtest)

    def score(self, X, y):
        """
        Return Spearman correlation (ranking quality).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        score : float
            Spearman correlation coefficient (higher is better).
        """
        y_pred = self.predict(X)
        corr, _ = stats.spearmanr(y, y_pred)
        return corr

    def get_feature_importance(self, importance_type='weight'):
        """
        Get feature importance scores.

        Parameters
        ----------
        importance_type : str, default='weight'
            Type of importance:
            - 'weight': Number of times feature is used
            - 'gain': Average gain when feature is used
            - 'cover': Average coverage when feature is used

        Returns
        -------
        importance : dict
            Feature importance scores.
        """
        check_is_fitted(self, ['model_'])
        return self.model_.get_score(importance_type=importance_type)


class LambdaRankXGBRegressor(BaseEstimator, RegressorMixin):
    """
    XGBoost with built-in LambdaRank objective.

    Uses XGBoost's native ranking objectives (rank:ndcg, rank:map) which
    are optimized for learning-to-rank tasks.

    This is the recommended approach for pure ranking when not concerned
    about absolute prediction values.

    Parameters
    ----------
    objective : str, default='rank:ndcg'
        Ranking objective:
        - 'rank:ndcg': NDCG (Normalized Discounted Cumulative Gain)
        - 'rank:map': MAP (Mean Average Precision)
        - 'rank:pairwise': Pairwise ranking (RankNet)

    n_estimators : int, default=100
        Number of boosting rounds.

    max_depth : int, default=6
        Maximum tree depth.

    learning_rate : float, default=0.1
        Boosting learning rate.

    **kwargs
        Additional arguments passed to xgboost.train().

    Examples
    --------
    >>> model = LambdaRankXGBRegressor(objective='rank:ndcg')
    >>> # For ranking, need to specify groups (all samples in one group here)
    >>> model.fit(X_train, y_train, group=[len(X_train)])
    >>> y_pred = model.predict(X_test)

    Notes
    -----
    LambdaRank requires group information (queries in search, dates in finance).
    For portfolio selection, each date is a group where we rank stocks.
    """

    def __init__(
        self,
        objective='rank:ndcg',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=None,
        **kwargs
    ):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")

        self.objective = objective
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(self, X, y, group=None, **fit_params):
        """
        Fit LambdaRank model.

        Parameters
        ----------
        X : array-like
            Features.
        y : array-like
            Targets (relevance scores).
        group : array-like, default=None
            Group sizes for ranking. Each group is ranked independently.
            If None, treats all samples as one group.
        **fit_params
            Additional fit parameters.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        self.n_features_in_ = X.shape[1]

        # If no groups provided, treat all as one group
        if group is None:
            group = [len(X)]

        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(group)

        params = {
            'objective': self.objective,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'seed': self.random_state,
            **self.kwargs
        }

        self.model_ = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            **fit_params
        )

        return self

    def predict(self, X):
        """Predict rankings."""
        check_is_fitted(self, ['model_'])
        X = check_array(X, accept_sparse=False)
        dtest = xgb.DMatrix(X)
        return self.model_.predict(dtest)

    def score(self, X, y):
        """Spearman correlation."""
        y_pred = self.predict(X)
        corr, _ = stats.spearmanr(y, y_pred)
        return corr


# Example usage
if __name__ == "__main__":
    if not HAS_XGBOOST:
        print("XGBoost not installed. Skipping examples.")
        print("Install with: pip install xgboost")
    else:
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score

        print("=" * 70)
        print("RANKING XGBoost REGRESSOR - EXAMPLE")
        print("=" * 70)

        # Generate synthetic data
        X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Standard XGBoost
        print("\n1. Standard XGBoost (MSE):")
        model_xgb = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        model_xgb.fit(X_train, y_train)
        y_pred_xgb = model_xgb.predict(X_test)

        print(f"   MSE: {mean_squared_error(y_test, y_pred_xgb):.4f}")
        print(f"   R²: {r2_score(y_test, y_pred_xgb):.4f}")
        spearman, _ = stats.spearmanr(y_test, y_pred_xgb)
        print(f"   Spearman: {spearman:.4f}")

        # Ranking XGBoost
        print("\n2. RankingXGBRegressor (rank_weight=0.5):")
        model_ranking = RankingXGBRegressor(
            rank_weight=0.5,
            n_estimators=100,
            random_state=42,
            verbose=True
        )
        model_ranking.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        y_pred_ranking = model_ranking.predict(X_test)

        print(f"   MSE: {mean_squared_error(y_test, y_pred_ranking):.4f}")
        print(f"   R²: {r2_score(y_test, y_pred_ranking):.4f}")
        spearman, _ = stats.spearmanr(y_test, y_pred_ranking)
        print(f"   Spearman: {spearman:.4f}")

        print("\n" + "=" * 70)
        print("✓ XGBoost ranking models ready!")
        print("=" * 70)