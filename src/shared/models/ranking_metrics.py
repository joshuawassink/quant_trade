"""
Ranking Loss Metrics

Standalone loss/metric functions for evaluating and optimizing rank-ordering
accuracy rather than absolute value prediction.

These metrics can be used with:
- SGDRegressor (via custom loss)
- XGBoost (via custom objective)
- Neural networks (via custom loss layer)
- Model evaluation

Key Insight:
    In portfolio selection, correctly ranking stocks matters more than
    predicting exact returns. These metrics quantify ranking quality.

Usage:
    from src.shared.models.ranking_metrics import rank_mse, rank_mae, spearman_loss

    # As evaluation metrics
    error = rank_mae(y_true, y_pred)

    # For gradient-based optimization (returns loss and gradient)
    loss, grad = rank_mse_with_grad(y_true, y_pred)
"""

import numpy as np
from scipy import stats
from typing import Tuple


# ============================================================================
# Pure Ranking Losses (Your Notebook Approach)
# ============================================================================

def rank_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Rank Error.

    Measures average position error in ranking. Intuitive metric: "On average,
    how many positions off are my predictions?"

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    mae : float
        Mean absolute rank error. Lower is better.
        - 0.0 = perfect ranking
        - 1.8 = average 2 positions off
        - n/2 = random ranking (approximately)

    Examples
    --------
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0])  # Good predictions
    >>> rank_mae(y_true, y_pred)
    0.0  # Perfect ranking

    >>> y_pred = np.array([5.0, 1.0, 3.0, 2.0, 4.0])  # Bad predictions
    >>> rank_mae(y_true, y_pred)
    2.4  # Average ~2 positions off
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Get ranks (using scipy for consistent tie-handling)
    true_ranks = stats.rankdata(y_true, method='average')
    pred_ranks = stats.rankdata(y_pred, method='average')

    # Rank errors
    rank_errors = np.abs(pred_ranks - true_ranks)

    return np.mean(rank_errors)


def rank_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Rank Error.

    Similar to rank_mae but penalizes large errors more heavily.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    mse : float
        Mean squared rank error. Lower is better.

    Notes
    -----
    MSE variant is better for optimization (smooth gradients) while MAE
    is more interpretable for evaluation.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_ranks = stats.rankdata(y_true, method='average')
    pred_ranks = stats.rankdata(y_pred, method='average')

    rank_errors = pred_ranks - true_ranks

    return np.mean(rank_errors ** 2)


def rank_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Rank Error.

    Same scale as rank_mae but with MSE's property of penalizing large errors.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    rmse : float
        Root mean squared rank error. Lower is better.
    """
    return np.sqrt(rank_mse(y_true, y_pred))


# ============================================================================
# Position-Weighted Variants
# ============================================================================

def weighted_rank_mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weight_fn: str = 'top_heavy'
) -> float:
    """
    Position-weighted rank MSE.

    Weights rank errors by position importance. Useful when top-K accuracy
    matters more than overall ranking.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.
    weight_fn : str, default='top_heavy'
        Weighting scheme:
        - 'top_heavy': 1/sqrt(rank) - Top positions weighted heavily
        - 'linear': (n - rank + 1) / n - Linear decay
        - 'quadratic': ((n - rank + 1) / n)² - Quadratic decay
        - 'top_k': 1 for top 10%, 0.1 for rest - Focus only on top

    Returns
    -------
    weighted_mse : float
        Weighted mean squared rank error. Lower is better.

    Examples
    --------
    >>> y_true = np.array([1, 2, 3, 4, 5])
    >>> y_pred = np.array([1, 2, 3, 5, 4])  # Swap #4 and #5
    >>> rank_mse(y_true, y_pred)  # Standard
    0.4
    >>> weighted_rank_mse(y_true, y_pred, 'top_heavy')  # Weighted
    0.2  # Lower because error is at bottom
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    true_ranks = stats.rankdata(y_true, method='average')
    pred_ranks = stats.rankdata(y_pred, method='average')
    rank_errors = (pred_ranks - true_ranks) ** 2

    # Calculate weights based on true ranks
    if weight_fn == 'top_heavy':
        weights = 1.0 / np.sqrt(true_ranks)
    elif weight_fn == 'linear':
        weights = (n - true_ranks + 1) / n
    elif weight_fn == 'quadratic':
        weights = ((n - true_ranks + 1) / n) ** 2
    elif weight_fn == 'top_k':
        top_k = int(n * 0.1)
        weights = np.where(true_ranks >= (n - top_k + 1), 1.0, 0.1)
    else:
        raise ValueError(f"Unknown weight_fn: {weight_fn}")

    # Normalize weights to sum to n (so scale matches unweighted)
    weights = weights / weights.sum() * n

    return np.mean(weights * rank_errors)


# ============================================================================
# Correlation-Based Metrics
# ============================================================================

def spearman_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman correlation as a loss (for minimization).

    Returns 1 - correlation so lower is better.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    loss : float
        1 - Spearman correlation. Range [0, 2].
        - 0.0 = perfect correlation (ρ=1)
        - 1.0 = no correlation (ρ=0)
        - 2.0 = perfect negative correlation (ρ=-1)

    Notes
    -----
    Not differentiable, so can't be used for gradient-based optimization.
    Use rank_mse for that instead.
    """
    corr, _ = stats.spearmanr(y_true, y_pred)
    return 1.0 - corr


# ============================================================================
# Hybrid Losses (MSE + Ranking)
# ============================================================================

def combined_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rank_weight: float = 0.5,
    mse_weight: float = 1.0
) -> float:
    """
    Combined MSE and ranking loss.

    Loss = mse_weight * MSE + rank_weight * RankMSE

    Balances:
    - Predicting reasonable magnitudes (MSE)
    - Getting relative ordering correct (rank)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.
    rank_weight : float, default=0.5
        Weight for ranking term.
    mse_weight : float, default=1.0
        Weight for MSE term.

    Returns
    -------
    loss : float
        Combined loss. Lower is better.

    Examples
    --------
    >>> # Pure MSE (standard regression)
    >>> loss = combined_loss(y_true, y_pred, rank_weight=0.0)

    >>> # Pure ranking (ignores magnitudes)
    >>> loss = combined_loss(y_true, y_pred, rank_weight=1.0, mse_weight=0.0)

    >>> # Balanced (default)
    >>> loss = combined_loss(y_true, y_pred, rank_weight=0.5, mse_weight=1.0)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # MSE component
    mse = np.mean((y_true - y_pred) ** 2)

    # Rank MSE component
    rank_mse_val = rank_mse(y_true, y_pred)

    return mse_weight * mse + rank_weight * rank_mse_val


# ============================================================================
# Top-K Specific Metrics
# ============================================================================

def top_k_overlap(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """
    Fraction of overlap between top-K true and predicted.

    Measures: "Of the K best stocks, how many did I correctly identify?"

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.
    k : int, default=10
        Number of top items to consider.

    Returns
    -------
    overlap : float
        Fraction of overlap, range [0.0, 1.0].
        - 1.0 = perfect overlap (all K correct)
        - 0.5 = half correct
        - 0.0 = no overlap (0 correct)

    Examples
    --------
    >>> y_true = np.array([1, 2, 3, 4, 5])
    >>> y_pred = np.array([1, 2, 3, 4, 5])  # Perfect
    >>> top_k_overlap(y_true, y_pred, k=2)
    1.0  # Got both top 2 correct

    >>> y_pred = np.array([5, 4, 3, 2, 1])  # Reversed
    >>> top_k_overlap(y_true, y_pred, k=2)
    0.0  # Got 0 of top 2 correct
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Get indices of top K
    true_top_k = set(np.argsort(y_true)[-k:])
    pred_top_k = set(np.argsort(y_pred)[-k:])

    # Count overlap
    overlap_count = len(true_top_k.intersection(pred_top_k))

    return overlap_count / k


def top_k_rank_mae(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """
    Rank MAE calculated only on top-K stocks.

    Measures: "Among the K best stocks, how accurate is my ranking?"

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.
    k : int, default=10
        Number of top items to consider.

    Returns
    -------
    mae : float
        Mean absolute rank error within top K. Lower is better.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Get indices of true top K
    true_top_k_idx = np.argsort(y_true)[-k:]

    # Calculate ranks only for those indices
    y_true_subset = y_true[true_top_k_idx]
    y_pred_subset = y_pred[true_top_k_idx]

    return rank_mae(y_true_subset, y_pred_subset)


# ============================================================================
# Decile Analysis
# ============================================================================

def decile_spread(y_true: np.ndarray, y_pred: np.ndarray, n_deciles: int = 10) -> float:
    """
    Difference between top and bottom decile returns.

    Measures signal strength: "Do my top predictions actually perform better?"

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.
    n_deciles : int, default=10
        Number of deciles to compute.

    Returns
    -------
    spread : float
        Mean return of top decile - mean return of bottom decile.
        - Positive = signal present (top > bottom)
        - Negative = inverted signal (bottom > top)
        - ~0 = no signal

    Examples
    --------
    >>> y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> y_pred = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Perfect
    >>> decile_spread(y_true, y_pred, n_deciles=10)
    9.0  # Top decile (10) - bottom decile (1)

    >>> y_pred = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])  # Inverted
    >>> decile_spread(y_true, y_pred, n_deciles=10)
    -9.0  # Negative spread = bad signal
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Get decile size
    decile_size = len(y_true) // n_deciles

    # Sort by predicted
    sorted_idx = np.argsort(y_pred)

    # Top and bottom decile indices
    top_decile_idx = sorted_idx[-decile_size:]
    bottom_decile_idx = sorted_idx[:decile_size]

    # Calculate actual returns for each decile
    top_return = np.mean(y_true[top_decile_idx])
    bottom_return = np.mean(y_true[bottom_decile_idx])

    return top_return - bottom_return


# ============================================================================
# All-in-one Evaluation
# ============================================================================

def ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> dict:
    """
    Calculate comprehensive ranking metrics.

    Convenience function that computes all relevant ranking metrics at once.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.
    k : int, default=10
        Number of top items for top-K metrics.

    Returns
    -------
    metrics : dict
        Dictionary with all ranking metrics:
        - rank_mae: Mean absolute rank error
        - rank_rmse: Root mean squared rank error
        - spearman: Spearman correlation
        - top_k_overlap: Fraction of top K identified correctly
        - top_k_rank_mae: Rank MAE within top K
        - decile_spread: Top decile - bottom decile returns

    Examples
    --------
    >>> metrics = ranking_metrics(y_true, y_pred, k=10)
    >>> print(f"Rank MAE: {metrics['rank_mae']:.2f}")
    >>> print(f"Top-10 overlap: {metrics['top_k_overlap']:.1%}")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    spearman_corr, _ = stats.spearmanr(y_true, y_pred)

    return {
        'rank_mae': rank_mae(y_true, y_pred),
        'rank_rmse': rank_rmse(y_true, y_pred),
        'spearman': spearman_corr,
        'top_k_overlap': top_k_overlap(y_true, y_pred, k=k),
        'top_k_rank_mae': top_k_rank_mae(y_true, y_pred, k=k),
        'decile_spread': decile_spread(y_true, y_pred, n_deciles=10),
    }


# ============================================================================
# Example / Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RANKING METRICS - EXAMPLES")
    print("=" * 70)

    # Example 1: Perfect ranking
    print("\n1. Perfect Ranking:")
    y_true = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_pred = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    metrics = ranking_metrics(y_true, y_pred, k=3)
    print(f"   Rank MAE: {metrics['rank_mae']:.2f}")
    print(f"   Spearman: {metrics['spearman']:.3f}")
    print(f"   Top-3 overlap: {metrics['top_k_overlap']:.1%}")
    print(f"   Decile spread: {metrics['decile_spread']:.2f}")

    # Example 2: Slightly off ranking
    print("\n2. Slightly Off (swap 4<->5, 9<->10):")
    y_pred = np.array([1, 2, 3, 5, 4, 6, 7, 8, 10, 9])

    metrics = ranking_metrics(y_true, y_pred, k=3)
    print(f"   Rank MAE: {metrics['rank_mae']:.2f}")
    print(f"   Spearman: {metrics['spearman']:.3f}")
    print(f"   Top-3 overlap: {metrics['top_k_overlap']:.1%}")
    print(f"   Decile spread: {metrics['decile_spread']:.2f}")

    # Example 3: Inverted ranking
    print("\n3. Inverted Ranking:")
    y_pred = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

    metrics = ranking_metrics(y_true, y_pred, k=3)
    print(f"   Rank MAE: {metrics['rank_mae']:.2f}")
    print(f"   Spearman: {metrics['spearman']:.3f}")
    print(f"   Top-3 overlap: {metrics['top_k_overlap']:.1%}")
    print(f"   Decile spread: {metrics['decile_spread']:.2f}")

    # Example 4: Random ranking
    print("\n4. Random Ranking:")
    np.random.seed(42)
    y_pred = np.random.permutation(y_true)

    metrics = ranking_metrics(y_true, y_pred, k=3)
    print(f"   Rank MAE: {metrics['rank_mae']:.2f}")
    print(f"   Spearman: {metrics['spearman']:.3f}")
    print(f"   Top-3 overlap: {metrics['top_k_overlap']:.1%}")
    print(f"   Decile spread: {metrics['decile_spread']:.2f}")

    print("\n" + "=" * 70)
    print("Note: Rank MAE ≈ n/2 for random ranking")
    print("=" * 70)