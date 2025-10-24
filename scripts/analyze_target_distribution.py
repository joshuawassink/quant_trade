"""
Analyze target variable distribution to understand negative error skew.

This helps us understand why the model underestimates positive moves.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO")

sns.set_style("whitegrid")


def analyze_target_distribution():
    """Analyze target variable distribution."""
    logger.info("Loading training data...")

    df = pl.read_parquet("data/training/training_data_30d_latest.parquet")
    target = df['target_return_30d_vs_market'].to_numpy()

    # Remove nulls
    target = target[~np.isnan(target)]

    logger.info(f"Analyzing {len(target):,} returns...")

    # Calculate statistics
    stats_dict = {
        'mean': np.mean(target),
        'median': np.median(target),
        'std': np.std(target),
        'skew': stats.skew(target),
        'kurtosis': stats.kurtosis(target),
        'min': np.min(target),
        'max': np.max(target),
        'q01': np.percentile(target, 1),
        'q05': np.percentile(target, 5),
        'q25': np.percentile(target, 25),
        'q75': np.percentile(target, 75),
        'q95': np.percentile(target, 95),
        'q99': np.percentile(target, 99),
    }

    # Print statistics
    print("\n" + "="*70)
    print("TARGET DISTRIBUTION ANALYSIS")
    print("="*70)
    print(f"Mean:           {stats_dict['mean']:.4f}")
    print(f"Median:         {stats_dict['median']:.4f}")
    print(f"Std Dev:        {stats_dict['std']:.4f}")
    print(f"Skewness:       {stats_dict['skew']:.4f}")
    print(f"Kurtosis:       {stats_dict['kurtosis']:.4f}")
    print()
    print("Percentiles:")
    print(f"  Min:          {stats_dict['min']:.4f}")
    print(f"  1%:           {stats_dict['q01']:.4f}")
    print(f"  5%:           {stats_dict['q05']:.4f}")
    print(f"  25%:          {stats_dict['q25']:.4f}")
    print(f"  50% (median): {stats_dict['median']:.4f}")
    print(f"  75%:          {stats_dict['q75']:.4f}")
    print(f"  95%:          {stats_dict['q95']:.4f}")
    print(f"  99%:          {stats_dict['q99']:.4f}")
    print(f"  Max:          {stats_dict['max']:.4f}")
    print()

    # Interpretation
    print("INTERPRETATION")
    print("="*70)
    if stats_dict['skew'] > 0.5:
        print(f"✓ Target is RIGHT-SKEWED ({stats_dict['skew']:.2f})")
        print("  → More extreme positive returns than negative")
        print("  → This is typical for stocks (unlimited upside, limited downside)")
    elif stats_dict['skew'] < -0.5:
        print(f"⚠ Target is LEFT-SKEWED ({stats_dict['skew']:.2f})")
        print("  → More extreme negative returns than positive")
        print("  → Unusual for stocks, may indicate data issue")
    else:
        print(f"✓ Target is approximately SYMMETRIC ({stats_dict['skew']:.2f})")

    print()
    if stats_dict['kurtosis'] > 5:
        print(f"⚠ Heavy tails detected (kurtosis={stats_dict['kurtosis']:.2f})")
        print("  → Many extreme returns (outliers)")
        print("  → Normal distribution has kurtosis=3")
        print("  → Suggests need for robust methods")

    print()
    print("IMPLICATIONS FOR MODEL ERROR SKEW:")
    print("="*70)
    print("Model error skew = -1.22 (underestimates positive moves)")
    print(f"Target skew = {stats_dict['skew']:.2f}")
    print()

    if stats_dict['skew'] > 0.5:
        print("DIAGNOSIS:")
        print("  Target IS right-skewed (positive tail)")
        print("  But model UNDERESTIMATES the positive tail")
        print()
        print("ROOT CAUSE:")
        print("  1. Ridge regression minimizes squared errors")
        print("     → Treats over/under prediction equally")
        print("  2. With right-skewed targets, MSE loss is conservative")
        print("     → Pulls predictions toward mean to avoid large squared errors")
        print("  3. Outlier clipping at 99.9% may remove big winners")
        print()
        print("SOLUTIONS:")
        print("  ✓ Use asymmetric loss (penalize underestimation more)")
        print("  ✓ Use quantile regression (predict 60th-75th percentile)")
        print("  ✓ Use log-transform (reduce skewness)")
        print("  ✓ Adjust clipping (keep more positive outliers)")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Histogram
    ax = axes[0, 0]
    ax.hist(target, bins=100, edgecolor='black', alpha=0.7)
    ax.axvline(stats_dict['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats_dict['mean']:.4f}")
    ax.axvline(stats_dict['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats_dict['median']:.4f}")
    ax.set_xlabel('Return', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Target Distribution (Skew={stats_dict["skew"]:.2f})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Box plot
    ax = axes[0, 1]
    ax.boxplot(target, vert=True)
    ax.set_ylabel('Return', fontsize=12)
    ax.set_title('Box Plot (Outlier Detection)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Q-Q plot
    ax = axes[1, 0]
    stats.probplot(target, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot vs Normal Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 4. Cumulative distribution
    ax = axes[1, 1]
    sorted_target = np.sort(target)
    cumulative = np.arange(1, len(sorted_target) + 1) / len(sorted_target)
    ax.plot(sorted_target, cumulative, linewidth=2)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Median')
    ax.axvline(0, color='green', linestyle='--', alpha=0.5, label='Zero return')
    ax.set_xlabel('Return', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = project_root / "reports" / "target_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "target_distribution.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"\n✓ Saved visualization to {output_file}")

    # Compare positive vs negative tail
    print("\nTAIL ANALYSIS:")
    print("="*70)
    positive_tail = target[target > stats_dict['q95']]
    negative_tail = target[target < stats_dict['q05']]

    print(f"Positive tail (>95th percentile):")
    print(f"  Size: {len(positive_tail):,} ({len(positive_tail)/len(target)*100:.1f}%)")
    print(f"  Mean: {np.mean(positive_tail):.4f}")
    print(f"  Max:  {np.max(positive_tail):.4f}")

    print(f"\nNegative tail (<5th percentile):")
    print(f"  Size: {len(negative_tail):,} ({len(negative_tail)/len(target)*100:.1f}%)")
    print(f"  Mean: {np.mean(negative_tail):.4f}")
    print(f"  Min:  {np.min(negative_tail):.4f}")

    print(f"\nAsymmetry:")
    print(f"  |Positive tail mean|: {abs(np.mean(positive_tail)):.4f}")
    print(f"  |Negative tail mean|: {abs(np.mean(negative_tail)):.4f}")
    print(f"  Ratio: {abs(np.mean(positive_tail)) / abs(np.mean(negative_tail)):.2f}x")

    if abs(np.mean(positive_tail)) > abs(np.mean(negative_tail)):
        print("\n  → Positive tail is LARGER (right-skewed)")
        print("  → Model should capture these big winners!")

    print("\n" + "="*70)

    return stats_dict


if __name__ == "__main__":
    stats_dict = analyze_target_distribution()
