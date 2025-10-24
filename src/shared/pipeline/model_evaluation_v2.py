"""
Model Evaluation Step (v2)

Evaluate predictions from test set.
This version focuses on evaluation only - predictions are generated separately.

Workflow:
1. Train on train set → save model
2. Predict on test set → save predictions.parquet
3. Evaluate predictions → generate metrics and reports
"""

from pathlib import Path
from datetime import datetime
from typing import Optional
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from loguru import logger

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ModelEvaluatorV2:
    """
    Evaluate model predictions (v2).

    This version takes pre-generated predictions and focuses purely on evaluation.
    Separates prediction from evaluation for cleaner workflow.
    """

    def __init__(self, predictions_path: Path, top_k: int = 10, market_data_path: Optional[Path] = None):
        """
        Initialize evaluator with predictions.

        Args:
            predictions_path: Path to predictions.parquet from ModelPredictor
            top_k: Number of top stocks to evaluate (for precision@K metrics)
            market_data_path: Path to market data parquet (for SPY benchmark).
                            If None, will try default location.
        """
        self.predictions_path = Path(predictions_path)
        self.top_k = top_k
        self.market_data_path = Path(market_data_path) if market_data_path else None
        self.predictions_df = None
        self.market_df = None
        self.metrics = {}

    def load_predictions(self):
        """Load predictions from parquet."""
        logger.info(f"Loading predictions from {self.predictions_path}...")
        self.predictions_df = pl.read_parquet(self.predictions_path)

        # Check required columns
        required = ['predicted_return', 'actual_return']
        missing = [col for col in required if col not in self.predictions_df.columns]

        if missing:
            raise ValueError(f"Predictions missing required columns: {missing}")

        logger.info(f"  ✓ Loaded {len(self.predictions_df):,} predictions")
        logger.info(f"  Date range: {self.predictions_df['date'].min()} to {self.predictions_df['date'].max()}")

    def load_market_data(self):
        """Load market data (SPY) for benchmark comparisons."""
        if self.market_data_path is None:
            # Try default location
            default_path = Path("data/market/daily/market_data_latest.parquet")
            if default_path.exists():
                self.market_data_path = default_path
            else:
                logger.warning("  ⚠ No market data path provided and default not found. Skipping benchmark metrics.")
                return

        logger.info(f"Loading market data from {self.market_data_path}...")
        market_df = pl.read_parquet(self.market_data_path)

        # Filter to SPY only and select relevant columns
        self.market_df = market_df.filter(pl.col('symbol') == 'SPY').select(['date', 'close']).rename({'close': 'spy_close'})

        # Calculate SPY returns (30d forward by default - matches typical prediction horizon)
        # Note: Adjust shift period if using different prediction horizon
        self.market_df = self.market_df.with_columns([
            (pl.col('spy_close').shift(-30) / pl.col('spy_close') - 1).alias('spy_return_30d')
        ])

        # Ensure date column is proper type for joins
        self.market_df = self.market_df.cast({'date': pl.Date})

        logger.info(f"  ✓ Loaded {len(self.market_df)} SPY data points")

    def calculate_metrics(self):
        """Calculate comprehensive ML performance metrics."""
        logger.info("Calculating metrics...")

        y_true = self.predictions_df['actual_return'].to_numpy()
        y_pred = self.predictions_df['predicted_return'].to_numpy()

        # Remove nulls (shouldn't be any in test set, but safety check)
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            raise ValueError("No valid predictions after removing nulls")

        # Basic metrics
        self.metrics['r2'] = r2_score(y_true, y_pred)
        self.metrics['mse'] = mean_squared_error(y_true, y_pred)
        self.metrics['rmse'] = np.sqrt(self.metrics['mse'])
        self.metrics['mae'] = mean_absolute_error(y_true, y_pred)
        self.metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100

        # Error statistics
        errors = y_pred - y_true
        self.metrics['error_mean'] = np.mean(errors)
        self.metrics['error_std'] = np.std(errors)
        self.metrics['error_skew'] = stats.skew(errors)
        self.metrics['error_kurtosis'] = stats.kurtosis(errors)

        # Prediction statistics
        self.metrics['pred_mean'] = np.mean(y_pred)
        self.metrics['pred_std'] = np.std(y_pred)
        self.metrics['true_mean'] = np.mean(y_true)
        self.metrics['true_std'] = np.std(y_true)

        # Directional accuracy
        correct_direction = np.sign(y_pred) == np.sign(y_true)
        self.metrics['directional_accuracy'] = np.mean(correct_direction) * 100

        # Add derived metrics
        self.metrics['n_samples'] = len(y_true)

        # Ranking metrics
        spearman_corr, spearman_p = stats.spearmanr(y_true, y_pred)
        self.metrics['spearman_correlation'] = spearman_corr
        self.metrics['spearman_pvalue'] = spearman_p

        # Top/bottom decile analysis
        n_deciles = 10
        decile_size = len(y_true) // n_deciles
        sorted_idx = np.argsort(y_pred)[::-1]  # Sort by predicted (descending)

        top_decile_idx = sorted_idx[:decile_size]
        bottom_decile_idx = sorted_idx[-decile_size:]

        self.metrics['top_decile_mean_return'] = np.mean(y_true[top_decile_idx])
        self.metrics['bottom_decile_mean_return'] = np.mean(y_true[bottom_decile_idx])
        self.metrics['top_vs_bottom_spread'] = self.metrics['top_decile_mean_return'] - self.metrics['bottom_decile_mean_return']

        logger.info(f"  ✓ Calculated {len(self.metrics)} metrics")

        return self.metrics

    def analyze_temporal_patterns(self):
        """Analyze performance over time."""
        logger.info("Analyzing temporal patterns...")

        # Add error column if not present
        if 'error' not in self.predictions_df.columns:
            self.predictions_df = self.predictions_df.with_columns([
                (pl.col('predicted_return') - pl.col('actual_return')).alias('error'),
                (pl.col('predicted_return') - pl.col('actual_return')).abs().alias('abs_error'),
            ])

        # Monthly performance
        monthly = self.predictions_df.with_columns([
            pl.col('date').dt.strftime('%Y-%m').alias('month')
        ]).group_by('month').agg([
            pl.len().alias('n_samples'),
            pl.col('error').mean().alias('mean_error'),
            pl.col('abs_error').mean().alias('mae'),
            pl.col('error').std().alias('error_std'),
        ]).sort('month')

        self.monthly_perf = monthly
        logger.info(f"  ✓ Analyzed {len(monthly)} months")

        return monthly

    def analyze_sector_patterns(self):
        """Analyze performance by sector."""
        if 'sector' not in self.predictions_df.columns:
            logger.warning("  ⚠ No sector information available")
            self.sector_perf = None
            return None

        logger.info("Analyzing sector patterns...")

        # Ensure error columns exist
        if 'error' not in self.predictions_df.columns:
            self.predictions_df = self.predictions_df.with_columns([
                (pl.col('predicted_return') - pl.col('actual_return')).alias('error'),
                (pl.col('predicted_return') - pl.col('actual_return')).abs().alias('abs_error'),
            ])

        sector_perf = self.predictions_df.group_by('sector').agg([
            pl.len().alias('n_samples'),
            pl.col('error').mean().alias('mean_error'),
            pl.col('abs_error').mean().alias('mae'),
            pl.col('error').std().alias('error_std'),
        ]).sort('mae')

        self.sector_perf = sector_perf
        logger.info(f"  ✓ Analyzed {len(sector_perf)} sectors")

        return sector_perf

    def calculate_precision_at_k(self):
        """
        Calculate precision@K metrics for top K predicted stocks.

        Evaluates:
        1. Average return of top K stocks (equal weight)
        2. % of top K with positive returns
        3. % of top K that beat the market (SPY)
        4. Average outperformance vs SPY

        Note: This calculates top K per date (daily rebalancing scenario)
        """
        logger.info(f"Calculating Precision@{self.top_k} metrics...")

        # For each date, get top K stocks by predicted return
        top_k_df = (
            self.predictions_df
            .sort(['date', 'predicted_return'], descending=[False, True])
            .group_by('date', maintain_order=True)
            .head(self.top_k)
        )

        # 1. Average return (equal weight across all top K picks)
        avg_return = top_k_df['actual_return'].mean()
        self.metrics[f'precision_at_{self.top_k}_avg_return'] = avg_return

        # 2. % positive returns
        pct_positive = (top_k_df['actual_return'] > 0).mean() * 100
        self.metrics[f'precision_at_{self.top_k}_pct_positive'] = pct_positive

        # 3. % beat market (if we have SPY data)
        if self.market_df is not None:
            # Join SPY returns to top K stocks
            top_k_with_spy = top_k_df.join(
                self.market_df.select(['date', 'spy_return_30d']),
                on='date',
                how='left'
            )

            # Calculate % that beat SPY
            pct_beat_market = (
                top_k_with_spy['actual_return'] > top_k_with_spy['spy_return_30d']
            ).mean() * 100

            self.metrics[f'precision_at_{self.top_k}_pct_beat_spy'] = pct_beat_market

            # 4. Average outperformance
            avg_outperformance = (
                top_k_with_spy['actual_return'] - top_k_with_spy['spy_return_30d']
            ).mean()
            self.metrics[f'precision_at_{self.top_k}_avg_outperformance'] = avg_outperformance

            logger.info(f"  ✓ Top {self.top_k} avg return: {avg_return:.4f} ({avg_return*100:.2f}%)")
            logger.info(f"  ✓ Top {self.top_k} % positive: {pct_positive:.1f}%")
            logger.info(f"  ✓ Top {self.top_k} % beat SPY: {pct_beat_market:.1f}%")
        else:
            logger.info(f"  ✓ Top {self.top_k} avg return: {avg_return:.4f} ({avg_return*100:.2f}%)")
            logger.info(f"  ✓ Top {self.top_k} % positive: {pct_positive:.1f}%")
            logger.warning("  ⚠ No market data available for SPY comparison")

        return self.metrics

    def plot_error_distribution(self, output_dir: Path):
        """Plot comprehensive error distribution analysis."""
        logger.info("Plotting error distribution...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        y_true = self.predictions_df['actual_return'].to_numpy()
        y_pred = self.predictions_df['predicted_return'].to_numpy()
        errors = y_pred - y_true

        # 1. Error histogram
        ax = axes[0, 0]
        ax.hist(errors, bins=100, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Prediction Error', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        stats_text = f"Mean: {self.metrics['error_mean']:.4f}\nStd: {self.metrics['error_std']:.4f}\nSkew: {self.metrics['error_skew']:.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 2. Q-Q plot
        ax = axes[0, 1]
        stats.probplot(errors, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Normal Distribution)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 3. Predicted vs Actual
        ax = axes[1, 0]
        ax.scatter(y_true, y_pred, alpha=0.3, s=10)

        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        ax.set_xlabel('Actual Return', fontsize=12)
        ax.set_ylabel('Predicted Return', fontsize=12)
        ax.set_title(f'Predicted vs Actual (R²={self.metrics["r2"]:.3f})', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Residuals vs Predicted
        ax = axes[1, 1]
        ax.scatter(y_pred, errors, alpha=0.3, s=10)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted Return', fontsize=12)
        ax.set_ylabel('Residual', fontsize=12)
        ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_file = output_dir / "error_distribution.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  ✓ Saved to {output_file}")

    def plot_temporal_performance(self, output_dir: Path):
        """Plot performance over time."""
        if not hasattr(self, 'monthly_perf') or self.monthly_perf is None:
            logger.warning("  ⚠ No temporal data, skipping plot")
            return

        logger.info("Plotting temporal performance...")

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        monthly_data = self.monthly_perf.to_pandas()

        # Monthly MAE
        ax = axes[0]
        ax.plot(monthly_data['month'], monthly_data['mae'], marker='o', linewidth=2)
        ax.axhline(self.metrics['mae'], color='red', linestyle='--', linewidth=2,
                   label=f'Overall MAE: {self.metrics["mae"]:.4f}')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Mean Absolute Error', fontsize=12)
        ax.set_title('Model Performance Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        # Monthly bias
        ax = axes[1]
        ax.plot(monthly_data['month'], monthly_data['mean_error'], marker='o', linewidth=2, color='orange')
        ax.axhline(0, color='red', linestyle='--', linewidth=2, label='No Bias')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Mean Error (Bias)', fontsize=12)
        ax.set_title('Model Bias Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        output_file = output_dir / "temporal_performance.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  ✓ Saved to {output_file}")

    def plot_sector_performance(self, output_dir: Path):
        """Plot performance by sector."""
        if not hasattr(self, 'sector_perf') or self.sector_perf is None:
            logger.warning("  ⚠ No sector data, skipping plot")
            return

        logger.info("Plotting sector performance...")

        fig, ax = plt.subplots(figsize=(12, 8))

        sector_data = self.sector_perf.to_pandas()

        bars = ax.barh(sector_data['sector'], sector_data['mae'])

        colors = ['green' if x < self.metrics['mae'] else 'orange' for x in sector_data['mae']]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            bar.set_alpha(0.7)

        ax.axvline(self.metrics['mae'], color='red', linestyle='--', linewidth=2,
                   label=f'Overall MAE: {self.metrics["mae"]:.4f}')

        ax.set_xlabel('Mean Absolute Error', fontsize=12)
        ax.set_ylabel('Sector', fontsize=12)
        ax.set_title('Model Performance by Sector', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        output_file = output_dir / "sector_performance.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  ✓ Saved to {output_file}")

    def generate_text_report(self, output_dir: Path):
        """Generate comprehensive text report."""
        logger.info("Generating text report...")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ML MODEL EVALUATION REPORT (Test Set)")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Predictions: {self.predictions_path}")
        report_lines.append("")

        # Dataset info
        report_lines.append("=" * 80)
        report_lines.append("TEST SET")
        report_lines.append("=" * 80)
        report_lines.append(f"Total samples:        {self.metrics['n_samples']:,}")
        report_lines.append(f"Date range:          {self.predictions_df['date'].min()} to {self.predictions_df['date'].max()}")
        report_lines.append(f"Unique symbols:      {self.predictions_df['symbol'].n_unique()}")
        report_lines.append("")

        # Performance metrics
        report_lines.append("=" * 80)
        report_lines.append("PERFORMANCE METRICS")
        report_lines.append("=" * 80)
        report_lines.append(f"R² Score:            {self.metrics['r2']:.4f}")
        report_lines.append(f"RMSE:                {self.metrics['rmse']:.4f} ({self.metrics['rmse']*100:.2f}%)")
        report_lines.append(f"MAE:                 {self.metrics['mae']:.4f} ({self.metrics['mae']*100:.2f}%)")
        report_lines.append(f"MAPE:                {self.metrics['mape']:.2f}%")
        report_lines.append(f"Directional Acc:     {self.metrics['directional_accuracy']:.2f}%")
        report_lines.append("")

        # Error statistics
        report_lines.append("=" * 80)
        report_lines.append("ERROR STATISTICS")
        report_lines.append("=" * 80)
        report_lines.append(f"Mean Error (Bias):   {self.metrics['error_mean']:.4f}")
        report_lines.append(f"Error Std Dev:       {self.metrics['error_std']:.4f}")
        report_lines.append(f"Error Skewness:      {self.metrics['error_skew']:.4f}")
        report_lines.append(f"Error Kurtosis:      {self.metrics['error_kurtosis']:.4f}")
        report_lines.append("")

        # Ranking metrics
        report_lines.append("=" * 80)
        report_lines.append("RANKING METRICS")
        report_lines.append("=" * 80)
        report_lines.append(f"Spearman Correlation: {self.metrics['spearman_correlation']:.4f} (p={self.metrics['spearman_pvalue']:.4e})")
        report_lines.append(f"Top Decile Return:   {self.metrics['top_decile_mean_return']:.4f} ({self.metrics['top_decile_mean_return']*100:.2f}%)")
        report_lines.append(f"Bottom Decile Return: {self.metrics['bottom_decile_mean_return']:.4f} ({self.metrics['bottom_decile_mean_return']*100:.2f}%)")
        report_lines.append(f"Top vs Bottom Spread: {self.metrics['top_vs_bottom_spread']:.4f} ({self.metrics['top_vs_bottom_spread']*100:.2f}%)")
        report_lines.append("")

        # Precision@K metrics
        report_lines.append("=" * 80)
        report_lines.append(f"PRECISION@{self.top_k} METRICS (Top {self.top_k} Stocks)")
        report_lines.append("=" * 80)
        report_lines.append(f"Average Return:      {self.metrics[f'precision_at_{self.top_k}_avg_return']:.4f} ({self.metrics[f'precision_at_{self.top_k}_avg_return']*100:.2f}%)")
        report_lines.append(f"% Positive Returns:  {self.metrics[f'precision_at_{self.top_k}_pct_positive']:.2f}%")
        if f'precision_at_{self.top_k}_pct_beat_spy' in self.metrics:
            report_lines.append(f"% Beat SPY:          {self.metrics[f'precision_at_{self.top_k}_pct_beat_spy']:.2f}%")
            report_lines.append(f"Avg Outperformance:  {self.metrics[f'precision_at_{self.top_k}_avg_outperformance']:.4f} ({self.metrics[f'precision_at_{self.top_k}_avg_outperformance']*100:.2f}%)")
        report_lines.append("")

        # Sector performance
        if hasattr(self, 'sector_perf') and self.sector_perf is not None:
            report_lines.append("=" * 80)
            report_lines.append("SECTOR PERFORMANCE")
            report_lines.append("=" * 80)
            for row in self.sector_perf.iter_rows(named=True):
                report_lines.append(f"{row['sector']:30s}: MAE={row['mae']:.4f}, n={row['n_samples']:,}")
            report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)

        # Save report
        output_file = output_dir / "evaluation_report.txt"
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"  ✓ Saved to {output_file}")

    def run_evaluation(self, output_dir: Optional[Path] = None) -> dict:
        """
        Run complete evaluation pipeline.

        Args:
            output_dir: Directory to save outputs. If None, creates timestamped dir.

        Returns:
            Dictionary with metrics
        """
        logger.info("=" * 70)
        logger.info("STARTING MODEL EVALUATION (Test Set)")
        logger.info("=" * 70)

        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("reports") / f"eval_{timestamp}"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Load and analyze
        self.load_predictions()
        self.load_market_data()
        metrics = self.calculate_metrics()
        self.analyze_temporal_patterns()
        self.analyze_sector_patterns()
        self.calculate_precision_at_k()

        # Generate visualizations
        self.plot_error_distribution(output_dir)
        self.plot_temporal_performance(output_dir)
        self.plot_sector_performance(output_dir)

        # Generate text report
        self.generate_text_report(output_dir)

        # Print key metrics
        print("\n" + "=" * 70)
        print("TEST SET EVALUATION METRICS")
        print("=" * 70)
        print(f"R² Score:            {metrics['r2']:.4f}")
        print(f"RMSE:                {metrics['rmse']:.4f} ({metrics['rmse']*100:.2f}%)")
        print(f"MAE:                 {metrics['mae']:.4f} ({metrics['mae']*100:.2f}%)")
        print(f"Directional Acc:     {metrics['directional_accuracy']:.2f}%")
        print(f"Error Skew:          {metrics['error_skew']:.4f}")
        print(f"Samples:             {metrics['n_samples']:,}")
        print("=" * 70)

        logger.info("\n✓ Evaluation complete")
        logger.info(f"Reports saved to: {output_dir}")

        return metrics
