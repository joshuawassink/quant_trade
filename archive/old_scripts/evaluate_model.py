"""
Model Evaluation and Reporting Script

Generates comprehensive report for trained models including:
- Performance metrics (R², RMSE, MAE, etc.)
- Error distribution analysis
- Temporal performance patterns
- Sector-specific performance
- Feature importance analysis
- Visualizations saved to reports/

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model-dir models/baseline
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import joblib
from loguru import logger

from src.shared.models.preprocessing import FeaturePreprocessor

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ModelEvaluator:
    """Comprehensive model evaluation and reporting."""

    def __init__(self, model_dir: Path):
        """
        Initialize evaluator.

        Args:
            model_dir: Directory containing trained model artifacts
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.predictions_df = None
        self.metrics = {}

    def load_model(self):
        """Load trained model and preprocessor."""
        logger.info("Loading model artifacts...")

        # Load model
        model_file = self.model_dir / "ridge_model.joblib"
        self.model = joblib.load(model_file)
        logger.info(f"  ✓ Loaded model from {model_file}")

        # Load preprocessor
        preprocessor_file = self.model_dir / "preprocessor.joblib"
        self.preprocessor = FeaturePreprocessor.load(preprocessor_file)
        logger.info(f"  ✓ Loaded preprocessor from {preprocessor_file}")

        # Load feature names
        features_file = self.model_dir / "feature_names.txt"
        with open(features_file, 'r') as f:
            self.feature_names = [line.strip() for line in f if line.strip()]
        logger.info(f"  ✓ Loaded {len(self.feature_names)} feature names")

    def load_data(self, data_path: str = "data/training/training_data_30d_latest.parquet"):
        """
        Load training data.

        Args:
            data_path: Path to training data
        """
        logger.info(f"Loading data from {data_path}...")
        self.data_df = pl.read_parquet(data_path)
        logger.info(f"  ✓ Loaded {len(self.data_df):,} rows")

    def prepare_features(self):
        """Prepare features for prediction."""
        logger.info("Preparing features...")

        # Exclude metadata columns
        metadata_cols = [
            'symbol', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume',
            'dividends', 'stock_splits', 'sector'
        ]
        categorical_cols = ['vix_regime']

        feature_cols = [
            col for col in self.data_df.columns
            if col not in metadata_cols
            and col not in categorical_cols
            and not col.startswith('target_')
            and not col.startswith('XL')
            and col != 'spy_close'
        ]

        # Keep only features that were in training
        feature_cols = [col for col in feature_cols if col in self.feature_names]

        X = self.data_df.select(feature_cols).to_numpy()
        y = self.data_df['target_return_30d_vs_market'].to_numpy()

        logger.info(f"  ✓ X shape: {X.shape}")
        logger.info(f"  ✓ y shape: {y.shape}")

        return X, y

    def generate_predictions(self):
        """Generate predictions on all data."""
        logger.info("Generating predictions...")

        X, y = self.prepare_features()

        # Preprocess and predict
        X_processed = self.preprocessor.transform(X)
        y_pred = self.model.predict(X_processed)

        # Create predictions dataframe
        self.predictions_df = self.data_df.select(['symbol', 'date', 'sector', 'target_return_30d_vs_market']).with_columns([
            pl.lit(y_pred).alias('predicted_return'),
            (pl.lit(y_pred) - pl.col('target_return_30d_vs_market')).alias('error'),
            (pl.lit(y_pred) - pl.col('target_return_30d_vs_market')).abs().alias('abs_error'),
        ])

        logger.info(f"  ✓ Generated {len(self.predictions_df):,} predictions")

    def calculate_metrics(self):
        """Calculate comprehensive performance metrics."""
        logger.info("Calculating metrics...")

        y_true = self.predictions_df['target_return_30d_vs_market'].to_numpy()
        y_pred = self.predictions_df['predicted_return'].to_numpy()

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

        # Directional accuracy (did we predict the right direction?)
        correct_direction = np.sign(y_pred) == np.sign(y_true)
        self.metrics['directional_accuracy'] = np.mean(correct_direction) * 100

        logger.info(f"  ✓ Calculated {len(self.metrics)} metrics")

    def analyze_temporal_patterns(self):
        """Analyze performance over time."""
        logger.info("Analyzing temporal patterns...")

        # Group by month
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

        # Group by year
        yearly = self.predictions_df.with_columns([
            pl.col('date').dt.year().alias('year')
        ]).group_by('year').agg([
            pl.len().alias('n_samples'),
            pl.col('error').mean().alias('mean_error'),
            pl.col('abs_error').mean().alias('mae'),
            pl.col('error').std().alias('error_std'),
        ]).sort('year')

        self.yearly_perf = yearly
        logger.info(f"  ✓ Analyzed {len(yearly)} years")

    def analyze_sector_patterns(self):
        """Analyze performance by sector."""
        logger.info("Analyzing sector patterns...")

        if 'sector' not in self.predictions_df.columns:
            logger.warning("  ⚠ No sector information available")
            self.sector_perf = None
            return

        sector_perf = self.predictions_df.group_by('sector').agg([
            pl.len().alias('n_samples'),
            pl.col('error').mean().alias('mean_error'),
            pl.col('abs_error').mean().alias('mae'),
            pl.col('error').std().alias('error_std'),
        ]).sort('mae')

        self.sector_perf = sector_perf
        logger.info(f"  ✓ Analyzed {len(sector_perf)} sectors")

    def plot_error_distribution(self, output_dir: Path):
        """Plot error distribution analysis."""
        logger.info("Plotting error distribution...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        errors = self.predictions_df['error'].to_numpy()
        y_true = self.predictions_df['target_return_30d_vs_market'].to_numpy()
        y_pred = self.predictions_df['predicted_return'].to_numpy()

        # 1. Error histogram
        ax = axes[0, 0]
        ax.hist(errors, bins=100, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.set_xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics text
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

        # Add perfect prediction line
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
        ax.set_ylabel('Residual (Error)', fontsize=12)
        ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_file = output_dir / "error_distribution.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  ✓ Saved to {output_file}")

    def plot_temporal_performance(self, output_dir: Path):
        """Plot performance over time."""
        logger.info("Plotting temporal performance...")

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Monthly MAE
        ax = axes[0]
        monthly_data = self.monthly_perf.to_pandas()
        ax.plot(monthly_data['month'], monthly_data['mae'], marker='o', linewidth=2)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Mean Absolute Error', fontsize=12)
        ax.set_title('Model Performance Over Time (Monthly MAE)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)

        # Add overall MAE line
        ax.axhline(self.metrics['mae'], color='red', linestyle='--', linewidth=2,
                   label=f'Overall MAE: {self.metrics["mae"]:.4f}')
        ax.legend()

        # Monthly error mean
        ax = axes[1]
        ax.plot(monthly_data['month'], monthly_data['mean_error'], marker='o', linewidth=2, color='orange')
        ax.axhline(0, color='red', linestyle='--', linewidth=2, label='No Bias')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Mean Error (Bias)', fontsize=12)
        ax.set_title('Model Bias Over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        ax.legend()

        plt.tight_layout()

        output_file = output_dir / "temporal_performance.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  ✓ Saved to {output_file}")

    def plot_sector_performance(self, output_dir: Path):
        """Plot performance by sector."""
        if self.sector_perf is None:
            logger.warning("  ⚠ Skipping sector plot (no data)")
            return

        logger.info("Plotting sector performance...")

        fig, ax = plt.subplots(figsize=(12, 8))

        sector_data = self.sector_perf.to_pandas()

        # Bar plot of MAE by sector
        bars = ax.barh(sector_data['sector'], sector_data['mae'])

        # Color bars by performance
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

    def analyze_feature_importance(self):
        """Analyze feature importance from Ridge coefficients."""
        logger.info("Analyzing feature importance...")

        # Get feature names after preprocessing
        feature_names_out = self.preprocessor.get_feature_names_out()

        if feature_names_out is None:
            logger.warning("  ⚠ Could not get feature names from preprocessor")
            self.feature_importance = None
            return

        # Get coefficients
        coefficients = self.model.coef_

        # Create importance dataframe
        importance_df = pl.DataFrame({
            'feature': feature_names_out,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort('abs_coefficient', descending=True)

        self.feature_importance = importance_df
        logger.info(f"  ✓ Analyzed {len(importance_df)} features")

    def plot_feature_importance(self, output_dir: Path, top_n: int = 20):
        """Plot feature importance."""
        if self.feature_importance is None:
            logger.warning("  ⚠ Skipping feature importance plot (no data)")
            return

        logger.info(f"Plotting top {top_n} features...")

        fig, ax = plt.subplots(figsize=(12, 10))

        # Get top features
        top_features = self.feature_importance.head(top_n).to_pandas()

        # Bar plot
        colors = ['green' if x > 0 else 'red' for x in top_features['coefficient']]
        bars = ax.barh(top_features['feature'], top_features['coefficient'], color=colors, alpha=0.7)

        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Coefficient Value', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance (Ridge Coefficients)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        output_file = output_dir / "feature_importance.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  ✓ Saved to {output_file}")

    def generate_text_report(self, output_dir: Path):
        """Generate text report with all metrics."""
        logger.info("Generating text report...")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Model: {self.model_dir}")
        report_lines.append("")

        # Dataset info
        report_lines.append("=" * 80)
        report_lines.append("DATASET")
        report_lines.append("=" * 80)
        report_lines.append(f"Total samples:        {len(self.predictions_df):,}")
        report_lines.append(f"Date range:          {self.predictions_df['date'].min()} to {self.predictions_df['date'].max()}")
        report_lines.append(f"Unique symbols:      {self.predictions_df['symbol'].n_unique()}")
        report_lines.append(f"Features used:       {len(self.preprocessor.get_feature_names_out() or self.feature_names)}")
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

        # Prediction statistics
        report_lines.append("=" * 80)
        report_lines.append("PREDICTION STATISTICS")
        report_lines.append("=" * 80)
        report_lines.append(f"Predicted Mean:      {self.metrics['pred_mean']:.4f}")
        report_lines.append(f"Predicted Std:       {self.metrics['pred_std']:.4f}")
        report_lines.append(f"Actual Mean:         {self.metrics['true_mean']:.4f}")
        report_lines.append(f"Actual Std:          {self.metrics['true_std']:.4f}")
        report_lines.append("")

        # Temporal performance
        report_lines.append("=" * 80)
        report_lines.append("TEMPORAL PERFORMANCE (Yearly)")
        report_lines.append("=" * 80)
        for row in self.yearly_perf.iter_rows(named=True):
            report_lines.append(f"{row['year']}: MAE={row['mae']:.4f}, Bias={row['mean_error']:.4f}, n={row['n_samples']:,}")
        report_lines.append("")

        # Sector performance
        if self.sector_perf is not None:
            report_lines.append("=" * 80)
            report_lines.append("SECTOR PERFORMANCE")
            report_lines.append("=" * 80)
            for row in self.sector_perf.iter_rows(named=True):
                report_lines.append(f"{row['sector']:30s}: MAE={row['mae']:.4f}, Bias={row['mean_error']:.4f}, n={row['n_samples']:,}")
            report_lines.append("")

        # Top features
        if self.feature_importance is not None:
            report_lines.append("=" * 80)
            report_lines.append("TOP 20 FEATURES (by |coefficient|)")
            report_lines.append("=" * 80)
            for row in self.feature_importance.head(20).iter_rows(named=True):
                report_lines.append(f"{row['feature']:40s}: {row['coefficient']:+.4f}")
            report_lines.append("")

        # Insights
        report_lines.append("=" * 80)
        report_lines.append("INSIGHTS & RECOMMENDATIONS")
        report_lines.append("=" * 80)

        insights = self.generate_insights()
        for insight in insights:
            report_lines.append(f"• {insight}")

        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)

        # Save report
        output_file = output_dir / "evaluation_report.txt"
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"  ✓ Saved to {output_file}")

        # Also print key metrics to console
        print("\n" + "=" * 80)
        print("KEY METRICS")
        print("=" * 80)
        print(f"R² Score:            {self.metrics['r2']:.4f}")
        print(f"RMSE:                {self.metrics['rmse']:.4f} ({self.metrics['rmse']*100:.2f}%)")
        print(f"MAE:                 {self.metrics['mae']:.4f} ({self.metrics['mae']*100:.2f}%)")
        print(f"Directional Acc:     {self.metrics['directional_accuracy']:.2f}%")
        print(f"Samples:             {len(self.predictions_df):,}")
        print("=" * 80)

    def generate_insights(self) -> list[str]:
        """Generate insights based on analysis."""
        insights = []

        # R² interpretation
        if self.metrics['r2'] < 0:
            insights.append("Model performs worse than predicting the mean - expected for stock returns")
        elif self.metrics['r2'] < 0.1:
            insights.append(f"Low R² ({self.metrics['r2']:.3f}) typical for stock prediction - consider ensemble methods")
        else:
            insights.append(f"Decent R² ({self.metrics['r2']:.3f}) for stock prediction - above baseline")

        # Bias check
        if abs(self.metrics['error_mean']) > 0.01:
            insights.append(f"Model has bias ({self.metrics['error_mean']:+.4f}) - systematically {'over' if self.metrics['error_mean'] > 0 else 'under'}predicting")
        else:
            insights.append("Model is well-calibrated with minimal bias")

        # Error distribution
        if abs(self.metrics['error_skew']) > 1:
            insights.append(f"Error distribution is {'right' if self.metrics['error_skew'] > 0 else 'left'}-skewed - consider robust loss functions")

        if self.metrics['error_kurtosis'] > 5:
            insights.append("Heavy-tailed errors detected - consider outlier handling")

        # Directional accuracy
        if self.metrics['directional_accuracy'] > 52:
            insights.append(f"Strong directional accuracy ({self.metrics['directional_accuracy']:.1f}%) - suitable for long/short strategies")
        elif self.metrics['directional_accuracy'] > 50:
            insights.append(f"Slight directional edge ({self.metrics['directional_accuracy']:.1f}%) - may be tradeable with proper risk management")
        else:
            insights.append(f"No directional edge ({self.metrics['directional_accuracy']:.1f}%) - focus on improving signal quality")

        # Sector analysis
        if self.sector_perf is not None and len(self.sector_perf) > 1:
            best_sector = self.sector_perf[0]
            worst_sector = self.sector_perf[-1]
            mae_range = worst_sector['mae'][0] - best_sector['mae'][0]

            if mae_range > 0.005:
                insights.append(f"Large sector performance variation - consider sector-specific models")
                insights.append(f"  Best: {best_sector['sector'][0]} (MAE={best_sector['mae'][0]:.4f})")
                insights.append(f"  Worst: {worst_sector['sector'][0]} (MAE={worst_sector['mae'][0]:.4f})")

        # Temporal patterns
        recent_mae = self.monthly_perf[-3:]['mae'].mean()
        early_mae = self.monthly_perf[:3]['mae'].mean()

        if recent_mae > early_mae * 1.2:
            insights.append("Performance degrading over time - model may need retraining")
        elif early_mae > recent_mae * 1.2:
            insights.append("Performance improving over time - good generalization to recent data")

        return insights

    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING MODEL EVALUATION")
        logger.info("=" * 80)

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "reports" / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Load artifacts
        self.load_model()
        self.load_data()

        # Generate predictions
        self.generate_predictions()

        # Analysis
        self.calculate_metrics()
        self.analyze_temporal_patterns()
        self.analyze_sector_patterns()
        self.analyze_feature_importance()

        # Generate visualizations
        self.plot_error_distribution(output_dir)
        self.plot_temporal_performance(output_dir)
        self.plot_sector_performance(output_dir)
        self.plot_feature_importance(output_dir, top_n=20)

        # Generate text report
        self.generate_text_report(output_dir)

        logger.info("\n" + "=" * 80)
        logger.info("✓ EVALUATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Reports saved to: {output_dir}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/baseline',
        help='Directory containing trained model'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/training/training_data_30d_latest.parquet',
        help='Path to training data'
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = ModelEvaluator(model_dir=args.model_dir)
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
