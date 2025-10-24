"""
30-Day Returns Prediction Workflow

Complete ML pipeline for predicting 30-day market-relative stock returns.

This workflow orchestrates all pipeline steps:
1. Data Loading - Load price, market, financials, metadata
2. Feature Engineering - Compute technical, fundamental, sector features
3. Target Generation - Create 30-day forward returns target
4. Data Filtering - Filter for training readiness
5. Preprocessing - Handle nulls, outliers, scaling
6. Model Training - Train with time-series CV
7. Model Evaluation - ML diagnostics (R², RMSE, error distribution)
8. Model Returns - Financial performance analysis

Usage:
    # Full pipeline (data → model → evaluation)
    python workflows/wf_30d_returns.py --full

    # Just training (assumes data exists)
    python workflows/wf_30d_returns.py --train

    # Just evaluation (assumes model exists)
    python workflows/wf_30d_returns.py --evaluate

    # Custom model type
    python workflows/wf_30d_returns.py --full --model-type quantile --quantile 0.6
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

# Import pipeline components (from shared/)
from src.shared.config.universe import get_universe
from src.shared.pipeline.data_loading import DataLoader
from src.shared.pipeline.feature_engineering import FeatureEngineer
from src.shared.pipeline.target_generation import TargetGenerator
from src.shared.pipeline.data_filtering import DataFilter
from src.shared.pipeline.model_training import ModelTrainer
from src.shared.pipeline.model_evaluation import ModelEvaluator
from src.shared.pipeline.model_returns import ModelReturnsAnalyzer

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")


class Workflow30dReturns:
    """30-day returns prediction workflow orchestrator."""

    def __init__(
        self,
        universe: str = 'production',
        horizon_days: int = 30,
        model_type: str = 'ridge',
        model_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize workflow.

        Args:
            universe: Stock universe ('production', 'test', etc.)
            horizon_days: Prediction horizon in days
            model_type: Type of model ('ridge', 'quantile', etc.)
            model_params: Model-specific parameters
        """
        self.universe = universe
        self.horizon_days = horizon_days
        self.model_type = model_type
        self.model_params = model_params or {}

        # Paths
        self.data_root = project_root / "data"
        self.training_dir = self.data_root / "training"
        self.model_dir = project_root / "models" / f"{model_type}_{horizon_days}d"
        self.reports_dir = project_root / "reports"

        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("30-DAY RETURNS PREDICTION WORKFLOW")
        logger.info("=" * 80)
        logger.info(f"Universe:      {universe}")
        logger.info(f"Horizon:       {horizon_days} days")
        logger.info(f"Model Type:    {model_type}")
        logger.info(f"Model Params:  {model_params}")
        logger.info("=" * 80)

    def step1_load_data(self) -> Dict:
        """
        Step 1: Load Data

        Load all data sources (price, market, financials, metadata).

        Returns:
            Dictionary with loaded dataframes
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: LOAD DATA")
        logger.info("=" * 80)

        symbols = get_universe(self.universe)
        logger.info(f"Universe: {len(symbols)} stocks")

        loader = DataLoader(data_root=self.data_root)
        data = loader.load_all(symbols)

        logger.info("✓ Step 1 complete: Data loaded")
        return data

    def step2_engineer_features(self, data: Dict):
        """
        Step 2: Feature Engineering

        Compute all features from raw data.

        Args:
            data: Dictionary with raw dataframes

        Returns:
            DataFrame with computed features
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("=" * 80)

        engineer = FeatureEngineer()
        features_df = engineer.compute_features(
            price_df=data['price'],
            market_df=data['market'],
            financials_df=data['financials'],
            metadata_df=data['metadata'],
        )

        logger.info("✓ Step 2 complete: Features computed")
        return features_df

    def step3_generate_target(self, features_df):
        """
        Step 3: Target Generation

        Create target variable (30-day forward returns).

        Args:
            features_df: DataFrame with features

        Returns:
            DataFrame with target variable added
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: TARGET GENERATION")
        logger.info("=" * 80)

        generator = TargetGenerator()
        df_with_target = generator.compute_forward_return(
            df=features_df,
            horizon_days=self.horizon_days,
            market_relative=True,
        )

        logger.info("✓ Step 3 complete: Target generated")
        return df_with_target

    def step4_filter_data(self, df_with_target):
        """
        Step 4: Data Filtering

        Filter dataset for training readiness and save.

        Args:
            df_with_target: DataFrame with features and target

        Returns:
            Path to saved training data
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: DATA FILTERING")
        logger.info("=" * 80)

        target_col = f'target_return_{self.horizon_days}d_vs_market'

        filter_step = DataFilter(output_dir=self.training_dir)
        filtered_df = filter_step.filter_for_training(
            df=df_with_target,
            target_col=target_col,
        )

        training_path = filter_step.save_training_data(
            df=filtered_df,
            horizon_days=self.horizon_days,
        )

        logger.info("✓ Step 4 complete: Data filtered and saved")
        return training_path

    def step5_train_model(self, training_path: Path):
        """
        Step 5: Model Training

        Train model with time-series cross-validation.

        Args:
            training_path: Path to training data

        Returns:
            Dictionary with CV scores
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: MODEL TRAINING")
        logger.info("=" * 80)

        import polars as pl

        # Load training data
        df = pl.read_parquet(training_path)
        target_col = f'target_return_{self.horizon_days}d_vs_market'

        # Initialize trainer
        trainer = ModelTrainer(
            model_type=self.model_type,
            model_params=self.model_params,
        )

        # Prepare data
        X, y, feature_names, metadata = trainer.prepare_features_and_target(
            df=df,
            target_col=target_col,
        )

        # Train with CV
        cv_models, cv_scores = trainer.train_with_cv(
            X=X,
            y=y,
            feature_names=feature_names,
            metadata_df=metadata,
            n_splits=5,
        )

        # Train final model on all data
        final_model, final_preprocessor = trainer.train_final_model(
            X=X,
            y=y,
            feature_names=feature_names,
        )

        # Save model
        trainer.save_model(
            model=final_model,
            preprocessor=final_preprocessor,
            feature_names=feature_names,
            output_dir=self.model_dir,
            model_info={
                'horizon_days': self.horizon_days,
                'target': target_col,
                'cv_val_r2': cv_scores['val_r2_mean'],
                'cv_val_mse': cv_scores['val_mse_mean'],
            },
        )

        logger.info("✓ Step 5 complete: Model trained and saved")
        return cv_scores

    def step6_evaluate_ml(self, training_path: Path):
        """
        Step 6: ML Model Evaluation

        Evaluate model with ML diagnostics (R², RMSE, error distribution).

        Args:
            training_path: Path to training data

        Returns:
            Dictionary with ML metrics
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: ML MODEL EVALUATION")
        logger.info("=" * 80)

        target_col = f'target_return_{self.horizon_days}d_vs_market'
        model_filename = f'{self.model_type}_model.joblib'

        evaluator = ModelEvaluator(model_dir=self.model_dir)
        metrics = evaluator.run_evaluation(
            data_path=str(training_path),
            target_col=target_col,
            model_filename=model_filename,
        )

        logger.info("✓ Step 6 complete: ML evaluation done")
        return metrics

    def step7_evaluate_returns(self, predictions_path: Path, strategy_type: str = 'quintile'):
        """
        Step 7: Financial Returns Analysis

        Translate predictions into trading returns.

        Args:
            predictions_path: Path to predictions.parquet from step 6
            strategy_type: Type of strategy ('quintile', 'top_n')

        Returns:
            Dictionary with financial metrics
        """
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: FINANCIAL RETURNS ANALYSIS")
        logger.info("=" * 80)

        analyzer = ModelReturnsAnalyzer.from_parquet(predictions_path)

        # Calculate strategy returns
        if strategy_type == 'quintile':
            analyzer.calculate_quintile_strategy(
                long_quintile=5,
                short_quintile=1,
                long_only=False,
            )
        elif strategy_type == 'top_n':
            analyzer.calculate_top_n_strategy(top_n=10, equal_weight=True)

        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        returns_dir = self.reports_dir / timestamp / "returns"
        analyzer.generate_report(returns_dir)

        logger.info("✓ Step 7 complete: Financial returns analyzed")
        return analyzer.calculate_performance_metrics()

    def run_full_pipeline(self):
        """Run complete pipeline from data to evaluation."""
        logger.info("\n" + "#" * 80)
        logger.info("RUNNING FULL PIPELINE")
        logger.info("#" * 80)

        start_time = datetime.now()

        # Steps 1-4: Data pipeline
        data = self.step1_load_data()
        features_df = self.step2_engineer_features(data)
        df_with_target = self.step3_generate_target(features_df)
        training_path = self.step4_filter_data(df_with_target)

        # Step 5: Training
        cv_scores = self.step5_train_model(training_path)

        # Steps 6-7: Evaluation
        ml_metrics = self.step6_evaluate_ml(training_path)

        # Find predictions file from step 6
        timestamp = datetime.now().strftime("%Y%m%d")
        reports_today = list(self.reports_dir.glob(f"{timestamp}*"))
        if reports_today:
            latest_report = sorted(reports_today)[-1]
            predictions_path = latest_report / "predictions.parquet"

            if predictions_path.exists():
                returns_metrics = self.step7_evaluate_returns(predictions_path)
            else:
                logger.warning("Predictions file not found, skipping returns analysis")
                returns_metrics = None
        else:
            logger.warning("No reports found, skipping returns analysis")
            returns_metrics = None

        elapsed = datetime.now() - start_time

        logger.info("\n" + "#" * 80)
        logger.info("FULL PIPELINE COMPLETE")
        logger.info("#" * 80)
        logger.info(f"Time elapsed: {elapsed}")
        logger.info(f"Model saved to: {self.model_dir}")
        logger.info(f"Reports saved to: {self.reports_dir}")
        logger.info("#" * 80)

        return {
            'cv_scores': cv_scores,
            'ml_metrics': ml_metrics,
            'returns_metrics': returns_metrics,
        }

    def run_train_only(self):
        """Run training only (assumes data exists)."""
        logger.info("\n" + "#" * 80)
        logger.info("RUNNING TRAINING ONLY")
        logger.info("#" * 80)

        training_path = self.training_dir / f"training_data_{self.horizon_days}d_latest.parquet"

        if not training_path.exists():
            raise FileNotFoundError(f"Training data not found: {training_path}")

        cv_scores = self.step5_train_model(training_path)

        logger.info("\n" + "#" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("#" * 80)
        logger.info(f"Model saved to: {self.model_dir}")

        return cv_scores

    def run_evaluate_only(self):
        """Run evaluation only (assumes model exists)."""
        logger.info("\n" + "#" * 80)
        logger.info("RUNNING EVALUATION ONLY")
        logger.info("#" * 80)

        training_path = self.training_dir / f"training_data_{self.horizon_days}d_latest.parquet"

        if not training_path.exists():
            raise FileNotFoundError(f"Training data not found: {training_path}")

        ml_metrics = self.step6_evaluate_ml(training_path)

        # Find latest predictions
        timestamp = datetime.now().strftime("%Y%m%d")
        reports_today = list(self.reports_dir.glob(f"{timestamp}*"))
        if reports_today:
            latest_report = sorted(reports_today)[-1]
            predictions_path = latest_report / "predictions.parquet"

            if predictions_path.exists():
                returns_metrics = self.step7_evaluate_returns(predictions_path)
            else:
                returns_metrics = None
        else:
            returns_metrics = None

        logger.info("\n" + "#" * 80)
        logger.info("EVALUATION COMPLETE")
        logger.info("#" * 80)

        return {
            'ml_metrics': ml_metrics,
            'returns_metrics': returns_metrics,
        }


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="30-day returns prediction workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Workflow mode
    parser.add_argument('--full', action='store_true', help='Run full pipeline (data → model → eval)')
    parser.add_argument('--train', action='store_true', help='Run training only')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation only')

    # Configuration
    parser.add_argument('--universe', type=str, default='production', help='Stock universe')
    parser.add_argument('--horizon', type=int, default=30, help='Prediction horizon (days)')
    parser.add_argument('--model-type', type=str, default='ridge', choices=['ridge', 'quantile'],
                       help='Model type')

    # Model parameters
    parser.add_argument('--alpha', type=float, default=1.0, help='Ridge alpha (L2 regularization)')
    parser.add_argument('--quantile', type=float, default=0.5, help='Quantile for quantile regression')

    args = parser.parse_args()

    # Build model params
    model_params = {}
    if args.model_type == 'ridge':
        model_params['alpha'] = args.alpha
    elif args.model_type == 'quantile':
        model_params['quantile'] = args.quantile
        model_params['alpha'] = args.alpha

    # Initialize workflow
    workflow = Workflow30dReturns(
        universe=args.universe,
        horizon_days=args.horizon,
        model_type=args.model_type,
        model_params=model_params,
    )

    # Run workflow
    if args.full:
        results = workflow.run_full_pipeline()
    elif args.train:
        results = workflow.run_train_only()
    elif args.evaluate:
        results = workflow.run_evaluate_only()
    else:
        # Default: full pipeline
        logger.info("No mode specified, running full pipeline (use --help for options)")
        results = workflow.run_full_pipeline()

    return results


if __name__ == "__main__":
    results = main()
