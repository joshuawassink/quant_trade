"""
30-Day Returns Prediction Workflow (v2)

Refactored workflow with proper train/test split and separated prediction/evaluation.

Pipeline Steps:
1. Data Loading - Load raw data
2. Feature Engineering - Compute features
3. Target Generation - Create targets
4. Data Filtering - Remove nulls
5. Data Splitting - Train/test split (80/20)
6. Model Training - Train on train set
7. Model Prediction - Predict on test set → save predictions
8. Model Evaluation - Evaluate predictions (ML metrics)
9. Model Returns - Financial performance analysis

Usage:
    # Full pipeline
    python workflows/wf_30d_returns_v2.py --full

    # Just predict + evaluate (model exists, test set exists)
    python workflows/wf_30d_returns_v2.py --predict-eval
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
from src.shared.pipeline.data_splitting import DataSplitter
from src.shared.pipeline.model_training import ModelTrainer
from src.shared.pipeline.model_prediction import ModelPredictor
from src.shared.pipeline.model_evaluation_v2 import ModelEvaluatorV2
from src.shared.pipeline.model_returns import ModelReturnsAnalyzer

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO")


class Workflow30dReturnsV2:
    """30-day returns prediction workflow (v2) with proper train/test split."""

    def __init__(
        self,
        universe: str = 'production',
        horizon_days: int = 30,
        model_type: str = 'ridge',
        model_params: Optional[Dict[str, Any]] = None,
        test_size: float = 0.2,
    ):
        """
        Initialize workflow.

        Args:
            universe: Stock universe
            horizon_days: Prediction horizon
            model_type: Model type ('ridge', 'quantile')
            model_params: Model parameters
            test_size: Test set size (0.2 = 20%)
        """
        self.universe = universe
        self.horizon_days = horizon_days
        self.model_type = model_type
        self.model_params = model_params or {}
        self.test_size = test_size

        # Paths
        self.data_root = project_root / "data"
        self.training_dir = self.data_root / "training"
        self.model_dir = project_root / "models" / f"{model_type}_{horizon_days}d"
        self.reports_dir = project_root / "reports"

        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("30-DAY RETURNS PREDICTION WORKFLOW (v2)")
        logger.info("=" * 80)
        logger.info(f"Universe:      {universe}")
        logger.info(f"Horizon:       {horizon_days} days")
        logger.info(f"Model Type:    {model_type}")
        logger.info(f"Test Size:     {test_size*100:.0f}%")
        logger.info("=" * 80)

    def step1_load_data(self) -> Dict:
        """Step 1: Load Data"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: LOAD DATA")
        logger.info("=" * 80)

        symbols = get_universe(self.universe)
        logger.info(f"Universe: {len(symbols)} stocks")

        loader = DataLoader(data_root=self.data_root)
        data = loader.load_all(symbols)

        logger.info("✓ Step 1 complete")
        return data

    def step2_engineer_features(self, data: Dict):
        """Step 2: Feature Engineering"""
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

        logger.info("✓ Step 2 complete")
        return features_df

    def step3_generate_target(self, features_df):
        """Step 3: Target Generation"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: TARGET GENERATION")
        logger.info("=" * 80)

        generator = TargetGenerator()
        df_with_target = generator.compute_forward_return(
            df=features_df,
            horizon_days=self.horizon_days,
            market_relative=True,
        )

        logger.info("✓ Step 3 complete")
        return df_with_target

    def step4_filter_data(self, df_with_target):
        """Step 4: Data Filtering"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: DATA FILTERING")
        logger.info("=" * 80)

        target_col = f'target_return_{self.horizon_days}d_vs_market'

        filter_step = DataFilter(output_dir=self.training_dir)
        filtered_df = filter_step.filter_for_training(
            df=df_with_target,
            target_col=target_col,
        )

        logger.info("✓ Step 4 complete")
        return filtered_df

    def step5_split_data(self, filtered_df):
        """Step 5: Train/Test Split"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: TRAIN/TEST SPLIT")
        logger.info("=" * 80)

        splitter = DataSplitter(output_dir=self.training_dir)
        train_path, test_path = splitter.split_and_save(
            df=filtered_df,
            horizon_days=self.horizon_days,
            test_size=self.test_size,
        )

        logger.info("✓ Step 5 complete")
        return train_path, test_path

    def step6_train_model(self, train_path: Path):
        """Step 6: Model Training"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: MODEL TRAINING")
        logger.info("=" * 80)

        import polars as pl

        # Load training data
        df = pl.read_parquet(train_path)
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

        # Train final model on all training data
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

        logger.info("✓ Step 6 complete")
        return cv_scores

    def step7_predict(self, test_path: Path):
        """Step 7: Generate Predictions on Test Set"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: GENERATE PREDICTIONS (Test Set)")
        logger.info("=" * 80)

        predictor = ModelPredictor(model_dir=self.model_dir)
        predictor.load_model()

        # Predict and save to model directory
        predictions_path = self.model_dir / "predictions_test.parquet"
        predictions_df = predictor.predict(
            data_path=str(test_path),
            output_path=predictions_path,
        )

        logger.info("✓ Step 7 complete")
        return predictions_path

    def step8_evaluate(self, predictions_path: Path):
        """Step 8: Evaluate Predictions (ML Metrics)"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 8: EVALUATE PREDICTIONS")
        logger.info("=" * 80)

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.reports_dir / timestamp

        evaluator = ModelEvaluatorV2(predictions_path=predictions_path)
        metrics = evaluator.run_evaluation(output_dir=output_dir)

        logger.info("✓ Step 8 complete")
        return metrics, output_dir

    def step9_returns(self, predictions_path: Path):
        """Step 9: Financial Returns Analysis"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 9: FINANCIAL RETURNS ANALYSIS")
        logger.info("=" * 80)

        analyzer = ModelReturnsAnalyzer.from_parquet(predictions_path)

        # Calculate strategy returns
        analyzer.calculate_quintile_strategy(
            long_quintile=5,
            short_quintile=1,
            long_only=False,
        )

        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        returns_dir = self.reports_dir / timestamp / "returns"
        analyzer.generate_report(returns_dir)

        logger.info("✓ Step 9 complete")
        return analyzer.calculate_performance_metrics()

    def run_full_pipeline(self):
        """Run complete pipeline from data to evaluation."""
        logger.info("\n" + "#" * 80)
        logger.info("RUNNING FULL PIPELINE")
        logger.info("#" * 80)

        start_time = datetime.now()

        # Steps 1-5: Data pipeline
        data = self.step1_load_data()
        features_df = self.step2_engineer_features(data)
        df_with_target = self.step3_generate_target(features_df)
        filtered_df = self.step4_filter_data(df_with_target)
        train_path, test_path = self.step5_split_data(filtered_df)

        # Step 6: Training
        cv_scores = self.step6_train_model(train_path)

        # Step 7: Prediction
        predictions_path = self.step7_predict(test_path)

        # Step 8-9: Evaluation
        ml_metrics, eval_dir = self.step8_evaluate(predictions_path)
        returns_metrics = self.step9_returns(predictions_path)

        elapsed = datetime.now() - start_time

        logger.info("\n" + "#" * 80)
        logger.info("FULL PIPELINE COMPLETE")
        logger.info("#" * 80)
        logger.info(f"Time elapsed: {elapsed}")
        logger.info(f"Model saved to: {self.model_dir}")
        logger.info(f"Predictions saved to: {predictions_path}")
        logger.info(f"Reports saved to: {eval_dir}")
        logger.info("#" * 80)

        return {
            'cv_scores': cv_scores,
            'ml_metrics': ml_metrics,
            'returns_metrics': returns_metrics,
        }

    def run_predict_eval(self):
        """Run prediction and evaluation (assumes model and test set exist)."""
        logger.info("\n" + "#" * 80)
        logger.info("RUNNING PREDICTION + EVALUATION")
        logger.info("#" * 80)

        test_path = self.training_dir / f"test_{self.horizon_days}d_latest.parquet"

        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")

        # Predict
        predictions_path = self.step7_predict(test_path)

        # Evaluate
        ml_metrics, eval_dir = self.step8_evaluate(predictions_path)
        returns_metrics = self.step9_returns(predictions_path)

        logger.info("\n" + "#" * 80)
        logger.info("PREDICTION + EVALUATION COMPLETE")
        logger.info("#" * 80)
        logger.info(f"Reports saved to: {eval_dir}")

        return {
            'ml_metrics': ml_metrics,
            'returns_metrics': returns_metrics,
        }


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="30-day returns prediction workflow (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Workflow mode
    parser.add_argument('--full', action='store_true', help='Run full pipeline')
    parser.add_argument('--predict-eval', action='store_true', help='Run prediction + evaluation only')

    # Configuration
    parser.add_argument('--universe', type=str, default='production', help='Stock universe')
    parser.add_argument('--horizon', type=int, default=30, help='Prediction horizon (days)')
    parser.add_argument('--model-type', type=str, default='ridge', choices=['ridge', 'quantile'],
                       help='Model type')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size, default 0.2 is 20 percent')

    # Model parameters
    parser.add_argument('--alpha', type=float, default=1.0, help='Ridge alpha')
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
    workflow = Workflow30dReturnsV2(
        universe=args.universe,
        horizon_days=args.horizon,
        model_type=args.model_type,
        model_params=model_params,
        test_size=args.test_size,
    )

    # Run workflow
    if args.full:
        results = workflow.run_full_pipeline()
    elif args.predict_eval:
        results = workflow.run_predict_eval()
    else:
        # Default: full pipeline
        logger.info("No mode specified, running full pipeline (use --help for options)")
        results = workflow.run_full_pipeline()

    return results


if __name__ == "__main__":
    results = main()
