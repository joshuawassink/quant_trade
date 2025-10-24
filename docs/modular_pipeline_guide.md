# Modular Pipeline Architecture Guide

## Overview

The ML pipeline has been refactored into a modular, Airflow-style architecture where each step is a standalone, reusable component. This makes it easy to:

- Customize pipelines for different use cases (30d returns, 60d returns, classification, etc.)
- Swap out components (e.g., use different models or preprocessing)
- Debug and optimize individual steps independently
- Run partial pipelines (just training, just evaluation, etc.)

## Architecture

### Pipeline Components (`src/pipeline/`)

Each component is a self-contained module with a clear interface:

1. **`data_loading.py`** - Load raw data from storage
   - Class: `DataLoader`
   - Purpose: Load price, market, financials, metadata from parquet files
   - Output: Dictionary with DataFrames

2. **`feature_engineering.py`** - Compute features from raw data
   - Class: `FeatureEngineer`
   - Purpose: Use `FeatureAligner` to compute technical, fundamental, sector features
   - Output: DataFrame with all features

3. **`target_generation.py`** - Create target variables
   - Class: `TargetGenerator`
   - Purpose: Compute forward returns (30d, 60d, etc.)
   - Output: DataFrame with target column added

4. **`data_filtering.py`** - Filter and save training data
   - Class: `DataFilter`
   - Purpose: Remove rows with null targets, critical features; save to parquet
   - Output: Path to saved training data

5. **`model_training.py`** - Train ML models
   - Class: `ModelTrainer`
   - Purpose: Train with time-series CV, save model artifacts
   - Output: CV scores, saved model

6. **`model_evaluation.py`** - ML diagnostics (Step 7)
   - Class: `ModelEvaluator`
   - Purpose: R², RMSE, error distribution, temporal/sector analysis
   - Output: Metrics, visualizations, predictions.parquet

7. **`model_returns.py`** - Financial performance (Step 8)
   - Class: `ModelReturnsAnalyzer`
   - Purpose: Translate predictions → trading returns (quintile, top-N strategies)
   - Output: Financial metrics (Sharpe, max drawdown, returns)

### Workflow Orchestrators (`workflows/`)

Each workflow file orchestrates a complete pipeline by composing components.

**`wf_30d_returns.py`** - 30-day returns prediction pipeline

```python
# Full pipeline
python workflows/wf_30d_returns.py --full

# Just training (assumes data exists)
python workflows/wf_30d_returns.py --train

# Just evaluation (assumes model exists)
python workflows/wf_30d_returns.py --evaluate

# Custom model
python workflows/wf_30d_returns.py --full --model-type quantile --quantile 0.6
```

## Pipeline Flow

```
Step 1: Load Data
  ├─ Load price data for universe symbols
  ├─ Load market data (SPY, VIX, sector ETFs)
  ├─ Load quarterly financials
  └─ Load company metadata (sectors)
      ↓
Step 2: Feature Engineering
  ├─ Technical features (RSI, SMA, volatility, etc.)
  ├─ Fundamental features (P/E, ROE, margins, etc.)
  └─ Sector/market features (relative performance, VIX regime)
      ↓
Step 3: Target Generation
  └─ Compute 30-day forward returns vs market
      ↓
Step 4: Data Filtering
  ├─ Filter nulls in target and critical features
  └─ Save training_data_30d_latest.parquet
      ↓
Step 5: Model Training
  ├─ Prepare features and target
  ├─ Time-series cross-validation (5-fold)
  ├─ Train final model on all data
  └─ Save model artifacts (model, preprocessor, feature names)
      ↓
Step 6: ML Evaluation
  ├─ Generate predictions
  ├─ Calculate metrics (R², RMSE, MAE, directional accuracy)
  ├─ Analyze temporal/sector patterns
  ├─ Generate visualizations
  └─ Save predictions.parquet
      ↓
Step 7: Financial Returns
  ├─ Load predictions
  ├─ Calculate strategy returns (quintile, top-N)
  ├─ Calculate financial metrics (Sharpe, drawdown)
  └─ Generate returns report
```

## Creating Custom Workflows

### Example: 60-day returns workflow

```python
# workflows/wf_60d_returns.py

from src.pipeline.data_loading import DataLoader
from src.pipeline.feature_engineering import FeatureEngineer
from src.pipeline.target_generation import TargetGenerator
# ... etc

class Workflow60dReturns:
    def __init__(self):
        self.horizon_days = 60  # Different horizon!
        self.model_type = 'xgboost'  # Different model!

    def step5_train_model(self, training_path):
        # Custom: Use XGBoost instead of Ridge
        trainer = ModelTrainer(
            model_type='xgboost',
            model_params={'max_depth': 5, 'n_estimators': 100}
        )
        # ... rest is the same
```

### Example: Classification workflow (predict direction)

```python
# workflows/wf_direction_classifier.py

class WorkflowDirectionClassifier:
    def step3_generate_target(self, features_df):
        # Custom: Binary target instead of regression
        generator = TargetGenerator()
        df = generator.compute_forward_return(
            df=features_df,
            horizon_days=30,
        )

        # Convert to binary: 1 if return > 0, else 0
        df = df.with_columns([
            (pl.col('forward_return_30d') > 0).cast(pl.Int32).alias('target_direction')
        ])

        return df
```

## Component Interface Design

Each component follows a consistent pattern:

```python
class ComponentName:
    """Component description."""

    def __init__(self, **config):
        """Initialize with configuration."""
        pass

    def main_method(self, input_data):
        """
        Main processing method.

        Args:
            input_data: Clear input specification

        Returns:
            Clear output specification
        """
        logger.info("Doing the thing...")
        # Process
        result = do_work(input_data)
        logger.info("✓ Done")
        return result
```

## Optimization Workflow

Now that the pipeline is modular, you can optimize each step independently:

### 1. Optimize Data Loading
```python
# Experiment with different universes
loader = DataLoader()
data = loader.load_all(get_universe('sp500'))  # Full S&P 500
```

### 2. Optimize Features
```python
# Add new features in feature_engineering.py
engineer = FeatureEngineer()
# Modify FeatureAligner to add Bollinger Bands, volume indicators, etc.
```

### 3. Optimize Target
```python
# Try different horizons, transformations
generator = TargetGenerator()
df = generator.compute_forward_return(horizon_days=60)  # 60-day instead
```

### 4. Optimize Filtering
```python
# Adjust what gets filtered
filter = DataFilter()
df = filter.filter_for_training(
    df=df,
    critical_features=['rsi_14', 'pe_ratio'],  # Custom critical features
)
```

### 5. Optimize Model
```python
# Try different models and hyperparameters
trainer = ModelTrainer(
    model_type='quantile',
    model_params={'quantile': 0.6, 'alpha': 0.5},
)
```

### 6. Optimize Evaluation
```python
# Customize metrics and visualizations
evaluator = ModelEvaluator(model_dir='models/quantile_30d')
# Add custom analysis methods
```

### 7. Optimize Strategy
```python
# Try different trading strategies
analyzer = ModelReturnsAnalyzer.from_parquet(predictions_path)

# Long-only top 10
analyzer.calculate_top_n_strategy(top_n=10)

# Long-short quintile
analyzer.calculate_quintile_strategy(long_quintile=5, short_quintile=1)
```

## Workflow Execution Modes

### Full Pipeline
Runs all steps from data loading to evaluation.

```bash
python workflows/wf_30d_returns.py --full
```

**When to use:**
- First time running
- Data has changed
- Features have changed

### Train Only
Assumes training data already exists.

```bash
python workflows/wf_30d_returns.py --train
```

**When to use:**
- Hyperparameter tuning
- Testing different model types
- Data hasn't changed

### Evaluate Only
Assumes model already exists.

```bash
python workflows/wf_30d_returns.py --evaluate
```

**When to use:**
- Generating new reports
- Running on new data
- Model hasn't changed

## Directory Structure

```
quant_trade/
├── src/
│   └── pipeline/           # Modular components
│       ├── __init__.py
│       ├── data_loading.py
│       ├── feature_engineering.py
│       ├── target_generation.py
│       ├── data_filtering.py
│       ├── model_training.py
│       ├── model_evaluation.py      # Step 7: ML diagnostics
│       └── model_returns.py         # Step 8: Financial returns
│
├── workflows/              # Orchestrators
│   ├── __init__.py
│   ├── wf_30d_returns.py  # 30-day returns pipeline
│   └── wf_60d_returns.py  # (future) 60-day returns
│
├── data/
│   └── training/
│       └── training_data_30d_latest.parquet
│
├── models/
│   ├── ridge_30d/         # Model artifacts by type and horizon
│   │   ├── ridge_model.joblib
│   │   ├── preprocessor.joblib
│   │   ├── feature_names.txt
│   │   └── model_info.txt
│   └── quantile_30d/
│
└── reports/               # Timestamped evaluation reports
    └── 20251024_120000/
        ├── predictions.parquet        # For step 8
        ├── evaluation_report.txt      # ML metrics
        ├── error_distribution.png
        ├── temporal_performance.png
        ├── sector_performance.png
        └── returns/                   # Financial analysis
            ├── returns_report.txt
            └── cumulative_returns.png
```

## Benefits of Modular Design

1. **Reusability** - Components work across different workflows
2. **Testability** - Test each component independently
3. **Maintainability** - Changes isolated to specific components
4. **Flexibility** - Easy to swap implementations (Ridge → XGBoost)
5. **Debugging** - Run individual steps to isolate issues
6. **Parallelization** - Future: Run multiple workflows concurrently
7. **Version Control** - Track changes to specific components
8. **Documentation** - Each component self-documenting

## Next Steps

1. **Add more model types** - XGBoost, LightGBM, Neural Networks
2. **Create new workflows** - 60d returns, volatility prediction, sector rotation
3. **Add preprocessing variants** - Different scaling, imputation strategies
4. **Add feature selection** - Automated feature importance filtering
5. **Add hyperparameter tuning** - Grid search, Bayesian optimization
6. **Add ensemble methods** - Combine multiple models
7. **Add live prediction** - Production inference pipeline
8. **Add monitoring** - Model drift detection, performance tracking

## Example: Running Your First Modular Workflow

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run full 30-day pipeline with Ridge
python workflows/wf_30d_returns.py --full

# 3. Check outputs
ls models/ridge_30d/
ls reports/

# 4. Try quantile regression (addresses negative skew)
python workflows/wf_30d_returns.py --train --model-type quantile --quantile 0.6

# 5. Evaluate the quantile model
python workflows/wf_30d_returns.py --evaluate

# 6. Compare results
cat reports/*/evaluation_report.txt
```

## Troubleshooting

### "Training data not found"
Run with `--full` to generate training data first.

### "Model not found"
Run `--train` before `--evaluate`.

### "Module not found"
Activate virtual environment: `source .venv/bin/activate`

### Step fails partway through
Use individual step methods in Python REPL for debugging:

```python
from workflows.wf_30d_returns import Workflow30dReturns

wf = Workflow30dReturns()

# Run steps individually
data = wf.step1_load_data()
features = wf.step2_engineer_features(data)
# ... etc
```
