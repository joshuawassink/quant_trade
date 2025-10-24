# Pipeline Quick Reference

## Run Workflows

```bash
# Full 30-day returns pipeline
python workflows/wf_30d_returns.py --full

# Train only (data exists)
python workflows/wf_30d_returns.py --train

# Evaluate only (model exists)
python workflows/wf_30d_returns.py --evaluate

# Quantile regression
python workflows/wf_30d_returns.py --full --model-type quantile --quantile 0.6

# Custom universe
python workflows/wf_30d_returns.py --full --universe sp500
```

## Pipeline Steps

| Step | Component | Purpose | Output |
|------|-----------|---------|--------|
| 1 | `data_loading.py` | Load raw data | Dict with DataFrames |
| 2 | `feature_engineering.py` | Compute features | DataFrame with features |
| 3 | `target_generation.py` | Create targets | DataFrame with target col |
| 4 | `data_filtering.py` | Filter & save | training_data.parquet |
| 5 | `model_training.py` | Train model | Saved model artifacts |
| 6 | `model_evaluation.py` | ML diagnostics | Metrics, plots, predictions |
| 7 | `model_returns.py` | Financial returns | Strategy performance |

## Use Components Directly

```python
from src.pipeline.data_loading import DataLoader
from src.pipeline.model_evaluation import ModelEvaluator
from src.pipeline.model_returns import ModelReturnsAnalyzer

# Load data
loader = DataLoader()
data = loader.load_all(symbols=['AAPL', 'MSFT', 'GOOGL'])

# Evaluate model
evaluator = ModelEvaluator(model_dir='models/ridge_30d')
metrics = evaluator.run_evaluation('data/training/training_data_30d_latest.parquet')

# Analyze returns
analyzer = ModelReturnsAnalyzer.from_parquet('reports/20251024_120000/predictions.parquet')
analyzer.calculate_quintile_strategy(long_quintile=5, short_quintile=1)
perf = analyzer.calculate_performance_metrics()
```

## Key Outputs

```
models/ridge_30d/
├── ridge_model.joblib          # Trained model
├── preprocessor.joblib         # Fitted preprocessor
├── feature_names.txt           # Feature list
└── model_info.txt              # Model metadata

reports/20251024_120000/
├── predictions.parquet         # Model predictions (for step 8)
├── evaluation_report.txt       # ML metrics
├── error_distribution.png      # Error analysis
├── temporal_performance.png    # Performance over time
├── sector_performance.png      # Performance by sector
└── returns/
    ├── returns_report.txt      # Financial metrics
    └── cumulative_returns.png  # Returns chart
```

## Model Types

```bash
# Ridge regression (L2 regularization)
--model-type ridge --alpha 1.0

# Quantile regression (for skewed targets)
--model-type quantile --quantile 0.6 --alpha 1.0
```

## Workflow Class Methods

```python
from workflows.wf_30d_returns import Workflow30dReturns

wf = Workflow30dReturns(
    universe='production',
    horizon_days=30,
    model_type='ridge',
    model_params={'alpha': 1.0}
)

# Run individual steps
data = wf.step1_load_data()
features = wf.step2_engineer_features(data)
df_with_target = wf.step3_generate_target(features)
training_path = wf.step4_filter_data(df_with_target)
cv_scores = wf.step5_train_model(training_path)
ml_metrics = wf.step6_evaluate_ml(training_path)
returns_metrics = wf.step7_evaluate_returns(predictions_path)

# Or run full pipeline
results = wf.run_full_pipeline()
```

## Common Customizations

### Different Target
```python
# In target_generation.py
generator = TargetGenerator()
df = generator.compute_forward_return(
    df=features_df,
    horizon_days=60,  # 60-day instead of 30
    market_relative=True
)
```

### Different Features
```python
# In feature_engineering.py
engineer = FeatureEngineer()
# Modify FeatureAligner to add new features
```

### Different Strategy
```python
# In model_returns.py
analyzer = ModelReturnsAnalyzer(predictions_df)

# Top 10 stocks
analyzer.calculate_top_n_strategy(top_n=10, equal_weight=True)

# Quintile spread
analyzer.calculate_quintile_strategy(long_quintile=5, short_quintile=1)
```

## Metrics Reference

### ML Metrics (Step 6)
- **R²** - Variance explained (0.065 = 6.5%)
- **RMSE** - Root mean squared error (0.0865 = 8.65%)
- **MAE** - Mean absolute error
- **Directional Accuracy** - % correct sign prediction (57% = edge)
- **Error Skew** - Error distribution skewness (-1.22 = underestimates big moves)

### Financial Metrics (Step 7)
- **Total Return** - Cumulative strategy return
- **Annual Return** - Annualized return
- **Sharpe Ratio** - Risk-adjusted return
- **Max Drawdown** - Largest peak-to-trough decline
- **Win Rate** - % of positive returns
- **Information Ratio** - Excess return per unit of tracking error
