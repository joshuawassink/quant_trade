# Quantitative Trading Pipeline Architecture

## Overview

This is a machine learning pipeline for predicting stock returns. The architecture follows a modular design where **reusable components** are composed into **workflow orchestrators** that execute end-to-end pipelines.

## Core Concepts

### 1. **DataProvider**
Abstracts data sources (yfinance, local parquet files, API endpoints). Each provider knows how to fetch and format its specific data type.

- **Location:** `src/shared/data/`
- **Key implementations:** `price.py`, `fundamentals.py`, `macro.py`, `technical.py`, `vix.py`
- **Usage:** Workflows call providers to fetch raw data

### 2. **FeatureAligner**
Ensures all feature DataFrames share the same `(date, ticker)` keys using time-series aware joins. Critical for preventing data leakage.

- **Location:** `src/shared/features/alignment.py`
- **Key method:** `align_features()` - outer join on date/ticker
- **Usage:** Step 2 of workflows (after data loading)

### 3. **Pipeline Component**
A single-responsibility module that performs one logical step (e.g., load data, engineer features, train model). Each component has clear input/output contracts.

- **Location:** `src/shared/pipeline/`
- **Pattern:** Class-based with a primary method (`load()`, `engineer()`, `train()`, etc.)
- **Composability:** Workflows chain components together

### 4. **Workflow Orchestrator**
An executable script that composes pipeline components into an end-to-end machine learning workflow. Think of it like an Airflow DAG, but in pure Python.

- **Location:** `workflows/`
- **Pattern:** Class with `stepN_*()` methods that call components
- **Main method:** `run_full_pipeline()` executes all steps in sequence
- **Customization:** Each workflow can customize component behavior or use workflow-specific versions

### 5. **Time-Series Train/Test Split**
A chronological split (no shuffle) that respects temporal ordering. Prevents lookahead bias by ensuring test data comes after training data.

- **Location:** `src/shared/pipeline/data_splitting.py`
- **Default:** 80% train / 20% test
- **Critical:** Must be sorted by date before splitting

### 6. **Prediction Artifact**
Predictions saved as parquet files (not just in-memory). This allows evaluation to be separate from prediction, enabling reusability and reproducibility.

- **Format:** Parquet with columns: `date`, `ticker`, `predicted_return`, `actual_return`
- **Location:** `data/predictions/`
- **Usage:** Step 7 generates, Step 8 evaluates

### 7. **Ranking-Aware Models**
Custom models optimized for rank-ordering accuracy rather than absolute prediction accuracy. Critical insight: in portfolio selection, correctly ranking stocks matters more than predicting exact returns.

- **Location:** `src/shared/models/`
- **Key modules:**
  - `ranking_metrics.py` - Standalone ranking metrics (rank MAE, rank MSE, Spearman, top-K overlap)
  - `ranking_sgd.py` - SGD with custom ranking loss
  - `ranking_xgb.py` - XGBoost with ranking objectives
- **Use case:** When selecting top K stocks for a portfolio, getting the relative ordering correct is more valuable than precise return predictions
- **Evaluation:** Combines traditional metrics (MSE, R²) with ranking metrics (Spearman correlation, precision@K)

## Directory Structure

```
quant_trade/
├── src/
│   ├── shared/                  # Reusable code (general-purpose)
│   │   ├── pipeline/            # 11 pipeline components
│   │   ├── features/            # Feature engineering modules
│   │   ├── models/              # Model configurations
│   │   ├── data/                # Data providers
│   │   └── config/              # Configuration files
│   └── workflows/               # Workflow-specific customizations
│       ├── returns_30d/         # (future) 30-day return workflow customizations
│       ├── returns_60d/         # (future) 60-day return workflow customizations
│       └── volatility/          # (future) volatility prediction customizations
├── workflows/                   # Executable workflow orchestrators
│   ├── wf_30d_returns_v2.py    # Main: 9-step train/test workflow
│   └── wf_30d_returns.py       # Legacy: original workflow (no test split)
├── scripts/                     # Utility scripts (data fetching, analysis)
├── data/                        # Data storage
│   ├── raw/                     # Raw data from providers
│   ├── processed/               # Intermediate pipeline outputs
│   └── predictions/             # Saved predictions for evaluation
├── models/                      # Trained model artifacts
│   └── baseline/                # Ridge regression model + preprocessor
├── docs/                        # Documentation
└── reports/                     # Analysis reports and visualizations
```

## Module Reference

### Pipeline Components (`src/shared/pipeline/`)

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `data_loading.py` | Load parquet data | Data paths | polars DataFrame |
| `feature_engineering.py` | Compute technical indicators | Price data | Features DataFrame |
| `target_generation.py` | Calculate forward returns | Price + metadata | Target column added |
| `data_filtering.py` | Remove invalid rows | Any DataFrame | Filtered DataFrame |
| `data_splitting.py` | Train/test split | DataFrame | Two parquet files (train/test) |
| `model_training.py` | Train sklearn models | Training data path | Trained model + CV scores |
| `model_prediction.py` | Generate predictions | Test data path | Predictions parquet |
| `model_evaluation_v2.py` | Evaluate predictions | Predictions path | Metrics dict |
| `returns_analysis.py` | Financial backtest | Predictions path | Returns metrics |
| `feature_selection.py` | Select top K features | Features + target | Reduced feature set |
| `model_persistence.py` | Save/load models | Model object | Serialized model |

### Feature Modules (`src/shared/features/`)

| File | Purpose | Key Functionality |
|------|---------|-------------------|
| `alignment.py` | Align feature DataFrames | `FeatureAligner.align_features()` |
| `fundamental.py` | Fundamental indicators | P/E ratio, debt/equity, ROE |
| `technical.py` | Technical indicators | RSI, MACD, Bollinger Bands |
| `macro.py` | Macro features | VIX, treasury yields, sentiment |
| `targets.py` | Target variable generation | Forward returns (30d, 60d, etc.) |

### Data Providers (`src/shared/data/`)

| File | Data Source | Provides |
|------|-------------|----------|
| `price.py` | yfinance | Daily OHLCV data |
| `fundamentals.py` | yfinance | Balance sheet, income statement |
| `macro.py` | FRED API | Economic indicators |
| `technical.py` | Computed | Technical indicators from price |
| `vix.py` | yfinance | VIX (volatility index) |
| `metadata.py` | Local | Stock universe definitions |

### Configuration (`src/shared/config/`)

| File | Purpose |
|------|---------|
| `universe.py` | Defines stock universe (e.g., S&P 500 tickers) |
| `settings.py` | Model hyperparameters, paths, constants |

### Ranking Models (`src/shared/models/`)

| File | Purpose | Key Functionality |
|------|---------|-------------------|
| `ranking_metrics.py` | Ranking evaluation metrics | `rank_mae()`, `rank_mse()`, `top_k_overlap()`, `decile_spread()` |
| `ranking_sgd.py` | SGD with ranking loss | `RankingSGDRegressor` - Combines MSE + rank MSE loss |
| `ranking_xgb.py` | XGBoost with ranking | `RankingXGBRegressor`, `LambdaRankXGBRegressor` |

**Key Ranking Metrics:**
- **rank_mae**: Mean absolute rank error (e.g., 1.8 = avg 2 positions off)
- **rank_mse**: Mean squared rank error (for optimization)
- **spearman**: Rank correlation (-1 to 1, higher = better ranking)
- **top_k_overlap**: Fraction of top K correctly identified (0 to 1)
- **decile_spread**: Top decile return - bottom decile return (signal strength)

**When to use:**
- Standard models (Ridge, XGBoost): When absolute predictions matter
- Ranking models: When selecting top K stocks for equal-weight portfolio
- Hybrid approach: Combine MSE + ranking loss for balanced performance

## Data Flow

### V2 Workflow (with train/test split)

```
Step 1: Load Data
  └─> Raw parquet files → polars DataFrame

Step 2: Engineer Features
  └─> Price data → Technical indicators + Fundamental features

Step 3: Generate Target
  └─> Price data → Forward returns (e.g., 30-day)

Step 4: Filter Data
  └─> Remove nulls, outliers, low-volume stocks

Step 5: Split Data
  └─> Chronological split → train.parquet + test.parquet

Step 6: Train Model
  └─> train.parquet → Trained model + CV scores

Step 7: Predict
  └─> test.parquet + trained model → predictions.parquet

Step 8: Evaluate (ML Metrics)
  └─> predictions.parquet → RMSE, MAE, R², correlation

Step 9: Returns Analysis (Financial Metrics)
  └─> predictions.parquet → Sharpe ratio, total return, drawdown
```

### Key Data Transformations

1. **Raw → Features**: Data providers fetch raw data, feature engineers compute indicators
2. **Features → Training Set**: Feature aligner ensures consistent keys, filter removes invalid data
3. **Training Set → Model**: Time-series CV trains model on historical data
4. **Model + Test Set → Predictions**: Model generates predictions on held-out data
5. **Predictions → Metrics**: Separate evaluation of ML performance vs financial performance

## Entry Points

### Main Workflow (Recommended)
```bash
cd /Users/jwassink/repos/quant_trade
source .venv/bin/activate
python workflows/wf_30d_returns_v2.py --test-size 0.2
```

### Utility Scripts
```bash
# Fetch fresh data
python scripts/fetch_production_data.py

# Analyze missing data
python scripts/analyze_missing_data.py

# Update daily data
python scripts/update_daily_data.py
```

## Design Patterns

### 1. Component Pattern
Each pipeline component is a class with:
- `__init__()` - Configuration
- Primary method (e.g., `load()`, `train()`, `predict()`)
- Clear input/output contracts (usually polars DataFrames or paths)

### 2. Workflow Orchestration Pattern
Workflows compose components using numbered step methods:
```python
class Workflow30dReturnsV2:
    def run_full_pipeline(self):
        data = self.step1_load_data()
        features = self.step2_engineer_features(data)
        target = self.step3_generate_target(features)
        # ... etc
```

### 3. Artifact Persistence Pattern
Save intermediate outputs to disk (parquet) for:
- **Debugging**: Inspect intermediate data
- **Reusability**: Re-run evaluation without re-training
- **Reproducibility**: Exact predictions are versioned

### 4. Shared + Workflow-Specific Pattern
- `src/shared/` contains general-purpose code
- `src/workflows/<workflow_name>/` can override with custom versions
- Workflows import from `src.shared.*` by default

## Import Conventions

### Standard Imports
```python
# Pipeline components
from src.shared.pipeline.data_loading import DataLoader
from src.shared.pipeline.model_training import ModelTrainer

# Features
from src.shared.features.alignment import FeatureAligner
from src.shared.features.fundamental import FundamentalFeatures

# Data providers
from src.shared.data.price import PriceProvider
from src.shared.data.macro import MacroProvider

# Config
from src.shared.config.universe import SP500_TICKERS
```

### Workflow-Specific Imports (Future)
```python
# Use custom version if exists, else fall back to shared
try:
    from src.workflows.returns_30d.custom_evaluator import CustomEvaluator
except ImportError:
    from src.shared.pipeline.model_evaluation_v2 import ModelEvaluatorV2 as CustomEvaluator
```

## Testing Strategy

### Component Testing
Each component can be tested independently:
```python
# Test data loader
loader = DataLoader(data_path="data/processed/training_data.parquet")
df = loader.load()
assert df is not None
assert 'date' in df.columns
```

### Integration Testing
Test workflow end-to-end:
```bash
python workflows/wf_30d_returns_v2.py --test-size 0.2
# Check outputs exist:
ls data/predictions/
ls models/baseline/
```

### Validation Checks
- **No lookahead bias**: Test data dates > train data dates
- **No data leakage**: Features computed only from historical data
- **Realistic performance**: Metrics computed on held-out test set

## Key Differences: V1 vs V2 Workflow

| Aspect | V1 (wf_30d_returns.py) | V2 (wf_30d_returns_v2.py) |
|--------|------------------------|---------------------------|
| Train/Test Split | ❌ No split (evaluates on training data) | ✅ 80/20 chronological split |
| Prediction Artifact | ❌ Not saved | ✅ Saved to parquet |
| Evaluation | Combined ML + financial | Separate: Step 8 (ML) + Step 9 (financial) |
| Steps | 7 steps | 9 steps |
| Production Ready | ❌ No (overfits) | ✅ Yes (realistic metrics) |

## Common Operations

### Adding a New Feature
1. Create feature function in `src/shared/features/<module>.py`
2. Update `step2_engineer_features()` in workflow to include new feature
3. Re-run workflow to retrain model with new feature

### Creating a New Workflow
1. Copy `workflows/wf_30d_returns_v2.py` to `workflows/wf_<target>.py`
2. Update target generation (step 3)
3. Optionally customize other steps (e.g., different model, filters)
4. Create `src/workflows/<target>/` for workflow-specific overrides

### Tuning Hyperparameters
1. Edit `src/shared/pipeline/model_training.py`
2. Modify `param_grid` in `train()` method
3. Re-run workflow to retrain with new hyperparameters

### Changing Test Size
```bash
python workflows/wf_30d_returns_v2.py --test-size 0.3  # 30% test set
```

## File Relationships

### Critical Dependencies
```
wf_30d_returns_v2.py
├── imports src.shared.pipeline.data_loading
├── imports src.shared.pipeline.feature_engineering
├── imports src.shared.pipeline.target_generation
├── imports src.shared.pipeline.data_filtering
├── imports src.shared.pipeline.data_splitting
├── imports src.shared.pipeline.model_training
├── imports src.shared.pipeline.model_prediction
├── imports src.shared.pipeline.model_evaluation_v2
└── imports src.shared.pipeline.returns_analysis

model_evaluation_v2.py
└── imports src.shared.pipeline.returns_analysis (for financial metrics)

feature_engineering.py
├── imports src.shared.features.technical
├── imports src.shared.features.fundamental
└── imports src.shared.features.alignment
```

## Performance Considerations

### Memory
- **polars**: Used for efficient DataFrame operations (faster than pandas)
- **Lazy evaluation**: Use `pl.scan_parquet()` for large files when possible
- **Chunking**: Not currently implemented (future optimization)

### Speed
- **Bottleneck**: Feature engineering (technical indicators on full history)
- **Optimization**: Consider caching computed features to parquet
- **Parallelization**: Time-series CV in model training can't be parallelized (sequential folds)

### Disk Space
- **Raw data**: ~500MB per year for S&P 500
- **Features**: ~2GB for full training set
- **Models**: ~10MB for Ridge regression (sparse)

## Troubleshooting

### Common Issues

**"ModuleNotFoundError"**
- Ensure virtual environment is activated
- Check imports use `src.shared.*` (not `src.*`)

**"No such file or directory"**
- Run data fetching scripts first: `python scripts/fetch_production_data.py`
- Check paths are absolute or relative to project root

**"ValueError: No common dates"**
- Features have misaligned dates
- Check `FeatureAligner.align_features()` is called in step 2

**Poor model performance**
- Check for data leakage (forward-looking features)
- Verify train/test split is chronological (no shuffle)
- Inspect feature distributions for outliers

## Next Steps / Roadmap

### Immediate (High Priority)
- [ ] Test V2 workflow end-to-end in virtual environment
- [ ] Validate predictions file exists and has expected schema
- [ ] Compare V1 vs V2 metrics to confirm V2 is more realistic

### Short Term
- [ ] Add workflow-specific customizations to `src/workflows/returns_30d/`
- [ ] Implement 60-day and 90-day return workflows
- [ ] Add feature importance analysis to evaluation

### Medium Term
- [ ] Implement volatility prediction workflow
- [ ] Add ensemble models (Random Forest, Gradient Boosting)
- [ ] Create dashboard for monitoring model performance

### Long Term
- [ ] Production deployment with automated retraining
- [ ] Real-time prediction API
- [ ] Multi-asset support (crypto, commodities, etc.)

## References

### Key Documentation Files
- `docs/v2_workflow_summary.md` - V2 workflow detailed guide
- `docs/modular_pipeline_guide.md` - Pipeline architecture deep dive
- `docs/pipeline_quick_reference.md` - Common commands cheatsheet
- `docs/missing_data_strategy.md` - Handling missing data
- `docs/model_evaluation_summary.md` - Model evaluation best practices

### External Resources
- [Polars Documentation](https://pola-rs.github.io/polars/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)
