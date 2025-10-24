# Pipeline Architecture Diagram

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         WORKFLOW LAYER                               │
│                      (workflows/wf_*.py)                             │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │         Workflow30dReturns (Orchestrator)                  │    │
│  │                                                            │    │
│  │  Composes components into complete pipeline               │    │
│  │  Manages state, paths, configuration                      │    │
│  │  Provides CLI interface                                   │    │
│  └────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ uses
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        COMPONENT LAYER                               │
│                      (src/pipeline/*.py)                             │
│                                                                      │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────────┐          │
│  │ DataLoader    │  │FeatureEngineer│  │TargetGenerator │          │
│  │               │  │               │  │                │          │
│  │ Load raw data │→ │ Compute       │→ │ Create target  │          │
│  │ from parquet  │  │ features      │  │ variables      │          │
│  └───────────────┘  └───────────────┘  └────────────────┘          │
│         │                   │                   │                   │
│         └───────────────────┴───────────────────┘                   │
│                             │                                       │
│                             ▼                                       │
│                    ┌────────────────┐                               │
│                    │  DataFilter    │                               │
│                    │                │                               │
│                    │  Filter & save │                               │
│                    │  training data │                               │
│                    └────────────────┘                               │
│                             │                                       │
│                             ▼                                       │
│                    ┌────────────────┐                               │
│                    │ ModelTrainer   │                               │
│                    │                │                               │
│                    │ Train model    │                               │
│                    │ with CV        │                               │
│                    └────────────────┘                               │
│                             │                                       │
│                             ▼                                       │
│           ┌─────────────────┴─────────────────┐                    │
│           │                                   │                    │
│           ▼                                   ▼                    │
│  ┌────────────────┐                  ┌────────────────┐            │
│  │ModelEvaluator  │                  │ModelReturns    │            │
│  │ (Step 6)       │                  │Analyzer        │            │
│  │                │                  │ (Step 7)       │            │
│  │ ML diagnostics │                  │ Financial      │            │
│  │ R², RMSE, MAE  │                  │ performance    │            │
│  │ Error analysis │                  │ Sharpe, returns│            │
│  └────────────────┘                  └────────────────┘            │
│         │                                     │                    │
│         └─────────────────┬───────────────────┘                    │
│                           │                                        │
│                           ▼                                        │
│                   ┌────────────────┐                               │
│                   │   Reports &    │                               │
│                   │   Metrics      │                               │
│                   └────────────────┘                               │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ uses
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      FOUNDATION LAYER                                │
│                    (src/features, src/models)                        │
│                                                                      │
│  ┌───────────────┐  ┌─────────────────┐  ┌────────────────────┐   │
│  │FeatureAligner │  │FeaturePreproc.  │  │ ML Models          │   │
│  │               │  │                 │  │ (Ridge, Quantile)  │   │
│  │ Technical     │  │ Imputation      │  │                    │   │
│  │ Fundamental   │  │ Scaling         │  │ sklearn            │   │
│  │ Sector        │  │ Outlier clipping│  │                    │   │
│  └───────────────┘  └─────────────────┘  └────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
Step 1: Load Data
┌──────────────────────────────────────────────┐
│ DataLoader.load_all(symbols)                 │
│                                              │
│ ┌─────────────┐  ┌──────────────┐          │
│ │ Price Data  │  │ Market Data  │          │
│ │ (OHLCV)     │  │ (SPY, VIX,   │          │
│ │             │  │  Sectors)    │          │
│ └─────────────┘  └──────────────┘          │
│                                              │
│ ┌─────────────┐  ┌──────────────┐          │
│ │ Financials  │  │  Metadata    │          │
│ │ (Quarterly) │  │  (Sectors)   │          │
│ └─────────────┘  └──────────────┘          │
└──────────────────────────────────────────────┘
                    │
                    ▼
Step 2: Feature Engineering
┌──────────────────────────────────────────────┐
│ FeatureEngineer.compute_features()           │
│                                              │
│ Technical: RSI, SMA, Volatility, Volume     │
│ Fundamental: P/E, ROE, Margins, Growth      │
│ Sector: Relative performance, correlations  │
│ Market: VIX regime, market breadth          │
│                                              │
│ Output: DataFrame (252K rows × 100 cols)    │
└──────────────────────────────────────────────┘
                    │
                    ▼
Step 3: Target Generation
┌──────────────────────────────────────────────┐
│ TargetGenerator.compute_forward_return()     │
│                                              │
│ Compute: forward_return_30d                  │
│ Market-relative: vs SPY                      │
│                                              │
│ Output: + target_return_30d_vs_market        │
└──────────────────────────────────────────────┘
                    │
                    ▼
Step 4: Data Filtering
┌──────────────────────────────────────────────┐
│ DataFilter.filter_for_training()             │
│                                              │
│ Drop: Null targets                          │
│ Drop: Null critical features                │
│ Keep: Null fundamentals (impute later)      │
│                                              │
│ Save: training_data_30d_latest.parquet       │
└──────────────────────────────────────────────┘
                    │
                    ▼
Step 5: Model Training
┌──────────────────────────────────────────────┐
│ ModelTrainer.train_with_cv()                 │
│                                              │
│ Preprocessing:                               │
│   - Filter null features (>70% nulls)       │
│   - Impute remaining nulls (median)         │
│   - Clip outliers (0.1, 99.9 percentile)   │
│   - Scale features (StandardScaler)         │
│                                              │
│ Train:                                       │
│   - 5-fold time-series CV                   │
│   - Ridge or Quantile regression            │
│                                              │
│ Save: model, preprocessor, features, info   │
└──────────────────────────────────────────────┘
                    │
                    ▼
Step 6: ML Evaluation
┌──────────────────────────────────────────────┐
│ ModelEvaluator.run_evaluation()              │
│                                              │
│ Metrics:                                     │
│   - R², RMSE, MAE, MAPE                     │
│   - Directional accuracy                    │
│   - Error skew, kurtosis                    │
│                                              │
│ Analysis:                                    │
│   - Temporal patterns (monthly)             │
│   - Sector patterns                         │
│   - Error distribution                      │
│                                              │
│ Output:                                      │
│   - evaluation_report.txt                   │
│   - error_distribution.png                  │
│   - temporal_performance.png                │
│   - sector_performance.png                  │
│   - predictions.parquet                     │
└──────────────────────────────────────────────┘
                    │
                    ▼
Step 7: Financial Returns
┌──────────────────────────────────────────────┐
│ ModelReturnsAnalyzer.from_parquet()          │
│                                              │
│ Strategies:                                  │
│   - Quintile: Long Q5, Short Q1             │
│   - Top-N: Long top 10 stocks               │
│   - Market-neutral                          │
│                                              │
│ Metrics:                                     │
│   - Total/Annual return                     │
│   - Sharpe ratio                            │
│   - Max drawdown                            │
│   - Win rate, avg win/loss                  │
│   - Information ratio                       │
│                                              │
│ Output:                                      │
│   - returns_report.txt                      │
│   - cumulative_returns.png                  │
└──────────────────────────────────────────────┘
```

## Component Interfaces

```
┌────────────────────────────────────────────────┐
│         DataLoader                             │
├────────────────────────────────────────────────┤
│ Input:  symbols: list[str]                     │
│ Output: dict[str, DataFrame]                   │
│         - 'price', 'market', 'financials',     │
│           'metadata'                           │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│         FeatureEngineer                        │
├────────────────────────────────────────────────┤
│ Input:  price_df, market_df, financials_df,    │
│         metadata_df                            │
│ Output: DataFrame with all features            │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│         TargetGenerator                        │
├────────────────────────────────────────────────┤
│ Input:  features_df, horizon_days              │
│ Output: DataFrame + target column              │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│         DataFilter                             │
├────────────────────────────────────────────────┤
│ Input:  df_with_target, target_col             │
│ Output: Path to saved parquet                  │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│         ModelTrainer                           │
├────────────────────────────────────────────────┤
│ Input:  training_path, model_type, params      │
│ Output: CV scores, saved model artifacts       │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│         ModelEvaluator (Step 6)                │
├────────────────────────────────────────────────┤
│ Input:  model_dir, data_path                   │
│ Output: ML metrics dict, predictions.parquet,  │
│         visualizations, reports                │
│                                                │
│ Focus: R², RMSE, error distribution            │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│         ModelReturnsAnalyzer (Step 7)          │
├────────────────────────────────────────────────┤
│ Input:  predictions.parquet, strategy_type     │
│ Output: Financial metrics dict, returns plots  │
│                                                │
│ Focus: Sharpe, returns, drawdown              │
└────────────────────────────────────────────────┘
```

## Execution Modes

```
┌─────────────────────────────────────────────────────────┐
│               --full (Run All Steps)                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Data Loading → Feature Engineering → Target Gen →     │
│  Data Filtering → Model Training → ML Eval →           │
│  Returns Analysis                                       │
│                                                         │
│  Use when:                                              │
│    - First time running                                 │
│    - Data has changed                                   │
│    - Features have changed                              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│               --train (Train Only)                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Skip data steps] → Model Training                     │
│                                                         │
│  Assumes: training_data_30d_latest.parquet exists       │
│                                                         │
│  Use when:                                              │
│    - Hyperparameter tuning                              │
│    - Testing different models                           │
│    - Data hasn't changed                                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│               --evaluate (Evaluate Only)                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Skip data/train] → ML Eval → Returns Analysis         │
│                                                         │
│  Assumes: Model artifacts exist                         │
│                                                         │
│  Use when:                                              │
│    - Generating new reports                             │
│    - Testing different strategies                       │
│    - Model hasn't changed                               │
└─────────────────────────────────────────────────────────┘
```

## File Dependencies

```
Input Files (Required)
├── data/price/daily/{SYMBOL}.parquet
├── data/market/daily/market_data_latest.parquet
├── data/financials/quarterly_financials_latest.parquet
└── data/metadata/company_metadata_latest.parquet

Intermediate Files (Generated)
└── data/training/training_data_30d_latest.parquet

Model Artifacts (Generated)
└── models/{model_type}_{horizon}d/
    ├── {model_type}_model.joblib
    ├── preprocessor.joblib
    ├── feature_names.txt
    └── model_info.txt

Report Files (Generated)
└── reports/{timestamp}/
    ├── predictions.parquet           ← Step 6 → Step 7
    ├── evaluation_report.txt
    ├── error_distribution.png
    ├── temporal_performance.png
    ├── sector_performance.png
    └── returns/
        ├── returns_report.txt
        └── cumulative_returns.png
```

## Key Design Patterns

### 1. Separation of Concerns
- **Data components** (1-4): Handle data loading, features, targets
- **Model component** (5): Handle training only
- **Evaluation components** (6-7): Separate ML metrics from financial metrics

### 2. Dependency Injection
```python
# Components don't know about each other
# Workflow injects dependencies

data = step1()
features = step2(data)           # Inject data
target = step3(features)         # Inject features
path = step4(target)             # Inject target
scores = step5(path)             # Inject path
```

### 3. Configuration Over Code
```python
# Behavior controlled by config, not code changes
workflow = Workflow30dReturns(
    model_type='quantile',       # Not 'ridge'
    model_params={'quantile': 0.6}  # Custom params
)
```

### 4. Single Responsibility
- Each component does ONE thing well
- Easy to test, debug, optimize
- Clear boundaries

### 5. Output as Input
- Each step outputs what next step needs
- `predictions.parquet` from Step 6 → Step 7
- Training path from Step 4 → Steps 5 & 6
