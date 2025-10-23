# Next Session TODO

## High Priority

### 1. Update Notebook for Feature Pipeline Investigation
**File**: `notebooks/01_data_exploration.ipynb`

**Tasks**:
- [ ] Add cells to load training dataset from `data/training/training_data_30d_latest.parquet`
- [ ] Visualize feature distributions (technical, fundamental, sector/market)
- [ ] Analyze feature correlations and multicollinearity
- [ ] Plot target variable distribution (30-day market-relative returns)
- [ ] Show feature importance from baseline Ridge model
- [ ] Add feature engineering step-by-step walkthrough:
  - Load raw data (price, financials, market, metadata)
  - Compute technical features
  - Compute fundamental features
  - Compute sector/market features
  - Align to daily frequency
  - Compute target variable
- [ ] Add visualizations for key features:
  - RSI distributions
  - Return correlations across horizons
  - VIX regime vs market returns
  - Fundamental metric trends

**Why**: Need interactive environment to explore features, understand data quality, and identify issues/opportunities

**Estimated time**: 2-3 hours

---

### 2. Expand Stock Universe
**Current**: 20 stocks (sample for development)
**Target**: 100-500 stocks (production universe)

**Tasks**:
- [ ] Define selection criteria:
  - Market cap > $10B (large/mega cap)
  - Average daily volume > 1M shares (liquid)
  - Sector diversification (maintain balance)
  - Data availability (complete financials)
- [ ] Update `src/config/universe.py` with expanded list
- [ ] Re-fetch data for expanded universe:
  - Price data
  - Financials
  - Market data (same - SPY, VIX, sector ETFs)
  - Metadata
- [ ] Test feature pipeline with expanded universe
- [ ] Retrain baseline model with more data
- [ ] Compare performance (20 stocks vs expanded)

**Why**: 
- More training data → better models
- Better diversification
- More realistic production scenario
- 20 stocks is too small for meaningful backtesting

**Estimated time**: 3-4 hours

---

### 3. Develop Model Evaluation Framework
**Goal**: Comprehensive benchmarking of model performance and stability

**Tasks**:
- [ ] Create `src/evaluation/` module with:
  - `metrics.py` - Custom metrics for return prediction
  - `backtester.py` - Simulate trading performance
  - `stability.py` - Track performance over time
  - `benchmark.py` - Compare to baselines

**Metrics to track**:
- **Prediction Metrics**:
  - R² (coefficient of determination)
  - MSE, RMSE, MAE
  - Information Coefficient (IC)
  - Hit rate (% correct direction predictions)
  
- **Trading Metrics** (from backtest):
  - Sharpe ratio
  - Maximum drawdown
  - Win rate
  - Profit factor
  - Average return per trade
  
- **Stability Metrics**:
  - Rolling R² over time
  - Feature importance stability
  - Prediction error distribution shifts
  - Out-of-sample decay

**Benchmarks**:
- Buy-and-hold SPY
- Equal-weight portfolio
- Random predictions
- Mean prediction (always predict mean return)

**Implementation**:
```python
# Example usage
from src.evaluation import Backtester, ModelEvaluator

# Evaluate predictions
evaluator = ModelEvaluator(y_true, y_pred)
metrics = evaluator.compute_all_metrics()
evaluator.plot_residuals()
evaluator.plot_predictions_vs_actual()

# Backtest trading strategy
backtester = Backtester(
    predictions=y_pred,
    actual_returns=y_true,
    dates=dates,
    symbols=symbols
)
results = backtester.run(
    top_n=10,  # Long top 10 stocks
    rebalance_freq='monthly'
)
backtester.plot_equity_curve()
backtester.print_summary()
```

**Why**: 
- Need rigorous evaluation beyond just R²
- Trading metrics more relevant than ML metrics
- Track model stability over time
- Compare multiple models objectively

**Estimated time**: 4-6 hours

---

## Medium Priority

### 4. Hyperparameter Tuning
- [ ] Grid search for Ridge alpha (0.1, 1.0, 10.0, 100.0)
- [ ] Try different regularization (Lasso, ElasticNet)
- [ ] Optimize on validation IC instead of MSE

### 5. Feature Engineering Improvements
- [ ] Add categorical encoding for `vix_regime`
- [ ] Add interaction features (e.g., RSI × volatility)
- [ ] Add lagged features (t-1, t-5 returns)
- [ ] Winsorize extreme values in fundamental features

### 6. Advanced Models
- [ ] Random Forest Regressor
- [ ] Gradient Boosting (XGBoost, LightGBM)
- [ ] Neural Network (simple MLP)
- [ ] Ensemble (combine multiple models)

---

## Current Status

**Completed**:
✅ Feature engineering (111 features)
✅ Feature alignment & training dataset creation
✅ Baseline Ridge regression model (R²=0.32)
✅ Time-series cross-validation
✅ Data conventions documentation
✅ pl.Date refactor (fixed VIX bug)

**In Progress**:
🔄 Model evaluation framework
🔄 Notebook updates for feature exploration

**Blocked**:
❌ None

**Next Milestone**: Production-ready evaluation framework + expanded universe

---

## Notes

- Current training data: 823 rows (small due to 94.5% null drop rate)
- Consider relaxing null-dropping rules to get more training data
- VIX features working perfectly after pl.Date refactor
- Baseline R²=0.32 is good but has room for improvement
- Training/validation gap suggests overfitting (alpha tuning needed)


---

## Additional Context & Helpful Notes

### Quick Start Commands (when you return)
```bash
# Activate virtual environment
source .venv/bin/activate  # or: .venv/bin/python

# View training dataset
.venv/bin/python -c "import polars as pl; df = pl.read_parquet('data/training/training_data_30d_latest.parquet'); print(df.describe())"

# Check model performance
.venv/bin/python scripts/train_baseline_model.py

# Launch Jupyter for exploration
jupyter lab notebooks/01_data_exploration.ipynb
```

### Key Files Reference
```
📁 quant_trade/
├── 📄 data/training/training_data_30d_latest.parquet  # Ready for ML
├── 📄 models/baseline/ridge_model.joblib              # Trained model
├── 📄 docs/development/data_conventions.md            # **READ THIS FIRST**
├── 📄 docs/README.md                                  # Documentation index
├── 📄 src/features/alignment.py                       # Feature pipeline
└── 📄 scripts/train_baseline_model.py                 # Training script
```

### Known Issues & Limitations

1. **Small Training Set (823 rows)**
   - 94.5% of data dropped due to nulls
   - Root cause: Warm-up periods for rolling features
   - **Fix**: Either get more historical data OR relax null-dropping rules
   - **Impact**: Model might not generalize well with limited data

2. **Fundamental Features High Null Rate (53%)**
   - Expected for quarterly data
   - YoY features need 4 quarters of history
   - **Consider**: Should we forward-fill more aggressively? Or drop some YoY features?

3. **Overfitting in Baseline Model**
   - Train R² (0.80) >> Val R² (0.32)
   - **Fix**: Increase alpha (try 10.0, 100.0) or use stronger regularization
   - **Or**: Get more training data (expand universe)

4. **Categorical Features Excluded**
   - `vix_regime` currently not used
   - **Fix**: Add one-hot encoding or ordinal encoding
   - **Benefit**: Could improve predictions (VIX regime is informative)

5. **Only Using Recent Data (Jun-Sep 2025)**
   - Due to warm-up + null filtering
   - **Risk**: Model trained only on recent market conditions
   - **Fix**: Longer historical data fetch (5 years instead of 3)

### Data Quality Checks to Run

When you return, verify these:
```bash
# Check for inf/nan in training data
.venv/bin/python -c "
import polars as pl
df = pl.read_parquet('data/training/training_data_30d_latest.parquet')
print('Inf values:', df.select(pl.all().is_infinite().sum()))
print('Null values:', df.select(pl.all().is_null().sum()))
"

# Check target distribution
.venv/bin/python -c "
import polars as pl
df = pl.read_parquet('data/training/training_data_30d_latest.parquet')
print(df['target_return_30d_vs_market'].describe())
"

# Check date coverage per symbol
.venv/bin/python -c "
import polars as pl
df = pl.read_parquet('data/training/training_data_30d_latest.parquet')
print(df.group_by('symbol').agg(pl.len().alias('rows')).sort('rows'))
"
```

### Performance Optimization Opportunities

1. **Feature Selection**
   - 91 features might be too many (curse of dimensionality)
   - Run feature importance analysis
   - Consider dropping low-importance features
   - **Tool**: sklearn's SelectKBest or recursive feature elimination

2. **Feature Engineering Improvements**
   - Add momentum indicators (e.g., 5d vs 20d MA crossover)
   - Add volume-price relationships
   - Add earnings surprise (actual vs expected)
   - Add relative strength vs sector

3. **Target Variable Alternatives**
   - Try different horizons (14d, 21d, 45d)
   - Try absolute returns instead of market-relative
   - Try ranking/classification (top/bottom quintile)

### Questions to Investigate

1. **Why does validation R² vary so much across folds?**
   - Fold 1: 0.03 (terrible)
   - Fold 5: 0.62 (great)
   - Possible: Market regime changes? Different sectors?
   - **Action**: Analyze predictions by date/sector

2. **Which features matter most?**
   - Ridge coefficients show feature importance
   - Are technical or fundamental features more predictive?
   - **Action**: Plot feature coefficients from trained model

3. **Is the model just learning sector rotation?**
   - Tech stocks dominated recent period
   - **Risk**: Model might just pick winners based on sector
   - **Check**: Predictions correlated with sector returns?

4. **How does model perform on different market conditions?**
   - Bull markets vs bear markets
   - High vs low volatility
   - **Action**: Stratify validation by VIX regime

### Dependencies to Install (if needed)
```bash
# Already installed:
# - polars, yfinance, pandas, numpy
# - scikit-learn, joblib
# - jupyter, plotly, loguru

# May need later:
pip install xgboost lightgbm  # Gradient boosting
pip install optuna             # Hyperparameter tuning
pip install shap               # Model interpretability
pip install alphalens          # Quantitative analysis
```

### Git Workflow Reminder
```bash
# Before starting work
git status
git pull  # If working across machines

# After making changes
git add -A
git status
git commit -m "type(scope): description"  # Use conventional commits

# See recent commits
git log --oneline -10

# See what changed
git diff HEAD~1
```

### Useful Documentation Links
- Data Conventions: `docs/development/data_conventions.md`
- Git Workflow: `docs/development/git_workflow.md`
- Feature Evaluation: `docs/reports/feature_evaluation_2025-10-23.md`
- Architecture Spec: `docs/architecture/regression_framework_spec.md`

---

## Session Goals (Prioritized)

**Day 1 Goal**: Feature exploration + expanded universe
1. Update notebook with feature analysis (2-3 hours)
2. Expand to 100-200 stocks (3-4 hours)
3. Retrain and compare

**Day 2 Goal**: Evaluation framework
1. Build backtester module (3-4 hours)
2. Implement key metrics (2-3 hours)
3. Run backtest on baseline model

**Day 3 Goal**: Model improvements
1. Hyperparameter tuning (2 hours)
2. Try advanced models (3-4 hours)
3. Ensemble and compare

**Success Criteria**:
- Training data > 5,000 rows (from 823)
- Validation R² > 0.4 (from 0.32)
- Sharpe ratio > 1.0 in backtest
- Drawdown < 20%

---

## Remember

- **Data quality > Model complexity** - Fix the 94.5% drop rate first!
- **Read data_conventions.md** before any data work
- **Use pl.Date for daily data** - Don't revert to datetime
- **Time-series CV** - Always respect temporal ordering
- **Trading metrics matter** - R² is nice but Sharpe/drawdown are what count

Good luck! The framework is in excellent shape. 🚀
