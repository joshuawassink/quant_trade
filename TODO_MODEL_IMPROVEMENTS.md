# TODO: Model Improvements

Systematic plan to improve baseline Ridge regression model based on evaluation insights.

## Current Performance
- R²: 0.065
- MAE: 8.65%
- RMSE: 12.32%
- Directional Accuracy: 57.0%
- **Problem**: Error skew = -1.22 (underestimates big positive moves)

---

## Phase 1: Baseline Linear Model Improvements (Current Focus)

### 🔴 HIGH PRIORITY - Fix Negative Skew

#### 1.1 Address Underestimation of Positive Swings
**Problem**: Skew = -1.22 means model misses big positive moves

**Root Causes to Investigate**:
- [ ] Target variable distribution (is it skewed?)
- [ ] Feature clipping (are we removing positive outliers?)
- [ ] Ridge penalty (is L2 shrinking predictions too much?)
- [ ] Training on log-returns vs raw returns

**Solutions to Try**:
- [ ] **Log-transform target** for heavy-tailed distribution
  - File: `scripts/create_training_dataset.py`
  - Change: Apply `log(1 + return)` transform

- [ ] **Asymmetric loss function** (penalize underestimation more)
  - File: `scripts/train_baseline_model.py`
  - Use: `sklearn.linear_model.QuantileRegressor` (75th percentile)

- [ ] **Adjust outlier clipping** (keep more positive outliers)
  - File: `src/models/preprocessing.py`
  - Change: Asymmetric percentiles (0.5, 99.5 instead of 0.1, 99.9)

- [ ] **Post-prediction adjustment** (shift predictions by median error)
  - File: `scripts/train_baseline_model.py`
  - Add: Bias correction term

**Expected Impact**: Reduce skew to <|0.5|, improve capture of big winners

---

### 🟡 MEDIUM PRIORITY - Feature Engineering

#### 1.2 Add Technical Indicators (Address Balance Sheet Bias)
**Problem**: Top features are all balance sheet (size metrics), missing momentum

**Features to Add**:
- [ ] **Bollinger Bands**
  - `bb_upper`, `bb_lower`, `bb_width`, `price_bb_position`

- [ ] **Volume-based**
  - `volume_sma_ratio` (volume / 20-day avg)
  - `obv` (On-Balance Volume)
  - `volume_price_trend`

- [ ] **Market microstructure**
  - `bid_ask_spread` (if available)
  - `intraday_range` (high - low)
  - `close_position` ((close - low) / (high - low))

- [ ] **Additional momentum**
  - `momentum_6m`, `momentum_9m`, `momentum_12m`
  - `returns_consistency` (% of positive days in last 20)

**File**: `src/features/technical.py`
**Expected Impact**: 10-15% MAE improvement

#### 1.3 Sector-Relative Features
**Problem**: Tech stocks have 11% MAE vs Utilities 5.5% MAE

**Features to Add**:
- [ ] `return_vs_sector` (performance vs sector median)
- [ ] `pe_ratio_vs_sector`
- [ ] `revenue_growth_vs_sector`
- [ ] `volatility_vs_sector`

**File**: `src/features/fundamental.py`
**Expected Impact**: Reduce sector MAE variation by 30%

#### 1.4 Size-Adjusted Fundamentals
**Problem**: `total_assets` has coefficient -0.73 (size bias)

**Features to Replace**:
- [ ] Replace `total_revenue` → `revenue_per_share`
- [ ] Replace `net_income` → `eps` (earnings per share)
- [ ] Replace `total_assets` → `book_value_per_share`
- [ ] Add `revenue_per_employee`
- [ ] Add `profit_margin_rank` (percentile vs all stocks)

**File**: `src/features/fundamental.py`
**Expected Impact**: Reduce size bias, improve small-cap performance

---

### 🟢 LOW PRIORITY - Hyperparameter Tuning

#### 1.5 Ridge Alpha Optimization
**Current**: Alpha = 1.0 (arbitrary)

**Approach**:
- [ ] Grid search: [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
- [ ] Use time-series CV for validation
- [ ] Plot validation R² vs alpha (regularization path)

**File**: `scripts/train_baseline_model.py`
**Expected Impact**: 5-10% R² improvement

#### 1.6 Feature Selection
**Problem**: Using all 81 features (some may be noise)

**Methods to Try**:
- [ ] **L1 regularization** (Lasso) for automatic selection
- [ ] **Recursive Feature Elimination** (RFE)
- [ ] **Correlation analysis** (remove highly correlated features)
- [ ] **Mutual information** (keep high MI features only)

**File**: New script `scripts/feature_selection.py`
**Expected Impact**: 5% MAE improvement, faster training

---

### 🔵 ANALYSIS - Understand Current Model

#### 1.7 Detailed Error Analysis
- [ ] **Error by return magnitude**
  - Do we underestimate large moves more than small?

- [ ] **Error by volatility**
  - Performance on high-vol vs low-vol stocks

- [ ] **Error by market regime**
  - Bull market vs bear market performance

- [ ] **Error by time to earnings**
  - Performance near earnings announcements

**File**: `scripts/analyze_model_errors.py` (create)

#### 1.8 Feature Interaction Analysis
- [ ] Which features work together?
- [ ] Which features conflict?
- [ ] Are there polynomial features worth adding?

**File**: `scripts/analyze_feature_interactions.py` (create)

---

## Phase 2: Data Expansion (Future Work)

### 2.1 Expand Stock Universe
**Current**: 377 stocks (S&P 500 with data availability filter)

**Expansion Options**:
- [ ] **Full S&P 500** (all 500 stocks)
  - Challenge: More missing data for recent additions
  - Expected gain: +30% more training data

- [ ] **S&P 400 Mid-Cap**
  - Challenge: Different dynamics than large-cap
  - Expected gain: 2x training data

- [ ] **Russell 2000 Small-Cap**
  - Challenge: Much higher volatility, less fundamental data
  - Expected gain: 5x training data

**Recommendation**: Start with full S&P 500, then add mid-cap

### 2.2 Extend Historical Data
**Current**: 2023-01-04 to 2025-09-11 (2.7 years)

**Extension Options**:
- [ ] **Back to 2020** (include COVID period)
  - Pros: 5 years of data, includes major regime shift
  - Cons: COVID was unusual, may hurt generalization

- [ ] **Back to 2015** (10 years)
  - Pros: Multiple market cycles
  - Cons: Older data may be less relevant

- [ ] **Back to 2010** (15 years)
  - Pros: Maximum data
  - Cons: Different market structure (pre-algo dominance)

**Recommendation**: Start with 2020, evaluate regime stability

### 2.3 Alternative Data Sources
- [ ] **Earnings call transcripts** (sentiment analysis)
- [ ] **News sentiment** (from financial news APIs)
- [ ] **Insider trading** (Form 4 filings)
- [ ] **Short interest** (from FINRA)
- [ ] **Options data** (implied volatility, put/call ratio)
- [ ] **Analyst estimates** (consensus forecasts)

**Challenge**: Data cost, integration complexity
**Expected Impact**: 20-30% R² improvement (if done well)

### 2.4 Higher Frequency Data
**Current**: Daily returns

**Options**:
- [ ] **Intraday** (hourly, minute)
  - Better capture of momentum
  - More complex to model

- [ ] **Tick data** (every trade)
  - Maximum information
  - Huge data volume

**Recommendation**: Stay with daily for baseline, revisit later

---

## Phase 3: Advanced Models (After Baseline is Optimized)

### 3.1 Ensemble Methods
- [ ] **Random Forest**
  - Handles non-linearity
  - Feature importance built-in

- [ ] **Gradient Boosting** (XGBoost, LightGBM)
  - State-of-the-art for tabular data
  - Asymmetric loss functions available

- [ ] **Stacking**
  - Combine Ridge, Random Forest, XGBoost
  - Best of all approaches

**Expected Impact**: 50-100% R² improvement

### 3.2 Sector-Specific Models
**Motivation**: Utilities (5.5% MAE) vs Tech (10.8% MAE)

- [ ] Train separate models per sector
- [ ] Ensemble sector models
- [ ] Sector rotation strategy

**Expected Impact**: 20-30% overall MAE improvement

### 3.3 Time-Series Models
- [ ] **LSTM/GRU** (sequential patterns)
- [ ] **Transformer** (attention mechanism)
- [ ] **Prophet** (seasonality, trends)

**Challenge**: More complex, harder to interpret
**Expected Impact**: 10-20% improvement (uncertain)

---

## Implementation Order

### Week 1 (Current Sprint)
1. ✅ Fix negative skew (asymmetric loss or log transform)
2. ✅ Add technical indicators (Bollinger Bands, volume)
3. ✅ Hyperparameter tuning (Ridge alpha)
4. ✅ Evaluate improvements

### Week 2
5. ✅ Add sector-relative features
6. ✅ Size-adjusted fundamentals
7. ✅ Feature selection (remove noise)
8. ✅ Evaluate improvements

### Week 3
9. ✅ Detailed error analysis
10. ✅ Expand to full S&P 500
11. ✅ Extend data to 2020
12. ✅ Re-evaluate with more data

### Week 4+
13. Try ensemble methods
14. Sector-specific models
15. Alternative data integration

---

## Success Metrics

Track these metrics after each improvement:

| Metric | Current | Target | Stretch Goal |
|--------|---------|--------|--------------|
| R² | 0.065 | 0.10 | 0.15 |
| MAE | 8.65% | 7.5% | 6.5% |
| RMSE | 12.32% | 11.0% | 10.0% |
| Directional Acc | 57.0% | 58.5% | 60.0% |
| Error Skew | -1.22 | -0.5 | 0.0 |
| Error Kurtosis | 13.1 | 8.0 | 5.0 |

**Key**: Directional accuracy is most important for trading!

---

## Notes

- **Focus**: Iterate quickly, measure after each change
- **Baseline first**: Get linear model as good as possible before ensembles
- **Avoid overfitting**: Always use time-series CV, never optimize on full dataset
- **Document**: Update evaluation report after each major change
- **Version control**: Tag models (v1.0, v1.1, etc.) for comparison

---

## Questions to Answer

1. **Why do we underestimate positive moves?**
   - Is target distribution skewed?
   - Are features clipped asymmetrically?
   - Is Ridge penalty too strong?

2. **Why do balance sheet features dominate?**
   - Are they truly predictive or just size proxies?
   - Do we need more technical features?
   - Should we use sector-relative metrics?

3. **Can we exploit the 57% directional accuracy?**
   - Build long/short portfolio
   - Rank-based allocation
   - Option strategies

4. **What's the ceiling for linear models?**
   - Test on perfect features
   - Compare to ensemble baseline
   - When to graduate to non-linear models?
