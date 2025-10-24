# Missing Data Strategy

## Overview

This document describes the comprehensive missing data handling strategy implemented to maximize training data retention while maintaining model stability.

## Problem Statement

Initial implementation used aggressive `drop_nulls()` which caused:
- **91.6% data loss** (14,387 rows from 240K expected)
- Only 101 days of data instead of 2.7 years
- YoY features had 90% nulls (requiring 4 quarters of history)
- QoQ features had 65% nulls (requiring 2 quarters of history)

## Solution: Multi-Stage Approach

### 1. Feature Engineering Changes

#### Removed YoY Features (8 features)
**Rationale**: YoY features require 4 quarters of history, causing 90% nulls.

Removed features:
- `roe_yoy_change`
- `roa_yoy_change`
- `gross_margin_yoy_change`
- `net_margin_yoy_change`
- `total_revenue_yoy_change`
- `total_revenue_yoy_pct`
- `net_income_yoy_change`
- `net_income_yoy_pct`

**Impact**: Reduced fundamental features from 22 to 14.

#### Kept QoQ Features
**Rationale**: QoQ features only require 1 quarter lookback, providing better coverage (35% valid data).

Location: `src/features/fundamental.py:compute_all()`

### 2. Training Dataset Creation

#### Selective Null Filtering
Instead of `drop_nulls()`, we now:

**Step 1**: Drop rows where target is null
```python
df = df.filter(pl.col('target_return_30d_vs_market').is_not_null())
```

**Step 2**: Drop rows where critical price features are null
```python
critical_features = [
    'adj_close', 'volume', 'return_5d', 'return_20d',
    'volatility_5d', 'rsi_14', 'sma_20', 'sma_50',
    'spy_return_5d', 'vix_level'
]
```

**Step 3**: Keep rows with null fundamentals (will be imputed during training)

**Results**:
- **Before**: 14,387 rows (6% retention)
- **After**: 252,132 rows (89% retention)
- **17.5x more training data**
- Date range: 2023-01-04 to 2025-09-11 (2.7 years)

Location: `scripts/create_training_dataset.py:filter_and_save()`

### 3. Preprocessing Pipeline

#### Architecture
Flexible sklearn-based pipeline with 4 stages:

```
1. Null Feature Filter → 2. Outlier Clipper → 3. Imputer → 4. Scaler
```

#### Stage 1: Null Feature Filter
**Purpose**: Remove features with too many nulls in training fold

Parameters:
- `max_null_pct`: 70% threshold
- Adaptive per fold (different features filtered in early vs late folds)

**Example** (from time-series CV):
- Fold 1 (2023 data): 50/83 features filtered → 33 features
- Fold 5 (2025 data): 23/83 features filtered → 60 features
- Final model (all data): 2/83 features filtered → 81 features

#### Stage 2: Outlier Clipper
**Purpose**: Handle infinity values and extreme outliers

Method:
- Replace `inf` with `nan`
- Clip to 0.1th and 99.9th percentiles per feature

**Rationale**: Financial ratios can have division by zero (e.g., debt_to_equity when equity is 0).

#### Stage 3: Imputer
**Strategy**: Median imputation

**Rationale**:
- Median is robust to outliers
- Preferable to mean for financial data with fat tails
- Better than constant fill for features with meaningful distributions

#### Stage 4: Scaler
**Method**: StandardScaler (zero mean, unit variance)

**Alternative options**:
- MinMaxScaler: For neural networks
- RobustScaler: For very heavy-tailed data

Location: `src/models/preprocessing.py:FeaturePreprocessor`

### 4. Time-Series Cross-Validation

#### Key Principle: No Data Leakage
**Each fold fits its own preprocessor on training data only**

```python
for fold in tscv.split(X):
    X_train, X_val = ...

    # Fit preprocessor on training fold only
    preprocessor = FeaturePreprocessor(...)
    X_train_processed = preprocessor.fit_transform(X_train)

    # Apply to validation (no fitting!)
    X_val_processed = preprocessor.transform(X_val)
```

**Why this matters**:
- Early folds have fewer fundamental features available
- Preprocessor adapts by filtering different features per fold
- Prevents overfitting by not using validation data statistics

## Results

### Dataset Improvements
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Rows | 14,387 | 252,132 | +1,653% |
| Retention | 6% | 89% | +83 pp |
| Date range | 101 days | 972 days | +871 days |
| Features (input) | 91 | 83 | -8 (YoY removed) |
| Null columns | 0 | 67 | Tolerated |

### Model Stability
| Metric | Before (broken) | After (fixed) |
|--------|-----------------|---------------|
| Val R² Mean | -39M | -0.26 |
| Val R² Std | 78M | 0.19 |
| Model crashes | Yes (inf errors) | No |
| Feature count varies | No | Yes (adaptive) |

### Cross-Validation Results
```
Fold 1: Train R²=0.15, Val R²=-0.56  (33 features)
Fold 2: Train R²=0.11, Val R²=-0.34  (36 features)
Fold 3: Train R²=0.05, Val R²=-0.08  (41 features)
Fold 4: Train R²=0.04, Val R²=-0.28  (42 features)
Fold 5: Train R²=0.04, Val R²=-0.05  (60 features)

Mean: Train R²=0.08 ± 0.04, Val R²=-0.26 ± 0.19
```

**Interpretation**:
- Negative validation R² is expected for stock prediction (hard problem!)
- Stable performance across folds (no catastrophic failures)
- Adaptive feature count based on data availability
- Model performs slightly worse than predicting mean return

## Key Learnings

### 1. Don't Use drop_nulls() on Time-Series Data
- Quarterly fundamentals naturally have gaps
- Aggressive null removal loses most data
- Better to handle nulls in preprocessing

### 2. Adapt to Data Availability Over Time
- Early historical periods: fewer fundamentals available
- Recent periods: more complete data
- Solution: Per-fold feature filtering

### 3. Handle Financial Data Peculiarities
- Infinity values from division by zero
- Extreme outliers (e.g., 1000% returns during outlier events)
- Heavy-tailed distributions
- Solution: Clip outliers before imputation

### 4. YoY Features Not Worth the Data Loss
- Require 4 quarters of history
- Cause 90% nulls
- QoQ features (1 quarter lookback) better tradeoff

## Next Steps

### Near-term Improvements
1. **Feature engineering**:
   - Create more technical indicators (lower null rates)
   - Experiment with shorter lookback periods
   - Add sector-relative metrics

2. **Imputation strategies**:
   - Forward-fill fundamentals before imputation
   - Group-based imputation (by sector)
   - MICE (Multivariate Imputation by Chained Equations)

3. **Model improvements**:
   - Try tree-based models (handle nulls natively)
   - Increase regularization (alpha > 1.0)
   - Feature selection based on importance

### Long-term
1. Separate models for different time periods
2. Ensemble combining tech-only and fundamental models
3. Alternative targets (e.g., rank prediction instead of regression)

## Files Modified

1. `src/features/fundamental.py` - Removed YoY features
2. `scripts/create_training_dataset.py` - Selective null filtering
3. `src/models/preprocessing.py` - New preprocessing pipeline
4. `scripts/train_baseline_model.py` - Integrated preprocessing
5. `scripts/analyze_missing_data.py` - Analysis tool

## References

- [Sklearn Imputation](https://scikit-learn.org/stable/modules/impute.html)
- [Time-series CV](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- [Handling Missing Data in Finance](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3265269)
