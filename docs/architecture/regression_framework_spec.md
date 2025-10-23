# Regression Framework for 30-Day Relative Returns

## Objective

Build a sophisticated regression model to predict **30-day relative returns** (stock return vs. market return) that:
1. Is modular and reusable across strategies
2. Supports multiple model types (linear, ensemble, neural networks)
3. Handles feature engineering systematically
4. Prevents overfitting through proper validation
5. Is production-ready for live deployment

## Target Variable Definition

### Relative Return (What We're Predicting)

```python
# Target for day t (30 days forward-looking)
relative_return_30d = (stock_return_30d - market_return_30d)

# Where:
stock_return_30d = (price[t+30] - price[t]) / price[t]
market_return_30d = (SPY[t+30] - SPY[t]) / SPY[t]
```

### Why Relative Returns?
- **Market-neutral**: Isolates stock-specific alpha
- **Comparable**: Works in bull/bear markets
- **Actionable**: Tells us which stocks to overweight
- **Risk-adjusted**: Naturally accounts for market conditions

### Alternative Formulations (To Test)

1. **Excess Return Over Risk-Free Rate**
   ```python
   excess_return = stock_return_30d - (risk_free_rate * 30/365)
   ```

2. **Sector-Relative Return**
   ```python
   sector_relative = stock_return_30d - sector_return_30d
   ```

3. **Risk-Adjusted Return (Sharpe-style)**
   ```python
   risk_adjusted = relative_return_30d / volatility_30d
   ```

## Model Architecture

### Modular Pipeline Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │  Price  │ │Earnings │ │ Insider │ │ Stock   │ │ Sector/ │          │
│  │  Data   │ │  Data   │ │  Data   │ │Metadata │ │ Market  │          │
│  │         │ │         │ │         │ │         │ │  Data   │          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
│       ↓           ↓           ↓           ↓           ↓                 │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │              + Alternative Data (GitHub, etc.)               │       │
│  └─────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │Technical │  │Fundament.│  │Alternative│  │  Stock   │  │  Sector/ │ │
│  │ Features │  │ Features │  │ Features  │  │ Features │  │  Market  │ │
│  │          │  │          │  │           │  │          │  │ Features │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
│         ↓                  ↓                   ↓             │
│  ┌────────────────────────────────────────────────────┐     │
│  │         Feature Transformation & Selection         │     │
│  │  (Normalization, Winsorization, PCA, etc.)        │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    MODEL LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Linear     │  │   Ensemble   │  │    Neural    │      │
│  │   Models     │  │   Models     │  │   Network    │      │
│  │ (Ridge,Lasso)│  │(XGBoost,RF)  │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         ↓                  ↓                   ↓             │
│  ┌────────────────────────────────────────────────────┐     │
│  │            Ensemble Combiner (Meta-model)          │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  PREDICTION LAYER                            │
│  ┌────────────────────────────────────────────────────┐     │
│  │  30-Day Relative Return Predictions + Confidence   │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Feature Categories

### 1. Technical Features (Price-Based)

#### Momentum Features
- **Returns**: 1d, 5d, 10d, 20d, 60d returns
- **Relative strength**: Return vs SPY over multiple horizons
- **52-week high**: Distance from 52-week high
- **Moving average crosses**: Price vs 20/50/200 day MAs
- **Acceleration**: Change in momentum (second derivative)

#### Volatility Features
- **Historical volatility**: 10d, 20d, 60d standard deviation
- **ATR**: Average True Range
- **Volatility regime**: Current vol vs historical percentile
- **Volatility trend**: Is volatility increasing/decreasing?

#### Volume Features
- **Volume ratio**: Current volume vs 20-day average
- **Dollar volume**: Price * volume (liquidity proxy)
- **Volume trend**: 5-day vs 20-day average volume
- **Accumulation/distribution**: Volume-weighted price changes

### 2. Fundamental Features (Company-Based)

#### Earnings Features
- **EPS surprise**: (Actual - Expected) / Expected
- **Revenue surprise**: (Actual - Expected) / Expected
- **Earnings growth**: YoY earnings growth rate
- **Earnings quality**: Operating earnings vs reported
- **Days since earnings**: Time since last report
- **Days until earnings**: Time until next report

#### Valuation Features
- **P/E ratio**: Price to earnings
- **P/E relative**: P/E vs sector median
- **P/B ratio**: Price to book
- **EV/EBITDA**: Enterprise value to EBITDA
- **PEG ratio**: P/E to growth rate

#### Quality Features
- **ROE**: Return on equity
- **Profit margin**: Net margin, gross margin
- **Debt/Equity**: Leverage ratio
- **Free cash flow**: FCF / Market cap
- **Piotroski F-Score**: Quality composite score

### 3. Alternative Data Features

#### Insider Trading
- **Insider buy cluster**: Count of insider buys (last 30 days)
- **Insider buy amount**: Total $ purchased
- **Insider sell ratio**: Sells / (Buys + Sells)
- **Days since last buy**: Recency of insider activity

#### Analyst Coverage
- **Analyst revision trend**: Upgrades - downgrades (30d)
- **Estimate revision**: % change in consensus EPS
- **Target price gap**: (Target - Current) / Current
- **Analyst coverage**: Number of analysts covering

#### GitHub Activity (Tech Stocks)
- **Commit velocity**: Recent commits / historical avg
- **Contributor growth**: New contributors (30d)
- **Star growth**: Star acceleration
- **Issue resolution rate**: Closed / opened issues

### 4. Stock Metadata Features (Company Characteristics)

#### Company Profile
- **Market cap**: Size (small/mid/large cap)
- **Market cap rank**: Percentile within universe
- **Age since IPO**: Years since listing (maturity)
- **Float**: Shares available for trading
- **Shares outstanding**: Total shares

#### Company Fundamentals
- **Employee count**: Total headcount
- **Employee growth**: YoY headcount change %
- **Revenue per employee**: Efficiency metric
- **Asset intensity**: Assets / Revenue

#### Classification
- **Sector**: GICS sector (categorical)
- **Industry**: GICS industry (more granular)
- **Sub-industry**: Most granular classification
- **Geography**: Primary revenue geography

#### Valuation Level (Cross-Sectional)
- **P/E percentile**: Where stock ranks vs sector
- **P/B percentile**: Book value ranking
- **Market cap percentile**: Size ranking
- **Value/Growth classification**: Based on factor scores

### 5. Sector & Market Features

#### Sector Performance
- **Sector momentum**: Sector ETF return (5d, 20d, 60d)
- **Sector relative strength**: Sector vs SPY
- **Sector volatility**: Recent sector volatility
- **Sector correlation**: How correlated is sector to market?
- **Sector breadth**: % of sector stocks in uptrend

#### Sector Flows & Sentiment
- **Sector ETF flows**: Net inflows to sector ETFs
- **Sector relative volume**: Trading volume vs average
- **Sector rotation score**: Money flowing in/out of sector

#### Market Regime
- **Market trend**: Bull/bear/sideways classification
- **VIX level**: Current VIX (fear gauge)
- **VIX percentile**: VIX vs historical range
- **Market breadth**: Advance/decline ratio
- **New highs/lows**: NYSE new highs vs new lows
- **Put/Call ratio**: Options sentiment

#### Cross-Asset Indicators
- **Bond yields**: 10Y treasury yield level & change
- **Yield curve**: 2Y-10Y spread (recession indicator)
- **Dollar strength**: DXY dollar index
- **Commodity prices**: Oil, gold (risk-on/off)
- **Credit spreads**: High yield spreads (risk appetite)

#### Relative Positioning
- **Beta**: Stock's beta to SPY
- **Sector beta**: Stock's beta to sector
- **Size factor exposure**: Small vs large cap tilt
- **Value factor exposure**: Value vs growth tilt
- **Momentum factor exposure**: Winner vs loser tilt

### 6. Event Features

#### Categorical/Binary
- **Earnings week**: Within 1 week of earnings?
- **Post-earnings**: 0-30 days after earnings
- **Option expiry**: Near monthly option expiry?
- **Dividend ex-date**: Near ex-dividend date?

#### Seasonal
- **Month of year**: January effect, etc.
- **Day of week**: Monday/Friday patterns
- **Quarter**: Q1/Q2/Q3/Q4 seasonality

## Model Types

### Tier 1: Linear Models (Baseline)

#### Ridge Regression
```python
# Advantages:
- Interpretable coefficients
- Handles multicollinearity
- Fast training
- Good baseline

# Hyperparameters:
- alpha: L2 regularization strength
- solver: 'auto', 'svd', 'cholesky', 'lsqr'
```

#### Lasso Regression
```python
# Advantages:
- Feature selection (sets coefficients to 0)
- Sparse solutions
- Interpretable

# Hyperparameters:
- alpha: L1 regularization strength
```

#### Elastic Net
```python
# Advantages:
- Combines Ridge + Lasso
- Best of both worlds

# Hyperparameters:
- alpha: Overall regularization
- l1_ratio: Mix of L1 vs L2
```

### Tier 2: Ensemble Models (Primary)

#### XGBoost Regressor
```python
# Advantages:
- Handles non-linear relationships
- Feature importance built-in
- Robust to outliers
- Industry standard

# Key Hyperparameters:
- learning_rate: 0.01 - 0.1
- max_depth: 3 - 8
- n_estimators: 100 - 1000
- subsample: 0.7 - 1.0
- colsample_bytree: 0.7 - 1.0
- min_child_weight: 1 - 10
- gamma: 0 - 5
```

#### LightGBM
```python
# Advantages:
- Faster than XGBoost
- Better with categorical features
- Lower memory usage

# Key Hyperparameters:
- Similar to XGBoost
- num_leaves: 20 - 50
- min_data_in_leaf: 10 - 100
```

#### Random Forest
```python
# Advantages:
- Less prone to overfitting
- Natural feature importance
- Robust

# Hyperparameters:
- n_estimators: 100 - 500
- max_depth: 5 - 20
- min_samples_split: 2 - 20
- max_features: 'sqrt', 'log2', or fraction
```

### Tier 3: Neural Network (Advanced)

#### Feed-Forward NN
```python
# Architecture:
Input → Dense(128, ReLU) → Dropout(0.3) →
Dense(64, ReLU) → Dropout(0.2) →
Dense(32, ReLU) → Dense(1, Linear)

# Advantages:
- Captures complex non-linearities
- Can learn interactions automatically

# Considerations:
- Requires more data
- Prone to overfitting
- Harder to interpret
```

### Meta-Model (Ensemble of Ensembles)

```python
# Combine predictions from multiple models
# Stack predictions as features for final model

predictions_df = pd.DataFrame({
    'ridge': ridge_pred,
    'xgboost': xgb_pred,
    'lightgbm': lgb_pred,
    'rf': rf_pred
})

# Meta-model (simple weighted average or another regression)
final_pred = meta_model.predict(predictions_df)
```

## Training Strategy

### Walk-Forward Validation

```python
# Avoid look-ahead bias with time-series aware splits

Month 1-24: Train
Month 25-30: Validate
Month 31: Test

Month 2-25: Train (rolling)
Month 26-31: Validate
Month 32: Test

... continue rolling forward
```

### Cross-Validation Approach

1. **Time-series split**: Use `TimeSeriesSplit` from sklearn
2. **Purged K-Fold**: Remove data near validation set (avoid leakage)
3. **Embargo period**: Gap between train/test to account for lookahead

### Preventing Overfitting

#### Data Leakage Prevention
- **Point-in-time data**: Only use data available at prediction time
- **Survivor bias**: Include delisted stocks in training
- **Forward-looking features**: Never use future data

#### Regularization
- L1/L2 penalties
- Early stopping
- Dropout (neural networks)
- Feature selection

#### Validation Metrics
- **R²**: Explained variance
- **MAE**: Mean absolute error
- **RMSE**: Root mean squared error
- **IC**: Information coefficient (correlation of predictions to actuals)
- **Rank IC**: Spearman correlation (more robust)

## Feature Engineering Pipeline

### Transformation Steps

1. **Handle Missing Data**
   - Forward fill (for time-series)
   - Median imputation
   - Indicator variables for missingness

2. **Winsorization**
   - Cap outliers at 1st/99th percentile
   - Prevents extreme values from dominating

3. **Normalization**
   - Z-score normalization: `(x - mean) / std`
   - Cross-sectional (across stocks at each time)
   - Time-series (for each stock over time)

4. **Feature Interactions**
   - Earnings surprise × Momentum
   - Insider buying × Valuation
   - GitHub activity × Analyst revisions

5. **Dimensionality Reduction** (Optional)
   - PCA for correlated features
   - Feature selection (mutual information, LASSO)

### Feature Selection Methods

#### Filter Methods
- Correlation with target
- Mutual information
- Variance threshold

#### Wrapper Methods
- Recursive feature elimination (RFE)
- Forward/backward selection

#### Embedded Methods
- LASSO coefficients
- Tree feature importance
- SHAP values

## Model Evaluation

### Performance Metrics

#### Regression Metrics
- **R² Score**: % of variance explained (target > 0.05 for alpha)
- **Mean Absolute Error**: Average prediction error
- **RMSE**: Penalizes large errors more

#### Trading-Specific Metrics
- **Information Coefficient**: Correlation of predictions to actuals
- **Rank IC**: Spearman correlation (robust to outliers)
- **Hit Rate**: % of positive returns predicted correctly
- **Long/Short Return**: Top quintile vs bottom quintile

### Backtesting Integration

```python
# Monthly rebalancing simulation
for month in test_months:
    # Get predictions for all stocks
    predictions = model.predict(features[month])

    # Select top N stocks (highest predicted relative returns)
    long_positions = predictions.nlargest(10)

    # Track 30-day forward returns
    actual_returns = calculate_returns(long_positions, days=30)

    # Calculate portfolio metrics
    portfolio_return = actual_returns.mean()
    sharpe = portfolio_return / actual_returns.std()
```

## Implementation Phases

### Phase 1: Baseline (Simple)
- Linear model (Ridge regression)
- Basic technical features only (momentum, volatility)
- Simple train/test split
- Goal: Establish baseline R² and IC

### Phase 2: Enhanced Features
- Add fundamental features (P/E, earnings)
- Add alternative data (insider, earnings surprise)
- Improve validation (walk-forward)
- Goal: Beat baseline by 20%+

### Phase 3: Advanced Models
- XGBoost / LightGBM
- Feature engineering (interactions, selection)
- Hyperparameter tuning
- Goal: R² > 0.05, Rank IC > 0.05

### Phase 4: Ensemble
- Combine multiple models
- Meta-model for final predictions
- Production pipeline
- Goal: Robust, deployable system

## Success Criteria

### Minimum Viable Model
- **R² > 0.03**: Explains at least 3% of variance
- **Rank IC > 0.03**: Predictions correlate with actuals
- **Hit rate > 52%**: Better than random
- **Top quintile alpha > 2%/month**: Long-only return beats market

### Good Model
- **R² > 0.05**: 5% variance explained
- **Rank IC > 0.05**: Meaningful correlation
- **Hit rate > 55%**: Clear edge
- **Top quintile alpha > 4%/month**: Strong performance

### Excellent Model
- **R² > 0.10**: 10% variance explained (very strong for stocks)
- **Rank IC > 0.08**: High correlation
- **Hit rate > 60%**: Consistent edge
- **Top quintile alpha > 6%/month**: Exceptional performance

## Next Steps

1. **Data sourcing investigation** ([data_sources_feasibility.md](data_sources_feasibility.md))
2. **Modular code architecture** ([modular_architecture.md](modular_architecture.md))
3. **Feature engineering implementation** (src/features/)
4. **Model training pipeline** (src/models/)
