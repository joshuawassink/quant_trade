# Project TODO Tracker

## Purpose
Track future action items, design decisions, and implementation tasks that need to be addressed as the project evolves. Review this document periodically to ensure nothing is forgotten.

---

## High Priority

### Data & Features

- [ ] **Develop comprehensive missing data strategy**
  - Model-specific handling (XGBoost native support vs linear models)
  - Company/sector-specific strategies (e.g., GitHub features only for tech)
  - Feature-specific approaches (forward-fill for slow-changing metadata, imputation for fundamentals)
  - Create missing value indicator features where appropriate
  - Document rules in `docs/architecture/missing_data_strategy.md`
  - **Context**: Different models handle missingness differently; strategy should vary by feature type
  - **Priority**: Before production model training

- [ ] **Build metadata time-series database**
  - Weekly snapshots of company info (sector, employees, market cap)
  - Handle sector reclassifications
  - Track corporate actions (mergers, delistings)
  - **Why**: Current APIs only provide snapshots, need historical context

- [ ] **Set up automated data pipeline**
  - Daily: Price data, sector ETFs, market indicators, insider data
  - Weekly: Metadata, GitHub activity, fundamentals
  - Error handling and data quality checks
  - **Priority**: After initial manual data collection

### Model Development

- [ ] **Implement walk-forward validation**
  - Time-series aware train/test splits
  - Purged k-fold to prevent lookahead bias
  - Embargo periods between train/val/test
  - **Priority**: Before any backtesting

- [ ] **Feature importance analysis framework**
  - SHAP values for model interpretability
  - Feature correlation analysis
  - Identify redundant features
  - **Why**: 115 features - need to understand what matters

- [ ] **Hyperparameter tuning pipeline**
  - Grid search / Bayesian optimization
  - Cross-validation strategy
  - Save best parameters per model type
  - **Priority**: After baseline models working

- [ ] **Use rank-ordering as loss metric for model training**
  - Current: Models optimize for prediction accuracy (MSE/MAE)
  - Better: Optimize for correct rank ordering of stocks
  - Benefit: In practice, we pick top K stocks regardless of exact return predictions
  - Implementation: Use ranking loss functions (e.g., LambdaRank, ListNet, pairwise ranking)
  - **Context**: Spearman correlation shows ranking is more important than exact values
  - **Priority**: After baseline models proven, before production

- [ ] **Implement market regime prediction model**
  - Parallel prediction of overall market movement (SPY returns)
  - Use for position sizing and strategy selection:
    - Strong positive: Aggressive (call options, high leverage)
    - Mixed/neutral: Focus on selection quality (long equity, maximize %-positive)
    - Strong negative: Defensive (cash, inverse ETFs)
  - **Context**: Strategy should adapt to market conditions
  - **Priority**: After stock selection model working

### Strategy Implementation

- [ ] **Manual validation of earnings momentum strategy**
  - Test on 12 months historical data
  - Document learnings in strategy doc
  - **Priority**: Can do in parallel with ML work

- [ ] **Manual validation of insider cluster strategy**
  - Download OpenInsider historical data
  - Test clustering signals manually
  - **Priority**: Can do in parallel with ML work

### Production & Operations

- [ ] **Model versioning system**
  - Track model versions, hyperparameters, performance
  - Ability to roll back to previous models
  - **Priority**: Before live trading

- [ ] **Monitoring and alerting**
  - Model performance degradation detection
  - Data quality alerts
  - Prediction distribution monitoring
  - **Priority**: Before live trading

---

## Medium Priority

### Data Enhancements

- [ ] **Investigate Financial Modeling Prep integration**
  - Cost: $14/mo
  - Benefit: 10+ years of fundamentals, earnings surprises, analyst data
  - **Decision point**: After validating free data works

- [ ] **Build SEC EDGAR scraper for insider data**
  - Replace manual OpenInsider downloads
  - Real-time Form 4 monitoring
  - **Priority**: After initial insider features working

- [ ] **GitHub Archive integration for extended history**
  - Currently limited to 52 weeks via API
  - GH Archive has full history since 2011
  - **Decision**: Only if 52 weeks insufficient

### Feature Engineering

- [ ] **Feature interaction terms**
  - Earnings surprise × Momentum
  - Insider buying × Valuation
  - Sector strength × Stock beta
  - **Priority**: After baseline features working

- [ ] **Dimensionality reduction exploration**
  - PCA on correlated features
  - Factor models (Fama-French style)
  - **Why**: Reduce overfitting with 115 features

- [ ] **Alternative target variables**
  - Risk-adjusted returns (Sharpe-style)
  - Sector-relative returns (vs SPY)
  - Ranking/classification instead of regression
  - **Priority**: After baseline working

### Testing & Quality

- [ ] **Comprehensive unit tests**
  - Feature computation correctness
  - Data provider error handling
  - Model training/prediction
  - **Priority**: As we build each component

- [ ] **Integration tests**
  - End-to-end pipeline (data → features → model → predictions)
  - Backtest accuracy verification
  - **Priority**: Before production

- [ ] **Data quality monitoring**
  - Missing data detection
  - Outlier detection
  - Data consistency checks (e.g., negative prices)
  - **Priority**: Before automated pipelines

---

## Low Priority / Future Enhancements

### Advanced Features

- [ ] **Sentiment analysis**
  - News sentiment (via APIs or scraping)
  - Social media sentiment (Twitter/Reddit)
  - **Cost**: Likely paid APIs required

- [ ] **Options data integration**
  - Implied volatility
  - Put/call ratios
  - Unusual options activity
  - **Cost**: Expensive data

- [ ] **Short interest data**
  - Available every 2 weeks
  - Potential signal for squeezes
  - **Source**: Some free sources available

### Model Enhancements

- [ ] **Neural network models**
  - Feed-forward architecture
  - LSTM for time-series
  - **Priority**: After tree-based models working

- [ ] **Ensemble meta-models**
  - Stack multiple model predictions
  - Weighted combinations
  - **Priority**: After individual models working

- [ ] **Online learning / model updating**
  - Retrain periodically on new data
  - Incremental learning approaches
  - **Priority**: After production deployment

### Infrastructure

- [ ] **Move from SQLite to PostgreSQL**
  - **When**: If dataset > 100GB or need concurrent access
  - Current: SQLite sufficient for single-user

- [ ] **Distributed backtesting**
  - Parallel processing across multiple cores/machines
  - **When**: If backtests take > 1 hour

- [ ] **Cloud deployment**
  - AWS/GCP for production
  - **Priority**: After proven profitability

### Strategy Expansion

- [ ] **Options strategies**
  - Long calls for high conviction
  - Spreads for defined risk
  - **Priority**: After stock strategies proven

- [ ] **Multi-timeframe strategies**
  - Weekly/quarterly rebalancing
  - Swing trading (3-7 days)
  - **Priority**: After 1-month strategy working

- [ ] **Portfolio optimization**
  - Modern portfolio theory
  - Risk parity
  - Factor-based allocation
  - **Priority**: After individual signals working

---

## Completed ✅

- [x] Initial project structure (src/, docs/, tests/, data/)
- [x] Documentation framework (README, claude.md)
- [x] Virtual environment setup (uv)
- [x] Git repository initialization
- [x] Regression framework specification
- [x] Data sources feasibility analysis
- [x] Modular architecture design
- [x] Data acquisition guide
- [x] Feature set definition (115 features)
- [x] Stock metadata and sector/market features added

---

## Notes

**Review Frequency**:
- Weekly during active development
- Monthly during stable operation

**Priority Definitions**:
- **High**: Blocking current work or critical for next milestone
- **Medium**: Important but not blocking, nice-to-have soon
- **Low**: Future enhancements, explore when time permits

**Adding Items**:
When you identify a future task during development:
1. Add to appropriate priority section
2. Include context/reasoning
3. Note any dependencies or decision points
