# Claude Context - Quant Trade Framework

## Project Overview
This is an algorithmic trading framework designed for researching, backtesting, and executing non-traditional trading strategies. The focus is on manual validation and systematic testing of novel approaches.

### Current Focus: 1-Month Holding Period Strategies
The framework currently prioritizes **fixed 1-month holding period strategies** with these characteristics:
- **Long-only positions** (stocks or call options)
- **No short selling** (avoid unlimited risk)
- **Monthly rebalancing** (enter on Day 1, exit on Day 30)
- **Target returns**: 3-8% per month with manageable risk
- **Manual validation first** before any coding/automation

## Architecture Philosophy
- **Modular Design**: Each component (data ingestion, strategy logic, execution, backtesting) should be independent
- **Strategy-First**: Framework validates strategies manually before automation
- **Non-Traditional Focus**: Prioritize unconventional strategies over standard indicators
- **Testability**: Every component must be easily testable in isolation

## Directory Structure
```
quant_trade/
├── src/
│   ├── data/           # Data ingestion and management
│   ├── strategies/     # Trading strategy implementations
│   ├── backtesting/    # Backtesting engine
│   ├── execution/      # Trade execution logic
│   ├── analysis/       # Performance analysis and metrics
│   └── utils/          # Shared utilities
├── tests/              # Unit and integration tests
├── data/               # Local data storage (gitignored)
├── docs/               # Additional documentation
│   ├── strategies/     # Strategy research and documentation
│   ├── architecture/   # Technical architecture docs
│   └── api/            # API documentation
├── notebooks/          # Jupyter notebooks for research
└── configs/            # Configuration files
```

## Strategy Development Workflow
1. **Research & Document**: Document strategy hypothesis in docs/strategies/
2. **Manual Testing**: Test strategy logic manually with sample data
3. **Implementation**: Implement strategy class following framework interface
4. **Backtesting**: Run historical backtests with various parameters
5. **Analysis**: Analyze performance metrics and edge cases
6. **Refinement**: Iterate based on results

## Non-Traditional Strategy Ideas
These are unconventional approaches to explore:

### Market Microstructure
- **Order Book Imbalance**: Trade based on bid-ask imbalances beyond simple spread
- **Trade Size Clustering**: Identify patterns in institutional trade sizes
- **Quote Stuffing Detection**: Capitalize on HFT noise patterns

### Alternative Data
- **Commit Activity**: GitHub commit patterns for tech stocks
- **Job Posting Volume**: Employee hiring as leading indicator
- **Satellite Imagery**: Parking lot fullness for retail stocks
- **Web Traffic**: Product page views as demand signal

### Temporal Patterns
- **Intraday Seasonality**: Non-standard time-of-day patterns
- **Calendar Anomalies**: Post-holiday, earnings season patterns
- **Time Decay Acceleration**: Non-linear option decay patterns

### Cross-Asset Relationships
- **Volatility Regime Switching**: Trade based on VIX state changes
- **Correlation Breakdown**: Profit from temporary correlation failures
- **Sector Rotation Leads**: Use leaders to predict laggards

### Behavioral
- **Social Sentiment Divergence**: When sentiment contradicts price action
- **Analyst Revision Momentum**: Speed of estimate changes
- **Insider Clustering**: Multiple insiders acting simultaneously

## Key Technical Considerations
- Use Python 3.11+
- Type hints throughout (mypy strict mode)
- Async support for real-time data
- Polars for data manipulation (performance-first), with NumPy for numerical operations
- Clean separation between research and production code
- Comprehensive logging for debugging

## Development Principles
1. **Document First**: Write strategy docs before code
2. **Test Manually**: Validate logic by hand before automation
3. **Measure Everything**: Track all metrics, even unexpected ones
4. **Fail Fast**: Catch errors early with assertions and validation
5. **Stay Organized**: Keep strategies, data, and analysis separate

## Priority 1-Month Strategies

Three non-traditional strategies documented and ready for manual testing:

1. **Earnings Momentum** ([docs/strategies/earnings_momentum_1m.md](docs/strategies/earnings_momentum_1m.md))
   - Buy stocks that beat earnings by 10%+ with 3-8% price reaction
   - Hold for 30 days to capture post-earnings drift
   - Expected: 5-8% monthly returns, 60-70% win rate
   - Priority: HIGH (well-studied, clear signals)

2. **Insider Clustering** ([docs/strategies/insider_cluster_1m.md](docs/strategies/insider_cluster_1m.md))
   - Buy when 3+ insiders purchase $50K+ within 10 days
   - Signals information asymmetry and confidence
   - Expected: 4-8% monthly returns, 60-70% win rate
   - Priority: HIGH (strong academic support, unique angle on clusters)

3. **GitHub Activity** ([docs/strategies/github_activity_1m.md](docs/strategies/github_activity_1m.md))
   - Buy tech stocks when commit activity accelerates 150%+
   - Leading indicator for product development
   - Expected: 6%+ monthly returns, 55%+ win rate
   - Priority: MEDIUM-HIGH (novel, less competition, tech-focused)

## Current Phase
**Phase 1**: Data pipeline development for ML-based momentum strategy
- Building yfinance data providers (price, metadata, sector/market)
- Implementing feature computation pipeline (technical, metadata, sector/market)
- Creating modular architecture for reusability
- Target: 70+ features, 3-5 years historical data

**Next**: Feature engineering → Model training → Backtesting

## Important Files to Review Periodically

### [TODO.md](TODO.md) - Project Task Tracker
**IMPORTANT**: Review this file regularly to track:
- Future action items (missing data strategy, automated pipelines, etc.)
- Design decisions that need to be made
- Implementation tasks by priority
- Items we've deferred but shouldn't forget

**Review frequency**: Weekly during active development, monthly during maintenance
