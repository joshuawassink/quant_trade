# Claude Context - Quant Trade Framework

## Project Overview
This is an algorithmic trading framework designed for researching, backtesting, and executing non-traditional trading strategies. The focus is on manual validation and systematic testing of novel approaches.

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

## Current Phase
Initial setup - establishing project structure and documentation framework.
