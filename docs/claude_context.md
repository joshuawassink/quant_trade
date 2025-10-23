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

### Core Documentation Map

**Project Management:**
- [../TODO_NEXT_SESSION.md](../TODO_NEXT_SESSION.md) - **START HERE** - Next session priorities & context
- [../TODO.md](../TODO.md) - Long-term task tracker (review weekly)
- [../README.md](../README.md) - Project overview
- [../QUICKSTART.md](../QUICKSTART.md) - Getting started guide

**Development Standards:**
- [development/git_workflow.md](development/git_workflow.md) - Git best practices and commit conventions
- [development/data_conventions.md](development/data_conventions.md) - **CRITICAL** Data formatting, naming, type standards
  - Symbol formatting (uppercase)
  - Date handling (datetime[ns], timezone-naive, truncated)
  - Column naming (snake_case with clear suffixes)
  - Null handling, winsorization, schema validation
  - **Read this before any data work!**

**Architecture & Design:**
- [architecture/regression_framework_spec.md](architecture/regression_framework_spec.md) - ML framework specification
- [architecture/feature_summary.md](architecture/feature_summary.md) - Feature categories and targets
- [architecture/modular_architecture.md](architecture/modular_architecture.md) - System design
- [architecture/data_acquisition_guide.md](architecture/data_acquisition_guide.md) - Data source guide

**Strategy Documentation:**
- [strategies/README.md](strategies/README.md) - Strategy index
- [strategies/earnings_momentum_1m.md](strategies/earnings_momentum_1m.md) - Earnings-based strategy
- [strategies/insider_cluster_1m.md](strategies/insider_cluster_1m.md) - Insider trading clusters
- [strategies/github_activity_1m.md](strategies/github_activity_1m.md) - Developer activity signals

**Reports & Evaluations:**
- [reports/feature_evaluation_2025-10-23.md](reports/feature_evaluation_2025-10-23.md) - Latest feature evaluation
- [reports/codebase_review_2025-10-23.md](reports/codebase_review_2025-10-23.md) - Latest codebase analysis

---

## Recent Changes & Important Findings (2025-10-23)

### Feature Engineering Pipeline Complete ✓
**Status**: 111 features implemented across 3 modules
- **Technical features** (33): Returns, volatility, volume, RSI, MACD, moving averages
- **Fundamental features** (22): QoQ/YoY changes, trend indicators, financial ratios
- **Sector/Market features** (56): Market-relative returns, VIX regime, sector ETF tracking

**Key Files:**
- [src/features/technical.py](../src/features/technical.py)
- [src/features/fundamental.py](../src/features/fundamental.py)
- [src/features/sector.py](../src/features/sector.py)
- [scripts/evaluate_features.py](../scripts/evaluate_features.py)

### Critical Bug Fixed: Date Join Mismatch
**Problem**: Market data (VIX, SPY, ETFs) had different hour timestamps than stock data
- VIX: `2025-10-22 01:00:00`
- Stocks: `2025-10-22 00:00:00`
- Result: 100% null VIX features

**Solution**: Truncate all dates to day before joining (`.dt.truncate('1d')`)
**Impact**: All market joins now working correctly
**Documentation**: See [development/data_conventions.md](development/data_conventions.md) Section 2

### Data Conventions Established
**Purpose**: Prevent subtle bugs like the date join issue above
**Key conventions:**
- Symbols: Always UPPERCASE
- Dates: datetime[ns], timezone-naive, truncated to day
- Columns: snake_case with clear suffixes (e.g., `return_20d_vs_market`)
- Booleans: Store as Int8 (0/1) for null handling
- See full conventions: [development/data_conventions.md](development/data_conventions.md)

### Documentation Reorganized
**Changes**: Moved detailed docs from root to `docs/` subdirectories
- `claude.md` → `docs/claude_context.md`
- `GIT_BEST_PRACTICES.md` → `docs/development/git_workflow.md`
- `DATA_CONVENTIONS.md` → `docs/development/data_conventions.md`
- Reports → `docs/reports/` (with dates in filenames)

**Root directory now contains only**: README, QUICKSTART, TODO

---

## Git Workflow & Best Practices

### Branching Strategy
- **Work on `main`** for tested, working changes
- **Use feature branches** only for experimental/risky work
- Branch naming: `feature/`, `fix/`, `refactor/`, `docs/`, `test/`, `data/`

### Commit Guidelines

**Frequency**: Commit every logical unit of work (~3-5 times per coding session)
- ✅ After completing a function/class/feature component
- ✅ When tests are passing
- ✅ Before switching tasks
- ✅ End of coding session

**Format**: Conventional Commits
```
<type>(<scope>): <description>

<optional body>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

**Types**: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `data`, `perf`

**Examples**:
```bash
feat(features): add RSI technical indicator
fix(providers): handle missing quarterly data gracefully
refactor(validation): extract common validation logic
docs(strategies): update earnings momentum documentation
test(providers): add unit tests for price provider
data(universe): expand stock universe to 50 symbols
```

### What to Commit

**✅ Always Commit**:
- Source code changes
- Tests for those changes
- Documentation updates
- Configuration changes (no secrets!)

**❌ Never Commit**:
- `.env` files or secrets
- `__pycache__/` or `.pyc` files
- Virtual environments (`.venv/`)
- Large data files (`.parquet` files >1MB)
- API keys, passwords, tokens
- Personal notes (unless for team)

**Notebooks**:
- Clear output before committing (clean diffs)
- Use: `jupyter nbconvert --clear-output --inplace notebook.ipynb`

### Claude's Workflow

**Before Every Commit**:
1. Show `git status` and explain changes
2. Show proposed commit message
3. Highlight any concerns (large files, secrets, etc.)
4. **Wait for your approval** before executing

**Commit Process**:
```bash
# Review changes
git status
git diff --staged

# Commit with conventional format
git commit -m "type(scope): description"

# Push to GitHub
git push origin main
```

### Pre-Commit Checklist

Before committing, verify:
- [ ] No debug `print()` statements (unless intentional)
- [ ] No `import pdb` or breakpoints
- [ ] No secrets or API keys in diff
- [ ] All staged files are intentional
- [ ] Tests pass (when we have them)
- [ ] Code is in working state

### File Reference
- Full workflow: [docs/development/git_workflow.md](development/git_workflow.md)
- Emergency procedures in git_workflow.md section 10
- Data conventions: [docs/development/data_conventions.md](development/data_conventions.md)
