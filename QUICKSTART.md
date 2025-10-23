# Quick Start Guide

## Setup (One Time)

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create and activate virtual environment
uv venv
source .venv/bin/activate

# 3. Install dependencies
uv pip install -e ".[dev]"
```

## Daily Workflow

### Start Interactive Analysis

```bash
# Option 1: Use the helper script
./start_jupyter.sh

# Option 2: Manual start
source .venv/bin/activate
cd notebooks
jupyter lab
```

Then open: `01_data_exploration.ipynb`

### Collect Fresh Data

```bash
source .venv/bin/activate
python scripts/fetch_sample_data.py
```

Data will be saved to `data/price/daily/`

### Run Tests (Coming Soon)

```bash
source .venv/bin/activate
pytest tests/
```

## Project Structure

```
quant_trade/
├── src/                    # Source code
│   ├── data/              # Data providers
│   ├── features/          # Feature engineering
│   ├── models/            # ML models
│   └── strategies/        # Trading strategies
├── notebooks/             # Jupyter notebooks
│   └── 01_data_exploration.ipynb
├── scripts/               # Utility scripts
│   └── fetch_sample_data.py
├── data/                  # Data storage (gitignored)
│   └── price/daily/       # Price data
├── tests/                 # Tests
└── docs/                  # Documentation
    ├── strategies/        # Strategy docs
    └── architecture/      # Tech specs
```

## Key Files

- **[TODO.md](TODO.md)** - Project task tracker (review weekly!)
- **[claude.md](claude.md)** - Project context for Claude
- **[README.md](README.md)** - Full documentation
- **[pyproject.toml](pyproject.toml)** - Dependencies and config

## Current Status

✅ **Completed:**
- Project structure and documentation
- YFinance data provider (refactored, vectorized)
- Data collection (20 stocks, 3 years)
- Jupyter notebook for exploration

🔄 **In Progress:**
- Interactive data validation

⏭️ **Next Steps:**
- Build metadata provider (sector, market cap, employees)
- Build sector/market data provider (ETFs, VIX)
- Technical feature engineering

## Common Commands

```bash
# Activate environment (do this first!)
source .venv/bin/activate

# Start Jupyter
./start_jupyter.sh

# Fetch data
python scripts/fetch_sample_data.py

# Run a specific script
python scripts/your_script.py

# Run tests
pytest

# Type check
mypy src/

# Lint
ruff check src/
```

## Getting Help

- Check [docs/](docs/) for detailed architecture and strategy docs
- Review [TODO.md](TODO.md) for tracked tasks
- See [notebooks/README.md](notebooks/README.md) for notebook best practices

## Tips

1. **Always activate the venv first**: `source .venv/bin/activate`
2. **Keep notebooks clean**: Clear large outputs before committing
3. **Review TODO.md weekly**: Don't forget deferred tasks
4. **Document as you go**: Update strategy docs when you learn something
5. **Use Polars for data**: It's much faster than Pandas
