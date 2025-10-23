# Quant Trade Framework

A modular algorithmic trading framework for researching, backtesting, and executing non-traditional trading strategies.

## Philosophy

This framework prioritizes:
- **Manual validation** before automation
- **Non-traditional strategies** over conventional technical indicators
- **Systematic testing** with clear metrics
- **Modular architecture** for easy experimentation

## Project Structure

```
quant_trade/
├── src/              # Core framework code
├── tests/            # Test suite
├── data/             # Local data storage
├── docs/             # Documentation
├── notebooks/        # Research notebooks
└── configs/          # Configuration files
```

## Strategy Development Process

1. **Research**: Document hypothesis and approach
2. **Manual Test**: Validate logic with sample data
3. **Implement**: Code the strategy following framework interfaces
4. **Backtest**: Run historical simulations
5. **Analyze**: Review metrics and refine

## Getting Started

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for fast package management

### Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repo-url>
cd quant_trade

# Create virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# .venv\Scripts\activate   # On Windows

# Install dependencies
uv pip install -e ".[dev]"
```

### Quick Start
```python
# Example coming soon
```

### Why uv?
This project uses [uv](https://github.com/astral-sh/uv) for dependency management because it's:
- 10-100x faster than pip
- Handles both virtual environments and packages
- Compatible with standard Python packaging (pyproject.toml)
- Becoming the industry standard for Python projects

## Strategy Categories

The framework supports exploration of non-traditional strategies including:

- **Market Microstructure**: Order book dynamics, trade size patterns
- **Alternative Data**: GitHub activity, job postings, satellite imagery
- **Temporal Patterns**: Non-standard seasonality and calendar effects
- **Cross-Asset**: Correlation breakdowns, volatility regime shifts
- **Behavioral**: Sentiment divergence, insider activity clustering

See [docs/strategies/](docs/strategies/) for detailed strategy documentation.

## Development Status

Currently in initial development. Core framework components being established.

## Documentation

- [claude.md](claude.md) - Project context and architecture details
- [docs/](docs/) - Additional documentation

## Contributing

This is a personal research project. Documentation and code standards are maintained for clarity and reproducibility.

## License

TBD
