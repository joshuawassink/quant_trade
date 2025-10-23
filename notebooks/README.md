# Research Notebooks

Jupyter notebooks for interactive data exploration, strategy development, and analysis.

## Notebooks

### Data Exploration
- **01_data_exploration.ipynb** - Initial data quality checks and visualization

### Strategy Development
(Coming soon)
- Feature engineering experiments
- Model training and evaluation
- Strategy backtesting

## Getting Started

### Launch Jupyter

```bash
# Activate virtual environment
source .venv/bin/activate

# Start Jupyter
jupyter notebook
```

Or for JupyterLab:
```bash
jupyter lab
```

### Kernel Setup

The notebooks use the project's virtual environment. If you need to set up the kernel:

```bash
python -m ipykernel install --user --name=quant_trade --display-name="Quant Trade"
```

## Best Practices

1. **Save outputs**: Include visualizations and key results in committed notebooks
2. **Clear outputs**: Clear outputs of large data before committing (git diff will be huge otherwise)
3. **Reproducibility**: Include random seeds and version info for reproducible experiments
4. **Documentation**: Add markdown cells explaining your analysis
5. **Modularity**: Move reusable code to `src/` modules

## Notebook Organization

- Start with data loading and validation
- Include summary statistics
- Add visualizations
- Document insights and next steps
- Keep notebooks focused on one topic
