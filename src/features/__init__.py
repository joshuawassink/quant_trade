"""
Feature Engineering Module

Transforms raw data into ML-ready features for regression models.

Modules:
- technical: Technical indicators (RSI, MACD, momentum, volatility)
- fundamental: Fundamental metrics (ROE changes, margin trends)
- sector: Sector/market relative features (relative strength)
- alignment: Utilities for merging and forward-filling data
- transformers: sklearn-compatible feature transformers
"""

from .technical import TechnicalFeatures

__all__ = [
    'TechnicalFeatures',
]
