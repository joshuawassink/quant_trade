# Modular Architecture for Regression Framework

## Design Principles

1. **Separation of Concerns**: Each module has one clear responsibility
2. **Reusability**: Components work across different strategies
3. **Testability**: Every module can be tested independently
4. **Extensibility**: Easy to add new data sources, features, models
5. **Type Safety**: Full type hints for better IDE support and fewer bugs

## Module Overview

```
src/
├── data/                    # Data acquisition and storage
│   ├── __init__.py
│   ├── providers/          # Data source implementations
│   │   ├── __init__.py
│   │   ├── base.py         # Abstract base class
│   │   ├── yfinance_provider.py
│   │   ├── insider_provider.py
│   │   ├── github_provider.py
│   │   └── fundamental_provider.py
│   ├── storage/            # Data persistence
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── parquet_storage.py
│   │   └── cache.py
│   └── universe.py         # Stock universe management
│
├── features/                # Feature engineering
│   ├── __init__.py
│   ├── base.py             # Base feature class
│   ├── technical.py        # Price-based features
│   ├── fundamental.py      # Company fundamentals
│   ├── alternative.py      # Alt data features
│   ├── market.py           # Market regime features
│   ├── transformers.py     # Feature transformations
│   └── pipeline.py         # Feature pipeline orchestration
│
├── models/                  # Machine learning models
│   ├── __init__.py
│   ├── base.py             # Base model interface
│   ├── linear.py           # Ridge, Lasso, ElasticNet
│   ├── ensemble.py         # XGBoost, LightGBM, RandomForest
│   ├── neural.py           # Neural networks
│   ├── meta.py             # Meta-model (ensemble of models)
│   └── evaluation.py       # Model evaluation metrics
│
├── backtesting/             # Backtesting engine
│   ├── __init__.py
│   ├── engine.py           # Main backtest runner
│   ├── portfolio.py        # Portfolio management
│   ├── universe.py         # Stock selection
│   └── metrics.py          # Performance metrics
│
├── strategies/              # Strategy implementations
│   ├── __init__.py
│   ├── base.py             # Base strategy class
│   ├── earnings_momentum.py
│   ├── insider_cluster.py
│   └── ml_momentum.py      # ML-based momentum (our focus)
│
├── execution/               # Trade execution (future)
│   ├── __init__.py
│   ├── broker.py
│   └── order_manager.py
│
├── analysis/                # Results analysis
│   ├── __init__.py
│   ├── performance.py      # Performance analytics
│   ├── attribution.py      # Feature attribution
│   └── visualization.py    # Plotting utilities
│
└── utils/                   # Shared utilities
    ├── __init__.py
    ├── dates.py            # Date utilities
    ├── config.py           # Configuration management
    ├── logging.py          # Logging setup
    └── validation.py       # Data validation
```

## Module Details

### 1. Data Module (`src/data/`)

#### Base Provider Protocol

```python
# src/data/providers/base.py
from typing import Protocol, Optional
import polars as pl
from datetime import datetime

class DataProvider(Protocol):
    """Base protocol for data providers"""

    def fetch(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pl.DataFrame:
        """Fetch data for given symbols and date range"""
        ...

    def validate(self, df: pl.DataFrame) -> bool:
        """Validate fetched data"""
        ...

    def get_available_symbols(self) -> list[str]:
        """Get list of available symbols"""
        ...
```

#### Example Implementation

```python
# src/data/providers/yfinance_provider.py
import yfinance as yf
import polars as pl
from datetime import datetime
from .base import DataProvider

class YFinanceProvider:
    """Provider for Yahoo Finance data"""

    def fetch(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pl.DataFrame:
        """Fetch OHLCV data from Yahoo Finance"""
        # Implementation
        data = yf.download(
            symbols,
            start=start_date,
            end=end_date,
            group_by='ticker'
        )
        return self._to_polars(data)

    def _to_polars(self, df) -> pl.DataFrame:
        """Convert pandas DataFrame to Polars"""
        # Conversion logic
        pass

    def validate(self, df: pl.DataFrame) -> bool:
        """Ensure data quality"""
        # Check for missing dates, prices > 0, etc.
        pass
```

#### Storage Interface

```python
# src/data/storage/base.py
from typing import Protocol
import polars as pl
from pathlib import Path

class DataStorage(Protocol):
    """Base protocol for data storage"""

    def save(self, df: pl.DataFrame, path: Path) -> None:
        """Save DataFrame to storage"""
        ...

    def load(self, path: Path) -> pl.DataFrame:
        """Load DataFrame from storage"""
        ...

    def exists(self, path: Path) -> bool:
        """Check if data exists"""
        ...
```

---

### 2. Features Module (`src/features/`)

#### Base Feature Class

```python
# src/features/base.py
from typing import Protocol
import polars as pl

class Feature(Protocol):
    """Base protocol for feature computation"""

    name: str
    dependencies: list[str]  # Required columns

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute feature and add as new column(s)

        Args:
            df: Input DataFrame with required dependencies

        Returns:
            DataFrame with new feature column(s) added
        """
        ...

    def validate_dependencies(self, df: pl.DataFrame) -> bool:
        """Check if all required columns exist"""
        return all(col in df.columns for col in self.dependencies)
```

#### Example Technical Features

```python
# src/features/technical.py
import polars as pl
from .base import Feature

class MomentumFeature:
    """Calculate price momentum over various windows"""

    name = "momentum"
    dependencies = ["close", "date", "symbol"]

    def __init__(self, windows: list[int] = [5, 10, 20, 60]):
        self.windows = windows

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add momentum columns for each window"""
        result = df.clone()

        for window in self.windows:
            result = result.with_columns([
                (
                    (pl.col("close") / pl.col("close").shift(window))
                    - 1.0
                ).over("symbol").alias(f"return_{window}d")
            ])

        return result


class RelativeStrengthFeature:
    """Calculate returns relative to market (SPY)"""

    name = "relative_strength"
    dependencies = ["close", "date", "symbol"]

    def __init__(self, market_symbol: str = "SPY"):
        self.market_symbol = market_symbol

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate stock return - market return"""
        # Get market returns
        market_returns = (
            df.filter(pl.col("symbol") == self.market_symbol)
            .select(["date", pl.col("close").alias("market_close")])
        )

        # Join and calculate relative strength
        result = df.join(market_returns, on="date", how="left")

        # Calculate for various windows
        for window in [5, 10, 20]:
            result = result.with_columns([
                (
                    (pl.col("close") / pl.col("close").shift(window) - 1.0)
                    - (pl.col("market_close") / pl.col("market_close").shift(window) - 1.0)
                ).over("symbol").alias(f"rel_strength_{window}d")
            ])

        return result.drop("market_close")
```

#### Feature Pipeline

```python
# src/features/pipeline.py
import polars as pl
from typing import Sequence
from .base import Feature

class FeaturePipeline:
    """Orchestrate feature computation"""

    def __init__(self, features: Sequence[Feature]):
        self.features = features

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all features in sequence"""
        result = df.clone()

        for feature in self.features:
            if not feature.validate_dependencies(result):
                raise ValueError(
                    f"Feature {feature.name} missing dependencies: "
                    f"{feature.dependencies}"
                )
            result = feature.compute(result)

        return result

    def get_feature_names(self) -> list[str]:
        """Get names of all computed features"""
        return [f.name for f in self.features]
```

---

### 3. Models Module (`src/models/`)

#### Base Model Interface

```python
# src/models/base.py
from typing import Protocol, Optional, Any
import polars as pl
import numpy as np
from pathlib import Path

class Model(Protocol):
    """Base protocol for regression models"""

    def fit(
        self,
        X: pl.DataFrame | np.ndarray,
        y: pl.Series | np.ndarray,
        **kwargs
    ) -> None:
        """Train the model"""
        ...

    def predict(self, X: pl.DataFrame | np.ndarray) -> np.ndarray:
        """Generate predictions"""
        ...

    def save(self, path: Path) -> None:
        """Save model to disk"""
        ...

    def load(self, path: Path) -> None:
        """Load model from disk"""
        ...

    def get_feature_importance(self) -> Optional[dict[str, float]]:
        """Get feature importance scores (if available)"""
        ...
```

#### Example Model Implementation

```python
# src/models/ensemble.py
import polars as pl
import numpy as np
import xgboost as xgb
from pathlib import Path
from .base import Model

class XGBoostRegressor:
    """XGBoost wrapper for relative return prediction"""

    def __init__(self, **params):
        """
        Initialize XGBoost model

        Args:
            params: XGBoost hyperparameters
        """
        default_params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 5,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        default_params.update(params)
        self.params = default_params
        self.model: Optional[xgb.Booster] = None
        self.feature_names: Optional[list[str]] = None

    def fit(
        self,
        X: pl.DataFrame | np.ndarray,
        y: pl.Series | np.ndarray,
        eval_set: Optional[tuple] = None,
        **kwargs
    ) -> None:
        """Train XGBoost model"""
        # Convert to numpy if needed
        if isinstance(X, pl.DataFrame):
            self.feature_names = X.columns
            X = X.to_numpy()
        if isinstance(y, pl.Series):
            y = y.to_numpy()

        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names)

        # Validation set if provided
        evals = []
        if eval_set is not None:
            X_val, y_val = eval_set
            if isinstance(X_val, pl.DataFrame):
                X_val = X_val.to_numpy()
            if isinstance(y_val, pl.Series):
                y_val = y_val.to_numpy()
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals = [(dtrain, 'train'), (dval, 'val')]

        # Train
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=kwargs.get('num_boost_round', 100),
            evals=evals,
            early_stopping_rounds=kwargs.get('early_stopping_rounds', 20),
            verbose_eval=kwargs.get('verbose_eval', False)
        )

    def predict(self, X: pl.DataFrame | np.ndarray) -> np.ndarray:
        """Generate predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        if isinstance(X, pl.DataFrame):
            X = X.to_numpy()

        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dtest)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        importance = self.model.get_score(importance_type='gain')
        return importance

    def save(self, path: Path) -> None:
        """Save model to disk"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        self.model.save_model(str(path))

    def load(self, path: Path) -> None:
        """Load model from disk"""
        self.model = xgb.Booster()
        self.model.load_model(str(path))
```

---

### 4. Strategy Module (`src/strategies/`)

#### Base Strategy

```python
# src/strategies/base.py
from typing import Protocol
import polars as pl
from datetime import datetime

class Strategy(Protocol):
    """Base protocol for trading strategies"""

    name: str

    def generate_signals(
        self,
        data: pl.DataFrame,
        date: datetime
    ) -> pl.DataFrame:
        """
        Generate trading signals for given date

        Args:
            data: Historical data up to (and including) date
            date: Current date for signal generation

        Returns:
            DataFrame with columns: [symbol, signal, confidence]
            where signal is predicted return and confidence is 0-1
        """
        ...

    def select_positions(
        self,
        signals: pl.DataFrame,
        n_positions: int = 10,
        min_confidence: float = 0.5
    ) -> list[str]:
        """
        Select positions based on signals

        Args:
            signals: Output from generate_signals
            n_positions: Number of positions to select
            min_confidence: Minimum confidence threshold

        Returns:
            List of symbols to trade
        """
        ...
```

#### ML Momentum Strategy

```python
# src/strategies/ml_momentum.py
import polars as pl
from datetime import datetime
from typing import Optional
from .base import Strategy
from ..features.pipeline import FeaturePipeline
from ..models.base import Model

class MLMomentumStrategy:
    """ML-based momentum strategy using regression predictions"""

    name = "ml_momentum"

    def __init__(
        self,
        feature_pipeline: FeaturePipeline,
        model: Model,
        lookback_days: int = 252  # 1 year of history
    ):
        self.feature_pipeline = feature_pipeline
        self.model = model
        self.lookback_days = lookback_days

    def generate_signals(
        self,
        data: pl.DataFrame,
        date: datetime
    ) -> pl.DataFrame:
        """Generate signals using trained model"""
        # Filter data up to current date
        historical = data.filter(pl.col("date") <= date)

        # Get most recent data point for each symbol
        latest = (
            historical
            .group_by("symbol")
            .agg(pl.all().sort_by("date").last())
        )

        # Compute features
        features = self.feature_pipeline.transform(latest)

        # Get feature columns (exclude metadata)
        feature_cols = [
            col for col in features.columns
            if col not in ["symbol", "date", "close", "volume"]
        ]

        # Generate predictions
        X = features.select(feature_cols)
        predictions = self.model.predict(X)

        # Create signals DataFrame
        signals = pl.DataFrame({
            "symbol": features["symbol"],
            "date": date,
            "predicted_return": predictions,
            "confidence": self._calculate_confidence(predictions)
        })

        return signals

    def _calculate_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """Convert predictions to confidence scores (0-1)"""
        # Use absolute value, normalized to 0-1
        abs_pred = np.abs(predictions)
        return np.clip(abs_pred / np.percentile(abs_pred, 95), 0, 1)

    def select_positions(
        self,
        signals: pl.DataFrame,
        n_positions: int = 10,
        min_confidence: float = 0.3
    ) -> list[str]:
        """Select top N positions with highest predicted returns"""
        filtered = signals.filter(pl.col("confidence") >= min_confidence)

        top_positions = (
            filtered
            .sort("predicted_return", descending=True)
            .head(n_positions)
        )

        return top_positions["symbol"].to_list()
```

---

## Configuration Management

```python
# src/utils/config.py
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class DataConfig:
    """Data-related configuration"""
    data_dir: Path
    cache_dir: Path
    providers: list[str]
    universe_file: Path

@dataclass
class ModelConfig:
    """Model-related configuration"""
    model_type: str  # 'xgboost', 'ridge', etc.
    hyperparameters: dict
    features: list[str]
    target: str

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: str
    end_date: str
    initial_capital: float
    n_positions: int
    rebalance_frequency: str  # 'monthly', 'weekly', etc.

@dataclass
class Config:
    """Main configuration"""
    data: DataConfig
    model: ModelConfig
    backtest: BacktestConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file"""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        # Parse and create config objects
        # ...
        return cls(...)
```

---

## Usage Example

```python
# Example: Train and backtest ML momentum strategy

from src.data.providers.yfinance_provider import YFinanceProvider
from src.features.technical import MomentumFeature, RelativeStrengthFeature
from src.features.pipeline import FeaturePipeline
from src.models.ensemble import XGBoostRegressor
from src.strategies.ml_momentum import MLMomentumStrategy
from src.backtesting.engine import BacktestEngine

# 1. Set up data provider
data_provider = YFinanceProvider()
symbols = ["AAPL", "MSFT", "GOOGL", "META", "NVDA"]
data = data_provider.fetch(symbols, start_date, end_date)

# 2. Set up feature pipeline
features = [
    MomentumFeature(windows=[5, 10, 20, 60]),
    RelativeStrengthFeature(),
    # ... more features
]
feature_pipeline = FeaturePipeline(features)

# 3. Train model
model = XGBoostRegressor(learning_rate=0.05, max_depth=5)
# ... training logic

# 4. Create strategy
strategy = MLMomentumStrategy(feature_pipeline, model)

# 5. Backtest
backtest = BacktestEngine(strategy, data)
results = backtest.run(start_date, end_date)
```

---

## Testing Strategy

### Unit Tests
```python
# tests/features/test_technical.py
import polars as pl
from src.features.technical import MomentumFeature

def test_momentum_feature():
    # Create sample data
    df = pl.DataFrame({
        "symbol": ["AAPL"] * 100,
        "date": pl.date_range(...),
        "close": [100, 101, 102, ...]  # Mock prices
    })

    # Compute feature
    feature = MomentumFeature(windows=[5])
    result = feature.compute(df)

    # Assertions
    assert "return_5d" in result.columns
    assert result["return_5d"][5] == pytest.approx(0.02, rel=1e-2)
```

### Integration Tests
Test full pipeline end-to-end

---

## Next Steps

1. **Implement base classes** (Protocol definitions)
2. **Build data providers** (yfinance first)
3. **Implement technical features** (momentum, volatility)
4. **Train baseline model** (Ridge regression)
5. **Set up backtesting** (simple monthly rebalance)

This modular design allows each component to be developed, tested, and improved independently!
