# Src/ Directory Reorganization Plan

## New Structure

```
src/
├── shared/                          # Shared, reusable code
│   ├── __init__.py
│   ├── pipeline/                    # Pipeline components (moved from src/pipeline/)
│   │   ├── __init__.py
│   │   ├── data_loading.py
│   │   ├── feature_engineering.py
│   │   ├── target_generation.py
│   │   ├── data_filtering.py
│   │   ├── data_splitting.py
│   │   ├── model_training.py
│   │   ├── model_prediction.py
│   │   ├── model_evaluation.py      # V1
│   │   ├── model_evaluation_v2.py   # V2
│   │   └── model_returns.py
│   │
│   ├── features/                    # Feature engineering (moved from src/features/)
│   │   ├── __init__.py
│   │   ├── alignment.py
│   │   ├── technical.py
│   │   ├── fundamental.py
│   │   └── sector.py
│   │
│   ├── models/                      # Model utilities (moved from src/models/)
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   │
│   ├── data/                        # Data providers (moved from src/data/)
│   │   ├── __init__.py
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── yfinance_provider.py
│   │       ├── yfinance_market_provider.py
│   │       ├── yfinance_financials_provider.py
│   │       └── yfinance_metadata_provider.py
│   │
│   └── config/                      # Configuration (moved from src/config/)
│       ├── __init__.py
│       └── universe.py
│
├── workflows/                       # Workflow-specific customizations
│   ├── __init__.py
│   │
│   ├── returns_30d/                 # 30-day returns workflow
│   │   ├── __init__.py
│   │   ├── custom_features.py       # Workflow-specific features (future)
│   │   ├── custom_preprocessing.py  # Workflow-specific preprocessing (future)
│   │   └── config.py                # Workflow-specific config (future)
│   │
│   ├── returns_60d/                 # 60-day returns workflow (future)
│   │   └── __init__.py
│   │
│   └── volatility/                  # Volatility prediction workflow (future)
│       └── __init__.py
│
├── analysis/                        # Keep as-is (empty placeholder)
│   └── __init__.py
│
├── backtesting/                     # Keep as-is (empty placeholder)
│   └── __init__.py
│
├── execution/                       # Keep as-is (empty placeholder)
│   └── __init__.py
│
├── strategies/                      # Keep as-is (empty placeholder)
│   └── __init__.py
│
├── utils/                           # Keep as-is (empty placeholder)
│   └── __init__.py
│
└── __init__.py
```

## Migration Plan

### Step 1: Create new directories
```bash
mkdir -p src/shared/pipeline
mkdir -p src/shared/features
mkdir -p src/shared/models
mkdir -p src/shared/data/providers
mkdir -p src/shared/config
mkdir -p src/workflows/returns_30d
mkdir -p src/workflows/returns_60d
mkdir -p src/workflows/volatility
```

### Step 2: Move files to shared/
```bash
# Pipeline components
mv src/pipeline/*.py src/shared/pipeline/

# Features
mv src/features/*.py src/shared/features/

# Models
mv src/models/*.py src/shared/models/

# Data providers
mv src/data/providers/*.py src/shared/data/providers/
mv src/data/__init__.py src/shared/data/

# Config
mv src/config/*.py src/shared/config/
```

### Step 3: Update imports in workflow files
```python
# Old imports
from src.pipeline.data_loading import DataLoader
from src.features.alignment import FeatureAligner
from src.models.preprocessing import FeaturePreprocessor
from src.config.universe import get_universe

# New imports
from src.shared.pipeline.data_loading import DataLoader
from src.shared.features.alignment import FeatureAligner
from src.shared.models.preprocessing import FeaturePreprocessor
from src.shared.config.universe import get_universe
```

### Step 4: Create workflow-specific __init__.py files

## Benefits

### 1. Clear Separation
- **shared/** = Reusable across all workflows
- **workflows/** = Workflow-specific customizations

### 2. Future Flexibility
```python
# Example: 60d returns might need custom features
# src/workflows/returns_60d/custom_features.py
class Returns60dFeatureEngineer(FeatureEngineer):
    def compute_features(self, *args, **kwargs):
        # Custom feature computation for 60d
        df = super().compute_features(*args, **kwargs)
        # Add 60d-specific features
        df = df.with_columns([
            # Long-term momentum indicators
            pl.col('return_60d').alias('long_term_momentum')
        ])
        return df
```

### 3. Workflow Isolation
- Each workflow can override shared components
- Changes to one workflow don't affect others
- Easy to experiment with workflow-specific optimizations

### 4. Import Clarity
```python
# Shared component (used by all workflows)
from src.shared.pipeline.data_loading import DataLoader

# Workflow-specific component (only for this workflow)
from src.workflows.returns_30d.custom_features import Custom30dFeatures
```

## Files to Keep vs Remove

### Keep (Active)
- **src/shared/pipeline/** - All 11 files (core pipeline)
- **src/shared/features/** - All 4 files (feature engineering)
- **src/shared/models/** - preprocessing.py
- **src/shared/data/** - All providers
- **src/shared/config/** - universe.py

### Consider Removing (Empty/Unused)
- **src/analysis/** - Empty, no files
- **src/backtesting/** - Empty, no files
- **src/execution/** - Empty, no files
- **src/strategies/** - Empty, no files
- **src/utils/** - Empty, no files

Decision: Keep as placeholders for future use, or remove now?

### Remove (Old directories after migration)
- **src/pipeline/** - Moved to shared/pipeline/
- **src/features/** - Moved to shared/features/
- **src/models/** - Moved to shared/models/
- **src/data/** - Moved to shared/data/
- **src/config/** - Moved to shared/config/
