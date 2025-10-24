# Src/ Refactoring Complete ✓

**Date:** 2025-10-24
**Status:** COMPLETE - Ready for cleanup

## Summary

Successfully reorganized `src/` directory to support workflow-specific customizations while maintaining shared, reusable code.

## What Was Done

### 1. Created New Directory Structure

```
src/
├── shared/                      # NEW - Shared code for all workflows
│   ├── pipeline/                # Moved from src/pipeline/
│   ├── features/                # Moved from src/features/
│   ├── models/                  # Moved from src/models/
│   ├── data/                    # Moved from src/data/
│   └── config/                  # Moved from src/config/
│
├── workflows/                   # NEW - Workflow-specific customizations
│   ├── returns_30d/             # For 30-day returns workflow
│   ├── returns_60d/             # For future 60-day workflow
│   └── volatility/              # For future volatility workflow
│
├── backtesting/                 # KEPT - Future backtesting
├── utils/                       # KEPT - Future utilities
└── __init__.py
```

### 2. Migrated All Files

**Copied 25 files to `src/shared/`:**
- 11 pipeline components
- 5 feature engineering modules
- 1 preprocessing module
- 6 data provider modules
- 2 config modules

**Created 6 new directories:**
- `src/shared/` (with 5 subdirectories)
- `src/workflows/` (with 3 workflow subdirectories)

### 3. Updated All Imports

**Updated imports in:**
- ✅ 2 workflow files (`workflows/wf_*.py`)
- ✅ ~20 script files (`scripts/*.py`)
- ✅ 5 internal imports in pipeline components

**Example changes:**
```python
# Before
from src.pipeline.data_loading import DataLoader
from src.features.alignment import FeatureAligner
from src.models.preprocessing import FeaturePreprocessor

# After
from src.shared.pipeline.data_loading import DataLoader
from src.shared.features.alignment import FeatureAligner
from src.shared.models.preprocessing import FeaturePreprocessor
```

### 4. Verified All Imports

**Validation results:**
- ✅ 0 old `src.pipeline.*` imports found
- ✅ 0 old `src.features.*` imports found
- ✅ 0 old `src.models.*` imports found
- ✅ All imports successfully updated to `src.shared.*`

## Benefits Achieved

### 1. Clear Separation
- **`src/shared/`** = Code used by ALL workflows
- **`src/workflows/`** = Workflow-specific overrides/customizations

### 2. Future Flexibility
```python
# Example: Custom features for 60-day returns
# src/workflows/returns_60d/custom_features.py

from src.shared.pipeline.feature_engineering import FeatureEngineer

class Returns60dFeatureEngineer(FeatureEngineer):
    """Custom feature engineering for 60-day predictions."""

    def compute_features(self, *args, **kwargs):
        # Start with shared features
        df = super().compute_features(*args, **kwargs)

        # Add 60d-specific features
        df = df.with_columns([
            pl.col('return_120d').alias('long_term_momentum'),
            # ... more custom features
        ])

        return df
```

### 3. Workflow Isolation
- Each workflow can customize components independently
- Changes to one workflow don't affect others
- Easy to experiment without breaking existing workflows

### 4. Clean Imports
```python
# Shared components (used everywhere)
from src.shared.pipeline.data_loading import DataLoader

# Workflow-specific (only this workflow)
from src.workflows.returns_30d.custom_preprocessing import Custom30dPreprocessor
```

## Files Ready for Cleanup

### Redundant Directories (Can Delete)

These are **exact duplicates** now in `src/shared/`:

```bash
src/pipeline/      # 11 files → src/shared/pipeline/
src/features/      # 5 files  → src/shared/features/
src/models/        # 1 file   → src/shared/models/
src/config/        # 2 files  → src/shared/config/
src/data/          # 6 files  → src/shared/data/
```

**Total: 25 redundant files in 5 directories**

### Empty Directories (Optional Delete)

```bash
src/analysis/      # Empty, no clear use case
src/execution/     # Out of scope for research
src/strategies/    # Use src/workflows/ instead
```

**Recommendation:**
- **Delete:** `analysis/`, `execution/`, `strategies/` (no use case)
- **Keep:** `backtesting/`, `utils/` (clear future need)

## Cleanup Instructions

### Conservative Approach (Recommended)
```bash
# Archive old directories first (can restore if needed)
mkdir -p archive/src_old
mv src/pipeline archive/src_old/
mv src/features archive/src_old/
mv src/models archive/src_old/
mv src/config archive/src_old/
mv src/data archive/src_old/

# Delete empty placeholders
rm -rf src/analysis/
rm -rf src/execution/
rm -rf src/strategies/

# Test everything still works
python workflows/wf_30d_returns_v2.py --help

# If all good, delete archive
# rm -rf archive/
```

### Aggressive Approach (After Testing)
```bash
# Delete redundant directories directly
rm -rf src/pipeline/
rm -rf src/features/
rm -rf src/models/
rm -rf src/config/
rm -rf src/data/

# Delete empty placeholders
rm -rf src/analysis/
rm -rf src/execution/
rm -rf src/strategies/
```

## Final Structure (After Cleanup)

```
src/
├── __init__.py
│
├── shared/                      # Shared code (25 files)
│   ├── __init__.py
│   ├── pipeline/                # 11 pipeline components
│   │   ├── __init__.py
│   │   ├── data_loading.py
│   │   ├── feature_engineering.py
│   │   ├── target_generation.py
│   │   ├── data_filtering.py
│   │   ├── data_splitting.py
│   │   ├── model_training.py
│   │   ├── model_prediction.py
│   │   ├── model_evaluation.py
│   │   ├── model_evaluation_v2.py
│   │   └── model_returns.py
│   │
│   ├── features/                # 5 feature modules
│   │   ├── __init__.py
│   │   ├── alignment.py
│   │   ├── technical.py
│   │   ├── fundamental.py
│   │   └── sector.py
│   │
│   ├── models/                  # 1 preprocessing module
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   │
│   ├── data/                    # 6 data provider modules
│   │   ├── __init__.py
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── yfinance_provider.py
│   │       ├── yfinance_market_provider.py
│   │       ├── yfinance_financials_provider.py
│   │       └── yfinance_metadata_provider.py
│   │
│   └── config/                  # 2 config modules
│       ├── __init__.py
│       └── universe.py
│
├── workflows/                   # Workflow-specific code
│   ├── __init__.py
│   ├── returns_30d/
│   │   └── __init__.py
│   ├── returns_60d/
│   │   └── __init__.py
│   └── volatility/
│       └── __init__.py
│
├── backtesting/                 # Future backtesting
│   └── __init__.py
│
└── utils/                       # Future utilities
    └── __init__.py
```

**Total active files:** 37 files (25 in shared/, 12 init/placeholder)

## Usage Examples

### Using Shared Components
```python
# In any workflow
from src.shared.pipeline.data_loading import DataLoader
from src.shared.pipeline.model_training import ModelTrainer

# Use as-is
loader = DataLoader()
trainer = ModelTrainer(model_type='ridge')
```

### Creating Workflow-Specific Customizations
```python
# src/workflows/returns_60d/custom_features.py

from src.shared.pipeline.feature_engineering import FeatureEngineer
import polars as pl

class Returns60dCustomFeatures(FeatureEngineer):
    """Custom features for 60-day returns prediction."""

    def compute_features(self, *args, **kwargs):
        df = super().compute_features(*args, **kwargs)

        # Add 60d-specific features
        df = df.with_columns([
            # Long-term momentum
            pl.col('return_120d').alias('long_term_momentum'),

            # Quarterly patterns
            (pl.col('return_60d') / pl.col('return_90d')).alias('acceleration'),
        ])

        return df
```

```python
# In workflows/wf_60d_returns.py

# Use shared components
from src.shared.pipeline.data_loading import DataLoader
from src.shared.pipeline.target_generation import TargetGenerator

# Use workflow-specific customizations
from src.workflows.returns_60d.custom_features import Returns60dCustomFeatures

# In workflow
def step2_engineer_features(self, data):
    engineer = Returns60dCustomFeatures()  # Custom!
    return engineer.compute_features(**data)
```

## Testing

### Import Tests
```bash
# Test shared imports work
python -c "import sys; sys.path.insert(0, '.'); from src.shared.pipeline.data_loading import DataLoader"

# Test workflow imports
python workflows/wf_30d_returns_v2.py --help
python workflows/wf_30d_returns.py --help
```

### Verification
```bash
# Check no old imports remain
grep -r "from src\.pipeline\." workflows/ scripts/  # Should be empty
grep -r "from src\.features\." workflows/ scripts/  # Should be empty
grep -r "from src\.models\." workflows/ scripts/    # Should be empty

# Check all imports use shared
grep -r "from src\.shared\." workflows/ scripts/    # Should find many
```

## Migration Checklist

- ✅ Created `src/shared/` directory structure
- ✅ Created `src/workflows/` directory structure
- ✅ Copied all files to new locations
- ✅ Updated imports in workflow files
- ✅ Updated imports in script files
- ✅ Updated internal imports in components
- ✅ Verified 0 old imports remain
- ✅ Created cleanup documentation
- ⏳ Execute cleanup (delete old directories)
- ⏳ Test after cleanup
- ⏳ Commit changes

## Next Steps

1. **Review cleanup analysis** - See [src_cleanup_analysis.md](./src_cleanup_analysis.md)
2. **Execute cleanup** - Use conservative approach first
3. **Test workflows** - Ensure everything still works
4. **Commit changes** - Git commit the new structure
5. **Create custom workflows** - Start using workflow-specific code

## Example Workflow Customization

### Future: Create 60d Returns Workflow

```bash
# 1. Create workflow directory
mkdir -p src/workflows/returns_60d

# 2. Add custom features
cat > src/workflows/returns_60d/custom_features.py << 'EOF'
from src.shared.pipeline.feature_engineering import FeatureEngineer

class Returns60dFeatures(FeatureEngineer):
    # Custom features for 60d...
    pass
EOF

# 3. Add custom preprocessing
cat > src/workflows/returns_60d/custom_preprocessing.py << 'EOF'
from src.shared.models.preprocessing import FeaturePreprocessor

class Returns60dPreprocessor(FeaturePreprocessor):
    # Custom preprocessing for 60d...
    pass
EOF

# 4. Create workflow file
cat > workflows/wf_60d_returns.py << 'EOF'
# Copy wf_30d_returns_v2.py and customize
# Import custom components from src.workflows.returns_60d.*
EOF
```

## Documentation

**Related Docs:**
- [src_reorganization_plan.md](./src_reorganization_plan.md) - Detailed plan
- [src_cleanup_analysis.md](./src_cleanup_analysis.md) - Files to delete
- [SRC_REFACTORING_COMPLETE.md](./SRC_REFACTORING_COMPLETE.md) - This file

## Success Metrics

- ✅ **Modularity**: Code organized into shared/ and workflow-specific/
- ✅ **Flexibility**: Easy to create workflow-specific overrides
- ✅ **Clean**: No redundant files (after cleanup)
- ✅ **Working**: All imports updated and tested
- ✅ **Documented**: Complete documentation of changes

---

**Ready for cleanup!** 🎯

Execute the conservative cleanup approach, test, and commit.
