# Src/ Cleanup Analysis

## Files Review After Reorganization

### Status: Post-Migration to shared/

All active code has been copied to `src/shared/` and imports updated.
The old directories are now **redundant** and can be removed.

## Redundant Files (Can Delete)

### 1. Old Pipeline Directory (11 files) - REDUNDANT
**Location:** `src/pipeline/`
**Status:** ❌ DELETE - Duplicates in `src/shared/pipeline/`

```
src/pipeline/__init__.py                → src/shared/pipeline/__init__.py
src/pipeline/data_loading.py           → src/shared/pipeline/data_loading.py
src/pipeline/feature_engineering.py    → src/shared/pipeline/feature_engineering.py
src/pipeline/target_generation.py      → src/shared/pipeline/target_generation.py
src/pipeline/data_filtering.py         → src/shared/pipeline/data_filtering.py
src/pipeline/data_splitting.py         → src/shared/pipeline/data_splitting.py
src/pipeline/model_training.py         → src/shared/pipeline/model_training.py
src/pipeline/model_prediction.py       → src/shared/pipeline/model_prediction.py
src/pipeline/model_evaluation.py       → src/shared/pipeline/model_evaluation.py
src/pipeline/model_evaluation_v2.py    → src/shared/pipeline/model_evaluation_v2.py
src/pipeline/model_returns.py          → src/shared/pipeline/model_returns.py
```

**Action:** Delete entire directory
```bash
rm -rf src/pipeline/
```

### 2. Old Features Directory (5 files) - REDUNDANT
**Location:** `src/features/`
**Status:** ❌ DELETE - Duplicates in `src/shared/features/`

```
src/features/__init__.py       → src/shared/features/__init__.py
src/features/alignment.py      → src/shared/features/alignment.py
src/features/technical.py      → src/shared/features/technical.py
src/features/fundamental.py    → src/shared/features/fundamental.py
src/features/sector.py         → src/shared/features/sector.py
```

**Action:** Delete entire directory
```bash
rm -rf src/features/
```

### 3. Old Models Directory (1 file) - REDUNDANT
**Location:** `src/models/`
**Status:** ❌ DELETE - Duplicate in `src/shared/models/`

```
src/models/preprocessing.py   → src/shared/models/preprocessing.py
```

**Action:** Delete entire directory
```bash
rm -rf src/models/
```

### 4. Old Config Directory (2 files) - REDUNDANT
**Location:** `src/config/`
**Status:** ❌ DELETE - Duplicates in `src/shared/config/`

```
src/config/__init__.py    → src/shared/config/__init__.py
src/config/universe.py    → src/shared/config/universe.py
```

**Action:** Delete entire directory
```bash
rm -rf src/config/
```

### 5. Old Data Directory (6 files) - REDUNDANT
**Location:** `src/data/`
**Status:** ❌ DELETE - Duplicates in `src/shared/data/`

```
src/data/__init__.py                                → src/shared/data/__init__.py
src/data/providers/base.py                          → src/shared/data/providers/base.py
src/data/providers/yfinance_provider.py             → src/shared/data/providers/yfinance_provider.py
src/data/providers/yfinance_market_provider.py      → src/shared/data/providers/yfinance_market_provider.py
src/data/providers/yfinance_financials_provider.py  → src/shared/data/providers/yfinance_financials_provider.py
src/data/providers/yfinance_metadata_provider.py    → src/shared/data/providers/yfinance_metadata_provider.py
```

**Action:** Delete entire directory
```bash
rm -rf src/data/
```

## Empty Placeholder Directories

### 6. Analysis (Empty) - KEEP OR DELETE?
**Location:** `src/analysis/`
**Status:** ⚠️ EMPTY - Only `__init__.py`
**Usage:** None currently

**Options:**
- **Keep:** As placeholder for future analysis code
- **Delete:** If no plans for analysis-specific code

**Recommendation:** **DELETE** - No clear use case. Analysis can go in scripts/ or notebooks/

### 7. Backtesting (Empty) - KEEP OR DELETE?
**Location:** `src/backtesting/`
**Status:** ⚠️ EMPTY - Only `__init__.py`
**Usage:** None currently

**Options:**
- **Keep:** As placeholder for backtesting engine
- **Delete:** If backtesting will be in workflows or scripts

**Recommendation:** **KEEP** - Backtesting is a clear future need, good placeholder

### 8. Execution (Empty) - KEEP OR DELETE?
**Location:** `src/execution/`
**Status:** ⚠️ EMPTY - Only `__init__.py`
**Usage:** None currently

**Options:**
- **Keep:** For trade execution code
- **Delete:** If not planning to build execution system

**Recommendation:** **DELETE** - Out of scope for research project. Can add later if needed.

### 9. Strategies (Empty) - KEEP OR DELETE?
**Location:** `src/strategies/`
**Status:** ⚠️ EMPTY - Only `__init__.py`
**Usage:** None currently

**Options:**
- **Keep:** For trading strategy implementations
- **Delete:** Strategies can go in `src/workflows/`

**Recommendation:** **DELETE** - Use `src/workflows/` instead for strategy-specific code

### 10. Utils (Empty) - KEEP OR DELETE?
**Location:** `src/utils/`
**Status:** ⚠️ EMPTY - Only `__init__.py`
**Usage:** None currently

**Options:**
- **Keep:** For utility functions
- **Delete:** If no utility functions needed

**Recommendation:** **KEEP** - Utilities are always useful. Add helper functions here as needed.

## Files to Keep

### Active Directories
```
src/shared/           ← All active shared code
src/workflows/        ← Workflow-specific customizations
src/__init__.py       ← Root init
```

### Recommended Placeholders to Keep
```
src/backtesting/      ← Future backtesting engine
src/utils/            ← Future utility functions
```

## Cleanup Commands

### Conservative Approach (Recommended First Time)
```bash
# Move old directories to archive (can restore if needed)
mkdir -p archive/src_old
mv src/pipeline archive/src_old/
mv src/features archive/src_old/
mv src/models archive/src_old/
mv src/config archive/src_old/
mv src/data archive/src_old/

# Delete empty placeholders (optional)
rm -rf src/analysis/
rm -rf src/execution/
rm -rf src/strategies/
```

### Aggressive Approach (After Confirming)
```bash
# Delete redundant directories
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

## Final Structure After Cleanup

```
src/
├── __init__.py
│
├── shared/                      # All shared code
│   ├── __init__.py
│   ├── pipeline/                # 11 files
│   ├── features/                # 5 files
│   ├── models/                  # 1 file
│   ├── data/                    # 6 files
│   └── config/                  # 2 files
│
├── workflows/                   # Workflow-specific
│   ├── __init__.py
│   ├── returns_30d/
│   ├── returns_60d/
│   └── volatility/
│
├── backtesting/                 # Placeholder for future
│   └── __init__.py
│
└── utils/                       # Placeholder for utilities
    └── __init__.py
```

**Total files to delete:** 25 redundant files
**Directories to remove:** 5 redundant, 3 empty (optional)

## Validation Before Deletion

### 1. Confirm all imports updated
```bash
# Should find NO references to old paths
grep -r "from src\.pipeline\." workflows/ scripts/
grep -r "from src\.features\." workflows/ scripts/
grep -r "from src\.models\." workflows/ scripts/
grep -r "from src\.config\." workflows/ scripts/
grep -r "from src\.data\." workflows/ scripts/

# Should all be updated to src.shared.*
```

### 2. Test imports work
```bash
python workflows/wf_30d_returns_v2.py --help
python workflows/wf_30d_returns.py --help
```

### 3. Check git status
```bash
git status
# Review what will be deleted before committing
```

## Summary

**Redundant (DELETE):**
- `src/pipeline/` (11 files) - Moved to shared/
- `src/features/` (5 files) - Moved to shared/
- `src/models/` (1 file) - Moved to shared/
- `src/config/` (2 files) - Moved to shared/
- `src/data/` (6 files) - Moved to shared/
- `src/analysis/` (empty) - No use case
- `src/execution/` (empty) - Out of scope
- `src/strategies/` (empty) - Use workflows/ instead

**Keep:**
- `src/shared/` - All active code
- `src/workflows/` - Workflow customizations
- `src/backtesting/` - Future use
- `src/utils/` - Future utilities
- `src/__init__.py` - Root init

**Space Saved:** ~25 duplicate files, 8 directories

**Next Steps:**
1. Review this analysis
2. Test that new imports work
3. Execute conservative cleanup (move to archive/)
4. Test again
5. If all good, delete archive/ or commit changes
