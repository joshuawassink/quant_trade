# Project Root Cleanup Analysis

**Date:** 2025-10-24
**After:** Src/ refactoring and pipeline modularization

## Overview

After refactoring to modular workflows, many old scripts and files are now **redundant** or **superseded** by the new workflow architecture.

---

## Files to Remove

### 1. Script Backup Files (15 files) - DELETE

**Location:** `scripts/*.bak`

These are sed backup files from the import update process.

```bash
scripts/analyze_missing_data.py.bak
scripts/analyze_target_distribution.py.bak
scripts/create_training_dataset.py.bak
scripts/evaluate_features.py.bak
scripts/evaluate_model.py.bak
scripts/fetch_financials.py.bak
scripts/fetch_market_data.py.bak
scripts/fetch_metadata.py.bak
scripts/fetch_production_data.py.bak
scripts/fetch_sample_data.py.bak
scripts/generate_sp500_universe.py.bak
scripts/train_baseline_model.py.bak
scripts/train_quantile_model.py.bak
scripts/train_with_target_transform.py.bak
scripts/update_daily_data.py.bak
```

**Action:**
```bash
rm scripts/*.bak
```

**Impact:** Removes 15 unnecessary backup files

---

### 2. Old Log Files (2 files) - DELETE

**Location:** Root directory

```
yfinance_metadata.log
yfinance_provider.log
```

These are old log files that should be in `logs/` directory instead.

**Action:**
```bash
rm yfinance_*.log
```

**Impact:** Cleaner root directory

---

### 3. Redundant Scripts - MOVE TO ARCHIVE OR DELETE

Now that we have modular workflows, several old scripts are **superseded**:

#### 3a. `scripts/train_baseline_model.py` - REDUNDANT
**Superseded by:** `workflows/wf_30d_returns_v2.py` (steps 5-6)

**What it does:**
- Load training data
- Prepare features
- Train Ridge model with CV
- Save model

**What's better now:**
- Workflow has modular components
- Better separation of concerns
- Can use different models easily

**Recommendation:**
- **Archive** (might want to reference old approach)
- OR **Delete** (workflow is clearly better)

#### 3b. `scripts/create_training_dataset.py` - REDUNDANT
**Superseded by:** `workflows/wf_30d_returns_v2.py` (steps 1-4)

**What it does:**
- Load raw data
- Compute features
- Create target
- Filter and save

**What's better now:**
- Workflow breaks this into 4 separate steps
- More modular and reusable
- Better error handling

**Recommendation:** **Archive** or **Delete**

#### 3c. `scripts/evaluate_model.py` - REDUNDANT
**Superseded by:** `src/shared/pipeline/model_evaluation_v2.py`

**What it does:**
- Load model and data
- Generate predictions
- Calculate metrics
- Create visualizations

**What's better now:**
- New evaluation step is more comprehensive
- Separates prediction from evaluation
- Better organized (prediction → evaluation → returns)

**Recommendation:** **Archive** or **Delete**

#### 3d. `scripts/train_quantile_model.py` - PARTIALLY REDUNDANT
**Superseded by:** `workflows/wf_30d_returns_v2.py --model-type quantile`

**What it does:**
- Train quantile regression model
- Similar to train_baseline_model.py but with quantile loss

**What's better now:**
- Workflow supports multiple model types
- Just pass `--model-type quantile`
- Same pipeline, different model

**Recommendation:** **Archive** (might want to reference quantile-specific code)

#### 3e. `scripts/train_with_target_transform.py` - KEEP FOR NOW
**Not superseded yet**

**What it does:**
- Train with target transformations (signed_log, yeo-johnson, etc.)
- Addresses negative skew problem

**Status:**
- Could be integrated into workflow as a preprocessing option
- Useful for experimentation

**Recommendation:** **Keep** until target transforms are integrated into workflow

---

### 4. Old TODO Files - CONSOLIDATE

**Current:**
```
TODO.md                      # General project TODOs
TODO_NEXT_SESSION.md         # Session-specific TODOs
TODO_MODEL_IMPROVEMENTS.md   # Model improvement plan
```

**Issues:**
- Multiple TODO files create confusion
- Some items are outdated after refactoring
- Session-specific file may be stale

**Recommendations:**

**Option A: Consolidate into one**
```bash
# Merge all into TODO.md, archive old ones
cat TODO_NEXT_SESSION.md >> TODO.md
cat TODO_MODEL_IMPROVEMENTS.md >> TODO.md
mkdir -p archive/old_docs
mv TODO_NEXT_SESSION.md archive/old_docs/
mv TODO_MODEL_IMPROVEMENTS.md archive/old_docs/
```

**Option B: Keep structure but update**
- Keep `TODO.md` - High-level project TODOs
- Keep `TODO_MODEL_IMPROVEMENTS.md` - Still relevant
- **Delete** `TODO_NEXT_SESSION.md` - Stale (refers to old structure)

**Recommendation:** Option B

---

### 5. Old Session Summary - ARCHIVE

**File:** `SESSION_SUMMARY.md`

This is from an old session before the refactoring.

**Recommendation:**
```bash
mkdir -p docs/sessions/
mv SESSION_SUMMARY.md docs/sessions/session_20251023.md
```

---

## Scripts to Keep (Still Useful)

### Data Fetching Scripts - KEEP
These are still useful for data collection:
```
scripts/fetch_financials.py
scripts/fetch_market_data.py
scripts/fetch_metadata.py
scripts/fetch_production_data.py
scripts/fetch_sample_data.py
scripts/update_daily_data.py
```

**Why:** Not replaced by workflow (data fetching is separate)

### Analysis Scripts - KEEP
```
scripts/analyze_missing_data.py
scripts/analyze_target_distribution.py
scripts/evaluate_features.py
```

**Why:** Useful for ad-hoc analysis and debugging

### Universe Management - KEEP
```
scripts/generate_sp500_universe.py
```

**Why:** Needed to update stock universe

---

## Empty/Minimal Directories

### `tests/` - EMPTY
**Status:** Empty directory
**Recommendation:** **Keep** - Will add tests later

### `configs/` - EMPTY
**Status:** Empty directory
**Recommendation:** **Delete** - Config is in `src/shared/config/`

---

## Summary of Actions

### Immediate Cleanup (Safe to do now)

```bash
# 1. Delete backup files
rm scripts/*.bak

# 2. Delete old logs
rm yfinance_*.log

# 3. Delete empty configs directory
rmdir configs/

# 4. Delete stale TODO
rm TODO_NEXT_SESSION.md

# 5. Move old session summary
mkdir -p docs/sessions
mv SESSION_SUMMARY.md docs/sessions/session_20251023.md

echo "✓ Basic cleanup complete"
```

**Impact:** Removes 18 files, 1 directory

### Archive Redundant Scripts (Conservative)

```bash
# Archive old training/evaluation scripts
mkdir -p archive/old_scripts
mv scripts/train_baseline_model.py archive/old_scripts/
mv scripts/create_training_dataset.py archive/old_scripts/
mv scripts/evaluate_model.py archive/old_scripts/
mv scripts/train_quantile_model.py archive/old_scripts/

echo "✓ Old scripts archived"
```

**Impact:** Archives 4 major scripts (still accessible if needed)

### Aggressive Cleanup (After Testing)

```bash
# If you're confident in new workflows, delete archived scripts
rm -rf archive/old_scripts/
rm -rf archive/src_old_20251024/

echo "✓ All archived files deleted"
```

---

## File Count Comparison

### Before Cleanup
```
Root files:           12
Scripts:              15 .py + 15 .bak = 30
Backup files:         15
Log files:            2
TODO files:           3
Total to clean:       ~32 files
```

### After Cleanup
```
Root files:           10 (-2 logs, -1 TODO, -1 moved)
Scripts:              15 .py (or 11 if archived)
Backup files:         0 (-15)
Log files:            0 (-2)
TODO files:           2 (-1)
Cleaned:              18-22 files
```

---

## Detailed Script Analysis

### Redundant Scripts (Superseded by Workflows)

| Script | Lines | Superseded By | Keep/Archive/Delete |
|--------|-------|---------------|---------------------|
| `train_baseline_model.py` | 334 | `workflows/wf_30d_returns_v2.py` steps 5-6 | Archive |
| `create_training_dataset.py` | 257 | `workflows/wf_30d_returns_v2.py` steps 1-4 | Archive |
| `evaluate_model.py` | 654 | `src/shared/pipeline/model_evaluation_v2.py` | Archive |
| `train_quantile_model.py` | 359 | `workflows/wf_30d_returns_v2.py --model-type quantile` | Archive |

**Total redundant:** ~1,604 lines (superseded by modular components)

### Useful Scripts (Keep)

| Script | Purpose | Status |
|--------|---------|--------|
| `fetch_*.py` | Data collection | Keep - still needed |
| `analyze_*.py` | Ad-hoc analysis | Keep - useful for debugging |
| `generate_sp500_universe.py` | Universe management | Keep - needed for updates |
| `train_with_target_transform.py` | Target transforms | Keep - not in workflow yet |

---

## Updated Project Structure

### After Cleanup

```
quant_trade/
├── src/                          # Source code
│   ├── shared/                   # Shared components
│   └── workflows/                # Workflow-specific
│
├── workflows/                    # Orchestrators
│   ├── wf_30d_returns.py        # V1
│   └── wf_30d_returns_v2.py     # V2
│
├── scripts/                      # Utility scripts (11 files)
│   ├── fetch_*.py               # Data fetching
│   ├── analyze_*.py             # Analysis
│   ├── generate_sp500_universe.py
│   ├── train_with_target_transform.py
│   └── update_daily_data.py
│
├── docs/                         # Documentation
│   ├── sessions/                # Session summaries
│   └── *.md                     # Guides
│
├── archive/                      # Archived code
│   ├── src_old_20251024/        # Old src/ structure
│   └── old_scripts/             # Superseded scripts
│
├── data/                         # Data files
├── models/                       # Trained models
├── reports/                      # Evaluation reports
├── notebooks/                    # Jupyter notebooks
├── logs/                         # Log files
├── tests/                        # Tests (future)
├── setup/                        # LaunchAgent configs
│
├── TODO.md                       # Project TODOs
├── TODO_MODEL_IMPROVEMENTS.md    # Model improvement plan
├── README.md                     # Project README
├── QUICKSTART.md                 # Quick start guide
├── requirements.txt              # Dependencies
└── pyproject.toml               # Project config
```

**Clean, organized, and maintainable!**

---

## Recommendations Summary

### High Priority (Do Now)
1. ✅ Delete `.bak` files (15 files)
2. ✅ Delete old log files (2 files)
3. ✅ Delete `configs/` directory (empty)
4. ✅ Delete `TODO_NEXT_SESSION.md` (stale)
5. ✅ Move `SESSION_SUMMARY.md` to `docs/sessions/`

### Medium Priority (This Week)
6. ⏳ Archive redundant scripts (4 files)
7. ⏳ Update `TODO.md` to reflect new architecture
8. ⏳ Update `README.md` to reference new workflows

### Low Priority (Later)
9. ⏳ Add tests to `tests/` directory
10. ⏳ Delete archives if confident (after 1-2 weeks)

---

## Commands to Execute

### Quick Cleanup (Safe)
```bash
cd /Users/jwassink/repos/quant_trade

# Delete backups and logs
rm scripts/*.bak
rm yfinance_*.log

# Delete stale TODO
rm TODO_NEXT_SESSION.md

# Move old session summary
mkdir -p docs/sessions
mv SESSION_SUMMARY.md docs/sessions/session_20251023.md

# Delete empty configs
rmdir configs/ 2>/dev/null || true

echo "✓ Quick cleanup complete"
```

### Archive Old Scripts (Optional)
```bash
# Archive superseded scripts
mkdir -p archive/old_scripts
mv scripts/train_baseline_model.py archive/old_scripts/
mv scripts/create_training_dataset.py archive/old_scripts/
mv scripts/evaluate_model.py archive/old_scripts/
mv scripts/train_quantile_model.py archive/old_scripts/

echo "✓ Old scripts archived"
```

---

## Benefits After Cleanup

1. ✅ **Cleaner root** - No backup files or old logs
2. ✅ **Clear scripts/** - Only useful, non-redundant scripts
3. ✅ **Better organization** - Archived files separated
4. ✅ **Less confusion** - One source of truth (workflows)
5. ✅ **Easier to find** - Fewer files to search through

---

**Next:** Execute quick cleanup commands and test workflows!
