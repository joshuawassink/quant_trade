# Quick Cleanup Guide

## TL;DR

After refactoring, we have:
- ✅ 15 `.bak` files (sed backups) - DELETE
- ✅ 2 old log files in root - DELETE
- ✅ 4 redundant scripts (superseded by workflows) - ARCHIVE
- ✅ 1 stale TODO file - DELETE
- ✅ 1 old session summary - MOVE

**Total cleanup:** ~22 files

---

## Execute Cleanup

### Step 1: Safe Cleanup (Do Now)
```bash
cd /Users/jwassink/repos/quant_trade

# Delete backup files from sed
rm scripts/*.bak

# Delete old root log files
rm yfinance_*.log

# Delete stale TODO
rm TODO_NEXT_SESSION.md

# Move old session summary
mkdir -p docs/sessions
mv SESSION_SUMMARY.md docs/sessions/session_20251023.md

# Try to delete empty configs (may not exist)
rmdir configs/ 2>/dev/null || true

echo "✓ Cleanup complete - 18 files removed/moved"
```

### Step 2: Archive Redundant Scripts (Optional)
```bash
# These scripts are superseded by workflows/wf_30d_returns_v2.py
mkdir -p archive/old_scripts

# Archive old training scripts
mv scripts/train_baseline_model.py archive/old_scripts/
mv scripts/create_training_dataset.py archive/old_scripts/
mv scripts/evaluate_model.py archive/old_scripts/
mv scripts/train_quantile_model.py archive/old_scripts/

echo "✓ 4 old scripts archived"
```

### Step 3: Delete Archives (After Testing)
```bash
# Once you've tested workflows and confirmed everything works
rm -rf archive/old_scripts/
rm -rf archive/src_old_20251024/

echo "✓ All archives deleted"
```

---

## What Gets Cleaned

### Backup Files (15) - SAFE TO DELETE
```
scripts/*.bak
```
These are sed backups from import updates. Not needed.

### Old Logs (2) - SAFE TO DELETE
```
yfinance_metadata.log
yfinance_provider.log
```
Old logs that should be in logs/ directory.

### Stale TODO (1) - SAFE TO DELETE
```
TODO_NEXT_SESSION.md
```
Refers to old architecture, no longer relevant.

### Old Session Summary (1) - MOVE
```
SESSION_SUMMARY.md → docs/sessions/session_20251023.md
```
Archive old session summary.

### Redundant Scripts (4) - ARCHIVE
```
scripts/train_baseline_model.py     → Use workflows/wf_30d_returns_v2.py
scripts/create_training_dataset.py  → Use workflows/wf_30d_returns_v2.py
scripts/evaluate_model.py           → Use src/shared/pipeline/model_evaluation_v2.py
scripts/train_quantile_model.py     → Use workflows/wf_30d_returns_v2.py --model-type quantile
```

---

## What to Keep

### Useful Scripts
```
scripts/fetch_*.py                  # Data fetching
scripts/analyze_*.py                # Analysis
scripts/generate_sp500_universe.py  # Universe management
scripts/train_with_target_transform.py  # Target transforms (not in workflow yet)
scripts/update_daily_data.py        # Data updates
```

### Documentation
```
docs/*.md                           # All guides
```

### Code
```
src/shared/                         # Shared components
src/workflows/                      # Workflow-specific
workflows/                          # Orchestrators
```

---

## Verification

### Before Cleanup
```bash
ls scripts/*.bak | wc -l          # 15
ls *.log 2>/dev/null | wc -l      # 2
ls TODO*.md | wc -l               # 3
```

### After Step 1
```bash
ls scripts/*.bak 2>/dev/null | wc -l    # 0
ls *.log 2>/dev/null | wc -l            # 0
ls TODO*.md | wc -l                     # 2
```

### After Step 2
```bash
ls scripts/*.py | wc -l                 # 11 (down from 15)
ls archive/old_scripts/*.py | wc -l     # 4
```

---

## File Count Summary

| Category | Before | After | Change |
|----------|--------|-------|--------|
| Backup files | 15 | 0 | -15 |
| Root logs | 2 | 0 | -2 |
| TODO files | 3 | 2 | -1 |
| Scripts | 15 | 11 | -4 (archived) |
| **Total** | **35** | **13** | **-22** |

---

## One-Command Cleanup

If you want to do it all at once:

```bash
cd /Users/jwassink/repos/quant_trade && \
rm scripts/*.bak yfinance_*.log TODO_NEXT_SESSION.md && \
mkdir -p docs/sessions archive/old_scripts && \
mv SESSION_SUMMARY.md docs/sessions/session_20251023.md && \
mv scripts/{train_baseline_model,create_training_dataset,evaluate_model,train_quantile_model}.py archive/old_scripts/ 2>/dev/null && \
rmdir configs/ 2>/dev/null || true && \
echo "✓ Complete cleanup done: 22 files removed/archived"
```

---

## Rollback (If Needed)

If you archived scripts and need them back:

```bash
# Restore archived scripts
mv archive/old_scripts/*.py scripts/

echo "✓ Scripts restored"
```

---

**Quick and clean!** 🎯
