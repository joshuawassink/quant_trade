# Quick Cleanup Commands

## TL;DR - Ready to Clean

All imports have been updated. Old directories are duplicates.
Safe to delete after testing.

## Step-by-Step Cleanup

### 1. Archive Old Directories (Safe - Can Restore)
```bash
# Create archive
mkdir -p archive/src_old

# Move old directories to archive
mv src/pipeline archive/src_old/
mv src/features archive/src_old/
mv src/models archive/src_old/
mv src/config archive/src_old/
mv src/data archive/src_old/

# Delete empty placeholder directories
rm -rf src/analysis/
rm -rf src/execution/
rm -rf src/strategies/

echo "✓ Old directories archived"
```

### 2. Test Everything Works
```bash
# Test V2 workflow
python workflows/wf_30d_returns_v2.py --help

# Test V1 workflow
python workflows/wf_30d_returns.py --help

# Test a script
python scripts/fetch_metadata.py --help

echo "✓ All imports working"
```

### 3. Delete Archive (After Confirming)
```bash
# Once you've confirmed everything works
rm -rf archive/

echo "✓ Cleanup complete"
```

## One-Command Cleanup (Aggressive)

**Only use after testing!**

```bash
# Delete old directories directly
rm -rf src/pipeline/ src/features/ src/models/ src/config/ src/data/ src/analysis/ src/execution/ src/strategies/

echo "✓ Old directories deleted"
```

## Rollback (If Needed)

If you archived first:
```bash
# Restore from archive
mv archive/src_old/pipeline src/
mv archive/src_old/features src/
mv archive/src_old/models src/
mv archive/src_old/config src/
mv archive/src_old/data src/

echo "✓ Restored old directories"
```

## Verification

### Before Cleanup
```bash
# Should show old directories
ls -d src/*/

# Should show ~50 total files
find src -name "*.py" -not -path "*/__pycache__/*" | wc -l
```

### After Cleanup
```bash
# Should only show shared/ and workflows/
ls -d src/*/

# Should show ~37 files (25 in shared, 12 inits)
find src -name "*.py" -not -path "*/__pycache__/*" | wc -l
```

## What Gets Deleted

**Redundant (Exact duplicates in shared/):**
- `src/pipeline/` (11 files)
- `src/features/` (5 files)
- `src/models/` (1 file)
- `src/config/` (2 files)
- `src/data/` (6 files)

**Empty (No use case):**
- `src/analysis/` (empty)
- `src/execution/` (empty)
- `src/strategies/` (empty)

**Total: 25 files + 8 directories**

## What Gets Kept

**Active:**
- `src/shared/` (25 files)
- `src/workflows/` (6 init files)
- `src/__init__.py`

**Placeholders:**
- `src/backtesting/` (future use)
- `src/utils/` (future use)

**Total: 37 files + 9 directories**
