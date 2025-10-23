# Git Best Practices for Quant Trade

## Proposed Workflow & Standards

This document outlines git best practices for the quant_trade project. Review and adjust as needed before adoption.

---

## 1. Branching Strategy

### Recommended: GitHub Flow (Simple & Effective)

**Main Branches:**
- `main` - Production-ready code, always deployable
- Feature branches - Short-lived, focused on specific tasks

**Workflow:**
```bash
# Create feature branch from main
git checkout main
git pull origin main
git checkout -b feature/add-technical-indicators

# Work on feature, commit frequently
git add <files>
git commit -m "Add RSI calculation"

# Keep branch updated with main
git checkout main
git pull origin main
git checkout feature/add-technical-indicators
git merge main  # or rebase if you prefer

# When ready, push and create PR
git push -u origin feature/add-technical-indicators
```

**Branch Naming:**
- `feature/description` - New features (e.g., `feature/add-rsi-indicator`)
- `fix/description` - Bug fixes (e.g., `fix/missing-data-handling`)
- `refactor/description` - Code improvements (e.g., `refactor/provider-validation`)
- `docs/description` - Documentation only (e.g., `docs/api-reference`)
- `test/description` - Test additions (e.g., `test/provider-unit-tests`)
- `data/description` - Data updates (e.g., `data/expand-universe`)

**Alternative: For solo development:**
- Work directly on `main` for small, tested changes
- Use feature branches for experimental/risky work
- Always ensure code works before committing to `main`

---

## 2. Commit Message Standards

### Format: Conventional Commits

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `refactor` - Code restructuring (no behavior change)
- `perf` - Performance improvement
- `docs` - Documentation only
- `test` - Adding/updating tests
- `chore` - Maintenance (dependencies, config)
- `style` - Code style/formatting (no logic change)
- `data` - Data updates or schema changes

**Examples:**

```bash
# Simple commit
feat(features): add RSI technical indicator

# Detailed commit
feat(features): add RSI technical indicator

Implemented Relative Strength Index calculation with:
- 14-period default (configurable)
- Polars-native computation (vectorized)
- Proper handling of series start (NaN for first 14 periods)
- Unit tests with known values

Closes #23
```

```bash
# Breaking change
feat(providers)!: change DataProvider.fetch signature

BREAKING CHANGE: DataProvider.fetch() now requires start_date and end_date
as mandatory parameters. Previously these were optional.

Migration guide:
- Before: provider.fetch(symbols)
- After:  provider.fetch(symbols, start_date, end_date)
```

```bash
# Bug fix
fix(financials): handle missing quarterly data gracefully

Some stocks (e.g., recent IPOs) have <5 quarters of data.
Provider now handles this without crashing.

Fixes #45
```

### Commit Message Rules:

1. **Subject line (first line):**
   - ≤50 characters
   - Imperative mood ("add" not "added" or "adds")
   - No period at end
   - Capitalize first letter

2. **Body (optional):**
   - Wrap at 72 characters
   - Explain WHAT and WHY, not HOW
   - Separate from subject with blank line

3. **Footer (optional):**
   - Reference issues/PRs
   - Note breaking changes
   - Add co-authors

---

## 3. When to Commit

### Commit Frequency Guidelines:

**✅ Good Times to Commit:**
- Completed a logical unit of work (function, class, feature component)
- Tests are passing
- Code is in a working state
- Before switching tasks
- End of coding session
- After successful refactoring

**❌ Don't Commit:**
- Broken/non-working code (unless on feature branch with WIP tag)
- Generated files (.pyc, __pycache__, .venv/)
- Sensitive data (.env, credentials)
- Large data files (>10MB without Git LFS)

**Guideline:** Commit 3-5 times per hour when actively coding. Each commit should be a meaningful checkpoint.

---

## 4. What to Stage

### File Selection Guidelines:

**✅ Always Stage:**
- Source code changes related to current task
- Tests for those changes
- Documentation updates for those changes
- Configuration changes if intentional

**⚠️ Carefully Consider:**
- Generated files (usually .gitignore these)
- Data files (use data/ directory, consider size)
- Config files (don't commit secrets)
- Multiple unrelated changes (split into separate commits)

**❌ Never Stage:**
- `.env` files with secrets
- `__pycache__/` or `.pyc` files
- `.venv/` or virtual environments
- IDE-specific files (.vscode/, .idea/) unless team standard
- Personal notes or TODO files (unless intended for team)
- Binary files >10MB (without Git LFS)
- API keys, passwords, tokens

### How to Stage Selectively:

```bash
# Stage specific files only
git add src/features/technical.py tests/test_technical.py

# Stage all files in a directory
git add src/features/

# Interactive staging (review each change)
git add -p

# Stage everything (use cautiously!)
git add -A

# Check what's staged before committing
git status
git diff --staged
```

---

## 5. Code Review Before Commit

### Pre-Commit Checklist:

**Every commit should pass:**
```bash
# 1. Review what's being committed
git diff --staged

# 2. Check for common issues
grep -r "TODO\|FIXME\|XXX\|HACK" <staged-files>  # Intentional?
grep -r "print(" <staged-files>  # Debug statements?
grep -r "import pdb" <staged-files>  # Breakpoints?

# 3. Run tests (when we have them)
pytest tests/

# 4. Run linter (optional but recommended)
ruff check src/
# or
pylint src/

# 5. Verify imports
python -m compileall src/

# 6. Check for secrets (basic)
git diff --staged | grep -i "password\|api_key\|secret\|token"
```

---

## 6. Pull Request Guidelines

### When to Create PRs:

**Use PRs for:**
- Feature branches merging to main
- Significant refactoring
- Breaking changes
- Code you want reviewed
- Collaborative work

**Skip PRs for (solo development):**
- Documentation fixes
- Typos
- Minor formatting
- Emergency hotfixes (review after)

### PR Template:

```markdown
## Summary
Brief description of changes

## Type of Change
- [ ] Feature
- [ ] Bug fix
- [ ] Refactoring
- [ ] Documentation
- [ ] Performance improvement

## Changes Made
- Added RSI indicator calculation
- Updated technical features documentation
- Added unit tests for RSI

## Testing
- [ ] Tests added/updated
- [ ] Manual testing completed
- [ ] Existing tests still pass

## Checklist
- [ ] Code follows project style
- [ ] Documentation updated
- [ ] No secrets committed
- [ ] Breaking changes documented

## Related Issues
Closes #23
Related to #45
```

---

## 7. Merge Strategy

**Options:**

1. **Merge Commit** (Recommended for features)
   ```bash
   git merge feature/branch --no-ff
   ```
   - Preserves full history
   - Shows when features were integrated
   - Easy to revert entire feature

2. **Squash Merge** (For messy branches)
   ```bash
   git merge feature/branch --squash
   ```
   - Cleans up commit history
   - Good for WIP commits
   - Loses intermediate history

3. **Rebase** (For clean linear history)
   ```bash
   git rebase main
   ```
   - Linear history
   - No merge commits
   - Can be risky (rewrites history)

**Recommendation for quant_trade:**
- Use **merge commits** for feature branches (preserves context)
- Use **squash** if branch has lots of "WIP" commits
- Avoid rebase on shared branches

---

## 8. Gitignore Best Practices

### Current .gitignore Should Include:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/

# Data (large files)
*.parquet
*.csv
*.h5
*.hdf5
data/*.parquet  # Unless small reference data

# Notebooks
.ipynb_checkpoints/
*.ipynb  # If you prefer to exclude (or include with output cleared)

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Secrets
.env
.env.local
credentials.json
*.key
*.pem

# Logs
logs/
*.log

# Testing
.pytest_cache/
.coverage
htmlcov/

# Build
build/
dist/
*.egg-info/
```

---

## 9. Workflow for Claude Code Assistant

### Guidelines for AI Commits:

**Before Committing:**
1. Show user `git status` and `git diff --staged`
2. Explain what's being committed and why
3. Propose commit message
4. Wait for user approval (unless standing instruction)

**Commit Message Format:**
```
<type>(<scope>): <clear description>

<details of what changed>
<why it changed>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**When to Batch vs. Split Commits:**
- **Batch**: Related changes (feature + tests + docs)
- **Split**: Unrelated changes (fix bug + add feature)
- **Split**: Large refactoring (easier to review/revert)

---

## 10. Emergency Procedures

### Undo Last Commit (not pushed):
```bash
# Keep changes, undo commit
git reset --soft HEAD~1

# Discard changes, undo commit
git reset --hard HEAD~1
```

### Undo Pushed Commit:
```bash
# Create revert commit (safe)
git revert HEAD
git push origin main
```

### Accidentally Committed Secret:
```bash
# Remove from last commit (not pushed)
git rm --cached .env
echo ".env" >> .gitignore
git add .gitignore
git commit --amend --no-edit

# If already pushed (severe - requires force push)
# Contact all collaborators first!
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/secret" \
  --prune-empty --tag-name-filter cat -- --all
git push origin --force --all

# Better: Rotate the secret immediately!
```

---

## Questions for You:

1. **Branching:** Feature branches or direct commits to main?
2. **Commit frequency:** Every logical unit, or batch related changes?
3. **Commit messages:** Strict conventional commits, or more relaxed?
4. **Data files:** Commit small datasets, or .gitignore all .parquet?
5. **Notebooks:** Commit .ipynb with output, or clear output first?
6. **My autonomy:** Should I auto-commit after successful work, or always ask?

---

## Recommendations Summary

**For solo research project (current state):**
- ✅ Work on `main`, use feature branches for risky experiments
- ✅ Commit every logical unit (3-5x per coding session)
- ✅ Use conventional commit format (good habit)
- ✅ **Always show and explain before committing**
- ✅ Commit code/docs, ignore large data files
- ✅ Clear notebook output before committing

**As project matures:**
- Feature branches + PRs for all changes
- Pre-commit hooks (black, ruff, pytest)
- CI/CD pipeline (GitHub Actions)
- Semantic versioning for releases
