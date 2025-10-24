# Claude Code Navigation Guide

## Quick Start: Understanding the Codebase

**FIRST STEP:** Read [ARCHITECTURE.md](../ARCHITECTURE.md) to understand:
- Core concepts (DataProvider, FeatureAligner, Pipeline Component, etc.)
- Directory structure and module purposes
- Data flow through the pipeline
- Entry points for running workflows

This will give you 80% of the context you need in 2-3 minutes.

## Navigation Strategy for AI Assistant

### When Starting a New Session

1. **Read ARCHITECTURE.md first** - This is your map of the entire codebase
2. **Read the main workflow** - [workflows/wf_30d_returns_v2.py](../workflows/wf_30d_returns_v2.py) to see how components are composed
3. **Read TODO.md** (if exists) - Current priorities and pending work

### Finding Code by Purpose

Use this decision tree to navigate efficiently:

#### "How does X work?" or "Where is Y implemented?"
→ Use **Explore agent** (subagent_type=Explore) for broad searches
- Example: "How does model training work?"
- Example: "Where are fundamental features computed?"
- The Explore agent will search multiple locations and naming patterns

#### "Find file matching pattern"
→ Use **Glob tool** for file pattern matching
- Example: Find all pipeline components: `src/shared/pipeline/*.py`
- Example: Find all workflows: `workflows/wf_*.py`
- Example: Find all feature modules: `src/shared/features/*.py`

#### "Find code containing specific text"
→ Use **Grep tool** for content search
- Example: Find where `FeatureAligner` is used: `pattern: "FeatureAligner"`
- Example: Find where Ridge model is configured: `pattern: "Ridge\("`
- Example: Find all time-series split calls: `pattern: "time_series_split"`

#### "Read a specific file"
→ Use **Read tool** directly (fastest)
- If you know the exact path from ARCHITECTURE.md, read it directly
- Don't waste time with Explore/Grep when you know the location

### Code Locations Cheat Sheet

| What You Need | Where to Look | Quick Command |
|---------------|---------------|---------------|
| Pipeline components | `src/shared/pipeline/` | See ARCHITECTURE.md Module Reference table |
| Feature engineering | `src/shared/features/` | See ARCHITECTURE.md Feature Modules table |
| Data providers | `src/shared/data/` | See ARCHITECTURE.md Data Providers table |
| Workflows (orchestrators) | `workflows/` | `wf_30d_returns_v2.py` is main |
| Model artifacts | `models/baseline/` | Ridge model + preprocessor |
| Documentation | `docs/` | See ARCHITECTURE.md References section |
| Utility scripts | `scripts/` | Data fetching, analysis utilities |

### Import Path Reference

All imports use the pattern: `from src.shared.<module>.<file> import Class`

**Examples:**
```python
# Pipeline components
from src.shared.pipeline.data_loading import DataLoader
from src.shared.pipeline.model_training import ModelTrainer
from src.shared.pipeline.model_evaluation_v2 import ModelEvaluatorV2

# Features
from src.shared.features.alignment import FeatureAligner
from src.shared.features.fundamental import FundamentalFeatures
from src.shared.features.technical import TechnicalFeatures

# Data providers
from src.shared.data.price import PriceProvider
from src.shared.data.macro import MacroProvider

# Config
from src.shared.config.universe import SP500_TICKERS
from src.shared.config.settings import MODEL_PARAMS
```

### Common Tasks and Where to Start

| Task | Start Here | Related Files |
|------|-----------|---------------|
| **Understand pipeline flow** | ARCHITECTURE.md → wf_30d_returns_v2.py | All src/shared/pipeline/*.py |
| **Add new feature** | src/shared/features/*.py | feature_engineering.py, alignment.py |
| **Modify model training** | src/shared/pipeline/model_training.py | model_evaluation_v2.py |
| **Change target variable** | src/shared/pipeline/target_generation.py | src/shared/features/targets.py |
| **Debug data issues** | scripts/analyze_missing_data.py | src/shared/features/alignment.py |
| **Add new workflow** | Copy wf_30d_returns_v2.py | src/workflows/<new_workflow>/ |
| **Fetch fresh data** | scripts/fetch_production_data.py | src/shared/data/*.py |
| **Evaluate model** | src/shared/pipeline/model_evaluation_v2.py | returns_analysis.py |
| **Use ranking models** | src/shared/models/ranking_*.py | ranking_metrics.py, ranking_sgd.py, ranking_xgb.py |
| **Add ranking metrics** | src/shared/models/ranking_metrics.py | model_evaluation_v2.py |

### Key Context: Ranking vs Regression

**Portfolio Selection Insight:**
When building portfolios, we select the top K stocks. What matters is:
- **Ranking accuracy** (get the top 10 correct) > absolute prediction accuracy
- A model with Spearman=0.85 and MSE=0.10 beats Spearman=0.60 and MSE=0.05

**When to use what:**
- **Standard models** (Ridge, XGBoost with MSE): When absolute predictions matter (e.g., position sizing)
- **Ranking models** (RankingSGD, RankingXGB): When selecting top K for equal-weight portfolio
- **Hybrid**: Combine MSE + rank loss (rank_weight=0.5) for balanced performance

**Ranking metrics available:**
- `rank_mae(y_true, y_pred)` - Mean absolute rank error (intuitive: "off by X positions")
- `rank_mse(y_true, y_pred)` - For optimization (smooth gradients)
- `top_k_overlap(y_true, y_pred, k=10)` - Fraction of top K correctly identified
- `decile_spread(y_true, y_pred)` - Signal strength (top decile - bottom decile)
- `spearman_loss(y_true, y_pred)` - 1 - Spearman correlation

See: `src/shared/models/ranking_metrics.py` for full API

### File Dependency Quick Reference

**Key dependency chains:**

1. **Workflow → Components:**
   ```
   wf_30d_returns_v2.py
   └─> Imports 9 pipeline components from src/shared/pipeline/
   ```

2. **Feature Engineering → Feature Modules:**
   ```
   feature_engineering.py
   ├─> technical.py (technical indicators)
   ├─> fundamental.py (fundamental ratios)
   └─> alignment.py (date/ticker alignment)
   ```

3. **Model Training → Model Evaluation:**
   ```
   model_training.py (trains model)
   └─> model_prediction.py (generates predictions)
       └─> model_evaluation_v2.py (evaluates predictions)
           └─> returns_analysis.py (financial metrics)
   ```

### Search Efficiency Tips

#### ❌ AVOID (Slow/Inefficient)
- Using Explore agent for specific file paths you already know
- Using Grep when you just need a file list (use Glob)
- Reading multiple files sequentially when you can parallel read
- Searching for code when ARCHITECTURE.md already documents the location

#### ✅ PREFER (Fast/Efficient)
- Read ARCHITECTURE.md first → then Read specific files directly
- Use Glob for file pattern matching (very fast)
- Use Grep with specific patterns, not broad searches
- Use Explore agent only for genuinely unclear locations
- Parallel tool calls when possible (multiple Reads, multiple Greps)

### Example Efficient Navigation

#### Scenario: "User asks to modify the model training step"

**Efficient approach:**
1. Read ARCHITECTURE.md (if not already read) - 30 seconds
2. Check Module Reference table → model_training.py is in src/shared/pipeline/
3. Read src/shared/pipeline/model_training.py directly - 20 seconds
4. Make changes

**Total time:** ~1 minute

**Inefficient approach:**
1. Use Explore agent to search for "model training" - 2 minutes
2. Grep for "train" keyword - 1 minute
3. Read multiple files to find the right one - 2 minutes
4. Make changes

**Total time:** ~5 minutes

#### Scenario: "User asks how features are aligned"

**Efficient approach:**
1. Read ARCHITECTURE.md → Core Concepts → FeatureAligner
2. Read src/shared/features/alignment.py directly
3. Answer question

**Inefficient approach:**
1. Grep for "align" across codebase
2. Read multiple matches
3. Infer from code

### Understanding Data Flow

If you need to understand how data flows through the pipeline, follow this path:

1. **Read ARCHITECTURE.md Data Flow section** - Shows 9 steps visually
2. **Read wf_30d_returns_v2.py** - Shows how steps are chained
3. **Read specific components** as needed for detail

```
Raw Data (parquet)
  ↓ [Step 1: data_loading.py]
Features (polars DataFrame)
  ↓ [Step 2: feature_engineering.py]
Features + Target (DataFrame)
  ↓ [Step 3: target_generation.py]
Filtered Data (DataFrame)
  ↓ [Step 4: data_filtering.py]
Train/Test Split (2 parquet files)
  ↓ [Step 5: data_splitting.py]
Trained Model (joblib)
  ↓ [Step 6: model_training.py]
Predictions (parquet)
  ↓ [Step 7: model_prediction.py]
ML Metrics (dict)
  ↓ [Step 8: model_evaluation_v2.py]
Financial Metrics (dict)
  ↓ [Step 9: returns_analysis.py]
```

## Project Context

### Current State
- **Main workflow:** V2 pipeline with proper train/test split (80/20)
- **Model:** Ridge regression with RobustScaler preprocessing
- **Target:** 30-day forward returns
- **Universe:** S&P 500 stocks (configurable in src/shared/config/universe.py)
- **Features:** Technical indicators + fundamental ratios
- **Evaluation:** Separate ML metrics (RMSE, R²) and financial metrics (Sharpe, returns)

### Recent Changes (from previous session)
- ✅ Refactored to modular pipeline architecture
- ✅ Created V2 workflow with train/test split
- ✅ Reorganized src/ into shared/ and workflows/ subdirectories
- ✅ Updated all imports to new structure
- ✅ Cleaned up 22 redundant files

### Known Issues / TODOs
Check [TODO.md](../TODO.md) for current priorities (if file exists)

## Architecture Principles

When modifying code, follow these principles:

1. **Modularity:** Each component does one thing well
2. **Composability:** Workflows compose components, don't duplicate code
3. **Time-series aware:** Always sort by date, never shuffle time-series data
4. **No lookahead bias:** Features computed only from historical data
5. **Artifact persistence:** Save intermediate outputs for debugging/reusability
6. **Clear contracts:** Each component has clear input/output types

## Performance Optimization

### When to Use Each Tool

**For navigation/search tasks:**
- ARCHITECTURE.md > Direct Read > Glob > Grep > Explore agent
- (Most efficient → Least efficient, but Explore is more thorough)

**For execution tasks:**
- Parallel tool calls when operations are independent
- Sequential only when operations depend on previous results
- Use specialized tools (Read, Edit, Write) over Bash for file operations

### Token Budget Management
- Read ARCHITECTURE.md early (4k tokens) to avoid multiple searches (10k+ tokens)
- Use Glob output to identify files, then Read only what you need
- Don't read the same file twice - take notes in todo list or messages

## Testing and Validation

### Before committing changes:
1. Check imports are correct (use Grep to verify no old imports)
2. Check file paths are absolute or relative to project root
3. Validate any DataFrame operations preserve required columns (date, ticker)
4. Consider time-series implications (no future data in features)

### How to test workflows:
```bash
cd /Users/jwassink/repos/quant_trade
source .venv/bin/activate
python workflows/wf_30d_returns_v2.py --test-size 0.2
```

## Git Context

**Current branch:** main
**No remote main branch** - First push will require `git push -u origin main`

**Typical workflow:**
1. User requests changes
2. Make changes using Edit/Write tools
3. Test changes (if requested)
4. User requests commit → Follow git safety protocol
5. User requests PR → Follow PR creation protocol

## Additional Resources

- **ARCHITECTURE.md** - Complete architecture reference (START HERE)
- **docs/v2_workflow_summary.md** - V2 workflow detailed guide
- **docs/modular_pipeline_guide.md** - Pipeline design patterns
- **docs/pipeline_quick_reference.md** - Common commands cheatsheet

---

**Remember:** ARCHITECTURE.md is your north star. Read it first in every session to understand the codebase structure. This will save 5-10 minutes of context gathering and prevent navigation mistakes.
