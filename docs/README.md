# Documentation Index

Welcome to the quant_trade documentation! This directory contains all project documentation organized by category.

---

## Quick Navigation

### 🚀 Getting Started
- [../README.md](../README.md) - Project overview
- [../QUICKSTART.md](../QUICKSTART.md) - Setup and first steps
- [../TODO.md](../TODO.md) - Current tasks and roadmap

### 🛠️ Development Standards
- **[development/data_conventions.md](development/data_conventions.md)** - **START HERE** for data work
  - Symbol formatting, date handling, column naming
  - Critical for preventing bugs (see VIX join issue)
  - Schema validation, outlier handling
- [development/git_workflow.md](development/git_workflow.md) - Git best practices, commit conventions
- [claude_context.md](claude_context.md) - Context for Claude AI assistant

### 🏗️ Architecture
- [architecture/regression_framework_spec.md](architecture/regression_framework_spec.md) - ML framework specification (30-day returns)
- [architecture/feature_summary.md](architecture/feature_summary.md) - Feature categories overview
- [architecture/modular_architecture.md](architecture/modular_architecture.md) - System design principles
- [architecture/data_acquisition_guide.md](architecture/data_acquisition_guide.md) - Data sources and providers
- [architecture/data_sources_feasibility.md](architecture/data_sources_feasibility.md) - Alternative data evaluation

### 📈 Trading Strategies
- [strategies/README.md](strategies/README.md) - Strategy documentation index
- [strategies/1_month_framework.md](strategies/1_month_framework.md) - 1-month holding period framework
- **Priority Strategies:**
  - [strategies/earnings_momentum_1m.md](strategies/earnings_momentum_1m.md) - Post-earnings drift (HIGH priority)
  - [strategies/insider_cluster_1m.md](strategies/insider_cluster_1m.md) - Insider trading clusters (HIGH priority)
  - [strategies/github_activity_1m.md](strategies/github_activity_1m.md) - Developer activity signals (MEDIUM priority)
- **Research Strategies:**
  - [strategies/order_book_imbalance.md](strategies/order_book_imbalance.md) - Microstructure approach

### 📊 Reports & Evaluations
- [reports/feature_evaluation_2025-10-23.md](reports/feature_evaluation_2025-10-23.md) - Latest feature quality assessment
- [reports/codebase_review_2025-10-23.md](reports/codebase_review_2025-10-23.md) - Latest codebase analysis

---

## Documentation Organization

```
docs/
├── README.md                    # This file
├── claude_context.md            # Claude AI context (important changes, file map)
├── development/                 # Development standards & practices
│   ├── data_conventions.md      # **CRITICAL** Data formatting rules
│   └── git_workflow.md          # Git & commit best practices
├── architecture/                # System design & specifications
│   ├── README.md
│   ├── regression_framework_spec.md
│   ├── feature_summary.md
│   ├── modular_architecture.md
│   ├── data_acquisition_guide.md
│   └── data_sources_feasibility.md
├── strategies/                  # Trading strategy documentation
│   ├── README.md
│   ├── 1_month_framework.md
│   ├── earnings_momentum_1m.md
│   ├── insider_cluster_1m.md
│   ├── github_activity_1m.md
│   └── order_book_imbalance.md
└── reports/                     # Dated evaluation reports
    ├── codebase_review_2025-10-23.md
    └── feature_evaluation_2025-10-23.md
```

---

## Documentation Standards

### When to Create New Docs

**Architecture docs**: Major design decisions, specifications
**Strategy docs**: New strategy research, backtests, parameters
**Reports**: Evaluations, reviews, audits (include date in filename)
**Development guides**: Reusable patterns, best practices, workflows

### File Naming

- Use `snake_case.md` for all markdown files
- Include dates in reports: `feature_evaluation_2025-10-23.md`
- Be descriptive: `regression_framework_spec.md` not `spec.md`

### Documentation Content

Each doc should include:
1. **Purpose**: Why does this doc exist?
2. **Context**: What problem does it solve?
3. **Details**: Specific information, examples, code snippets
4. **References**: Links to related docs, code files

### Updating Docs

**When code changes significantly**:
- Update relevant architecture docs
- Update claude_context.md "Recent Changes" section
- Create dated report if major evaluation/review

**When strategies evolve**:
- Update strategy docs with new findings
- Document backtest results
- Note parameter changes

---

## Key Concepts

### Current Phase: Feature Engineering ✓
We've completed building 111 features across 3 categories:
- Technical (33): Price/volume patterns, indicators
- Fundamental (22): Financial metrics, QoQ/YoY changes
- Sector/Market (56): Relative performance, VIX regime

**Next**: Model training, backtesting

### Data Conventions (CRITICAL)
Before doing any data work, read [development/data_conventions.md](development/data_conventions.md)

**Why it matters**: We encountered a subtle bug where VIX data had hour timestamps (`01:00:00`) while stock data had (`00:00:00`), causing all joins to produce nulls. The conventions document prevents issues like this.

**Key rules**:
- Symbols: UPPERCASE
- Dates: datetime[ns], timezone-naive, truncated to day
- Columns: snake_case with clear suffixes
- Nulls: Use properly, never NaN/inf

### Strategy Focus
Currently focusing on **1-month holding period strategies**:
- Long-only positions
- Monthly rebalancing
- Target: 3-8% monthly returns
- Manual validation before automation

---

## Contributing to Documentation

### Adding a New Doc

1. Choose appropriate directory (architecture/, strategies/, development/, reports/)
2. Follow naming conventions (snake_case, dates for reports)
3. Use markdown format with clear structure
4. Link from relevant README.md files
5. Update claude_context.md if significant

### Updating Existing Docs

1. Keep original structure intact
2. Add "Updated: YYYY-MM-DD" at top if major changes
3. Preserve historical context (don't delete old decisions)
4. Update related docs' links if filename changes

### Creating Reports

1. Always include date in filename: `report_name_2025-10-23.md`
2. Save in `reports/` directory
3. Include: Purpose, Methods, Findings, Recommendations
4. Link from claude_context.md "Recent Changes" section

---

## Questions?

- For git workflow questions: See [development/git_workflow.md](development/git_workflow.md)
- For data formatting questions: See [development/data_conventions.md](development/data_conventions.md)
- For architecture decisions: See [architecture/](architecture/)
- For strategy research: See [strategies/](strategies/)
- For recent changes: See [claude_context.md](claude_context.md) "Recent Changes" section

---

**Last Updated**: 2025-10-23
