# Data Conventions & Standards

## Purpose
Centralized standards for data formatting, naming, and type conventions across the quant_trade framework. These conventions ensure consistency across data providers, feature engineering, and model training.

---

## 1. Stock Symbols

### Format: UPPERCASE, no special handling
```python
# ✅ Correct
'AAPL', 'MSFT', 'GOOGL', 'BRK.B'

# ❌ Incorrect
'aapl', 'Aapl', ' AAPL ', 'aapl.us'
```

### Rules:
- **Always uppercase**: Use `symbol.upper()` when ingesting
- **No whitespace**: Trim with `.strip()`
- **Preserve punctuation**: Keep dots (e.g., 'BRK.B'), hyphens (e.g., 'BRK-B')
- **Consistent delimiter**: Use dots for share classes (not hyphens)

### Special Symbols:
```python
# Market indices and ETFs with special prefixes
MARKET_SYMBOLS = {
    '^VIX': 'CBOE Volatility Index',
    '^GSPC': 'S&P 500 Index',
    '^DJI': 'Dow Jones Industrial Average',
    # ... ETFs do not need prefixes
    'SPY': 'S&P 500 ETF',
    'QQQ': 'Nasdaq 100 ETF',
}
```

### Validation:
```python
def validate_symbol(symbol: str) -> str:
    """Validate and normalize stock symbol."""
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("Symbol cannot be empty")
    if len(symbol) > 10:
        raise ValueError(f"Symbol too long: {symbol}")
    return symbol
```

---

## 2. Date & Time Formats

### Storage Format: `pl.Date` (YYYY-MM-DD, no time component)
```python
# ✅ Correct - Daily data
pl.Date  # Polars Date type (YYYY-MM-DD only)
'2025-10-23'  # ISO 8601 date string

# ❌ Incorrect - Daily data
pl.Datetime('ns')  # Overkill for daily data
pl.Datetime('us', 'America/New_York')  # Timezone complications
pd.Timestamp('2025-10-23 00:00:00')  # Unnecessary time component
```

### Why pl.Date for Daily Data?
1. **Prevents timestamp bugs**: No hour/minute mismatches (00:00:00 vs 01:00:00)
2. **Simpler**: Just dates, no timezones or precision issues
3. **More efficient**: 4 bytes vs 8 bytes for datetime
4. **Clearer intent**: Explicitly daily granularity

### Rules:
1. **Use `pl.Date` for all daily data**
   - Price data, financials, market data
   - Cast at ingestion: `.cast(pl.Date)`
   - No time component, no timezone

2. **Use `pl.Datetime` only for intraday data** (future)
   - Tick data, minute bars, order book
   - Keep intraday data in separate datasets
   - Always timezone-naive if possible

### Date Normalization Pipeline:
```python
def normalize_date_column(df: pl.DataFrame, col: str = 'date') -> pl.DataFrame:
    """
    Normalize date column for consistent joins.

    For daily data:
    - Cast to pl.Date (removes time component entirely)
    """
    return df.with_columns([
        pl.col(col).cast(pl.Date).alias(col)
    ])
```

### Why This Matters:
**Problem we encountered**: VIX data had timestamps like `2025-10-22 01:00:00` while stock data had `2025-10-22 00:00:00`. This caused all VIX joins to produce nulls because dates didn't match exactly.

**Solution**: Use `pl.Date` instead of `pl.Datetime`. No time component = no timestamp bugs.

---

## 3. Column Naming

### Format: `snake_case`, descriptive, consistent suffixes

```python
# ✅ Correct
'return_20d'           # Clear: 20-day return
'return_20d_rank'      # Suffix describes transformation
'volatility_60d'       # Horizon embedded in name
'return_5d_vs_market'  # Relative feature clearly labeled
'roe_qoq_change'       # Change frequency specified
'margin_expanding'     # Binary indicator (boolean)

# ❌ Incorrect
'ret20'                # Ambiguous abbreviation
'return20days'         # Inconsistent format (no underscore)
'Return_20D'           # CamelCase or mixed case
'20d_return'           # Leading with number (poor for sorting)
'vs_market_return_5d'  # Inconsistent suffix order
```

### Naming Patterns:

**Momentum/Returns:**
- Base: `return_{horizon}d` (e.g., `return_20d`)
- Rank: `return_{horizon}d_rank` (percentile, 0-1)
- Relative: `return_{horizon}d_vs_{benchmark}` (e.g., `return_20d_vs_market`)

**Volatility:**
- Historical: `volatility_{horizon}d` (annualized std dev)
- Rank: `volatility_{horizon}d_rank`
- Regime: `volatility_regime` (categorical: 'low', 'medium', 'high')

**Volume:**
- Ratio: `volume_ratio_{horizon}d` (vs N-day average)
- Trend: `volume_trend` (short-term vs long-term)
- Dollar volume: `dollar_volume` (price × volume)

**Fundamental Changes:**
- Quarter-over-quarter: `{metric}_qoq_change` (absolute change)
- Year-over-year: `{metric}_yoy_change` (absolute change)
- QoQ percentage: `{metric}_qoq_pct` (percentage change)
- YoY percentage: `{metric}_yoy_pct` (percentage change)
- Binary trends: `{metric}_improving` (boolean, e.g., `roe_improving`)

**Technical Indicators:**
- Standard: `rsi_{period}` (e.g., `rsi_14`)
- MACD: `macd`, `macd_signal`, `macd_histogram`
- Moving averages: `sma_{period}`, `ema_{period}`
- Price vs MA: `price_vs_sma_{period}` (percentage deviation)

**Market Features:**
- Level: `vix_level`, `spy_close`
- Regime: `vix_regime`, `market_trend_bullish` (boolean)
- Percentile: `vix_percentile` (0-100 scale)

---

## 4. Data Types

### Type Standards by Column Purpose:

**Prices & Returns:**
```python
'close': pl.Float64        # Price levels
'return_20d': pl.Float64   # Returns (can be negative)
'volume': pl.Int64          # Share counts (integer)
```

**Identifiers:**
```python
'symbol': pl.Utf8   # Stock ticker (string)
'date': pl.Date     # Date (YYYY-MM-DD, no time component)
'sector': pl.Utf8   # Categorical, but store as string
```

**Categorical Variables:**
```python
# Store as string, not Polars Categorical
'vix_regime': pl.Utf8      # 'low', 'medium', 'high'
'sector': pl.Utf8          # 'Technology', 'Healthcare', etc.

# Reason: Easier to join, filter, and serialize
# Convert to Categorical only for ML training if needed
```

**Boolean Indicators:**
```python
# Use Int8 (0/1) instead of Boolean for better null handling
'market_trend_bullish': pl.Int8  # 1 = bullish, 0 = bearish
'roe_improving': pl.Int8         # 1 = improving, 0 = not
'margin_expanding': pl.Int8      # 1 = expanding, 0 = not

# Reason: Allows null representation (warm-up periods)
# Reason: Easier arithmetic (sum to count positives)
```

**Percentiles & Ranks:**
```python
'return_20d_rank': pl.Float64   # 0.0 to 1.0 (cross-sectional percentile)
'vix_percentile': pl.Float64    # 0.0 to 100.0 (time-series percentile)

# Note: Choose scale consistently per use case
# Cross-sectional: 0-1 scale (easier for interpretation)
# Time-series: 0-100 scale (matches typical percentile reporting)
```

---

## 5. Missing Data Handling

### Null Values vs Special Values

**Use `null` for:**
- Warm-up periods (rolling windows not full yet)
- Missing source data
- Invalid computations (divide by zero → null, not inf)

**Never use:**
- `NaN` - convert to `null`
- `inf` or `-inf` - cap/winsorize instead
- Magic numbers like `-999` or `0` - use `null`

### Forward-Fill Policy:
```python
# Fundamental data: Forward-fill to daily
# Reason: Quarterly metrics stay constant until next report
financials_df = financials_df.sort(['symbol', 'date']).with_columns([
    pl.col('roe').forward_fill().over('symbol')
])

# Price data: No forward-fill
# Reason: Missing prices indicate no trading (gaps should remain null)
```

### Null Thresholds:
- **<5% nulls**: Acceptable, likely warm-up periods
- **5-20% nulls**: Review carefully, document reason
- **>20% nulls**: Red flag, investigate

### Warm-up Period Documentation:
```python
# Always document expected null count for rolling features
def add_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """
    Add RSI indicator.

    Warm-up period: First {period} rows per symbol will be null.
    Expected nulls: {period} × {num_symbols}
    """
    ...
```

---

## 6. Outlier Handling

### When to Winsorize:
- Fundamental ratios (growth rates, margins)
- Returns (only if extreme and confirmed erroneous)
- Volume ratios (spike detection vs data error)

### When NOT to Winsorize:
- Price levels (actual market prices)
- Volume levels (actual share counts)
- Technical indicators (RSI, MACD - naturally bounded)
- VIX (actual volatility measure)

### Winsorization Standards:
```python
def winsorize_feature(
    df: pl.DataFrame,
    col: str,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pl.DataFrame:
    """
    Winsorize at 1st and 99th percentiles (default).

    Use 0.05/0.95 for more aggressive winsorization.
    Use 0.01/0.99 for light winsorization (recommended).
    """
    lower = df[col].quantile(lower_pct)
    upper = df[col].quantile(upper_pct)

    return df.with_columns([
        pl.col(col).clip(lower, upper).alias(col)
    ])
```

### Document Winsorization:
```python
# In docstring or comments, note:
# - Which features are winsorized
# - At what percentiles
# - Reason for winsorization
```

---

## 7. File Naming & Storage

### Directory Structure:
```
data/
├── price/
│   └── daily/
│       └── {universe}_{start_date}_to_{end_date}.parquet
├── financials/
│   └── quarterly_financials_{fetch_date}.parquet
├── market/
│   └── daily/
│       └── market_data_{fetch_date}.parquet
└── metadata/
    └── company_metadata_{fetch_date}.parquet
```

### File Naming Convention:
```python
# Format: {data_type}_{frequency}_{descriptor}_{date_range}.parquet

# ✅ Correct
'sample_universe_2022-10-24_to_2025-10-23.parquet'  # Price data
'quarterly_financials_latest.parquet'               # Latest fetch
'market_data_2025-10-23.parquet'                     # Market data

# ❌ Incorrect
'data.parquet'                    # Not descriptive
'AAPL_price_2023.parquet'         # Don't split by symbol
'financials_2023_Q1.parquet'      # Don't split by quarter
```

### Date Format in Filenames: `YYYY-MM-DD`
- Sortable alphabetically
- ISO 8601 standard
- Unambiguous internationally

---

## 8. Schema Validation

### Enforce Schema at Ingestion:
```python
# Define expected schema
PRICE_SCHEMA = {
    'symbol': pl.Utf8,
    'date': pl.Date,  # Daily data uses pl.Date
    'open': pl.Float64,
    'high': pl.Float64,
    'low': pl.Float64,
    'close': pl.Float64,
    'adj_close': pl.Float64,
    'volume': pl.Int64,
}

def validate_schema(df: pl.DataFrame, expected: dict) -> None:
    """Validate DataFrame matches expected schema."""
    actual_schema = {col: dtype for col, dtype in df.schema.items()}

    for col, expected_dtype in expected.items():
        if col not in actual_schema:
            raise ValueError(f"Missing column: {col}")
        if actual_schema[col] != expected_dtype:
            raise TypeError(
                f"Column '{col}' has type {actual_schema[col]}, "
                f"expected {expected_dtype}"
            )
```

---

## 9. Cross-Dataset Joins

### Join Keys:
Primary join keys across datasets:
1. **`symbol`** (always present, uppercase, Utf8)
2. **`date`** (always present, datetime[ns], timezone-naive, truncated to day)

### Before Every Join:
```python
# 1. Normalize both DataFrames
df1 = normalize_date_column(df1)
df2 = normalize_date_column(df2)

# 2. Verify join keys exist
assert 'symbol' in df1.columns and 'symbol' in df2.columns
assert 'date' in df1.columns and 'date' in df2.columns

# 3. Verify data types match
assert df1.schema['symbol'] == df2.schema['symbol']
assert df1.schema['date'] == df2.schema['date']

# 4. Perform join
result = df1.join(df2, on=['symbol', 'date'], how='left')
```

---

## 10. Feature Documentation

### Feature Metadata:
Every feature computation should include:
```python
def add_return_features(df: pl.DataFrame, horizons: list[int]) -> pl.DataFrame:
    """
    Add return features over multiple horizons.

    Features created:
        - return_{N}d: N-day simple return
        - return_{N}d_rank: Cross-sectional percentile rank (0-1)

    Warm-up period: {max(horizons)} days per symbol
    Expected nulls: {max(horizons)} × {num_symbols}

    Args:
        df: Price DataFrame with 'close' column
        horizons: List of lookback periods in days

    Returns:
        DataFrame with return features added
    """
    ...
```

---

## 11. Version Control for Data

### What to Commit:
✅ Small reference datasets (<1MB)
✅ Sample/test data
✅ Configuration files with universe definitions
✅ Schema definitions

### What NOT to Commit:
❌ Large datasets (>1MB) - use `.gitignore`
❌ Fetched market data (regenerable)
❌ Intermediate computation results
❌ Model checkpoints (unless explicitly versioned)

### Data Versioning:
```python
# Include fetch date in filename for reproducibility
'market_data_2025-10-23.parquet'  # Fetched on 2025-10-23

# Or use descriptive names for curated datasets
'sample_universe_3yr_history.parquet'  # Clear time range
```

---

## 12. Code Review Checklist

Before committing feature engineering code, verify:

**Date Handling:**
- [ ] All date columns use `pl.Date` for daily data
- [ ] Dates cast to pl.Date at ingestion (`.cast(pl.Date)`)
- [ ] No datetime/timezone used for daily data

**Column Naming:**
- [ ] snake_case format
- [ ] Descriptive names with clear suffixes
- [ ] Horizon embedded in name (e.g., `_20d`)
- [ ] Consistent with existing features

**Data Types:**
- [ ] Prices/returns as Float64
- [ ] Volume as Int64
- [ ] Symbols as Utf8 (uppercase)
- [ ] Boolean indicators as Int8 (0/1)
- [ ] Nulls used properly (no inf, NaN, or magic numbers)

**Documentation:**
- [ ] Docstring describes features created
- [ ] Warm-up period documented
- [ ] Expected null count noted
- [ ] Outlier handling described

**Testing:**
- [ ] Schema validation passes
- [ ] Join keys verified before joins
- [ ] Null rates checked (<20% except warm-up)
- [ ] Outliers inspected (winsorize if needed)

---

## Implementation

### Create Validation Utilities:
```python
# src/utils/validation.py

def validate_price_data(df: pl.DataFrame) -> None:
    """Validate price data meets standards."""
    # Check required columns
    required = ['symbol', 'date', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Check data types
    assert df.schema['symbol'] == pl.Utf8
    assert df.schema['date'] == pl.Date  # Daily data uses pl.Date
    assert df.schema['close'] == pl.Float64

    # Check symbol format
    assert df['symbol'].str.to_uppercase().equals(df['symbol'])

def validate_feature_names(df: pl.DataFrame) -> None:
    """Validate feature names meet conventions."""
    for col in df.columns:
        # Check snake_case
        if col != col.lower() or ' ' in col:
            raise ValueError(f"Column '{col}' not in snake_case")

        # Check no leading numbers
        if col[0].isdigit():
            raise ValueError(f"Column '{col}' starts with number")
```

---

## Summary

**Key Principles:**
1. **Consistency**: Same format everywhere (uppercase symbols, pl.Date, snake_case)
2. **Simplicity**: Use pl.Date for daily data, avoid timezones/datetime complexity
3. **Explicit**: Document warm-up periods, outlier handling, null reasons
4. **Validation**: Check schemas, types, and formats at ingestion
5. **Joins**: Dates automatically match with pl.Date (no truncation needed)

**Most Critical:**
- **Symbols**: Always uppercase
- **Dates**: Use `pl.Date` for daily data (not datetime)
- **Nulls**: Use properly, document expected counts
- **Column names**: snake_case with clear suffixes

Following these conventions will prevent subtle bugs (like the VIX join issue we encountered with datetime mismatches) and make the codebase more maintainable as it grows.
