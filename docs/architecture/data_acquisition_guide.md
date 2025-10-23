# Data Acquisition Guide - Historical & Current

## Overview

Detailed guide for acquiring **historical** (for training) and **current** (for live predictions) data for all 115+ features in our regression framework.

## Status Legend

- ✅ **READY**: Easy to obtain, API available, well-documented
- ⚠️ **MODERATE**: Obtainable but requires scraping or workarounds
- ❌ **DIFFICULT**: Limited free access, may need paid tier
- 🔄 **COMPUTED**: Calculated from other data sources

---

## 1. Technical Features (~20 features)

### Data Source: yfinance
**Status**: ✅ READY

### Historical Data
```python
import yfinance as yf
from datetime import datetime

# Get 5 years of historical data
ticker = yf.Ticker("AAPL")
hist = ticker.history(start="2019-01-01", end="2024-01-01")

# Returns DataFrame with:
# - Date (index)
# - Open, High, Low, Close, Volume
# - Dividends, Stock Splits
```

**Available History**:
- Most stocks: Back to IPO date
- Typical: 10-20 years easily
- Adjusted for splits/dividends automatically

### Current Data
```python
# Get most recent data (real-time during market hours)
current = ticker.history(period="1d")

# Or get last N days
recent = ticker.history(period="5d")
```

**Update Frequency**:
- Free tier: ~15 minute delay
- Sufficient for daily rebalancing

### Features Computed 🔄

From OHLCV data:
- ✅ **Momentum**: Returns over 1d, 5d, 10d, 20d, 60d
- ✅ **Volatility**: Rolling std dev (10d, 20d, 60d)
- ✅ **Volume ratios**: Current vs average
- ✅ **Moving averages**: 20/50/200 day
- ✅ **52-week high/low**: Max/min over 252 days
- ✅ **ATR**: Average True Range
- ✅ **Acceleration**: Rate of change of momentum

**Code Example**:
```python
import polars as pl

def compute_technical_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute all technical features from OHLCV"""

    # Momentum (returns)
    for window in [5, 10, 20, 60]:
        df = df.with_columns([
            ((pl.col("close") / pl.col("close").shift(window)) - 1.0)
            .over("symbol")
            .alias(f"return_{window}d")
        ])

    # Volatility
    for window in [10, 20, 60]:
        df = df.with_columns([
            pl.col("close").pct_change()
            .rolling_std(window)
            .over("symbol")
            .alias(f"volatility_{window}d")
        ])

    # Volume ratio
    df = df.with_columns([
        (pl.col("volume") / pl.col("volume").rolling_mean(20))
        .over("symbol")
        .alias("volume_ratio_20d")
    ])

    # Distance from moving averages
    for ma in [20, 50, 200]:
        df = df.with_columns([
            ((pl.col("close") / pl.col("close").rolling_mean(ma)) - 1.0)
            .over("symbol")
            .alias(f"dist_from_ma{ma}")
        ])

    # 52-week high
    df = df.with_columns([
        ((pl.col("close") / pl.col("close").rolling_max(252)) - 1.0)
        .over("symbol")
        .alias("dist_from_52w_high")
    ])

    return df
```

**Availability**: ✅ 100% available historically and currently

---

## 2. Fundamental Features (~25 features)

### Data Source: yfinance (basic) + Financial Modeling Prep (enhanced)
**Status**: ✅ READY (free tier) / ⚠️ MODERATE (full historical)

### Historical Data - Basic (FREE)

```python
import yfinance as yf

ticker = yf.Ticker("AAPL")

# Annual financials
income_stmt = ticker.financials          # Income statement
balance_sheet = ticker.balance_sheet     # Balance sheet
cash_flow = ticker.cashflow              # Cash flow statement

# Quarterly (more timely)
quarterly_financials = ticker.quarterly_financials
quarterly_balance_sheet = ticker.quarterly_balance_sheet

# Key metrics
info = ticker.info
pe_ratio = info.get('trailingPE')
pb_ratio = info.get('priceToBook')
roe = info.get('returnOnEquity')
debt_equity = info.get('debtToEquity')
```

**Available History**:
- Annual: Usually 4-5 years
- Quarterly: Usually 4-8 quarters
- Current metrics: Real-time

**Limitation**:
- Historical quarterly data limited
- No point-in-time snapshots (as-reported data)

### Historical Data - Enhanced (Financial Modeling Prep - $14/mo)

```python
import requests

API_KEY = "your_key"
symbol = "AAPL"

# Historical earnings with surprises
url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{symbol}"
params = {"apikey": API_KEY}
earnings_history = requests.get(url, params=params).json()

# Returns:
# - date, eps_actual, eps_estimate, revenue_actual, revenue_estimate
# - Goes back 5+ years

# Historical financial ratios
url = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}"
ratios_history = requests.get(url, params=params).json()

# Returns quarterly/annual:
# - P/E, P/B, ROE, ROA, debt/equity, margins, etc.
# - 10+ years of history
```

**Available History**:
- Earnings: 5-10 years
- Financials: 10+ years
- Ratios: 10+ years quarterly

### Current Data

```python
# Current metrics (yfinance)
ticker = yf.Ticker("AAPL")
current_pe = ticker.info['trailingPE']
current_pb = ticker.info['priceToBook']

# Next earnings date
next_earnings = ticker.calendar  # Earnings date, estimate
```

### Features Available

**From yfinance (FREE)**:
- ✅ Current P/E, P/B, EV/EBITDA
- ✅ Current ROE, margins, debt/equity
- ⚠️ Limited historical quarterly data
- ✅ Next earnings date

**From FMP ($14/mo)**:
- ✅ Historical earnings surprises (5+ years)
- ✅ Historical financial ratios (10+ years)
- ✅ Quarterly point-in-time data
- ✅ Analyst estimates (current + historical revisions)

**Recommendation**:
- Start with yfinance (free)
- Add FMP later for historical backtesting depth

---

## 3. Alternative Data Features (~15 features)

### 3a. Insider Trading

**Data Source**: SEC EDGAR / OpenInsider
**Status**: ✅ READY (both historical and current)

#### Historical Data - SEC EDGAR (FREE, official)

```python
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def get_insider_filings(cik: str, start_date: datetime, end_date: datetime):
    """Get Form 4 filings from SEC EDGAR"""

    # SEC EDGAR search
    url = "https://www.sec.gov/cgi-bin/browse-edgar"
    params = {
        'action': 'getcompany',
        'CIK': cik,
        'type': '4',  # Form 4 = insider transactions
        'dateb': end_date.strftime('%Y%m%d'),
        'owner': 'include',
        'count': '100'
    }

    response = requests.get(url, params=params)
    # Parse HTML to get filing links
    # Download and parse XML for each filing
    # Extract: date, insider name, transaction type (P/S), shares, price

    return transactions_df
```

**Available History**:
- Complete history back to ~2003
- All insider transactions publicly filed
- 2-day filing delay by law

**Challenge**:
- Requires XML parsing
- No clean API (scraping HTML/XML)

#### Historical Data - OpenInsider (FREE, pre-processed)

**Easier Alternative**:
- Website: http://openinsider.com
- Has screening tools built-in
- Shows "clusters" automatically
- Can export to CSV

**Manual Process**:
1. Use OpenInsider screener
2. Filter for clusters (3+ insiders)
3. Download historical data as CSV
4. Build database

**Available History**:
- Several years of data
- Already aggregated (saves work)

#### Current Data

```python
# Check OpenInsider daily for new clusters
# Or set up SEC EDGAR scraper to run daily

# Pseudo-code for daily update:
def check_daily_clusters():
    # Get yesterday's Form 4 filings
    filings = get_recent_form4s(days=1)

    # Group by company
    clusters = filings.groupby('cik').filter(lambda x: len(x) >= 3)

    # Alert if cluster found
    if not clusters.empty:
        send_alert(clusters)
```

**Features Available**:
- ✅ Historical insider clusters (3+ years via OpenInsider)
- ✅ Current clusters (daily monitoring)
- ✅ Transaction amounts, dates, insider titles
- ✅ Buy/sell ratios

**Recommendation**:
- Use OpenInsider for initial historical data (manual download)
- Build SEC scraper for live monitoring
- Store in local database

---

### 3b. GitHub Activity (Tech Stocks)

**Data Source**: GitHub API
**Status**: ✅ READY (free tier sufficient)

#### Historical Data

```python
import requests
from datetime import datetime, timedelta

def get_commit_activity(org: str, repo: str):
    """Get weekly commit activity (52 weeks)"""

    url = f"https://api.github.com/repos/{org}/{repo}/stats/commit_activity"

    # No auth needed for public repos (60 requests/hour)
    # With auth: 5000 requests/hour
    response = requests.get(url)

    # Returns 52 weeks of data:
    # [{week: unix_timestamp, total: commits, days: [...]}, ...]
    return response.json()

def get_contributor_stats(org: str, repo: str):
    """Get contributor statistics"""

    url = f"https://api.github.com/repos/{org}/{repo}/stats/contributors"
    response = requests.get(url)

    # Returns per-contributor commit history
    return response.json()

def get_repo_metadata(org: str, repo: str):
    """Get stars, forks, watchers"""

    url = f"https://api.github.com/repos/{org}/{repo}"
    response = requests.get(url)

    data = response.json()
    return {
        'stars': data['stargazers_count'],
        'forks': data['forks_count'],
        'watchers': data['watchers_count'],
        'open_issues': data['open_issues_count']
    }
```

**Available History**:
- Commit activity: 52 weeks (rolling)
- Contributors: All-time history
- Stars/forks: Current count only (no history)

**Limitation**:
- Commit activity only goes back 52 weeks via API
- For deeper history, need to use GitHub Archive (more complex)

#### Historical Data - Extended (GitHub Archive)

For > 1 year of history:
- Source: https://www.gharchive.org/
- Contains all GitHub events since 2011
- Requires downloading large JSON files
- More complex to parse

**Recommendation**:
- Use API for 52 weeks (sufficient for most backtests)
- Skip GH Archive unless needed

#### Current Data

```python
# Update weekly
def update_github_data(companies_repos: dict):
    """
    companies_repos = {
        'MDB': [('mongodb', 'mongo')],
        'DDOG': [('DataDog', 'datadog-agent')],
        # ...
    }
    """

    for symbol, repos in companies_repos.items():
        for org, repo in repos:
            commits = get_commit_activity(org, repo)
            metadata = get_repo_metadata(org, repo)

            # Store in database
            save_github_data(symbol, org, repo, commits, metadata)
```

**Features Available**:
- ✅ Commit velocity (52 weeks historical)
- ✅ Contributor counts (all-time)
- ✅ Current stars/forks (snapshot)
- ⚠️ Star growth rate (need to track over time)

**Recommendation**:
- Build weekly scraper for ~30 tech stocks
- Store time-series locally
- 52 weeks sufficient for momentum signals

---

### 3c. Analyst Data

**Data Source**: Seeking Alpha (scraping) or FMP ($14/mo)
**Status**: ⚠️ MODERATE (scraping) / ✅ READY (paid)

#### Free Option - Seeking Alpha (Legal Gray Area)

**Caution**: Check terms of service

```python
# Pseudo-code (scraping required)
def scrape_analyst_ratings(symbol: str):
    url = f"https://seekingalpha.com/symbol/{symbol}/ratings"
    # Parse HTML for:
    # - Analyst ratings (buy/hold/sell)
    # - Target prices
    # - Estimate revisions
```

**Available History**: Limited via scraping

#### Paid Option - Financial Modeling Prep ($14/mo)

```python
# Analyst estimates and ratings
url = f"https://financialmodelingprep.com/api/v3/analyst-estimates/{symbol}"
estimates = requests.get(url, params={'apikey': API_KEY}).json()

# Analyst ratings (upgrades/downgrades)
url = f"https://financialmodelingprep.com/api/v3/upgrades-downgrades/{symbol}"
ratings = requests.get(url, params={'apikey': API_KEY}).json()

# Target price consensus
url = f"https://financialmodelingprep.com/api/v3/price-target-consensus"
targets = requests.get(url, params={'apikey': API_KEY, 'symbol': symbol}).json()
```

**Available History**:
- Estimates: 5+ years
- Ratings: 5+ years
- Target prices: 3+ years

**Recommendation**:
- Skip for MVP (not critical)
- Add FMP tier if budget allows
- Focus on free features first

---

## 4. Stock Metadata Features (~15 features)

**Data Source**: yfinance
**Status**: ✅ READY

### Historical Data - Challenge

**Problem**: Most metadata is current snapshot only
**Workaround**: Build time-series database

```python
import yfinance as yf

ticker = yf.Ticker("AAPL")
info = ticker.info

# Available metadata (CURRENT snapshot):
metadata = {
    'symbol': 'AAPL',
    'sector': info['sector'],                      # ✅ Stable (rarely changes)
    'industry': info['industry'],                   # ✅ Stable
    'market_cap': info['marketCap'],               # 🔄 Changes daily (calculate from price)
    'employees': info.get('fullTimeEmployees'),    # ✅ Changes quarterly
    'shares_outstanding': info['sharesOutstanding'],# ✅ Changes occasionally
    'float': info.get('floatShares'),              # ✅ Changes occasionally
    'ipo_date': None,  # Not directly available    # ⚠️ Need workaround
}
```

### Historical Workarounds

#### Market Cap (Easy) 🔄
```python
# Calculate from historical prices
market_cap_historical = price * shares_outstanding
```

#### Sector/Industry (Stable) ✅
```python
# Assume current sector applies historically
# Changes are rare (1-2% per year)
# Can manually track major reclassifications if needed
```

#### Employee Count (Moderate) ⚠️
```python
# Available in quarterly financials for many companies
ticker.quarterly_financials  # Sometimes includes employee count

# Or scrape from 10-K/10-Q filings (SEC EDGAR)
# Usually in "Item 1 - Business" section
```

#### IPO Date (Workaround) ⚠️
```python
# Use first available trading date as proxy
hist = ticker.history(period="max")
ipo_date_approx = hist.index.min()

# Age since IPO
age_years = (datetime.now() - ipo_date_approx).days / 365
```

### Current Data

```python
def get_current_metadata(symbol: str) -> dict:
    """Get current stock metadata"""
    ticker = yf.Ticker(symbol)
    info = ticker.info

    return {
        'symbol': symbol,
        'sector': info.get('sector'),
        'industry': info.get('industry'),
        'market_cap': info.get('marketCap'),
        'employees': info.get('fullTimeEmployees'),
        'shares_outstanding': info.get('sharesOutstanding'),
        'float': info.get('floatShares'),
        'country': info.get('country'),
    }

# Update weekly (metadata changes slowly)
```

### Features Available

- ✅ **Sector/Industry**: Current (assume stable historically)
- ✅ **Market cap**: Calculate from price * shares
- ⚠️ **Employee count**: Current + some historical
- ✅ **Shares outstanding**: Current (track changes)
- ⚠️ **Age since IPO**: Approximate from first trade date
- 🔄 **Percentile rankings**: Compute cross-sectionally

**Recommendation**:
- Capture metadata snapshots weekly
- Build time-series database
- Use current sector/industry for historical (acceptable approximation)
- Calculate market cap from price history

---

## 5. Sector & Market Features (~30 features)

**Data Source**: yfinance (sector ETFs) + FRED (economic)
**Status**: ✅ READY

### Historical Data - Sector ETFs

```python
import yfinance as yf

# Download all sector ETFs
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Financials': 'XLF',
    'Healthcare': 'XLV',
    'Energy': 'XLE',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Industrials': 'XLI',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Utilities': 'XLU',
    'Communication Services': 'XLC'
}

# Get historical data
tickers = list(SECTOR_ETFS.values())
sector_data = yf.download(tickers, start="2019-01-01", end="2024-01-01")

# Also get SPY for market reference
spy_data = yf.download("SPY", start="2019-01-01", end="2024-01-01")
```

**Available History**:
- Most sector ETFs: 20+ years
- XLRE (Real Estate): Since 2015
- XLC (Communication): Since 2018
- **Workaround**: Use sector indices before ETF inception

### Historical Data - Market Indicators

```python
# VIX, yields, commodities
market_tickers = {
    'vix': '^VIX',
    'spy': 'SPY',
    'tnx': '^TNX',      # 10-year yield
    'dxy': 'DX-Y.NYB',  # Dollar index
    'gold': 'GC=F',
    'oil': 'CL=F',
}

market_data = yf.download(
    list(market_tickers.values()),
    start="2019-01-01",
    end="2024-01-01"
)

# Economic data from FRED
from pandas_datareader import data as web

gdp = web.DataReader('GDP', 'fred', '2019-01-01', '2024-01-01')
unemployment = web.DataReader('UNRATE', 'fred', '2019-01-01', '2024-01-01')
```

**Available History**:
- VIX: Since 1990
- Yields: Decades
- SPY: Since 1993
- Commodities: 20+ years
- FRED data: Varies (usually decades)

### Current Data

```python
# Update daily
def update_market_data():
    # Sector ETFs
    sector_etfs = list(SECTOR_ETFS.values())
    current_sectors = yf.download(sector_etfs, period="5d")

    # Market indicators
    current_vix = yf.Ticker("^VIX").history(period="5d")
    current_yields = yf.Ticker("^TNX").history(period="5d")

    # Save to database
    save_market_data(current_sectors, current_vix, current_yields)
```

### Features Computed 🔄

From sector ETF + market data:
- ✅ **Sector momentum**: 5d, 20d, 60d returns per sector
- ✅ **Sector relative**: Sector return - SPY return
- ✅ **Sector volatility**: Rolling std dev
- ✅ **VIX level & percentile**: Current vs historical
- ✅ **Yield levels**: 10Y treasury, changes
- ✅ **Stock beta**: Stock return correlation to SPY/sector

**Code Example**:
```python
def compute_sector_features(stock_data, sector_data, spy_data):
    """Compute sector-relative features"""

    # Get stock's sector
    sector = get_stock_sector(symbol)  # e.g., "Technology"
    sector_etf = SECTOR_ETFS[sector]    # e.g., "XLK"

    # Sector momentum
    sector_return_20d = (
        sector_data[sector_etf].pct_change(20).iloc[-1]
    )

    # Sector relative to market
    spy_return_20d = spy_data.pct_change(20).iloc[-1]
    sector_relative = sector_return_20d - spy_return_20d

    # Stock beta to sector
    stock_returns = stock_data.pct_change()
    sector_returns = sector_data[sector_etf].pct_change()
    sector_beta = stock_returns.corr(sector_returns)

    return {
        'sector_momentum_20d': sector_return_20d,
        'sector_relative_20d': sector_relative,
        'sector_beta': sector_beta,
        # ... more features
    }
```

### Features Available

- ✅ All sector features (10+ years history)
- ✅ All market regime features (20+ years)
- ✅ Cross-asset indicators (20+ years)
- ✅ Relative positioning (computed from above)

**Availability**: ✅ 100% available historically and currently

---

## 6. Event Features (~10 features)

**Status**: 🔄 COMPUTED from other data

### Earnings Proximity

```python
# From yfinance
ticker = yf.Ticker("AAPL")
earnings_dates = ticker.earnings_dates  # Historical earnings dates

# Compute features
def compute_earnings_features(current_date, earnings_dates):
    # Days since last earnings
    last_earnings = earnings_dates[earnings_dates < current_date].max()
    days_since = (current_date - last_earnings).days

    # Days until next earnings
    next_earnings = earnings_dates[earnings_dates > current_date].min()
    days_until = (next_earnings - current_date).days

    return {
        'days_since_earnings': days_since,
        'days_until_earnings': days_until,
        'within_earnings_week': days_until <= 7,
        'post_earnings_30d': days_since <= 30,
    }
```

**Available History**: ✅ Via earnings dates from yfinance

### Seasonal Features

```python
def compute_seasonal_features(date):
    return {
        'month': date.month,
        'day_of_week': date.dayofweek,
        'quarter': (date.month - 1) // 3 + 1,
        'is_january': date.month == 1,
        'is_monday': date.dayofweek == 0,
    }
```

**Available History**: ✅ Trivially computed from dates

---

## Summary Table: Data Availability

| Feature Category | Historical | Current | Source | Status | Cost |
|-----------------|-----------|---------|--------|--------|------|
| **Technical** | 10-20 years | Real-time | yfinance | ✅ READY | FREE |
| **Fundamental (basic)** | 4-5 years | Current | yfinance | ✅ READY | FREE |
| **Fundamental (full)** | 10+ years | Current | FMP | ✅ READY | $14/mo |
| **Insider Trading** | 5+ years | Daily | OpenInsider/SEC | ✅ READY | FREE |
| **GitHub Activity** | 52 weeks | Weekly | GitHub API | ✅ READY | FREE |
| **Analyst Data** | Limited | Current | FMP / Scraping | ⚠️ MODERATE | $14/mo |
| **Stock Metadata** | Approximate | Current | yfinance | ✅ READY | FREE |
| **Sector/Market** | 10-20 years | Daily | yfinance/FRED | ✅ READY | FREE |
| **Event Features** | Full | Computed | Derived | ✅ READY | FREE |

---

## Implementation Priority

### Phase 1: Core Data (Week 1)
**Goal**: Get 80% of features working

1. ✅ Price data (yfinance) - all stocks
2. ✅ Sector ETF data (yfinance) - 11 ETFs
3. ✅ Market indicators (yfinance) - VIX, SPY, yields
4. ✅ Stock metadata (yfinance) - weekly snapshots
5. 🔄 Compute technical features
6. 🔄 Compute sector/market features

**Outcome**: ~70 features available, 3-5 years of history

### Phase 2: Alternative Data (Week 2)
**Goal**: Add unique signals

1. ⚠️ Insider data (OpenInsider manual download + SEC scraper)
2. ⚠️ GitHub data (API scraper for 30 tech stocks)
3. 🔄 Compute alternative features

**Outcome**: ~90 features available

### Phase 3: Enhanced Fundamentals (Week 3)
**Goal**: Deeper history

1. Optional: Add FMP ($14/mo) for earnings history
2. 🔄 Compute fundamental features
3. Build point-in-time database

**Outcome**: All ~115 features, 5-10 years history

---

## Next Steps

1. ✅ Data sources identified and validated
2. ⏭️ Build data providers for Phase 1 (yfinance)
3. ⏭️ Create data storage schema (Parquet files)
4. ⏭️ Implement feature computation pipeline
5. ⏭️ Set up automated daily/weekly updates

**Recommendation**: Start with Phase 1 (all free, easy to obtain, 80% of features)
