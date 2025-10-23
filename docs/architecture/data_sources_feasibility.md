# Data Sources - Feasibility Analysis

## Overview

Analysis of data availability for building regression models to predict 30-day relative returns. Focus on **free/affordable data sources** suitable for individual traders.

## Data Categories & Sources

### 1. Price & Volume Data (HIGH PRIORITY)

#### ✅ yfinance (FREE)
**What**: Historical OHLCV data for stocks
**Coverage**:
- All US stocks
- Daily data back to IPO
- Intraday available (limited history)

**Features Available**:
- Open, High, Low, Close, Volume
- Adjusted prices (split/dividend adjusted)
- Dividends and stock splits

**Pros**:
- Free and unlimited
- Easy Python API
- Widely used and maintained

**Cons**:
- Yahoo can change/break API
- Occasional data gaps
- Rate limiting on bulk downloads

**Code Example**:
```python
import yfinance as yf

# Single stock
aapl = yf.Ticker("AAPL")
hist = aapl.history(period="5y")  # Get 5 years of data

# Multiple stocks
tickers = yf.download(["AAPL", "MSFT", "GOOGL"], start="2020-01-01")
```

**Cost**: FREE
**Feasibility**: ✅ EXCELLENT

---

#### ✅ pandas-datareader (FREE)
**What**: Aggregates multiple free data sources
**Sources Included**:
- Yahoo Finance
- FRED (Economic data)
- World Bank
- OECD

**Pros**:
- Multiple sources in one API
- Economic indicators (GDP, inflation, etc.)
- Free

**Cons**:
- Some sources deprecated over time
- Less reliable than paid sources

**Code Example**:
```python
from pandas_datareader import data as web
import datetime

start = datetime.datetime(2020, 1, 1)
end = datetime.datetime(2024, 1, 1)

# Stock prices
df = web.DataReader("AAPL", "yahoo", start, end)

# Economic data from FRED
gdp = web.DataReader("GDP", "fred", start, end)
```

**Cost**: FREE
**Feasibility**: ✅ EXCELLENT

---

### 2. Fundamental Data (MEDIUM PRIORITY)

#### ⚠️ Financial Modeling Prep (FREEMIUM)
**What**: Fundamentals, earnings, analyst estimates
**Free Tier**:
- 250 API calls/day
- All endpoints available
- Historical data limited

**Data Available**:
- Income statements
- Balance sheets
- Cash flow statements
- Financial ratios
- Earnings calendar
- Analyst estimates (!!!)

**Pros**:
- Free tier is generous
- Well-documented API
- Includes estimates

**Cons**:
- 250 calls/day limits bulk downloads
- Need to cache aggressively

**Code Example**:
```python
import requests

API_KEY = "your_key"
symbol = "AAPL"

# Earnings calendar
url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{symbol}?apikey={API_KEY}"
earnings = requests.get(url).json()

# Financial ratios
url = f"https://financialmodelingprep.com/api/v3/ratios/{symbol}?apikey={API_KEY}"
ratios = requests.get(url).json()
```

**Cost**: FREE (250 calls/day) or $14-29/month
**Feasibility**: ✅ GOOD (with caching)

---

#### ⚠️ Alpha Vantage (FREEMIUM)
**What**: Stock data, fundamentals, technical indicators
**Free Tier**:
- 25 API calls/day (very limited!)
- 500 calls/day for $49/month

**Data Available**:
- Real-time quotes
- Fundamentals (annual/quarterly)
- Technical indicators (pre-calculated)
- Earnings data

**Pros**:
- Clean API
- Pre-calculated technical indicators

**Cons**:
- 25 calls/day is very limiting
- Need paid tier for serious use

**Cost**: FREE (25/day) or $49/month (500/day)
**Feasibility**: ⚠️ LIMITED (free tier too restrictive)

---

#### ✅ Yahoo Finance Scraping (FREE)
**What**: Scrape Yahoo Finance pages directly
**Data Available**:
- P/E, P/B, EV/EBITDA ratios
- Analyst estimates
- Target prices
- Earnings dates
- Key statistics

**Pros**:
- Free and comprehensive
- More data than yfinance API
- Analyst estimates available

**Cons**:
- Scraping (can break)
- Need to be respectful (rate limiting)
- Legal gray area

**Code Example**:
```python
import yfinance as yf

ticker = yf.Ticker("AAPL")

# Get fundamentals
info = ticker.info  # Dict with P/E, market cap, etc.
earnings = ticker.earnings  # Annual earnings
financials = ticker.financials  # Income statement
balance_sheet = ticker.balance_sheet
cash_flow = ticker.cashflow

# Analyst info
recommendations = ticker.recommendations
analyst_price_targets = ticker.analyst_price_target
```

**Cost**: FREE
**Feasibility**: ✅ EXCELLENT

---

### 3. Earnings & Estimates (HIGH PRIORITY)

#### ✅ Earnings Whisper (FREE)
**What**: Earnings announcements, whisper numbers
**Access**: Public website (scraping needed)
**Data**:
- Earnings dates
- Consensus estimates
- Whisper numbers
- Historical surprises

**Pros**:
- Free
- Includes "whisper" numbers (crowd expectations)
- Historical data

**Cons**:
- Requires scraping
- No official API

**URL**: https://www.earningswhispers.com/calendar

**Feasibility**: ✅ GOOD (requires scraping)

---

#### ✅ Seeking Alpha (FREE - Web Scraping)
**What**: Earnings data, analyst revisions
**Access**: Public pages (scraping)
**Data**:
- Earnings transcripts
- Analyst ratings
- Estimate revisions
- News sentiment

**Pros**:
- Comprehensive analyst data
- Free to scrape

**Cons**:
- Requires scraping
- May violate ToS (check carefully)
- Can break if site changes

**Feasibility**: ⚠️ MODERATE (legal concerns)

---

### 4. Insider Trading Data (HIGH PRIORITY)

#### ✅ SEC EDGAR (FREE - Official)
**What**: Form 4 filings (insider transactions)
**Access**: Official SEC API
**Data**:
- All insider purchases/sales
- Transaction dates and amounts
- Insider names and titles
- Real-time (filed within 2 days)

**Pros**:
- 100% free and legal
- Official source of truth
- Comprehensive

**Cons**:
- XML parsing required
- No rate limits but need to be respectful
- Data format complex

**URL**: https://www.sec.gov/edgar/searchedgar/companysearch.html

**Code Example**:
```python
import requests
from bs4 import BeautifulSoup

# Get recent Form 4 filings for a company
cik = "0000789019"  # Microsoft CIK
url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=4&dateb=&owner=include&count=40"
```

**Cost**: FREE
**Feasibility**: ✅ EXCELLENT

---

#### ✅ OpenInsider (FREE - Scraped)
**What**: Pre-processed insider trading data
**Access**: Public website with screening tools
**Data**:
- Insider clusters (our key signal!)
- Transaction aggregations
- Screening filters
- Historical data

**Pros**:
- Already aggregated (saves work)
- Cluster detection built-in
- Free and easy to use

**Cons**:
- Requires scraping
- No official API
- May have slight delays vs SEC

**URL**: http://openinsider.com/screener

**Feasibility**: ✅ EXCELLENT

---

### 5. Alternative Data - GitHub (MEDIUM PRIORITY)

#### ✅ GitHub API (FREE)
**What**: Repository activity data
**Free Tier**:
- 60 requests/hour (unauthenticated)
- 5,000 requests/hour (authenticated)

**Data Available**:
- Commit counts (by week)
- Contributors (count and changes)
- Stars, forks, watchers
- Issues (opened/closed)
- Pull requests

**Pros**:
- Free and official
- Rich data for tech stocks
- Well-documented API

**Cons**:
- Only useful for companies with OSS presence
- Rate limits (need caching)
- Doesn't show private repo activity

**Code Example**:
```python
import requests

# Get commit activity (52 weeks)
url = "https://api.github.com/repos/mongodb/mongo/stats/commit_activity"
response = requests.get(url)
commit_data = response.json()

# Get contributors
url = "https://api.github.com/repos/mongodb/mongo/contributors"
contributors = requests.get(url).json()
```

**Cost**: FREE
**Feasibility**: ✅ EXCELLENT (for tech stocks)

---

### 6. Analyst Data (LOW PRIORITY - Expensive)

#### ❌ Bloomberg Terminal
**Cost**: $24,000/year
**Feasibility**: ❌ TOO EXPENSIVE

#### ❌ FactSet
**Cost**: $12,000-30,000/year
**Feasibility**: ❌ TOO EXPENSIVE

#### ⚠️ Seeking Alpha API (PAID)
**Cost**: $29.99/month
**Data**: Analyst ratings, estimates
**Feasibility**: ⚠️ MAYBE (if needed)

---

### 7. Stock Metadata (HIGH PRIORITY)

#### ✅ yfinance - Company Info (FREE)
**What**: Company profile and metadata
**Data Available**:
- Market cap, shares outstanding, float
- Sector, industry classification
- Employee count (for some companies)
- Company description
- IPO date (via first trading date)

**Code Example**:
```python
import yfinance as yf

ticker = yf.Ticker("AAPL")
info = ticker.info

# Key metadata
market_cap = info['marketCap']
sector = info['sector']
industry = info['industry']
employees = info.get('fullTimeEmployees', None)
shares_outstanding = info['sharesOutstanding']
float_shares = info.get('floatShares', None)
```

**Pros**:
- Free and comprehensive
- Updated regularly
- Most metadata available

**Cons**:
- Employee count not always available
- Some fields missing for smaller stocks
- IPO date requires workaround

**Cost**: FREE
**Feasibility**: ✅ EXCELLENT

---

#### ✅ Sector ETF Data (FREE)
**What**: Sector performance via ETFs
**Sector ETFs** (via yfinance):
- XLK: Technology
- XLF: Financials
- XLV: Healthcare
- XLE: Energy
- XLY: Consumer Discretionary
- XLP: Consumer Staples
- XLI: Industrials
- XLB: Materials
- XLRE: Real Estate
- XLU: Utilities
- XLC: Communication Services

**Code Example**:
```python
import yfinance as yf

# Get all sector ETFs
sectors = {
    'Technology': 'XLK',
    'Financials': 'XLF',
    'Healthcare': 'XLV',
    # ... etc
}

# Download sector data
sector_data = yf.download(list(sectors.values()), start="2020-01-01")
```

**Pros**:
- Free and accurate
- Easy to calculate sector momentum
- Good proxy for sector performance

**Cons**:
- ETF performance may differ slightly from index
- Need to map stocks to sectors

**Cost**: FREE
**Feasibility**: ✅ EXCELLENT

---

### 8. Market Data (Supporting)

#### ✅ Market Indicators (FREE)
**Source**: Yahoo Finance, FRED
**Data Available**:

**Via Yahoo Finance**:
- ^VIX: Volatility index
- ^SPX: S&P 500
- ^DJI: Dow Jones
- ^IXIC: NASDAQ
- ^TNX: 10-Year Treasury Yield
- GC=F: Gold futures
- CL=F: Oil futures
- DX-Y.NYB: Dollar index

**Via FRED**:
- Treasury yields (all maturities)
- Economic indicators
- Credit spreads
- Inflation data

**Code Example**:
```python
import yfinance as yf
from pandas_datareader import data as web

# Market indicators from Yahoo
vix = yf.Ticker("^VIX").history(period="1y")
spy = yf.Ticker("SPY").history(period="5y")
tnx = yf.Ticker("^TNX").history(period="1y")  # 10Y yield

# Economic data from FRED
start = datetime(2020, 1, 1)
gdp = web.DataReader("GDP", "fred", start)
unemployment = web.DataReader("UNRATE", "fred", start)
```

**Pros**:
- Completely free
- Comprehensive coverage
- Real-time (or near real-time)

**Cons**:
- Need to combine multiple sources
- Some exotic indicators not available

**Cost**: FREE
**Feasibility**: ✅ EXCELLENT

---

### 9. Market Breadth & Sentiment (MEDIUM PRIORITY)

#### ⚠️ NYSE Advance/Decline (LIMITED FREE)
**What**: Market breadth indicators
**Free Sources**:
- FinViz: Shows daily breadth (scraping needed)
- Barchart: Some breadth data free
- StockCharts: Limited free breadth indicators

**Data**:
- Advance/decline ratio
- New highs/new lows
- Up volume / down volume

**Pros**:
- Valuable for market regime detection
- Leading indicator

**Cons**:
- Not all data freely available via API
- May need scraping
- Historical data limited

**Cost**: FREE (with effort) or $20-50/month for clean data
**Feasibility**: ⚠️ MODERATE (scraping required)

---

#### ✅ Put/Call Ratio (FREE)
**Source**: CBOE website (scraping) or via some brokers
**What**: Options sentiment indicator

**Code Example**:
```python
# Put/Call ratio available from CBOE
# http://www.cboe.com/data/historical-options-data/market-statistics

# Or calculate from options data (if available)
# Total put volume / total call volume
```

**Feasibility**: ⚠️ MODERATE (requires scraping or broker API)

---

## Recommended Data Stack (MVP)

### Tier 1: Free & Essential
1. **yfinance**: Price, volume, fundamentals, stock metadata (sector, market cap, employees)
2. **yfinance**: Sector ETFs (XLK, XLF, etc.) for sector performance
3. **yfinance**: Market indicators (VIX, SPY, TNX, etc.)
4. **OpenInsider**: Insider trading clusters
5. **SEC EDGAR**: Backup for insider data
6. **GitHub API**: Tech stock alternative data
7. **FRED**: Economic indicators (GDP, unemployment, yields)

**Total Cost**: $0/month
**Coverage**: ~85% of features needed (including metadata & sector/market data!)

### Tier 2: Enhanced (Low Cost)
Add if budget allows:
1. **Financial Modeling Prep** ($14/month): Better fundamentals, analyst estimates
2. **Seeking Alpha scraping**: Analyst revisions (legal risk)

**Total Cost**: $14/month
**Coverage**: ~95% of features

### Tier 3: Premium (If Scaling)
1. **Quandl/Nasdaq Data Link** ($50-200/month): Professional-grade data
2. **Polygon.io** ($99/month): Real-time data, better coverage

**Total Cost**: $100-300/month
**Coverage**: 100%

---

## Data Storage Strategy

### Local Storage (Recommended Start)

```
data/
├── price/
│   ├── daily/
│   │   ├── AAPL.parquet
│   │   ├── MSFT.parquet
│   │   └── ...
│   └── metadata.json
├── metadata/                # Stock metadata
│   ├── company_info.parquet # Sector, industry, market cap, employees
│   ├── sector_mapping.parquet
│   └── universe.parquet     # List of tradeable stocks
├── sector/                  # Sector-level data
│   ├── etf_prices/         # XLK, XLF, etc.
│   │   ├── XLK.parquet
│   │   ├── XLF.parquet
│   │   └── ...
│   └── sector_metrics.parquet
├── market/                  # Market-level indicators
│   ├── vix.parquet
│   ├── spy.parquet
│   ├── yields.parquet      # Treasury yields
│   ├── commodities.parquet # Gold, oil
│   └── economic.parquet    # GDP, unemployment from FRED
├── fundamentals/
│   ├── ratios/
│   ├── earnings/
│   └── estimates/
├── insider/
│   ├── transactions/
│   └── clusters/
├── github/
│   ├── commits/
│   └── contributors/
└── cache/
    └── api_responses/
```

**Format**: Parquet files (fast, compressed)
**Update Frequency**:
- Price: Daily (end of day)
- Metadata: Weekly (changes rarely)
- Sector/Market: Daily (end of day)
- Fundamentals: Weekly
- Insider: Daily
- GitHub: Weekly

### Database (If Scaling)

#### SQLite (Local)
- Good for < 100GB data
- No server needed
- Easy setup

#### PostgreSQL (Production)
- Handle larger datasets
- Better query performance
- Support concurrent access

---

## Data Quality Considerations

### Survivorship Bias
**Problem**: Only including currently-listed stocks biases results
**Solution**:
- Include delisted stocks in historical data
- Use datasets that track delistings

### Point-in-Time Data
**Problem**: Using revised data (e.g., restated earnings) causes look-ahead bias
**Solution**:
- Careful timestamping
- Only use data available at prediction time
- For free sources, less of an issue (they don't revise)

### Missing Data
**Problem**: Not all stocks have all data
**Solution**:
- Forward fill (careful!)
- Median imputation by sector
- Create "missing" indicator features

---

## Implementation Priority

### Phase 1: MVP (Week 1-2)
- [x] yfinance for price data
- [ ] Calculate technical features
- [ ] Build simple linear model
- [ ] Backtest on 1 year of data

### Phase 2: Enhanced (Week 3-4)
- [ ] Add OpenInsider data
- [ ] Add yfinance fundamentals
- [ ] Build XGBoost model
- [ ] Backtest on 3 years

### Phase 3: Alternative Data (Week 5-6)
- [ ] GitHub API integration
- [ ] Earnings calendar scraping
- [ ] Enhanced feature engineering
- [ ] Ensemble model

### Phase 4: Production (Week 7-8)
- [ ] Automated data pipeline
- [ ] Database storage
- [ ] Live predictions
- [ ] Monitoring & alerting

---

## Cost Summary

| Tier | Monthly Cost | Data Coverage | Feasibility |
|------|-------------|---------------|-------------|
| Free | $0 | 80% | ✅ Recommended Start |
| Low-Cost | $14 | 95% | ✅ Good Value |
| Premium | $100-300 | 100% | ⚠️ If Scaling |
| Enterprise | $1000+ | 100%+ | ❌ Not Needed |

---

## Conclusion

**Recommendation**: Start with **100% free tier**
- yfinance, OpenInsider, GitHub API, FRED
- Sufficient to build robust model
- Can always add paid sources later
- Validate model works before spending money

**Total MVP Cost**: $0/month
**Feasibility**: ✅ EXCELLENT
**Next Step**: Build data ingestion pipeline
