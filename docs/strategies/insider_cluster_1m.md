# Insider Clustering - 1 Month Strategy

**Category**: Behavioral / Alternative Data
**Risk Level**: Low-Medium
**Data Requirements**: SEC Form 4 filings (insider transactions)
**Timeframe**: 1 month after cluster detected
**Position Type**: Long stocks

## Hypothesis

When multiple insiders (C-suite, directors, 10%+ owners) buy stock within a short timeframe, it signals:
1. **Information asymmetry**: Insiders know something positive market doesn't
2. **Confidence**: Executives putting personal capital at risk
3. **Timing**: Insiders typically buy before positive catalysts
4. **Legal front-running**: Buying before they can't (blackout periods)

This is NOT about single insider purchases, but **clusters** of multiple insiders buying simultaneously.

## Strategy Logic

### Entry Criteria - The "Cluster Signal"
1. **Multiple insiders buying**: ≥3 different insiders within 30 days
2. **Meaningful size**: Each purchase > $50K (shows conviction)
3. **Open market purchases**: Only "P" transactions (not option exercises/grants)
4. **Recent activity**: All purchases within last 10 trading days
5. **Company quality**:
   - Market cap: $500M - $50B (avoid too small/large)
   - Not distressed (no bankruptcy risk)
   - Trading above $10

### What Counts as "Insider"
- CEO, CFO, COO (C-suite executives)
- Directors
- 10%+ beneficial owners
- **NOT** lower-level employees or stock-based comp

### Entry Timing
- Enter within 5 days of detecting cluster
- Use limit orders to avoid chasing

### Position Sizing
- Equal weight: 10% per position
- Target: 5-8 positions per month (this signal is rarer)
- More concentrated than other strategies (higher conviction)

### Exit Rules
- **Primary**: Sell exactly 30 days after entry
- **Stop-loss**: Exit if down -20% from entry
- **Early exit**: Exit if insiders start selling (Form 4 "S" transactions)

## Manual Testing Plan

### Data Sources (Free)
1. **SEC EDGAR**: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=4
2. **OpenInsider**: http://openinsider.com/screener (excellent free tool)
3. **Finviz Insider**: https://finviz.com/insidertrading.ashx

### Manual Test Process

1. **Find clusters** (use OpenInsider):
   - Filter for "Cluster Buys" (3+ insiders)
   - Filter for "Open Market" purchases
   - Filter for last 10 days
   - Filter for purchase size > $50K

2. **For each cluster found**:
   - Record company ticker
   - Count # of insiders buying
   - Total $ amount purchased
   - Entry date (day cluster detected)
   - Entry price

3. **Track 30-day performance**:
   - Exit price after 30 days
   - Any major news during period
   - Did stock have earnings/catalyst?

4. **Calculate metrics**:
   - Return per position
   - Portfolio return (equal weight)
   - Win rate
   - Compare to SPY return same period

### Sample Size
- Test 12 months of historical data
- Expect 5-10 clusters per month
- Total sample: 60-120 positions

### Expected Characteristics
- **Win rate**: 60-70% (insiders have edge)
- **Average return**: 4-8% per month
- **Best performers**: Small/mid caps (more info asymmetry)
- **Timing**: Often precedes earnings or product launches

## Enhanced Filters (Test Variations)

### Version 1: Basic Cluster (Start Here)
- 3+ insiders, $50K+ each, 10-day window

### Version 2: Strong Cluster
- 5+ insiders OR
- $500K+ total purchases OR
- CEO is one of the buyers (higher signal)

### Version 3: Quality Filter Added
- Only companies with positive earnings
- Only companies with institutional ownership 20-70%
- Avoid heavily shorted stocks (>20% short interest)

### Version 4: Sector Focused
- Only tech/healthcare (growth sectors)
- Hypothesis: Insiders have more info advantage in complex businesses

## Red Flags to Avoid

### Don't Buy If:
1. **Recent poor earnings**: Stock down >20% in last quarter
2. **Regulatory issues**: Company under investigation
3. **Distressed**: Debt/equity ratio > 2.0
4. **Illiquid**: Average volume < $1M/day
5. **All insiders buying tiny amounts**: $10K each = no conviction

### Suspicious Patterns:
- Insider buying right before they announce bad news (rare but happens)
- Buying in companies about to dilute shareholders
- Buying in related-party transactions

## Implementation Notes

### Automation Potential
This strategy can be semi-automated:
1. **API**: SEC provides API for Form 4 filings
2. **Python parsing**: Parse XML Form 4s for transaction data
3. **Alert system**: Email/SMS when cluster detected
4. **Manual review**: Still review each cluster before entering

### Data Challenges
- Form 4s filed within 2 days of transaction (slight lag)
- Need to parse XML or scrape data
- Some filings are amended/corrected
- Option exercises vs. open market purchases

### Competitive Advantage
- Most retail investors don't track insider buying systematically
- Institutional investors may be restricted from using insider data
- Clusters are rarer than single purchases (less noise)

## Manual Test Template

```
Month: February 2024

Position 1: CRWD
- Entry Date: 2/5/2024
- Entry Price: $245.00
- Cluster Details:
  * CEO bought $200K
  * 2 Directors bought $100K each
  * Total: 3 insiders, $400K
- 30-Day Exit: 3/5/2024
- Exit Price: $268.50
- Return: +9.6%
- Notes: Preceded strong earnings beat 2 weeks later
- Catalyst: Product launch announced during hold period

[Repeat for 5-8 positions]

Portfolio Return: [Average]
Win Rate: [%]
Comparison to SPY: [+/- %]
Best Signal: [e.g., "CEO + directors combination"]
Lessons: [Key insights about what clusters work best]
```

## Case Studies to Research

### Historical Examples (to manually verify):
1. **NVDA (2022)**: Insiders bought before AI boom
2. **CRWD (multiple times)**: Clusters often before earnings
3. **SQ/SHOP**: Small-cap fintech with insider buying clusters

Look these up on OpenInsider historical data and see 30-day returns.

## Performance Tracking

### Key Metrics:
- Return by cluster size (3 vs 5+ insiders)
- Return by purchase amount ($50K vs $200K+)
- Return by company size (market cap)
- Return by sector
- Return when CEO is involved vs. not

### Success Criteria:
- Win rate > 60%
- Average return > 5% per month
- Outperforms SPY in same periods
- Low correlation to market (alpha generation)

## Next Steps

1. **Manual test**: Pull 3 months of clusters from OpenInsider
2. **Calculate returns**: Track 30-day performance
3. **Identify best signals**: What cluster characteristics predict success?
4. **Refine filters**: Based on manual test results
5. **Build scraper**: Automate cluster detection
6. **Live paper trade**: Test 2-3 months real-time before real money

## Status
**Phase**: Documentation complete - ready for manual testing
**Priority**: High (strong academic support, clear signal, legal)
**Estimated Time**: 3-4 hours for 12-month manual backtest
**Unique Angle**: Focus on clusters (not individual purchases)
