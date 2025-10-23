# Earnings Momentum - 1 Month Strategy

**Category**: Event-Driven / Momentum
**Risk Level**: Medium
**Data Requirements**: Earnings dates, EPS surprises, price reactions
**Timeframe**: 1 month post-earnings
**Position Type**: Long stocks

## Hypothesis

Stocks that significantly beat earnings expectations tend to continue outperforming for 30 days due to:
1. **Post-Earnings Announcement Drift (PEAD)**: Well-documented phenomenon
2. **Analyst estimate revisions**: Take 2-4 weeks to fully reflect
3. **Institutional rebalancing**: Funds adjust positions gradually
4. **Momentum cascades**: Initial beat attracts more buyers

## Strategy Logic

### Entry Criteria
1. **Earnings beat**: EPS surprise ≥ 10% above consensus
2. **Revenue beat**: Revenue surprise ≥ 5% above consensus
3. **Positive guidance**: Management raises forward guidance (qualitative check)
4. **Price reaction**: Stock up 3-8% on earnings day (not too much, not too little)
5. **Quality screen**:
   - Market cap > $1B
   - Average volume > $10M/day
   - Price > $20 (avoid penny stocks)

### Entry Timing
- Enter at market close on earnings day OR next morning
- Maximum 1-2 days after earnings release

### Position Sizing
- Equal weight: 10% per position
- Target: 8-10 positions per month
- Diversify across sectors

### Exit Rules
- **Primary**: Sell exactly 30 days after entry
- **Stop-loss**: Exit if down -15% from entry
- **Early exit**: Exit if company pre-announces negative news

## Manual Testing Plan

### Data Collection
For the past 12 months, collect:
- Companies reporting earnings each month
- EPS and revenue surprises
- Stock price on earnings day and 30 days later
- Any major news during the 30-day period

### Suggested Test Stocks (High Volume)
- Tech: AAPL, MSFT, NVDA, META, GOOGL
- Consumer: AMZN, TGT, WMT, NKE
- Finance: JPM, BAC, V, MA
- Healthcare: UNH, LLY, JNJ

### Manual Backtest Process
1. For each month (last 12 months):
   - Find all companies with 10%+ EPS beat
   - Filter for 3-8% price reaction
   - Check if they meet quality screens
2. Record entry price (close on earnings day)
3. Record exit price (30 days later)
4. Calculate return for each position
5. Calculate portfolio return (equal weighted)

### Expected Results
- **Win rate**: 60-70%
- **Average return**: 5-8% per month
- **Best months**: Post Q4 earnings (Jan-Feb)
- **Worst months**: Low earnings volume months (May, Aug)

## Implementation Notes

### Key Risks
1. **Earnings calendar clustering**: Too many opportunities at once
2. **Sector concentration**: Tech often dominates in bull markets
3. **Macro events**: Fed announcements can override company-specific news
4. **Liquidity**: Ensure can exit full position without slippage

### Data Sources
- **Earnings data**: Yahoo Finance, earnings whisper
- **Analyst estimates**: FactSet, Bloomberg (expensive) or Seeking Alpha (free)
- **Price data**: yfinance Python library

### Enhancements to Test
1. **Analyst upgrade filter**: Only stocks with analyst upgrades post-earnings
2. **Short interest**: Prioritize stocks with high short interest (squeeze potential)
3. **Institutional ownership**: Focus on stocks with increasing institutional ownership
4. **Sector rotation**: Overweight sectors showing relative strength

## Performance Tracking

### Metrics to Calculate
- Monthly return (portfolio level)
- Win rate by sector
- Average winning vs. losing trade
- Correlation to market (SPY)
- Performance in up vs. down months

### Red Flags
- Win rate < 50% (strategy not working)
- Average loss > average win (poor risk/reward)
- High correlation to market (no alpha)
- Concentrated losses in one sector (diversification issue)

## Manual Test Template

```
Month: January 2024
Earnings Season: Q4 2023

Position 1: NVDA
- Entry Date: 1/24/2024
- Entry Price: $145.50
- EPS Beat: 15% (Strong)
- Revenue Beat: 8% (Strong)
- 30-Day Exit: 2/24/2024
- Exit Price: $158.20
- Return: +8.7%
- Notes: AI momentum, analyst upgrades

[Repeat for 8-10 positions]

Portfolio Return: [Average of all positions]
Win Rate: [% of profitable positions]
Best Performer: [Stock with highest return]
Worst Performer: [Stock with lowest return]
Lessons Learned: [Key insights]
```

## Next Steps

1. **Manual test 3 months** of historical data
2. **Identify patterns**: What characteristics predict success?
3. **Refine entry criteria**: Tighten or loosen filters based on results
4. **Code the strategy** once confident in logic
5. **Full backtest**: 3+ years of data

## Status
**Phase**: Documentation complete - ready for manual testing
**Priority**: High (well-studied phenomenon, clear entry/exit)
**Estimated Time**: 2-3 hours for 12-month manual backtest
