# 1-Month Holding Period Strategy Framework

**Category**: Fixed Duration / Medium-Term
**Risk Level**: Low to Medium (long-only, no shorts)
**Data Requirements**: Monthly rebalancing data, fundamental metrics, alternative signals
**Timeframe**: Exactly 1 month (30 days)
**Position Type**: Long stocks or long call options

## Philosophy

The 1-month holding period strikes an optimal balance:
- **Long enough** to avoid overtrading and noise
- **Short enough** to capture momentum and react to new information
- **Tax efficient** (vs. day trading)
- **Practical** for manual validation and monthly reviews

## Risk Management Principles

### Position Types
- **Stocks (long only)**: Primary vehicle, lower risk
- **Call options**: For higher conviction plays, defined risk
- **NO SHORTS**: Avoid unlimited downside risk

### Portfolio Constraints
- Maximum single position: 10-15% of portfolio
- Maximum sector concentration: 30% of portfolio
- Minimum diversification: 8-10 positions
- Cash buffer: 10-20% for opportunities

## General 1-Month Strategy Structure

### Monthly Cycle
1. **Day 0**: Scan for signals, rank opportunities
2. **Day 1**: Enter positions (beginning of month)
3. **Days 2-29**: Monitor, no action unless stop-loss triggered
4. **Day 30**: Exit all positions, calculate returns
5. **Repeat**: New scan and ranking for next month

### Entry Criteria Template
- **Signal strength**: Quantifiable metric above threshold
- **Risk/reward**: Expected upside > 2x expected downside
- **Liquidity**: Minimum daily volume (e.g., $10M+)
- **Quality filter**: Avoid distressed/speculative companies

### Exit Rules
- **Time-based (primary)**: Exit after exactly 30 days
- **Stop-loss**: -15% to -20% (protect against major losses)
- **Early exit**: Only on fundamental breakdown (earnings miss, fraud, etc.)

## Strategy Categories for 1-Month Holding

### 1. Momentum-Based
- Post-earnings announcement drift
- 52-week high breakouts
- Relative strength vs. sector

### 2. Mean Reversion
- Oversold quality stocks
- Sector rotation opportunities
- Event-driven recoveries

### 3. Event-Driven
- Pre-earnings momentum
- Post-analyst upgrade clusters
- Merger arbitrage opportunities

### 4. Alternative Data
- Product launch cycles
- Hiring momentum
- Social sentiment inflections

### 5. Technical + Fundamental Hybrid
- Value stocks breaking resistance
- Quality growth on pullbacks
- Dividend stocks with momentum

## Performance Metrics

### Target Returns (Stocks)
- **Conservative**: 3-5% per month (43-79% annualized)
- **Moderate**: 5-8% per month (79-151% annualized)
- **Aggressive**: 8%+ per month (151%+ annualized)

### Risk Metrics
- **Max drawdown**: -20% portfolio level
- **Win rate target**: 55-65%
- **Risk/reward**: Minimum 1.5:1

### Tracking
- Monthly return by strategy
- Monthly return by position
- Win rate and average win/loss
- Sharpe ratio (rolling 12-month)
- Maximum consecutive losses

## Options Considerations

### When to Use Calls
- High conviction (top 2-3 ideas)
- Stock has high implied volatility compression
- Defined risk is beneficial

### Call Option Selection
- **Strike**: At-the-money (ATM) or slightly OTM
- **Expiration**: 45-60 days (sell after 30)
- **Delta**: 0.50-0.70 range
- **Implied volatility**: Below historical average

### Position Sizing for Options
- Max 20% of portfolio in options
- Individual option position: 2-5% of portfolio
- Account for total loss potential

## Backtesting Requirements

### Historical Testing
- Minimum 3-year backtest
- Test across different market regimes:
  - Bull market (2017-2019)
  - Crash (Mar 2020)
  - Recovery (2020-2021)
  - Rate hikes (2022-2023)

### Metrics to Track
- Month-by-month returns
- Maximum drawdown periods
- Correlation to SPY
- Consistency (% of profitable months)
- Tail risk (worst 5% of months)

## Manual Validation Process

Before coding any 1-month strategy:

1. **Document hypothesis** clearly
2. **Manual backtest** 12 months of data:
   - Pick entry dates manually
   - Calculate 30-day forward returns
   - Track what worked/didn't work
3. **Identify edge cases**:
   - Earnings releases during hold
   - Major news events
   - Low volume periods
4. **Validate risk management**:
   - Would stop-loss have helped?
   - Was diversification sufficient?
5. **Only then code** the strategy

## Next Steps

See individual strategy documents for specific implementations:
- [Earnings Momentum Strategy](earnings_momentum_1m.md)
- [Quality Value Breakout Strategy](quality_value_breakout_1m.md)
- [Alternative Data Hiring Signal](hiring_signal_1m.md)
