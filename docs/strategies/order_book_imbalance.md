# Order Book Imbalance Strategy

**Category**: Market Microstructure
**Risk Level**: Medium
**Data Requirements**: Level 2 order book data (bid/ask quantities at multiple price levels)
**Timeframe**: Intraday (1-60 second holding periods)

## Hypothesis

When there's a significant imbalance between bid and ask volume in the order book, it indicates short-term directional pressure. Large bid volume relative to ask volume suggests buying pressure (bullish), while the opposite suggests selling pressure (bearish).

## Logic

### Entry Conditions
1. Calculate order book imbalance ratio:
   ```
   Imbalance = (Total Bid Volume - Total Ask Volume) / (Total Bid Volume + Total Ask Volume)
   ```
2. Enter LONG when Imbalance > threshold (e.g., 0.3)
3. Enter SHORT when Imbalance < -threshold (e.g., -0.3)

### Position Sizing
- Fixed size per trade initially
- Scale with imbalance magnitude in advanced version

### Exit Conditions
- Time-based: Exit after N seconds (e.g., 30 seconds)
- Reversal: Exit if imbalance reverses beyond neutral
- Profit target: +X basis points
- Stop loss: -Y basis points

### Risk Management
- Maximum position size per symbol
- Maximum total exposure across all positions
- No new positions if existing position open in same symbol

## Manual Testing Plan

### Sample Data Needed
- Order book snapshots for a liquid stock (e.g., AAPL) for 1 trading day
- At least 1-second granularity
- Include at least top 5 price levels on each side

### Manual Test Steps
1. Calculate imbalance for each snapshot
2. Identify entry signals (imbalance > 0.3 or < -0.3)
3. Record entry price
4. Track price movement for next 30 seconds
5. Calculate P&L for each trade
6. Analyze win rate, average profit/loss

### Expected Behavior
- Higher absolute imbalance → stronger price movement
- Strategy should capture short-term momentum
- Win rate likely 50-60% with positive risk/reward ratio

### Edge Cases
1. What happens during market open/close volatility?
2. How does strategy perform around news events?
3. Does imbalance persist or mean-revert quickly?
4. Are there false signals during low volume periods?

## Implementation Notes

### Key Considerations
- Order book data must be real-time and accurate
- Need to handle order book updates efficiently
- Transaction costs critical at this timeframe
- Latency matters - signals decay quickly

### Potential Pitfalls
- Phantom liquidity (orders that disappear when hit)
- Wide spreads reducing profitability
- Overfitting threshold parameters
- Data quality issues

### Performance Expectations
- Sharpe ratio: 1.5-2.5 (if strategy works)
- Win rate: 50-60%
- Average holding period: 10-60 seconds
- Requires high-frequency data and execution

## Backtest Parameters

### Historical Period
- Start with 1-2 weeks of data
- Test across different market regimes

### Universe Selection
- Start with single highly liquid stock (e.g., SPY)
- Expand to top 10-20 liquid stocks

### Transaction Costs
- Assume 0.5-1 basis point per trade (maker/taker fees)
- Include slippage (0.5-1 basis point)

### Variations to Test
1. Different imbalance thresholds (0.2, 0.3, 0.4)
2. Different time horizons (10s, 30s, 60s)
3. Volume-weighted imbalance vs. simple sum
4. Including deeper order book levels (top 10 vs. top 5)
5. Time-of-day filters (avoid open/close)

## Status
**Phase**: Documentation complete - ready for manual testing
**Next Steps**: Obtain sample order book data and manually validate logic
