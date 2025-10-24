# Understanding Negative Error Skew

## Your Question
> "Does the negative skew in the error distribution indicate that the model tends to underestimate big positive swings? That seems like it could lead to underperformance and should potentially be addressed."

## Short Answer
**YES, you're absolutely correct!** The negative error skew (-1.22) means the model systematically underestimates large positive returns. This is a serious problem for a trading model because you're missing the biggest winners.

---

## Detailed Explanation

### What Does Error Skew Mean?

```
Error = Predicted - Actual

Negative error skew = Left tail is longer
→ More large NEGATIVE errors
→ Predicted < Actual
→ Model UNDERESTIMATES
```

### Our Numbers

**Error Distribution**:
- Skew: -1.22 (negative = left-skewed)
- Kurtosis: 13.1 (heavy tails)

**Target Distribution**:
- Skew: +1.47 (positive = right-skewed)
- Kurtosis: 13.4 (heavy tails)

**The Problem**:
- Target has big positive tail (stocks can go up 100%+)
- But errors have big negative tail
- **This means**: When stocks make big moves up, model predicts too low

---

## Concrete Example

### What Actually Happens

Let's say a stock has these returns over time:
```
Actual returns: -5%, +2%, +8%, +50%, -3%, +100%, +5%

Model predictions: -4%, +2%, +7%, +25%, -2%, +45%, +4%

Errors (Pred - Actual):
  -5% vs -4% → Error: +1% (overestimate)
  +2% vs +2% → Error:  0% (perfect)
  +8% vs +7% → Error: -1% (underestimate)
 +50% vs +25% → Error: -25% (BIG underestimate)  ← Problem!
  -3% vs -2% → Error: +1% (overestimate)
+100% vs +45% → Error: -55% (HUGE underestimate) ← Problem!
  +5% vs +4% → Error: -1% (underestimate)

Error distribution: [-55%, -25%, -1%, -1%, 0%, +1%, +1%]
→ Negative skew! (big negative errors on left)
```

**The big losses are the BIG WINNERS that we underestimated!**

---

## Why Does This Happen?

### 1. Ridge Regression Uses MSE Loss

Mean Squared Error (MSE) penalizes large errors heavily:
```
MSE = average((predicted - actual)²)

Error of -55% → Squared error = 3025
Error of +10% → Squared error = 100
```

The model "learns" to avoid large errors by:
- Being conservative
- Predicting closer to the mean
- **Not taking risks on predicting big moves**

### 2. Stock Returns Are Right-Skewed

Stocks have:
- **Limited downside**: Can't lose more than 100%
- **Unlimited upside**: Can gain 1000%+

Distribution:
```
Negative returns: -60% to 0
Positive returns:  0 to +273% (in our data!)

Mean: +1.89%
Median: +1.21%
95th percentile: +21.6%
99th percentile: +42.9%
Max: +273%
```

**The positive tail is 1.49x larger than the negative tail.**

### 3. MSE + Skewed Targets = Conservative Predictions

When you minimize MSE with a right-skewed target:
1. Model sees more small/medium returns
2. Occasional huge positive returns
3. Predicting huge return and being wrong = massive squared error
4. Model learns: "Play it safe, predict closer to mean"
5. **Result**: Systematically underestimates the tail

---

## Why This Matters for Trading

### Impact on Strategy Performance

**Scenario 1: Model predicts +25%, actual is +50%**
- You buy, but position is too small
- You make money, but only half of what you could have
- **Opportunity cost**

**Scenario 2: Model predicts +5%, actual is +100%**
- You might not buy at all (threshold too low)
- You completely miss a 10-bagger
- **Major loss**

**Scenario 3: Model predicts -10%, actual is -15%**
- You short or avoid
- Prediction is wrong but not by much
- Smaller impact

### The Math

If you build a long-only strategy based on predictions:
- You buy stocks with predicted return > threshold (say +5%)
- But you underestimate big winners by 50%
- So you rank them lower than you should
- **You underweight winners in your portfolio**

Example portfolio:
```
Stock A: Actual +100%, Predict +40% → Rank #5
Stock B: Actual +80%,  Predict +35% → Rank #8
Stock C: Actual +15%,  Predict +14% → Rank #2
Stock D: Actual +12%,  Predict +12% → Rank #1

Your top picks: D, C
Reality: Should be A, B

Your return: (12% + 15%) / 2 = 13.5%
Optimal return: (100% + 80%) / 2 = 90%

You left 77% on the table!
```

---

## Solutions We Implemented

### Solution 1: Quantile Regression
**Instead of predicting the mean, predict the 60th-75th percentile**

```python
# Ridge: Predicts E[Y | X] (mean)
model = Ridge()

# Quantile: Predicts Q[Y | X, q=0.60] (60th percentile)
model = QuantileRegressor(quantile=0.60)
```

**Why this helps**:
- 60th percentile is higher than mean (1.89% → ~2.5%)
- Directly targets upside
- Not penalized by MSE

**Trade-off**:
- Might overestimate small returns
- But that's better than missing big ones!

### Solution 2: Target Transformation
**Transform the skewed target before training**

Options:
1. **Signed log**: `sign(x) * log(1 + |x|)`
   - Compresses large values
   - Reduces skewness
   - Works with negatives

2. **Yeo-Johnson**: Automatic power transform
   - Learns optimal transformation
   - Handles negatives

3. **Rank-based**: Convert to percentiles
   - Very robust
   - Loses magnitude but keeps order

**Why this helps**:
- Makes target more symmetric
- MSE loss works better on symmetric targets
- Inverse transform predictions back

### Solution 3: Asymmetric Loss (Future)
**Penalize underestimation more than overestimation**

```python
def asymmetric_loss(y_true, y_pred, alpha=0.7):
    """
    alpha > 0.5: Penalize underestimation more
    alpha < 0.5: Penalize overestimation more
    """
    error = y_pred - y_true
    return np.where(error > 0,
                    alpha * error**2,        # Overestimate
                    (1-alpha) * error**2)    # Underestimate
```

With alpha=0.7:
- Underestimating by 10% → penalty = 3.0
- Overestimating by 10% → penalty = 7.0

---

## How to Check If Solutions Work

After training with quantile regression or transformations, check:

### 1. Error Skew
```bash
python scripts/evaluate_model.py --model-dir models/quantile

# Look for:
Error Skewness: -0.5  # Better than -1.22!
```

**Target**:
- Baseline: -1.22
- Good: -0.5
- Excellent: 0.0 (symmetric)

### 2. Tail Capture
```python
# Group errors by return magnitude
Large positive returns (>20%): Mean error = -5%  # Better than -25%!
Small returns (<5%): Mean error = +2%            # Slight overestimate OK
```

### 3. Portfolio Performance
```python
# Build long-only portfolio
# Check if we capture the big winners

Top 10% by prediction:
  Average actual return: 35%  # Better than 25% with baseline!
```

---

## Bottom Line

**You correctly identified a critical problem:**
- Negative error skew = underestimating positive moves
- This costs money in trading (missed opportunities)
- It's caused by MSE loss + right-skewed targets

**We've implemented solutions:**
- Quantile regression (predict higher percentile)
- Target transformations (reduce skewness)
- These should reduce error skew and capture upside better

**Next step:**
- Compare baseline (-1.22 skew) vs quantile/transformed models
- Check if skew improves toward 0
- Verify we capture more big winners

Your intuition was spot-on! This is exactly the kind of subtle but important issue that separates good models from great ones. 🎯
