# GitHub Activity Signal - 1 Month Strategy

**Category**: Alternative Data / Tech-Focused
**Risk Level**: Medium
**Data Requirements**: GitHub API (commit frequency, stars, forks, contributors)
**Timeframe**: 1 month after activity spike
**Position Type**: Long stocks (tech companies only)

## Hypothesis

For public companies with open-source projects, accelerating GitHub activity signals:
1. **Product development velocity**: More commits = faster innovation
2. **Developer mindshare**: Rising stars/forks = growing ecosystem
3. **Leading indicator**: Code changes lead product releases by weeks/months
4. **Talent attraction**: Active repos attract better engineers

This works best for companies where core product IS their open-source project (e.g., MongoDB, Databricks via open-source projects, HashiCorp).

## Strategy Logic

### Universe Selection
Public companies with significant open-source presence:
- **Databases**: SNOW, MDB, ESTC
- **DevOps**: GTLB, PD (PagerDuty)
- **Cloud/Infrastructure**: FSLY, NET, DDOG
- **AI/ML**: PLTR (some OSS), AI-focused companies
- **Other**: RBLX (public SDKs), U (Unity)

(~20-30 companies to monitor)

### Entry Criteria - The "Velocity Signal"

1. **Commit acceleration**:
   - 30-day commit count > 150% of 90-day average
   - Sustained for 2+ consecutive weeks

2. **Quality indicators**:
   - New contributors joining (+3 in past month)
   - Stars/forks growing faster than trend
   - Issues being closed (not accumulating)

3. **Company fundamentals**:
   - Market cap > $1B
   - Not currently in downtrend (-20% from highs)
   - Earnings within 45-60 days (time for catalyst)

4. **Confirmation**:
   - Check commits are substantive (not just docs/typos)
   - Multiple repos showing activity (not just one)
   - Activity across time zones (global team engaged)

### Entry Timing
- Enter when signal confirmed (2 weeks of data)
- Avoid entering right before earnings (wait until after)

### Position Sizing
- Equal weight: 12.5% per position
- Target: 5-8 positions per month
- More concentrated (higher conviction on signal quality)

### Exit Rules
- **Primary**: Sell exactly 30 days after entry
- **Stop-loss**: Exit if down -18% from entry
- **Early exit**: Exit if GitHub activity suddenly drops off

## Manual Testing Plan

### Data Collection Process

#### Step 1: Identify Public Company Repos
For each company in universe:
1. Find main GitHub organization (e.g., mongodb, snowflakedb, elastic)
2. Identify top 3-5 most important repos
3. Note which are core product vs. tools/SDKs

#### Step 2: Collect Historical Data (Free via GitHub API)
```python
# Use GitHub API (free tier: 60 requests/hour)
# For each repo, collect monthly:
- Commit count
- Contributors (new vs. returning)
- Stars growth rate
- Forks growth rate
- Issues opened/closed
```

#### Step 3: Manual Backtest (12 months)
For each month:
1. Calculate commit velocity for each company
2. Identify which companies show 150%+ acceleration
3. Record entry price (when signal triggered)
4. Record exit price (30 days later)
5. Note any earnings/product releases during period

### Suggested Test Period
- **Start**: January 2023 - December 2023
- **Why**: Post-COVID normalization, interesting tech year
- **Expected signals**: 3-8 per month

### Free Data Sources
1. **GitHub API**: https://api.github.com
   - Rate limit: 60/hour (sufficient for manual testing)
   - No auth required for public data

2. **GitHub Archive**: https://www.gharchive.org
   - Historical GitHub events
   - Download by month/day

3. **Built-in GitHub Insights**:
   - Each repo has "Insights" tab
   - Shows commits, contributors over time
   - Can manually count for backtest

### Manual Workflow (No Coding First)
1. Visit each company's GitHub org monthly
2. Count commits in top repos
3. Compare to 90-day average (spreadsheet)
4. When 150%+ acceleration detected, record entry
5. Track stock price for 30 days
6. Calculate returns

## Enhanced Filters (Test Variations)

### Version 1: Basic Activity (Start Here)
- Commit count acceleration only
- 150%+ threshold
- Top 20 tech stocks

### Version 2: Quality Weighted
- Weight by:
  * Lines of code changed (bigger commits)
  * Contributor seniority (new vs. core team)
  * Which repo (main product vs. tools)

### Version 3: Combined with Fundamentals
- Only buy if:
  * Positive earnings last quarter
  * Revenue growth > 20% YoY
  * Gross margins > 60%

### Version 4: Momentum Confirmation
- Stock must also show:
  * Relative strength vs. QQQ
  * Above 50-day moving average
  * Volume increasing

## Signal Examples (Hypothetical)

### Strong Signal:
```
MongoDB (MDB) - March 2023
- Main repo commits: 450 (vs. 90-day avg: 280) = +61%
- Drivers repo commits: 180 (vs. avg: 90) = +100%
- New contributors: 8 (vs. avg: 3)
- Stars: +1,200 in month (vs. avg +400)
- Interpretation: Major feature development, growing community
- Entry: $250
- Catalyst: 30 days later announces Atlas revenue growth
- Exit: $285 (+14%)
```

### Weak Signal (Avoid):
```
Company X - May 2023
- Commits: +200% BUT mostly documentation updates
- Contributors: Same core 5 people
- Stars: Flat
- Interpretation: Maintenance work, not new features
- Skip this signal
```

## Implementation Notes

### GitHub API Specifics

```python
# Example API calls (pseudo-code)
import requests

# Get commit activity (weekly stats)
url = f"https://api.github.com/repos/{org}/{repo}/stats/commit_activity"
response = requests.get(url)
commits_per_week = response.json()

# Get contributors
url = f"https://api.github.com/repos/{org}/{repo}/contributors"
contributors = requests.get(url).json()

# Calculate velocity
recent_30d_commits = sum(last_4_weeks)
prior_90d_commits = sum(prior_12_weeks)
acceleration = recent_30d_commits / (prior_90d_commits / 3)
```

### Data Challenges
- **API rate limits**: Need to cache/space out requests
- **Commit quality**: Need to filter bot commits, doc updates
- **Private repos**: Can't see enterprise customer activity
- **Noise**: Some commits are minor (typos, formatting)

### Competitive Edge
- Very few investors monitor GitHub systematically
- Data is public but requires effort to track
- Leading indicator (changes code before announcing features)
- Particularly powerful for developer-focused products

## Red Flags / False Positives

### Avoid These Patterns:
1. **Commit spam**: Lots of tiny commits (automated)
2. **Documentation only**: No actual code changes
3. **Single contributor**: One person pushing lots of code
4. **Fork activity without commits**: Stars rising but no development
5. **Pre-IPO lockup expiry**: Insiders about to sell

### Macro Conditions:
- Avoid in tech bear markets (signal works but overwhelmed by sector)
- Best in stable/bull markets
- Best in "risk-on" environments

## Manual Test Template

```
Month: June 2023

Position 1: DDOG (Datadog)
- Entry Date: 6/5/2023
- Entry Price: $95.50
- Signal Details:
  * Main repo commits: 380 (vs. avg 220) = +73%
  * Integrations repo: 150 (vs. avg 75) = +100%
  * New contributors: 6
  * Combined acceleration: +165%
- 30-Day Exit: 7/5/2023
- Exit Price: $104.20
- Return: +9.1%
- Notes: Strong activity preceded new integrations announcement
- Verification: Check GitHub manually at github.com/DataDog

[Repeat for 5-8 positions]

Portfolio Return: [Average]
Win Rate: [%]
Comparison to QQQ: [+/- %]
Best Sector: [e.g., "Databases outperformed"]
Signal Strength: [What acceleration % worked best?]
```

## Performance Tracking

### Metrics to Analyze:
- Return by commit acceleration level (150% vs 200%+ vs 300%+)
- Return by company size (large vs mid cap)
- Return by repo type (core product vs SDKs)
- Return by contributor growth
- Timing: Days until stock reacts

### Success Criteria:
- Win rate > 55% (this is riskier/newer signal)
- Average return > 6% per month
- Low correlation to QQQ (finding alpha)
- Works best combined with other signals

## Academic Support

### Research Papers:
- "Developer Activity as a Leading Indicator" (hypothetical - verify)
- "Open Source Contribution and Firm Value" (finance research)

### Known Patterns:
- MongoDB's Atlas releases correlate with repo activity spikes
- GitLab's own product updates visible in commits
- Databricks activity in Apache Spark predicts features

## Next Steps

1. **Manual test 3 months** first:
   - Pick 3 months from 2023
   - Track top 10 companies
   - Calculate commit velocity manually
   - Record returns

2. **Validate signal quality**:
   - Does acceleration predict returns?
   - What threshold works best?
   - Which repos matter most?

3. **Build simple scraper**:
   - Python script to hit GitHub API
   - Calculate velocity automatically
   - Email alerts on signals

4. **Combine with other data**:
   - Add earnings estimates
   - Add relative strength
   - Add insider buying

5. **Live paper trade** 2 months before real money

## Status
**Phase**: Documentation complete - ready for manual testing
**Priority**: Medium-High (novel signal, less competition)
**Estimated Time**: 4-6 hours for initial 3-month backtest
**Unique Angle**: Rarely used by retail, requires coding knowledge to track
**Best For**: Tech-focused portfolio, higher risk tolerance
