## Quant Data Challenge

Prepathon 2026

## Statistical & Strategy Design Exercise

Duration: 7–10 days | Format: Individual or Team submission (max 2 members)

## 0. Overview

You have been given 108 daily files of high-frequency data for a masked equity ("EBX") [Link]. Strictly use Day 1 to Day 85 only for this exercise. The data is sampled once per second from market open (00:00:00 in file-local time) to close (~06:28:59). Note that some days may be shorter. Each file contains: [URL 🔗](https://www.kaggle.com/datasets/hawkwild/ebx-20-days)

- Time — HH:MM:SS

- Price — Traded price of EBX

- PBi_T1 ... PBi_T12 — Price-based features across 12 rolling windows (masked construction)

- VBi_T1 ... VBi_T12 — Volume-based features across 8 rolling windows (masked construction)

- Additional masked feature types.

You are not told exactly how these features are constructed. Figuring out what they plausibly represent—and proving it with evidence rather than guessing—is part of the exercise. For example, column 2 could represent a rolling return. Each correct decoding with proof earns you points.

This challenge has five parts of increasing difficulty, plus a bonus section. Parts 1–3 are compulsory for everyone. Parts 4–5 are intended for the later stages of evaluation, but if you finish the initial parts early, there is no harm in progressing further.

The bonus section is optional and exists purely to reward creativity. It will not be held against anyone who skips it, but it is an excellent way to earn top marks.

You are free to use Python (pandas/numpy/scipy/statsmodels are expected). Anything else—polars, GARCH libraries, ML frameworks—is fair game and encouraged where it strengthens your analysis.

## 1. Data Hygiene & Descriptive Statistics

Goal: Prove you can handle a large, messy, multi-file dataset correctly before doing anything clever with it. (Hint: Before starting, read up on time series stationarity and understand the rationale behind applying techniques to the first difference of prices (returns) rather than raw prices).

## 1. Ingestion & Sanity Checks

- Load files 1 through 85 (exclude days 86 to 108). Report row counts per file, missing timestamps, duplicate timestamps, and any obvious data errors (zero/negative prices, NaNs, flat-lined stretches, or price jumps that look like bad ticks).


- Decide on and justify a cleaning policy (drop vs. forward-fill vs. interpolate). State it explicitly—you will be judged on your reasoning, not just your choice.

## 2. Descriptive Statistics

- Per-day and pooled: Compute the mean, median, standard deviation, skew, and kurtosis of price levels, as well as 1-second, 1-minute, and 5-minute returns.

- Autocorrelation function (ACF) of returns at short lags (1s–60s): Is there evidence of microstructure noise?

- Intraday seasonality: Does volatility or volume follow a U-shape or any other recognizable pattern over the ~6.5-hour session, aggregated across all 85 days?

Deliverable: A clean summary table and a short (½–1 page) write-up of anything surprising in the raw data.

## 2. Distributional & Tail Analysis

Goal: Test the "returns are normal" assumption rigorously and quantify what happens when it breaks. Ensure you account for the assumptions of any model you apply; if you break an assumption, provide your rationale.

## 1. Normality Testing

- Test 1-min and 5-min return distributions (pooled and for a sample of individual days) using at least two of the following: Jarque–Bera, Shapiro–Wilk, Anderson–Darling, or D'Agostino K².

- Report the inferences of each test and whether they were useful. Don't just print a p-value; explain what it means for someone using a Gaussian VaR model on this stock.

## 2. Sigma-Event Analysis

- Assuming a normal distribution fit to the data, compute how many observations should fall outside ±1σ, ±2σ, and ±3σ, and compare this to how many actually do (both pooled and per-day).

- Identify the days with the most 3σ+ events. Do they cluster (volatility clustering / GARCH-type behavior), or are they scattered randomly across the 85 days?

- Estimate tail heaviness independently of the normal assumption (e.g., a Hill estimator for the tail index, or empirical excess kurtosis against 0).

## 3. Rare-Event Catalogue

- List the top 10–20 most extreme 1-minute moves across the dataset. Check if they coincide with a spike in volume-based features as a sanity check.

Deliverable: Distribution plots (histogram vs. fitted normal, QQ-plots), a sigma-event table, and a short discussion of what a naive Gaussian risk model would get wrong here.


## 3. Regime Classification: Mean-Reversion vs. Momentum

Goal: Move beyond "is it normal" to "how does it behave dynamically."

For each of the 85 days independently, classify the day's price behavior using at least two independent quantitative tests. Examples:

- Variance Ratio test: VR(q) significantly < 1 → mean-reverting; > 1 → trending/momentum.

- Hurst exponent (R/S analysis or DFA): H < 0.5 mean-reverting; H > 0.5 trending; H ≈ 0.5 random walk.

- Lag-k autocorrelation: Returns at horizons like 1min, 5min, or 30min with a significance test.

- (Optional) Augmented Dickey-Fuller test on price levels.

## Tasks:

- Report, out of 85 days: How many classify as mean-reverting, momentum, or random walk? Use consistent, pre-declared thresholds.

- Check agreement between your tests. If they disagree, discuss why.

- Identify any sequential patterns (e.g., does a mean-reverting day tend to follow another?). A simple transition-probability table is sufficient.

Deliverable: A per-day regime table (85 rows), a summary breakdown, and a written discussion on how an intraday strategy designer should use this information.

## 4. Feature Forensics on `PB_T*` / `VB_T*`

This section rewards effort and creativity. The features are masked but not arbitrary. Reverse-engineer what they plausibly represent using evidence.

- 1. Naming Hypothesis Testing: PB likely = price-based, VB likely = volume-based. Test this by computing correlations against hand-built rolling features (e.g., rolling volatility, EMA deviations). Which hand-built feature matches each PB_Tn best?

- 2. Predictive Content: Compute the correlation / mutual information of each feature with the forward N-second return. Rank them by predictive power.

- 3. Redundancy: Run a correlation matrix / PCA across all 20 features. How many effective independent dimensions exist?

- 4. (Optional) Granger-causality test: Does any VB_Tn lead PB_Tn or price?

Deliverable: A feature "dossier" (one paragraph + supporting chart per feature or family) stating your best hypothesis, evidence, and confidence level. Honest uncertainty is scored better than confident overreach.


## FOR LEVEL 2 (If Applicable)

## 5. Strategy Design & Backtest

Build and backtest a systematic intraday strategy on EBX using the provided features. Precision of implementation matters as much as the idea.

## Requirements (Execute exactly as specified):

- 1. Signal & Rebalance Rule: Define a clear rule for going long/flat/short and a fixed rebalancing frequency (e.g., 1-min or 5-min bars).

- 2. Position Sizing: State starting capital, sizing rule, and max exposure.

- 3. Costs: Apply a transaction cost assumption (e.g., 5–10 bps). No frictionless backtests.

- 4. No Look-Ahead Bias: Features used at time t must only use information available at or before t. Explicitly check this.

- 5. In-Sample / Out-of-Sample Split: Tune parameters on a subset of the 85 days, and report performance on a held-out subset.

- 6. Performance Metrics: Sharpe ratio, Sortino ratio, max drawdown, hit rate, average trade PnL, turnover.

- 7. Trade Log: A CSV ( trade_log.csv ) with columns: timestamp, side, entry_price, exit_price, quantity, pnl, pnl_pct .

Deliverable: Backtesting script/notebook, trade_log.csv , and a performance report.

## 6. Bonus: Alpha Hunting

For students who want to push further:

- Novel Features: Construct your own feature that predicts short-horizon returns and backtest it as a standalone signal. Check robustness across two halves of the dataset.

- Market-Impact Modeling: Using VB_T* as an order-flow proxy, estimate how price impact scales with trade size, and propose an execution schedule.

- Volatility Profiling: Profile the volatility of various days to find patterns. Plot these days to find the most efficient trades. Identify which feature changes (delta) over that window consistently signaled an event.

- Free Reign: Regime-conditional strategies, ML-based signals, etc. Originality and rigorous validation are key.

## Deliverables Checklist

- [ ] Well-commented code/notebooks for all parts (must run end-to-end without errors)


- [ ] Written report (~6–10 pages total)

- [ ] Regime table for Part 3 (85 rows)

- [ ] Feature dossier for Part 4

- [ ] trade_log.csv for Part 5 (If applying for Level 2)

- [ ] (Optional) GitHub link

## Grading Rubric

| Section | Weight | What's Rewarded |
| --- | --- | --- |
| Part 1 — Data hygiene & stats | 10% | Correctness, thoroughness |
| Part 2 — Distribution & tails | 15% | Correct test usage, interpretation |
| Part 3 — Regime classification | 20% | Rigor, multiple methods, honest thresholds |
| Part 4 — Feature forensics | 20% | Evidence-based reasoning, creativity |
| Part 5 — Strategy & backtest | 25% | Correct implementation, no look-ahead |
| Bonus — Alpha hunting | up to +15% | Originality + validation rigor |

Penalties apply for look-ahead bias, unreproducible code, unsupported claims, and copy-pasted boilerplate.
