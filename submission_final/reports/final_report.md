# EBX Quantitative Analysis

## 1. Executive Summary

The supplied dataset specifies 85 development days, but only 70 were available:
Days 1–64 and 80–85. Days 65–79 are explicit missing days and were never
fabricated. Days 86–108 are reserved holdout data and are not included in this
development package. Days 109–123 remain out of scope.

Parts 1–4 found heavy-tailed, non-normal returns; persistent/inconclusive
intraday regimes under the frozen methodology; substantial nominal-vs-actual
window deviations; strong redundancy; and a low-dimensional feature panel.
The frozen predictive screen contained 543 feature-horizon rows, of which 535
strongly generalized, 7 partially generalized, and 1 did not generalize.

The development package does not claim holdout generalization. It reports
available-day statistical evidence, frozen model comparisons, and one negative
after-costs baseline backtest.

## 2. Dataset and Methodology

Each day is processed independently at one-second frequency. The schema contains
`Time`, `Price`, and 691 masked features across PB, BB, PV, V, and VB families.
Two nominal ladders are preserved as hypotheses: PB uses 15–10800 seconds and
BB/PV/V/VB use 5–3600 seconds.

Structural NaNs are preserved. Unexpected internal/trailing missingness is
flagged, not imputed. Returns, rolling calculations, ACF, regime tests, and
forward-return alignment are day-local. Holdout Days 86–108 remain reserved
and are not used for new development analysis.

## 3. Data Hygiene

The 70 available development days were converted to lossless per-day Parquet
with validity masks. Holdout Days 86–108 are outside the scope of this
development report and are not analyzed here.

Development taxonomy identified 594/691 feature-level nominal-vs-actual window
deviations. The holdout comparison preserved these exceptions and found median
warm-up agreement for all 691 features.

## 4. Distribution and Tails

Development pooled 1-minute and 5-minute returns rejected normality under both
Jarque–Bera and Anderson–Darling. Excess kurtosis was 16.2943 at 1m and
11.4267 at 5m. The >3σ empirical/theoretical ratios were 6.2054 and 5.9569.

No holdout distributional conclusion is claimed in this development package.
The available-day results reject normality and show heavy tails; exact
generalization to Days 86–108 is reserved for a separately authorized phase.

## 5. Regime Classification

The frozen method uses variance ratio q=5, rescaled-range Hurst, lag-1 return
ACF, and ADF on log price, with predeclared thresholds. Development classified
61/70 days as momentum/persistent and 9/70 as random-walk/inconclusive; no day
was classified mean-reverting.

No holdout regime classification is claimed. The available-day regime table
contains 61 momentum/persistent and 9 random-walk/inconclusive classifications,
with Days 65–79 explicitly marked missing.

## 6. Feature Forensics

The taxonomy covers all 691 features. Candidate formulas were evaluated using
multiple metrics: Pearson, Spearman, affine-normalized RMSE, first-difference
correlation, sign agreement, and lagged correlation. Volume candidates remained
explicitly unavailable because no validated raw-volume semantic source was
present.

The 407 frozen best-match hypotheses are reported as development hypotheses. A
candidate match is a feature identity hypothesis, not a confirmed identity.
Correlation with a candidate formula alone is not sufficient evidence, and no
holdout identity-generalization claim is made here.

## 7. Predictive Relevance

The development screen used exact `feature(t) → return(t+h)` alignment at 1s,
5s, 30s, 60s, and 300s, with Benjamini–Hochberg FDR alpha 0.05. It retained
543 frozen feature-horizon rows using the declared development screen.

This is frozen statistical predictive relevance on the available development
scope, not ML model performance, causal evidence, or a trading result. No
holdout significance re-selection was performed.

## 8. PCA and Redundancy

Development pooled PCA reached 50%, 80%, and 90% variance at 3, 15, and 35
components. Holdout reached those levels at 3, 14, and 34 components. Median
per-day counts remained 3, 11, and 22.

Development median absolute Pearson/Spearman redundancy was 0.1319/0.1582;
holdout was 0.1227/0.1414. The pooled first-component variance changed from
0.2468 to 0.2021. The broad low-dimensional and redundant structure therefore
generalized, with some magnitude variation.

## 9. ML Development and Part 5

### 9.1 Part 4 figure completion

The Part 4 visualization suite is complete under `figures/part4/` and is
generated from frozen results only. It includes feature-IC distributions,
family summaries, top/bottom features, the eligibility funnel, and the
per-family redundancy heatmap. No figure changes the underlying conclusions.

### 9.2 Feature semantics and missingness

The feature taxonomy treats PB, VB, BB, PV, and V as distinct families. For a
fixed indicator/type `i`, changing the suffix `j` in `PB{i}_T{j}` denotes a
window variant within that subfamily; it is not treated as a new semantic
identity. Early rolling-window NaNs are structural warm-up values. They are
preserved and classified upstream, while the ML dataset builder applies
complete-case filtering downstream without imputation. Feature formulas remain
hypotheses unless independently supported by evidence.

### 9.3 ML Phase 0 and Ridge baseline

The model-ready pipeline consumed the 70 available development days only
(Days 1–64 and 80–85), retaining Days 65–79 as explicit missing days. It used
the frozen 197-feature screen, exact within-day 300-second future returns, and
training-only preprocessing with no imputation or clipping. The deterministic
Ridge baseline used `alpha=1.0`, trained on 745,700 rows from Days 1–64, and
validated on 73,452 rows from Days 80–85. Its pooled Pearson IC was 0.071416,
Spearman IC 0.060726, directional accuracy 0.508781, and R² -0.023186.
These are predictive-association results, not evidence of economic utility.

### 9.4 Training-only feature selection

The controlled selection experiment refit the existing screen on training days
only across 691 candidate features and 3,455 feature-horizon hypotheses. It
selected 198 300-second features, retaining all 197 frozen features plus one
additional feature. Experiment B produced Pearson IC 0.070712, Spearman IC
0.055737, directional accuracy 0.507651, and R² -0.023414 on the same six-day
validation period. The small decrease versus the frozen-screen baseline was
reported without tuning or validation-driven compensation.

### 9.5 Temporal robustness

The three pre-specified chronological windows produced Pearson IC values of
0.038912 (W1, validation Days 45–54), 0.031759 (W2, Days 55–64), and 0.070712
(W3, Days 80–85). Pooled R² was negative in all three windows. Training-only
selected feature counts were 195, 194, and 198, with pairwise Jaccard overlap
above 0.979. This supports some temporal association but not stable economic
performance.

### 9.6 Day-84 forensic analysis

Day 84 is retained in the primary W3 result. Removing it is a post-hoc
sensitivity diagnostic only: W3 Pearson IC falls from 0.070712 to 0.011037,
and Day 84 accounts for approximately 51.5% of pooled W3 IC while representing
16.7% of observations. The forensic report attributes this influence to
outlier-driven leverage on extreme target returns. Under the frozen Part 3
classification rule, Day 84 is classified as random-walk / inconclusive with
low confidence (conflicting persistent VR/Hurst/ACF vs. mean-reverting ADF
evidence); its regime classification does not independently validate the ML
result. No feature, target, threshold, or model was changed.

### 9.7 Part 5 baseline strategy

Part 5 tests one pre-specified economic-utility implementation of the existing
Ridge signal. Prediction sign determines long/short/flat direction, exposure is
fixed at unit notional, one position is held at a time, entry uses the first raw
observation at or after the signal, and exit uses the exact same-day timestamp
300 seconds after entry. The documented cost assumption is 5 bps at entry plus
5 bps at exit, with zero fee; it is not an empirically measured spread model.

| Window | Trades | Gross P&L | Costs | Net P&L | Sharpe |
|---|---:|---:|---:|---:|---:|
| W1 | 410 | 0.008677675 | 0.410000 | -0.401322325 | -186.162764 |
| W2 | 285 | 0.014409138 | 0.285000 | -0.270590862 | -32.994427 |
| W3 | 246 | 0.021281444 | 0.246000 | -0.224718556 | -106.280913 |
| Pooled development | 941 | 0.044368257 | 0.941000 | -0.896631743 | -52.575858 |

The pooled result is descriptive because validation days are reused across the
three temporal experiments. The positive gross P&L did not survive the
documented transaction-cost assumption. W3 including Day 84 remains the
primary result; excluding Day 84 is reported only as a sensitivity, with net
P&L -0.197612194.

### 9.8 Part 5 audit and development conclusion

The independent audit reconciled every trade's cost, gross P&L, net P&L,
timestamp alignment, 300-second holding period, day-local execution, and
position overlap. Stable research-bearing outputs were reproducible; only the
timestamped run manifest changes on rerun. The audit found no audited
look-ahead, and manifests record `holdout_days_loaded: []`.

The development evidence therefore distinguishes modest predictive association
from economic utility. The development baseline does not demonstrate economic
viability after costs, profitability, alpha, or production readiness. No final
production model, strategy optimization, or additional backtest is included.

### 9.9 Final model comparison

The controlled development model comparison is:

| Model | Pearson IC | Directional accuracy | Net P&L |
|---|---:|---:|---:|
| Ridge | 0.0707122893 | 0.5076512552 | -0.2247185557 |
| Elastic Net | 0.0336523173 | 0.5082094429 | -0.2431015522 |
| LightGBM | 0.0433187087 | 0.5213608887 | -0.2445651284 |

Ridge remains the strongest tested model by Pearson IC and net P&L. Elastic
Net and LightGBM do not improve Ridge. LightGBM used one fixed W3
configuration with 198 existing training-only selected features; no tuning or
additional model search was performed. The comparison is development-only and
does not establish profitability, alpha, production readiness, or holdout
generalization.

## 10. Out-of-Sample Validation

Days 86–108 are reserved holdout days and were not accessed, analyzed, or used for model selection, validation, or performance claims in the development experiments.

## 11. Limitations

- Development coverage is 70 days, not the specified 85; Days 65–79 remain missing.
- The dataset is masked and feature semantics are inferred hypotheses, not confirmed identities.
- Exact tail-index estimates are descriptive and unstable by construction.
- Some feature hypotheses remain uncertain because the original feature
  generator is unavailable.
- The ML experiments and Part 5 are development-only baselines, not production
  models or trading recommendations.
- Statistical persistence is not proof of economic value, causal effect, or deployability.

## Reproducibility

See [reports/reproducibility.md](reproducibility.md) and the
[artifact index](artifact_index.md). The development freeze must remain
unchanged for these conclusions to retain their recorded scope. Holdout Days
86–108 are reserved and are not part of the development reproducibility run.
