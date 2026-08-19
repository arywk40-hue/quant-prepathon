# HFT Quantitative Research Challenge

## Governing Specification

docs/quant.md

## Submission Contents

### Part 1 — Data Hygiene & Statistics
- Complete pipeline for raw to model-ready validation.
- Missingness heatmaps and statistics (`results/missingness/`).
- Diagnostic profiles for target validity (`results/diagnostics/`).
- Volatility U-shape documentation and structural profiling.

### Part 2 — Distribution & Tails
- Comprehensive distribution tests across price and volume.
- Extreme event analysis and sigma tables (`results/distributions/`).
- QQ-plots and log-probability profiles (`figures/part2/`).

### Part 3 — Regime Classification
- Robust statistical regime classification (volatility, momentum, volume).
- Final 85-row regime identification log (`results/regimes/regime_table.csv`).
- Regime stability and transition analysis.

### Part 4 — Feature Forensics
- Interpretative feature dossier (`reports/feature_semantics_audit.md`).
- Exhaustive predictive ranking by Pearson and Spearman IC (`results/predictive/`).
- PCA and inter-family redundancy clustering (`results/redundancy/`).
- Feature family visual summaries and eligibility funnels (`figures/part4/`).

### Part 5 — Strategy & Backtest
- Fully automated deterministic trading simulation.
- Validated chronological execution enforcing zero look-ahead bias.
- Incorporated transaction cost model (5 bps entry/exit).
- Final `trade_log.csv` and backtest metric summary (`results/ml/backtest_baseline/`).

## Final Report

Link to:

    reports/final_report.pdf

## Data Scope

Development:
70 available days

Unavailable:
Days 65–79

Reserved holdout:
Days 86–108

No new holdout analysis was performed.

## Reproducibility

The source code (`src/`) and required scripts (`scripts/`) strictly execute the logic required for Parts 1–5. The Part 5 strategy isolates the required Ridge prediction generation (`scripts/ml/baseline_model_training.py`, `temporal_robustness_validation.py`, `strategy_backtest.py`) to reproduce the signal and simulate the explicit rules exactly. All included findings in the final report derive directly from these audited components.
