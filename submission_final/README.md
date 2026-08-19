# HFT Quantitative Research Challenge

## Governing Specification

`docs/quant.md`

## Submission Contents

### Part 1 — Data Hygiene & Statistics
- Data-quality checks, missingness analysis, diagnostics, and validation of the available development data.
- Missingness heatmaps and statistics (`results/missingness/`).
- Diagnostic profiles for target validity (`results/diagnostics/`).
- Volatility U-shape documentation and structural profiling.

### Part 2 — Distribution & Tails
- Distribution tests across price and volume.
- Extreme event analysis and sigma tables (`results/distributions/`).
- QQ-plots and log-probability profiles (`figures/part2/`).

### Part 3 — Regime Classification
- Statistical regime classification using the specified return and dependence diagnostics.
- Final 85-row regime identification log (`results/regimes/regime_table.csv`).
- Regime stability and transition analysis.

### Part 4 — Feature Forensics
- Interpretative feature dossier (`reports/feature_semantics_audit.md`).
- Predictive ranking by Pearson and Spearman IC (`results/predictive/`).
- PCA and inter-family redundancy analysis (`results/redundancy/`).
- Feature family visual summaries and eligibility funnels (`figures/part4/`).

### Part 5 — Strategy & Backtest
- Deterministic trading simulation.
- Chronological execution with explicit look-ahead checks.
- Transaction cost model of 5 bps per entry and exit.
- Final `trade_log.csv` and backtest metric summary (`results/ml/backtest_baseline/`).

## Final Report

See:

`reports/final_report.pdf`

## Data Scope

Development:
70 available days

Unavailable:
Days 65–79

Reserved holdout:
Days 86–108

No new holdout analysis was performed.

## Reproducibility

The source code (`src/`) and scripts (`scripts/`) contain the logic used for Parts 1–5. The Part 5 Ridge prediction and backtest workflow is documented in the corresponding scripts under `scripts/ml/`. The findings in the final report come from these audited components.
