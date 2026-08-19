# EBX Quant Challenge — Master Implementation Specification

**Reference:** `architecture.md`  
**Development data:** Days 1–85  
**Final holdout:** Days 86–108  
**Current objective:** Implement the EBX analysis pipeline phase-by-phase, with tests and evidence gates.

---

# 0. Operating Rules

Before doing anything:

1. Read `architecture.md` completely.
2. Inspect the repository.
3. Locate the actual dataset.
4. Determine which day files exist.
5. Inspect existing code before creating new files.
6. Do not assume placeholder paths.

## Non-negotiable rules

- Use real data only.
- Never fabricate results.
- Never modify raw CSV files.
- Never silently drop observations.
- Never silently impute missing values.
- Never use Days 86–108 during development.
- Never cross day boundaries with rolling/lagged calculations.
- Never concatenate all 85 days by default.
- Never implement Part 5 before Parts 1–4 are complete.
- Do not redesign `architecture.md` unless the real data contradicts an architectural decision.

If required data is unavailable, report exactly what is missing.

---

# 1. Global Phase Protocol

For every phase:

```text
READ SPECIFICATION
      ↓
INSPECT EXISTING CODE
      ↓
IMPLEMENT
      ↓
RUN UNIT TESTS
      ↓
RUN ON REAL DATA
      ↓
INSPECT OUTPUTS
      ↓
VERIFY ACCEPTANCE CRITERIA
      ↓
REPORT RESULTS
      ↓
STOP AT PHASE BOUNDARY
```

Do not automatically advance to the next phase.

---

# 2. Master Roadmap

```text
PHASE 0  → Repository audit + environment + data discovery
PHASE 1  → Dataset reconnaissance
PHASE 2  → Ingestion + integrity validation
PHASE 3  → Structural missingness + processed dataset
PHASE 4  → Part 1: Data Hygiene
PHASE 5  → Part 2: Distribution & Tails
PHASE 6  → Part 3: Regime Classification
PHASE 7  → Part 4A: Feature Taxonomy
PHASE 8  → Part 4B: Feature Reverse Engineering
PHASE 9  → Part 4C: Predictive Relevance
PHASE 10 → Part 4D: Redundancy + PCA
PHASE 11 → Integrated evidence review
PHASE 12 → Final report + freeze development conclusions
PHASE 13 → Holdout validation on Days 86–108
```

---

# PHASE 0 — Repository Audit and Environment Setup

## Objective

Understand the existing repository and execution environment.

## Tasks

### 0.1 Repository inspection

Identify:

- existing source files
- existing notebooks
- existing tests
- existing config files
- existing data directories
- existing scripts
- requirements/environment files
- Git status

Do not overwrite existing work blindly.

### 0.2 Dataset discovery

Find the actual:

```text
day1.csv
day2.csv
...
```

Determine:

- dataset directory
- available day IDs
- missing day IDs
- file sizes
- total storage
- whether Days 1–85 exist
- whether Days 86–108 exist

### 0.3 Environment inventory

Record versions of:

- Python
- pandas
- NumPy
- SciPy
- statsmodels
- scikit-learn
- PyArrow
- pytest

Do not install unnecessary packages.

### 0.4 Baseline project structure

Ensure these exist:

```text
architecture.md
implementation.md
README.md
config/
src/
scripts/
tests/
results/
figures/
data/
```

## Outputs

```text
results/phase0/environment.txt
results/phase0/dataset_inventory.csv
results/phase0/repository_audit.txt
```

## Acceptance criteria

- [ ] repository inspected
- [ ] dataset located
- [ ] available days identified
- [ ] environment recorded
- [ ] raw files untouched
- [ ] project baseline established

---

# PHASE 1 — Dataset Reconnaissance

## Objective

Understand the real data before imposing cleaning rules.

Initially inspect:

- Day 1
- Day 21
- Day 81
- several additional sampled days

## Tasks

### 1.1 Basic profiling

For each sampled day measure:

- row count
- column count
- numeric column count
- timestamp range
- timestamp cadence
- price range
- price mean
- price standard deviation
- NaN count
- infinity count
- duplicate timestamps

### 1.2 Feature inventory

Classify columns:

```text
Time
Price
PB
BB
PV
V
VB
unknown
```

Parse:

```text
family
subfamily
suffix
```

### 1.3 Warm-up reconnaissance

Calculate first-valid position for every feature.

Test `_Tn` as a rolling-window hypothesis.

Do not assume the hypothesis is correct.

### 1.4 Window-ladder reconnaissance

Test separately:

PB:

```text
15, 30, 90, 180, 270, 360,
900, 1800, 2700, 4500, 5400, 10800
```

BB/PV/V/VB:

```text
5, 10, 30, 60, 90, 120,
300, 600, 900, 1500, 1800, 3600
```

Do not force individual features to match.

## Outputs

```text
results/phase1/schema_profile.csv
results/phase1/feature_inventory.csv
results/phase1/sample_day_profile.csv
results/phase1/warmup_profile.csv
results/phase1/ladder_reconnaissance.csv
```

## Acceptance criteria

- [ ] feature families identified
- [ ] schema behavior documented
- [ ] warm-up behavior measured
- [ ] Day 1 compared with other sampled days
- [ ] no unsupported feature semantics claimed

---

# PHASE 2 — Ingestion and Integrity Validation

## Objective

Create the production ingestion pipeline.

Process one day at a time:

```text
load
→ validate
→ analyze
→ write results
→ release memory
→ next day
```

## 2.1 Dataset discovery

Implement:

```python
discover_days(input_dir)
```

It must:

- discover day files
- parse numeric IDs
- reject malformed names
- detect duplicate day IDs
- detect missing expected days

## 2.2 Loader

Implement:

```python
load_day(day_id)
iter_days()
```

Each loaded day must provide:

```text
day_id
dataframe
source_path
row_count
columns
```

The loader must:

- parse timestamps
- preserve source values
- preserve NaNs
- avoid silent sorting
- never modify raw files

## 2.3 Schema validation

Compare every day with the reference schema.

Check:

- missing columns
- unexpected columns
- dtype mismatches
- row count
- numeric column count

Column ordering differences alone are not schema failures.

Output:

```text
results/quality/schema_validation.csv
```

## 2.4 Timestamp validation

Check:

- ordering
- duplicate timestamps
- timestamp gaps
- interval distribution
- non-1-second intervals

An example:

```text
12:00:01
12:00:02
12:00:04
```

contains one missing second.

Output:

```text
results/quality/day_integrity.csv
```

Required fields include:

```text
day
rows
start_time
end_time
expected_rows
frequency_mode
non_one_second_intervals
missing_seconds
duplicate_timestamps
out_of_order
status
```

## 2.5 Price validation

Calculate:

```text
price_min
price_max
price_mean
price_std
zero_count
negative_count
nan_count
inf_count
zero_return_count
```

Do not remove volatile days.

Do not label a large valid move as an invalid price merely because it is unusual.

## Phase 2 acceptance criteria

- [ ] all expected days discovered
- [ ] schema checked for every available day
- [ ] timestamps validated
- [ ] prices validated
- [ ] raw files untouched

---

# PHASE 3 — Structural Missingness and Processed Dataset

## Objective

Create a lossless processed representation and determine structural NaN behavior.

## 3.1 Feature parsing

For every feature except `Time` and `Price`, parse:

```text
feature
family
subfamily
suffix
nominal_window_seconds
```

Example:

```text
PB18_T12
→ family=PB
→ subfamily=PB18
→ suffix=T12
→ nominal_window_seconds=10800
```

Also test:

```text
VB3_T7
BB12_T4
PV1_T2
V5_T10
```

## 3.2 Missingness metrics

Calculate:

```text
first_valid_index
first_valid_timestamp
last_valid_index
leading_nan_count
internal_nan_count
trailing_nan_count
total_nan_count
missing_fraction
actual_warmup_seconds
```

## 3.3 Classification

Distinguish:

```text
structural leading NaNs
unexpected internal NaNs
trailing NaNs
all-NaN features
no-missingness features
```

Do not impute.

For this phase:

```text
unexpected NaN → FLAG ONLY
```

## 3.4 Cross-day warm-up

For every feature across Days 1–85 calculate:

```text
days_present
mean_warmup_sec
median_warmup_sec
std_warmup_sec
min_warmup_sec
max_warmup_sec
days_matching_nominal
days_deviating
internal_nan_days
stability_class
```

Output:

```text
results/missingness/cross_day_warmup.csv
```

## 3.5 Window-ladder reconstruction

Empirically validate:

### PB

```text
15, 30, 90, 180, 270, 360,
900, 1800, 2700, 4500, 5400, 10800
```

### BB/PV/V/VB

```text
5, 10, 30, 60, 90, 120,
300, 600, 900, 1500, 1800, 3600
```

Important:

```text
nominal window ≠ guaranteed actual warm-up
```

Preserve exceptions.

Output:

```text
results/missingness/window_ladder_validation.csv
```

## 3.6 Validity masks

Create:

```text
dayN_validity_mask.parquet
```

with a boolean validity column per feature.

Do not modify the underlying values.

## 3.7 Parquet conversion

Create:

```text
data/processed/dayN.parquet
```

Verify:

- rows preserved
- columns preserved
- values preserved
- NaNs preserved
- timestamps preserved

Perform a round-trip check.

## Phase 3 acceptance criteria

- [ ] feature parser complete
- [ ] missingness quantified
- [ ] structural/internal/trailing NaNs separated
- [ ] cross-day warm-up quantified
- [ ] all five family ladders evaluated
- [ ] validity masks generated
- [ ] Parquet generated
- [ ] round-trip verification passes
- [ ] tests pass

**Do not begin Part 1 until this phase passes.**

---

# PHASE 4 — PART 1: DATA HYGIENE

## Objective

Produce formal data-quality analysis for Days 1–85.

## 4.1 Cleaning policy

Use Phase 3 results before deciding treatment.

Do not invent an imputation strategy prematurely.

Explicitly distinguish:

- structural NaNs
- unexpected feature NaNs
- invalid prices
- timestamp errors
- isolated gaps
- longer gaps

## 4.2 Returns

Calculate:

```text
simple returns
log returns
```

at relevant horizons such as:

```text
1s
1m
5m
```

All lagged calculations must remain within a day.

## 4.3 Descriptive statistics

For price and returns:

- count
- mean
- median
- std
- min
- max
- skewness
- kurtosis
- quantiles

Report:

```text
per-day
pooled
```

## 4.4 ACF

Compute short-lag return ACF.

Never cross day boundaries.

## 4.5 Intraday seasonality

Analyze:

- volatility seasonality
- volume seasonality where a valid volume-like feature has been established

## Outputs

Write to:

```text
results/quality/
results/diagnostics/
figures/part1/
```

## Acceptance criteria

- [ ] cleaning policy justified
- [ ] returns correct
- [ ] descriptive statistics complete
- [ ] ACF complete
- [ ] seasonality complete
- [ ] no holdout data used

---

# PHASE 5 — PART 2: DISTRIBUTION & TAILS

## Objective

Characterize normality and tail behavior.

## 5.1 Normality

Use:

- Jarque-Bera
- Shapiro-Wilk or Anderson-Darling

Do not rely on p-values alone.

Also report:

- skewness
- excess kurtosis
- QQ plots

## 5.2 Sigma events

Compare empirical:

```text
|r| > 1σ
|r| > 2σ
|r| > 3σ
```

against Gaussian expectations.

Report theoretical probability, empirical probability and ratio.

## 5.3 Tail estimation

Where appropriate, use:

- Hill estimator
- QQ plots
- empirical tail quantiles

Check assumptions before using Hill.

## 5.4 Extreme events

Catalogue major moves with:

```text
day
timestamp
return
price_before
price_after
rolling_volatility
relevant_volume_features
```

Investigate volume behavior around extreme events.

## Acceptance criteria

- [ ] normality tests complete
- [ ] practical significance discussed
- [ ] sigma events quantified
- [ ] tail behavior quantified
- [ ] extreme events catalogued
- [ ] no holdout leakage

---

# PHASE 6 — PART 3: REGIME CLASSIFICATION

## Objective

Classify each of Days 1–85.

Candidate regimes:

```text
mean-reverting
momentum / persistent
random-walk / inconclusive
```

## 6.1 Tests

Use at least two independent methods from:

- Variance Ratio
- Hurst exponent
- return autocorrelation
- ADF
- KPSS

## 6.2 Per-day table

Produce:

```text
day
VR
VR_pvalue
Hurst
ACF
ADF
ADF_pvalue
KPSS
KPSS_pvalue
regime
confidence
```

## 6.3 Classification

Define thresholds before inspecting the final distribution.

Do not tune thresholds day-by-day.

Conflicting evidence must result in a documented tie-break or `inconclusive`.

## 6.4 Persistence

Calculate:

- regime counts
- regime proportions
- transition matrix
- persistence probabilities
- average regime durations

## Acceptance criteria

- [ ] 85-day table
- [ ] two or more independent tests
- [ ] conflicts documented
- [ ] transition analysis complete
- [ ] thresholds frozen
- [ ] day boundaries respected

---

# PHASE 7 — PART 4A: FEATURE TAXONOMY

## Objective

Create a forensic inventory for all masked features.

For every feature record:

```text
feature
family
subfamily
suffix
nominal_window
actual_warmup
scale
variance
missingness
```

Do not treat feature names as proof of semantics.

---

# PHASE 8 — PART 4B: FEATURE REVERSE ENGINEERING

## Objective

Test concrete candidate formulas against masked features.

## 8.1 Candidate library

Implement configurable candidates.

### Price

- rolling mean
- rolling median
- rolling std
- rolling variance
- rolling min/max
- price-minus-mean
- normalized deviation
- z-score
- momentum
- cumulative return
- EMA
- distance from rolling high/low

### Return

- rolling return mean
- realized variance
- realized volatility
- absolute-return mean
- downside volatility
- upside volatility

### Volume

- rolling volume mean
- rolling volume std
- volume z-score
- volume change
- price-volume covariance
- imbalance proxies

Use the feature's inferred window where appropriate.

## 8.2 Scoring

Use:

- Pearson
- Spearman
- normalized RMSE
- first-difference correlation
- sign agreement
- lagged correlation where relevant

Evidence tiers:

```text
strong evidence
moderate evidence
weak evidence
no convincing match
```

Never identify a feature from one correlation metric alone.

---

# PHASE 9 — PART 4C: PREDICTIVE RELEVANCE

## Objective

Measure relationship with future returns without look-ahead.

Use horizons:

```text
1s
5s
30s
60s
300s
```

Enforce:

```text
feature(t) → return(t+h)
```

The feature may not use information from `t+1` onward.

For each feature/horizon calculate per-day:

- Pearson IC
- Spearman IC

Aggregate:

```text
mean_IC
IC_std
pct_same_sign
```

Use FDR correction for multiple hypothesis testing.

Freeze FDR alpha before final interpretation.

---

# PHASE 10 — PART 4D: REDUNDANCY + PCA

## Objective

Estimate how many independent dimensions exist in the masked feature panel.

Use:

- Pearson correlation
- Spearman correlation
- clustering where helpful
- PCA

Account for:

- NaNs
- unequal validity windows
- day-to-day scale differences

Use per-day z-scoring where appropriate.

Report components explaining:

```text
50%
80%
90%
```

of variance.

Compare pooled and per-day PCA stability when useful.

---

# PHASE 11 — INTEGRATED EVIDENCE REVIEW

Review Parts 1–4 together.

Check:

- whether window conclusions agree with warm-up evidence
- whether feature hypotheses are stable across days
- whether regime results are consistent with volatility behavior
- whether extreme moves align with relevant feature behavior
- whether predictive relationships are stable
- whether any conclusion is driven by only a few days

Separate:

```text
observed fact
statistical result
hypothesis
interpretation
```

Never state an inference as a confirmed feature identity unless evidence supports it.

---

# PHASE 12 — Freeze Development Conclusions

Before touching Days 86–108, freeze:

- feature hypotheses
- candidate formulas
- regime thresholds
- FDR alpha
- analysis parameters
- evidence-tier criteria
- selected plots/tables
- final development conclusions

Create a record such as:

```text
results/freeze/development_freeze.json
```

containing configuration hashes and relevant analysis parameters.

No retrospective tuning is allowed after holdout results are seen.

---

# PHASE 13 — FINAL HOLDOUT VALIDATION

## Objective

Test generalization on Days 86–108.

These days were not used during development.

Do not use them to change:

- thresholds
- features
- candidate formulas
- feature hypotheses
- classification logic

## Validate

### Window ladder

Does the inferred ladder persist?

### Feature hypotheses

Do development-period feature/candidate relationships remain?

### Regimes

Do the same metrics remain interpretable?

### Predictive relevance

Do features selected from Days 1–85 retain their relationship with future returns?

## Failure handling

If a development conclusion fails:

1. record the failure
2. analyze its cause
3. do not keep tuning until it passes
4. report the lack of robustness honestly

---

# PHASE 14 — FINAL REPORT

Create:

```text
reports/report.md
```

Structure:

## Executive Summary

Main findings.

## Part 1 — Data Hygiene

- integrity
- cleaning policy
- descriptive statistics
- returns
- ACF
- seasonality

## Part 2 — Distribution & Tails

- normality
- kurtosis
- sigma events
- tails
- extreme moves

## Part 3 — Regimes

- methodology
- 85-day table
- regime distribution
- transitions
- persistence
- disagreements

## Part 4 — Feature Forensics

- taxonomy
- window reconstruction
- hypothesis matches
- predictive relevance
- FDR
- PCA
- independent dimensions

## Holdout

Days 86–108.

## Limitations

State unresolved ambiguities and statistical limitations.

---

# PHASE 15 — Reproducibility

The final project must be reproducible.

Required:

- configuration-driven parameters
- deterministic behavior wherever randomness exists
- tests
- documented commands
- result paths
- cached intermediate artifacts
- README instructions

Core analytical logic must remain in `src/`.

Notebooks must be presentation/orchestration only.

---

# Global Leakage Rules

## Rule 1 — Day boundaries

No rolling, lagged, ACF or future-return calculation crosses from one day into another.

## Rule 2 — Raw data

Never overwrite raw CSVs.

## Rule 3 — Missingness

Structural NaNs are preserved.

## Rule 4 — Holdout

Days 86–108 remain untouched until the development freeze.

## Rule 5 — Statistical evidence

Correlation is evidence, not proof.

Statistical significance is not automatically practical significance.

## Rule 6 — Multiple testing

Large feature searches require FDR control.

## Rule 7 — Transparency

Dropped days, excluded features, thresholds and transformations must be logged.

## Rule 8 — Reproducibility

All results must be reproducible from source data + configuration + code.

---

# Final Definition of Done

The complete project is finished only when:

- [ ] Phase 0 passed
- [ ] Phase 1 passed
- [ ] Phase 2 passed
- [ ] Phase 3 passed
- [ ] Part 1 passed
- [ ] Part 2 passed
- [ ] Part 3 passed
- [ ] Part 4A passed
- [ ] Part 4B passed
- [ ] Part 4C passed
- [ ] Part 4D passed
- [ ] integrated evidence review passed
- [ ] development freeze created
- [ ] Days 86–108 holdout validation completed
- [ ] final report generated
- [ ] tests pass
- [ ] raw data remains unchanged
- [ ] no fabricated results exist

---

# Codex Execution Protocol

Whenever Codex is invoked:

1. Read `architecture.md`.
2. Read `implementation.md`.
3. Inspect repository state.
4. Determine the current phase.
5. Implement only that phase.
6. Run tests.
7. Run against real data when available.
8. Inspect outputs.
9. Verify acceptance criteria.
10. Report exact results.
11. Stop at the phase boundary.

Do not automatically continue into the next phase.
