# Feature Semantics Audit

## Executive summary

The current repository implementation is already consistent with the authoritative clarification:

- `PB{i}_T{j}` is parsed as a price-based family (`PB`), a fixed indicator/type bucket (`i`), and a rolling window suffix (`j`).
- `VB`, `BB`, `PV`, and `V` are treated as distinct families.
- Structural warm-up NaNs are preserved and classified separately from unexpected missingness.
- ML model-ready datasets intentionally remove non-finite rows only after the structural/validity layer has identified them.

No code change is required to preserve correctness. The main adjustment, if any, is explanatory wording: structural warm-up NaNs should be described as expected evidence of rolling-window warm-up, not as bad data.

The audit did not inspect holdout Days 86–108.

## 1. Current feature taxonomy implementation

The active feature parser is [`src/common/features.py`](../src/common/features.py).

It uses a single family regex:

`PB|VB|PV|BB|V`

and extracts:

- `family`
- `body`
- `suffix`

It then derives:

- `subfamily = family + body`
- `nominal_window_seconds` from the family-specific ladder

This is exactly the right level of abstraction for the clarification:

- fixed `i` values remain grouped as the same underlying indicator type via `subfamily`
- varying `j` values map to different suffixes within that same type
- the implementation does not try to infer the hidden mathematical formula

The parser is used consistently by:

- [`src/analytics/taxonomy.py`](../src/analytics/taxonomy.py)
- [`src/ebx/ml/feature_selection.py`](../src/ebx/ml/feature_selection.py)
- [`src/ebx/ml/temporal_robustness.py`](../src/ebx/ml/temporal_robustness.py)
- [`scripts/analysis/phase7_part4a.py`](../scripts/analysis/phase7_part4a.py)

## 2. Current missing-value handling

The structural-missingness logic is in [`src/cleaning/missingness.py`](../src/cleaning/missingness.py).

It explicitly separates:

- leading NaNs
- internal NaNs
- trailing NaNs
- all-NaN features

It also computes:

- `actual_warmup_seconds`
- `stability_class`
- `unexpected_internal_nan`

The key point is that the code treats early missing values as structural when they appear before the first valid observation. That matches the clarification that warm-up NaNs are expected and should not be equated with bad data.

Representative evidence from a real processed development day:

- [`data/processed/day1.parquet`](../data/processed/day1.parquet) contains columns such as `PB1_T1`, `PB1_T2`, `PB1_T3`, ...
- the first rows of those columns are NaN, which is consistent with rolling warm-up rather than corruption

## 3. Structural warm-up handling

Structural warm-up rows are handled correctly in the forensic layer.

Evidence:

- [`src/cleaning/missingness.py`](../src/cleaning/missingness.py) classifies leading-only missingness separately from internal/trailing missingness.
- [`src/analytics/taxonomy.py`](../src/analytics/taxonomy.py) preserves the nominal-vs-actual warm-up relationship and explicitly marks `nominal_window_status`.
- [`tests/test_phase7.py`](../tests/test_phase7.py) asserts that nominal deviations are retained rather than overwritten.
- [`reports/phase2_audit.md`](./phase2_audit.md) already states that structural NaNs are preserved and that deviations are retained.
- [`reports/final_report.md`](./final_report.md) also says structural NaNs are preserved.

Conclusion:

- structural warm-up NaNs are recognized as expected evidence
- they are not being silently imputed
- they are not being reinterpreted as data-quality failures in the forensic outputs

## 4. Whether any rows are incorrectly excluded

No incorrect exclusion was found in the raw/forensic pipeline.

What does happen is deliberate and expected:

- [`src/ebx/ml/dataset_builder.py`](../src/ebx/ml/dataset_builder.py) uses `complete_case_mask(...)`
- [`src/ebx/ml/preprocessing.py`](../src/ebx/ml/preprocessing.py) requires finite values and rejects non-finite validation rows
- [`src/ebx/ml/validation.py`](../src/ebx/ml/validation.py) validates that model-ready partitions contain only finite targets and features

So the model-ready ML dataset excludes rows with warm-up NaNs, but that exclusion is intentional because the downstream model is trained only on finite feature rows. The raw structural information is still preserved in the missingness artifacts.

That means:

- the feature semantics are correct
- the ML partitions are complete-case by design
- structural missingness is not being confused with data corruption

## 5. Impact on existing ML datasets

There is no evidence that this clarification changes the already-generated ML datasets.

Why:

- feature selection consumes the frozen screen, not raw formula inference
- preprocessing is train-only and finite-value only
- dataset building already uses day-local validity filtering

The clarification mainly improves interpretation, not behavior.

## 6. Impact on Part 4 figures

No material impact was found.

Reason:

- Part 4 taxonomy and redundancy work off parsed family/subfamily labels and observed warm-up behavior
- [`scripts/analysis/phase7_part4a.py`](../scripts/analysis/phase7_part4a.py) groups by `family` and retains nominal-vs-actual deviation status
- [`scripts/plot_part4.py`](../scripts/plot_part4.py) consumes those outputs; it does not try to infer hidden formulas from the NaN pattern

The clarification strengthens the interpretation of the figures, but does not require recomputation.

## 7. Impact on frozen artifacts

No frozen artifact needs to change for correctness.

Relevant frozen or audited outputs remain semantically consistent:

- `results/missingness/structural_missingness.csv`
- `results/missingness/cross_day_warmup.csv`
- `results/features/feature_taxonomy.csv`
- `results/features/family_summary.csv`
- `results/redundancy/*`
- `results/predictive/*`
- ML frozen feature and split artifacts

The taxonomy artifact already records:

- `nominal_window_status`
- `semantic_identity_status = name_and_observed_behavior_only; identity_unconfirmed`

Those fields are already aligned with the clarification.

## 8. Recommended changes, if any

No functional change is required.

If you want the repository wording to mirror the clarification more explicitly, the only changes I would recommend are documentation-only:

- [`src/common/features.py`](../src/common/features.py): clarify that `subfamily` groups fixed-`i` variants and that the parser does not infer formulas
- [`src/cleaning/missingness.py`](../src/cleaning/missingness.py): add a short comment that leading NaNs are expected warm-up evidence, not quality failures
- [`src/analytics/taxonomy.py`](../src/analytics/taxonomy.py): keep the current “identity unconfirmed” phrasing and note that nominal windows are hypotheses
- [`reports/final_report.md`](../reports/final_report.md): if desired, add one sentence that structural warm-up NaNs are expected and preserved

No pipeline edits are required.

## 9. Whether existing experiments remain valid

Yes.

Existing Part 4 and ML experiments remain valid because:

- the feature naming semantics are already parsed consistently
- structural warm-up NaNs were preserved in the forensic layer
- model-ready datasets were intentionally complete-case
- no frozen artifact or result depends on treating warm-up NaNs as bad data

This clarification improves interpretation but does not invalidate the existing results.

## 10. Exact files that would need modification

If no wording changes are desired, then no files need modification.

If the team wants to make the clarification explicit in the repository narrative, the only files that would benefit are:

- `src/common/features.py`
- `src/cleaning/missingness.py`
- `src/analytics/taxonomy.py`
- `reports/final_report.md`

## Final assessment

The current implementation is already semantically aligned with the clarification:

- taxonomy: correct
- family/subfamily grouping: correct
- structural warm-up handling: correct
- model-ready filtering: correct and intentional
- Part 4 and ML experiments: still valid

No pipeline changes are necessary at this stage.

