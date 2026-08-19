"""Feature taxonomy assembly without semantic identity claims."""

from __future__ import annotations

import pandas as pd


def assemble_taxonomy(
    structural: pd.DataFrame,
    warmup: pd.DataFrame,
    value_stats: pd.DataFrame,
    coverage: dict[str, object],
) -> pd.DataFrame:
    """Join forensic metadata, missingness, warm-up and value-scale evidence."""

    required_structural = {
        "feature", "family", "subfamily", "suffix", "nominal_window_seconds",
        "leading_nan_count", "internal_nan_count", "trailing_nan_count",
        "total_nan_count", "total_inf_count", "missing_fraction", "stability_class",
    }
    required_warmup = {
        "feature", "days_expected", "days_present", "days_with_feature",
        "missing_days", "mean_warmup_sec", "median_warmup_sec", "std_warmup_sec",
        "min_warmup_sec", "max_warmup_sec", "days_matching_nominal", "days_deviating",
        "internal_nan_days", "stability_class",
    }
    if not required_structural.issubset(structural.columns):
        raise ValueError("structural missingness output is missing required fields")
    if not required_warmup.issubset(warmup.columns):
        raise ValueError("cross-day warm-up output is missing required fields")
    first = structural.sort_values(["feature", "day"]).groupby("feature", as_index=False).first()
    miss = structural.groupby("feature", as_index=False).agg(
        days_observed=("day", "nunique"),
        leading_nan_total=("leading_nan_count", "sum"),
        internal_nan_total=("internal_nan_count", "sum"),
        trailing_nan_total=("trailing_nan_count", "sum"),
        total_nan_total=("total_nan_count", "sum"),
        total_inf_total=("total_inf_count", "sum"),
        average_missing_fraction=("missing_fraction", "mean"),
    )
    result = first[["feature", "family", "subfamily", "suffix", "nominal_window_seconds"]].merge(
        warmup.drop(columns=["family", "subfamily", "suffix", "nominal_window_seconds", "stability_class"], errors="ignore"),
        on="feature", how="left", validate="one_to_one",
    ).merge(miss, on="feature", how="left", validate="one_to_one")
    result = result.merge(value_stats, on="feature", how="left", validate="one_to_one")
    result["nominal_window_status"] = result.apply(
        lambda row: "unavailable" if pd.isna(row["nominal_window_seconds"]) or row["nominal_window_seconds"] == "" else (
            "matches_actual_warmup" if row["days_deviating"] == 0 else "actual_deviations_retained"
        ), axis=1,
    )
    result["semantic_identity_status"] = "name_and_observed_behavior_only; identity_unconfirmed"
    for key, value in coverage.items():
        result[key] = value
    return result.sort_values(["family", "subfamily", "feature"]).reset_index(drop=True)
