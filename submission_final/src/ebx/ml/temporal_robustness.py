"""Pre-specified temporal robustness helpers for ML Phase 3."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.common.features import parse_feature

from .schemas import DevelopmentScope


TEMPORAL_WINDOWS: dict[str, dict[str, tuple[int, ...]]] = {
    "W1": {"training_days": tuple(range(1, 45)), "validation_days": tuple(range(45, 55))},
    "W2": {"training_days": tuple(range(1, 55)), "validation_days": tuple(range(55, 65))},
    "W3": {"training_days": tuple(range(1, 65)), "validation_days": tuple(range(80, 86))},
}


def validate_temporal_windows(
    windows: Mapping[str, Mapping[str, Sequence[int]]],
    scope: DevelopmentScope,
) -> None:
    """Validate fixed whole-day chronological windows and boundaries."""

    if set(windows) != set(TEMPORAL_WINDOWS):
        raise ValueError("temporal windows must be exactly W1, W2, and W3")
    for name, window in windows.items():
        train = tuple(int(day) for day in window["training_days"])
        validation = tuple(int(day) for day in window["validation_days"])
        if not train or not validation:
            raise ValueError(f"{name} has an empty training or validation set")
        if set(train) & set(validation):
            raise ValueError(f"{name} training and validation overlap")
        if max(train) >= min(validation):
            raise ValueError(f"{name} is not chronological")
        scope.assert_development_days((*train, *validation))
        if set(train) & set(scope.missing_development_days) or set(validation) & set(scope.missing_development_days):
            raise ValueError(f"{name} uses unavailable development days")
        if set(train) & set(scope.holdout_days) or set(validation) & set(scope.holdout_days):
            raise ValueError(f"{name} uses holdout days")
    expected = {name: {key: tuple(value) for key, value in window.items()} for name, window in TEMPORAL_WINDOWS.items()}
    observed = {name: {key: tuple(int(day) for day in window[key]) for key in ("training_days", "validation_days")} for name, window in windows.items()}
    if observed != expected:
        raise ValueError("temporal windows do not match the pre-specified design")


def feature_family_counts(features: Sequence[str]) -> dict[str, int]:
    """Count selected features by parsed family."""

    return dict(sorted(Counter(parse_feature(feature).family for feature in features).items()))


def feature_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    """Return Jaccard overlap of two feature sets."""

    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 1.0


def feature_overlap_matrix(feature_sets: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    """Return a deterministic pairwise Jaccard matrix."""

    names = sorted(feature_sets)
    return pd.DataFrame(
        [[feature_jaccard(feature_sets[left], feature_sets[right]) for right in names] for left in names],
        index=names,
        columns=names,
    )


def cross_window_summary(
    pooled_metrics: Mapping[str, Mapping[str, object]],
    daily_metrics: Mapping[str, pd.DataFrame],
    feature_sets: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    """Summarize descriptive robustness statistics without tuning."""

    names = sorted(pooled_metrics)
    pooled_ic = np.asarray([float(pooled_metrics[name]["pearson_ic"]) for name in names], dtype=float)
    feature_counts = np.asarray([len(feature_sets[name]) for name in names], dtype=float)
    per_window_daily = {
        name: {
            "mean_daily_pearson_ic": float(daily_metrics[name]["pearson_ic"].mean()),
            "median_daily_pearson_ic": float(daily_metrics[name]["pearson_ic"].median()),
            "positive_validation_day_fraction": float((daily_metrics[name]["pearson_ic"] > 0).mean()),
            "positive_validation_days": int((daily_metrics[name]["pearson_ic"] > 0).sum()),
            "negative_validation_days": int((daily_metrics[name]["pearson_ic"] < 0).sum()),
        }
        for name in names
    }
    return {
        "window_order": names,
        "mean_window_pearson_ic": float(np.mean(pooled_ic)),
        "median_window_pearson_ic": float(np.median(pooled_ic)),
        "std_window_pearson_ic": float(np.std(pooled_ic, ddof=1)) if len(pooled_ic) > 1 else float("nan"),
        "minimum_window_pearson_ic": float(np.min(pooled_ic)),
        "maximum_window_pearson_ic": float(np.max(pooled_ic)),
        "window_pearson_ic_range": float(np.max(pooled_ic) - np.min(pooled_ic)),
        "positive_window_count": int((pooled_ic > 0).sum()),
        "negative_window_count": int((pooled_ic < 0).sum()),
        "feature_count_mean": float(np.mean(feature_counts)),
        "feature_count_std_population": float(np.std(feature_counts, ddof=0)),
        "feature_count_coefficient_of_variation": float(np.std(feature_counts, ddof=0) / np.mean(feature_counts)),
        "daily_metrics_by_window": per_window_daily,
    }
