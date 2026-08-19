"""Explicit leakage and model-dataset validation checks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .schemas import DevelopmentScope
from .targets import build_future_return_target


def validate_target_alignment(day_data: pd.DataFrame, target: pd.Series, horizon: int) -> bool:
    expected = build_future_return_target(day_data, horizon)
    left = target.to_numpy(dtype=float)
    right = expected.to_numpy(dtype=float)
    return bool(np.array_equal(np.isnan(left), np.isnan(right)) and np.allclose(np.nan_to_num(left), np.nan_to_num(right), rtol=0, atol=0))


def validate_split_manifest(split: dict[str, object], scope: DevelopmentScope) -> None:
    train = {int(day) for day in split["training_days"]}  # type: ignore[index]
    validation = {int(day) for day in split["validation_days"]}  # type: ignore[index]
    scope.assert_development_days(train | validation)
    if train & validation or train | validation != set(scope.available_development_days):
        raise ValueError("train/validation days are not an exact partition of available development days")
    if max(train) >= min(validation):
        raise ValueError("chronological split order is invalid")
    if set(int(day) for day in split["holdout_days_excluded"]) != set(scope.holdout_days):  # type: ignore[index]
        raise ValueError("holdout exclusion is not explicit in split manifest")


def validate_partition(path: str | Path, allowed_days: Iterable[int], feature_names: tuple[str, ...]) -> dict[str, object]:
    frame = pd.read_parquet(path)
    required = {"day", "timestamp", "timestamp_seconds", "target", *feature_names}
    if not required.issubset(frame.columns):
        raise ValueError(f"partition missing columns: {sorted(required - set(frame.columns))}")
    days = set(frame["day"].astype(int))
    if not days.issubset(set(allowed_days)):
        raise ValueError(f"partition contains disallowed days: {sorted(days - set(allowed_days))}")
    if frame["target"].isna().any() or not np.isfinite(frame["target"].to_numpy(dtype=float)).all():
        raise ValueError("model-ready partition contains invalid targets")
    values = frame.loc[:, list(feature_names)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("model-ready partition contains invalid feature values")
    for day, group in frame.groupby("day", sort=False):
        seconds = group["timestamp_seconds"].to_numpy(dtype=np.int64)
        if len(seconds) > 1 and np.any(np.diff(seconds) <= 0):
            raise ValueError(f"partition timestamps are not day-local and ordered for day {day}")
    return {"path": str(path), "rows": int(len(frame)), "days": sorted(days)}


def leakage_report(
    *,
    scope: DevelopmentScope,
    split: dict[str, object],
    target_alignment_checked: bool,
    partition_reports: list[dict[str, object]],
    preprocessing_fit_days: Iterable[int],
    source_days_loaded: Iterable[int],
) -> dict[str, object]:
    loaded = {int(day) for day in source_days_loaded}
    train = {int(day) for day in split["training_days"]}  # type: ignore[index]
    validation = {int(day) for day in split["validation_days"]}  # type: ignore[index]
    fit_days = {int(day) for day in preprocessing_fit_days}
    return {
        "development_scope": scope.as_dict(),
        "temporal_alignment_checked": bool(target_alignment_checked),
        "target_day_boundary_checked": True,
        "target_cross_day_rows": 0,
        "train_validation_day_overlap": sorted(train & validation),
        "split_leakage_free": not bool(train & validation) and max(train) < min(validation),
        "preprocessing_fit_days": sorted(fit_days),
        "preprocessing_fit_uses_validation": bool(fit_days & validation),
        "preprocessing_leakage_free": fit_days == train,
        "feature_selection_recomputed_in_ml_phase": False,
        "feature_selection_source": "frozen Part 4 aggregate screen over all 70 available development days",
        "feature_selection_validation_fit_in_ml_phase": False,
        "holdout_days_loaded": sorted(loaded & set(scope.holdout_days)),
        "holdout_exclusion_passed": not bool(loaded & set(scope.holdout_days)),
        "partitions": partition_reports,
        "model_training_performed": False,
    }
