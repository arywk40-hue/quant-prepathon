"""Day-wise construction of the development-only model-ready dataset."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.common.day_boundary import parse_time_seconds

from .cache import sha256_file, write_json, write_partition
from .feature_selection import FROZEN_SCREEN_RULE
from .preprocessing import TrainOnlyStandardizer, complete_case_mask
from .schemas import DevelopmentScope
from .targets import build_future_return_target, profile_target


def _read_day(processed_dir: Path, day: int, columns: list[str]) -> pd.DataFrame:
    path = processed_dir / f"day{day}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(path, columns=columns).to_pandas()


def _seconds(frame: pd.DataFrame) -> np.ndarray:
    parsed = [parse_time_seconds(value) for value in frame["Time"]]
    if any(value is None for value in parsed):
        raise ValueError("invalid timestamp in processed Parquet")
    return np.asarray(parsed, dtype=np.int64)


def build_target_profiles(
    *,
    processed_dir: str | Path,
    scope: DevelopmentScope,
    horizons: Iterable[int],
    output_path: str | Path,
    recommendation_path: str | Path,
    frozen_screen: pd.DataFrame,
) -> pd.DataFrame:
    """Profile all candidate targets one day at a time."""

    processed = Path(processed_dir)
    horizons = tuple(int(horizon) for horizon in horizons)
    rows: list[dict[str, object]] = []
    pooled: dict[int, list[np.ndarray]] = {horizon: [] for horizon in horizons}
    pooled_total_rows: dict[int, int] = {horizon: 0 for horizon in horizons}
    for day in scope.available_development_days:
        frame = _read_day(processed, day, ["Time", "Price"])
        for horizon in horizons:
            target = build_future_return_target(frame, horizon)
            values = target.to_numpy(dtype=float)
            pooled[horizon].append(values[np.isfinite(values)])
            pooled_total_rows[horizon] += len(values)
            rows.append({
                **scope.as_dict(),
                **profile_target(values, scope="day", day=day, horizon=horizon),
            })
    for horizon in horizons:
        pooled_values = np.concatenate(pooled[horizon]) if pooled[horizon] else np.asarray([], dtype=float)
        pooled_row = profile_target(pooled_values, scope="pooled", day=None, horizon=horizon)
        pooled_row["missing_target_observations"] = int(pooled_total_rows[horizon] - pooled_row["valid_observations"])
        rows.append({
            **scope.as_dict(),
            **pooled_row,
        })
    output = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    selected = frozen_screen[frozen_screen["eligible_for_ml"]].copy()
    horizon_counts = selected.groupby("horizon_seconds").size().to_dict()
    horizon_strength = selected.assign(abs_ic=selected["mean_pearson_ic"].abs()).groupby("horizon_seconds")["abs_ic"].mean().to_dict()
    primary = max(horizons, key=lambda horizon: (int(horizon_counts.get(horizon, 0)), float(horizon_strength.get(horizon, 0.0)), horizon))
    recommendation = {
        "candidate_horizons_seconds": list(horizons),
        "primary_horizon_seconds": int(primary),
        "recommendation_basis": "target profiles are all emitted; frozen eligible feature-horizon count is the primary tie-broken criterion, followed by mean absolute frozen Pearson IC",
        "frozen_eligible_rows_by_horizon": {str(key): int(value) for key, value in horizon_counts.items()},
        "frozen_mean_absolute_pearson_ic_by_horizon": {str(key): float(value) for key, value in horizon_strength.items()},
        "no_holdout_access": True,
        **scope.as_dict(),
    }
    write_json(recommendation, recommendation_path)
    return output


def build_model_dataset(
    *,
    processed_dir: str | Path,
    output_root: str | Path,
    scope: DevelopmentScope,
    split: dict[str, object],
    frozen_screen: pd.DataFrame,
    target_horizon: int,
    frozen_paths: dict[str, str | Path],
) -> dict[str, object]:
    """Build train/validation Parquet partitions using two day-wise passes."""

    eligible = frozen_screen[
        frozen_screen["eligible_for_ml"] & (frozen_screen["horizon_seconds"].astype(int) == int(target_horizon))
    ]
    feature_names = tuple(sorted(eligible["feature"].unique()))
    if not feature_names:
        raise ValueError("frozen screen contains no eligible features at the selected horizon")
    processed = Path(processed_dir)
    root = Path(output_root)
    standardizer = TrainOnlyStandardizer(feature_names)
    train_days = tuple(int(day) for day in split["training_days"])  # type: ignore[index]
    validation_days = tuple(int(day) for day in split["validation_days"])  # type: ignore[index]
    all_days = train_days + validation_days
    first_pass: dict[int, dict[str, int]] = {}
    source_columns = ["Time", "Price", *feature_names]

    for day in train_days:
        frame = _read_day(processed, day, source_columns)
        target = build_future_return_target(frame, target_horizon)
        target_valid, feature_valid, complete = complete_case_mask(frame, feature_names, target)
        standardizer.update(frame.loc[complete, list(feature_names)])
        first_pass[day] = _counts(target_valid, feature_valid, complete, len(frame))
    standardizer.finalize()

    preprocessing_manifest = {
        **scope.as_dict(),
        "preprocessing_version": "train-only-standardization-v1",
        "fit_days": list(train_days),
        "validation_days_not_used_for_fit": list(validation_days),
        "target_horizon_seconds": int(target_horizon),
        "feature_count": len(feature_names),
        "screen_rule": FROZEN_SCREEN_RULE,
        **standardizer.manifest(),
    }
    preprocessing_path = root / "preprocessing" / "preprocessing_manifest.json"
    write_json(preprocessing_manifest, preprocessing_path)

    partition_reports: list[dict[str, object]] = []
    totals = {"rows": 0, "valid_target_count": 0, "valid_model_rows": 0, "excluded_row_count": 0, "invalid_target_rows": 0, "invalid_feature_rows": 0}
    for day in all_days:
        frame = _read_day(processed, day, source_columns)
        target = build_future_return_target(frame, target_horizon)
        target_valid, feature_valid, complete = complete_case_mask(frame, feature_names, target)
        counts = _counts(target_valid, feature_valid, complete, len(frame))
        if not np.array_equal(target_valid, np.isfinite(target.to_numpy(dtype=float))):
            raise AssertionError("target validity changed between passes")
        transformed = standardizer.transform(frame.loc[complete, list(feature_names)])
        timestamps = frame.loc[complete, "Time"].astype(str).to_numpy()
        timestamp_seconds = _seconds(frame.loc[complete, ["Time"]])
        base_partition = pd.DataFrame({
            "day": np.full(len(transformed), day, dtype=np.int16),
            "timestamp": timestamps,
            "timestamp_seconds": timestamp_seconds,
            "target": target.loc[complete].to_numpy(dtype=np.float32),
        })
        partition = pd.concat(
            [base_partition.reset_index(drop=True), transformed.reset_index(drop=True)],
            axis=1,
        )
        split_name = "train" if day in train_days else "validation"
        path = root / "datasets" / split_name / f"day{day}.parquet"
        write_partition(partition, path)
        partition_report = {"split": split_name, "day": day, "path": str(path), **counts, "written_rows": int(len(partition))}
        partition_reports.append(partition_report)
        totals["rows"] += len(frame)
        totals["valid_target_count"] += counts["valid_target_count"]
        totals["valid_model_rows"] += counts["valid_model_rows"]
        totals["excluded_row_count"] += counts["excluded_row_count"]
        totals["invalid_target_rows"] += counts["invalid_target_rows"]
        totals["invalid_feature_rows"] += counts["invalid_feature_rows"]

    input_hashes = {name: sha256_file(path) for name, path in frozen_paths.items()}
    feature_set_path = root / "features" / "frozen_feature_set.csv"
    dataset_manifest = {
        **scope.as_dict(),
        "data_version": "validated-development-parquet-v1",
        "feature_set_version": sha256_file(feature_set_path) if feature_set_path.exists() else "",
        "target_definition": "P(t+h) / P(t) - 1 using exact within-day timestamp t+h",
        "target_horizon": int(target_horizon),
        "training_days": list(train_days),
        "validation_days": list(validation_days),
        "missing_days": list(scope.missing_development_days),
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "row_count": totals["valid_model_rows"],
        "valid_target_count": totals["valid_target_count"],
        "excluded_row_count": totals["excluded_row_count"],
        "excluded_by_reason": {
            "invalid_target": totals["invalid_target_rows"],
            "invalid_feature_after_valid_target": totals["invalid_feature_rows"],
        },
        "preprocessing_version": "train-only-standardization-v1",
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "frozen_input_sha256": input_hashes,
        "screen_rule": FROZEN_SCREEN_RULE,
        "model_training_performed": False,
        "partition_count": len(partition_reports),
        "partitions": partition_reports,
    }
    dataset_path = root / "datasets" / "dataset_manifest.json"
    write_json(dataset_manifest, dataset_path)
    return {
        "feature_names": feature_names,
        "preprocessing_manifest": preprocessing_manifest,
        "dataset_manifest": dataset_manifest,
        "partition_reports": partition_reports,
        "preprocessing_fit_days": train_days,
        "source_days_loaded": all_days,
        "first_pass_counts": first_pass,
    }


def _counts(target_valid: np.ndarray, feature_valid: np.ndarray, complete: np.ndarray, rows: int) -> dict[str, int]:
    invalid_target = ~target_valid
    invalid_feature = target_valid & ~feature_valid
    return {
        "source_rows": int(rows),
        "valid_target_count": int(target_valid.sum()),
        "valid_model_rows": int(complete.sum()),
        "excluded_row_count": int(rows - complete.sum()),
        "invalid_target_rows": int(invalid_target.sum()),
        "invalid_feature_rows": int(invalid_feature.sum()),
    }
