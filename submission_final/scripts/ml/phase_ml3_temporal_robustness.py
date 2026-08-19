"""ML Phase 3: pre-specified blocked temporal robustness experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.features import parse_feature  # noqa: E402
from src.ebx.ml.baseline import RidgeBaseline, validation_metrics  # noqa: E402
from src.ebx.ml.cache import sha256_file, write_json, write_partition  # noqa: E402
from src.ebx.ml.schemas import TARGET_HORIZONS_SECONDS, audited_scope  # noqa: E402
from src.ebx.ml.temporal_robustness import (  # noqa: E402
    TEMPORAL_WINDOWS,
    cross_window_summary,
    feature_family_counts,
    feature_overlap_matrix,
    validate_temporal_windows,
)
from src.ebx.ml.train_only_selection import (  # noqa: E402
    build_training_only_partitions,
    fit_training_only_screen,
    load_training_daily_ic,
)


def _write_feature_table(selected: pd.DataFrame, path: Path) -> pd.DataFrame:
    result = selected.copy()
    metadata = [parse_feature(feature) for feature in result["feature"]]
    result.insert(1, "family", [item.family for item in metadata])
    result.insert(2, "subfamily", [item.subfamily for item in metadata])
    result.insert(3, "nominal_window_seconds", [item.nominal_window_seconds for item in metadata])
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result


def _run_window(
    *,
    name: str,
    window: dict[str, tuple[int, ...]],
    root: Path,
    scope,
    output: Path,
) -> tuple[dict[str, object], pd.DataFrame, set[str]]:
    window_root = output / name
    window_root.mkdir(parents=True, exist_ok=True)
    train_days = tuple(window["training_days"])
    validation_days = tuple(window["validation_days"])
    daily_ic = load_training_daily_ic(
        root / "results/predictive/per_day_ic.csv",
        training_days=train_days,
        horizons=TARGET_HORIZONS_SECONDS,
        scope=scope,
    )
    aggregate, selected = fit_training_only_screen(
        daily_ic,
        training_days=train_days,
        target_horizon=300,
        scope=scope,
    )
    if selected.empty:
        raise ValueError(f"{name} selected no 300-second features")
    selected = _write_feature_table(selected, window_root / "selected_features.csv")
    daily_ic.to_csv(window_root / "selection_daily_ic.csv", index=False)
    aggregate.to_csv(window_root / "selection_aggregate_ic.csv", index=False)
    features = tuple(selected["feature"].astype(str))

    scaler, partition_reports = build_training_only_partitions(
        processed_dir=root / "data/processed",
        output_root=window_root,
        training_days=train_days,
        validation_days=validation_days,
        features=features,
        target_horizon=300,
        scope=scope,
    )
    write_json({
        **scope.as_dict(),
        "window": name,
        "preprocessing_version": "train-only-standardization-v1",
        "fit_days": list(train_days),
        "validation_days_not_used_for_fit": list(validation_days),
        "target_horizon_seconds": 300,
        "feature_count": len(features),
        **scaler.manifest(),
    }, window_root / "preprocessing_manifest.json")

    train_paths = [window_root / "datasets/train" / f"day{day}.parquet" for day in train_days]
    validation_paths = [window_root / "datasets/validation" / f"day{day}.parquet" for day in validation_days]
    model = RidgeBaseline(features, alpha=1.0, fit_intercept=True).fit_partition_paths(train_paths)
    model.save(window_root / "ridge_model.pkl")

    prediction_frames: list[pd.DataFrame] = []
    prediction_partitions: list[dict[str, object]] = []
    for day, path in zip(validation_days, validation_paths):
        frame = pd.read_parquet(path)
        if set(frame["day"].astype(int).unique()) != {day}:
            raise ValueError(f"{name} validation partition day mismatch: {path}")
        prediction = model.predict(frame)
        result = frame[["day", "timestamp", "timestamp_seconds", "target"]].copy()
        result["prediction"] = prediction
        result["residual"] = result["prediction"] - result["target"]
        destination = window_root / "predictions" / f"day{day}.parquet"
        write_partition(result, destination)
        prediction_frames.append(result)
        prediction_partitions.append({"day": day, "path": str(destination), "rows": int(len(result))})
    predictions = pd.concat(prediction_frames, ignore_index=True)
    pooled, daily = validation_metrics(predictions)
    write_json(pooled, window_root / "validation_metrics.json")
    daily.to_csv(window_root / "daily_metrics.csv", index=False)
    write_json({
        "window": name,
        "model": model.summary(),
        "target_horizon_seconds": 300,
        "training_days": list(train_days),
        "validation_days": list(validation_days),
        "selected_feature_count": len(features),
        "selected_feature_names": list(features),
        "feature_family_counts": feature_family_counts(features),
        "alpha": 1.0,
        "fit_intercept": True,
    }, window_root / "model_config.json")

    run_manifest = {
        "window": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "expected_development_days": scope.expected_development_days,
        "available_development_days": len(scope.available_development_days),
        "missing_development_days": list(scope.missing_development_days),
        "training_days": list(train_days),
        "validation_days": list(validation_days),
        "source_days_loaded": [*train_days, *validation_days],
        "holdout_days_loaded": [],
        "holdout_accessed": False,
        "candidate_feature_count": int(daily_ic["feature"].nunique()),
        "candidate_hypothesis_count": int(len(aggregate)),
        "selected_feature_count": len(features),
        "selected_feature_names": list(features),
        "feature_family_counts": feature_family_counts(features),
        "selection_days_only": list(train_days),
        "selection_rule": "pearson_fdr_reject AND pearson_pct_same_sign >= 0.70 AND abs(mean_pearson_ic) >= 0.05",
        "fdr_scope": "all candidate feature-horizon hypotheses, refit on training-day ICs",
        "target_horizon_seconds": 300,
        "model": "ridge",
        "alpha": 1.0,
        "fit_intercept": True,
        "training_rows": int(model.n_train_samples_),
        "validation_rows": int(len(predictions)),
        "partition_reports": partition_reports,
        "prediction_partitions": prediction_partitions,
    }
    write_json(run_manifest, window_root / "run_manifest.json")
    write_json({
        "deterministic_selection": True,
        "deterministic_fit": True,
        "random_seed": None,
        "selection_days": list(train_days),
        "fit_days": list(train_days),
        "prediction_days": list(validation_days),
        "holdout_days_loaded": [],
        "model_artifact_sha256": sha256_file(window_root / "ridge_model.pkl"),
    }, window_root / "reproducibility.json")
    return pooled, daily, set(features)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    scope = audited_scope(root / "results/freeze/development_freeze.json")
    validate_temporal_windows(TEMPORAL_WINDOWS, scope)
    output = root / "results/ml/temporal_robustness"
    output.mkdir(parents=True, exist_ok=True)

    pooled_metrics: dict[str, dict[str, object]] = {}
    daily_metrics: dict[str, pd.DataFrame] = {}
    feature_sets: dict[str, set[str]] = {}
    window_manifests: dict[str, dict[str, object]] = {}
    for name in ("W1", "W2", "W3"):
        pooled, daily, features = _run_window(
            name=name,
            window={key: tuple(value) for key, value in TEMPORAL_WINDOWS[name].items()},
            root=root,
            scope=scope,
            output=output,
        )
        pooled_metrics[name] = pooled
        daily_metrics[name] = daily
        feature_sets[name] = features
        window_manifests[name] = json.loads((output / name / "run_manifest.json").read_text())

    overlap = feature_overlap_matrix(feature_sets)
    overlap.to_csv(output / "feature_overlap_matrix.csv")
    summary = cross_window_summary(pooled_metrics, daily_metrics, feature_sets)
    write_json(summary, output / "aggregate_robustness.json")
    pd.DataFrame([
        {"window": name, **pooled_metrics[name], **summary["daily_metrics_by_window"][name]}
        for name in ("W1", "W2", "W3")
    ]).to_csv(output / "window_metrics.csv", index=False)

    w3_predictions = pd.concat([
        pd.read_parquet(output / "W3/predictions" / f"day{day}.parquet")
        for day in TEMPORAL_WINDOWS["W3"]["validation_days"]
    ], ignore_index=True)
    normal, _ = validation_metrics(w3_predictions)
    excluding_day84, _ = validation_metrics(w3_predictions[w3_predictions["day"] != 84].reset_index(drop=True))
    difference = {
        key: (float(excluding_day84[key]) - float(normal[key]))
        for key in normal
        if key != "day" and isinstance(normal[key], (int, float)) and np.isfinite(normal[key]) and np.isfinite(excluding_day84.get(key, np.nan))
    }
    write_json({
        "diagnostic": "post-hoc W3 validation aggregation excluding Day 84",
        "retrained": False,
        "feature_selection_changed": False,
        "normal_w3": normal,
        "excluding_day84": excluding_day84,
        "difference_excluding_day84_minus_normal": difference,
    }, output / "day84_sensitivity.json")

    input_paths = {
        "development_freeze.json": root / "results/freeze/development_freeze.json",
        "split_manifest.json": root / "results/ml/splits/split_manifest.json",
        "per_day_ic.csv": root / "results/predictive/per_day_ic.csv",
        "config.yaml": root / "config/config.yaml",
    }
    top_manifest = {
        "phase": "ML Phase 3 — Temporal Robustness Experiment",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "window_order": ["W1", "W2", "W3"],
        "windows": {name: {key: list(value) for key, value in TEMPORAL_WINDOWS[name].items()} for name in TEMPORAL_WINDOWS},
        "expected_development_days": scope.expected_development_days,
        "available_development_days": len(scope.available_development_days),
        "missing_development_days": list(scope.missing_development_days),
        "holdout_days_loaded": [],
        "holdout_accessed": False,
        "target_horizon_seconds": 300,
        "model": "ridge",
        "alpha": 1.0,
        "fit_intercept": True,
        "window_manifests": window_manifests,
        "input_sha256": {name: sha256_file(path) for name, path in input_paths.items()},
        "day84_sensitivity_post_hoc": True,
        "frozen_artifacts_modified": False,
    }
    write_json(top_manifest, output / "run_manifest.json")
    write_json({
        "deterministic_selection": True,
        "deterministic_fit": True,
        "random_seed": None,
        "window_order": ["W1", "W2", "W3"],
        "holdout_days_loaded": [],
        "input_sha256": top_manifest["input_sha256"],
    }, output / "reproducibility.json")
    print(json.dumps({
        "phase": top_manifest["phase"],
        "windows": {name: {"training_rows": window_manifests[name]["training_rows"], "validation_rows": pooled_metrics[name]["validation_observations"], "pearson_ic": pooled_metrics[name]["pearson_ic"], "selected_features": len(feature_sets[name])} for name in ("W1", "W2", "W3")},
        "holdout_days_loaded": [],
    }, indent=2))


if __name__ == "__main__":
    main()
