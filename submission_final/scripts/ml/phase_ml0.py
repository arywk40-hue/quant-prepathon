"""ML Phase 0: build development-only model-ready data, without training."""

from __future__ import annotations

import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ebx.ml.cache import write_json  # noqa: E402
from src.ebx.ml.dataset_builder import build_model_dataset, build_target_profiles  # noqa: E402
from src.ebx.ml.feature_selection import load_frozen_feature_screen, write_frozen_feature_set  # noqa: E402
from src.ebx.ml.schemas import TARGET_HORIZONS_SECONDS, audited_scope  # noqa: E402
from src.ebx.ml.splits import chronological_split, write_split_manifest  # noqa: E402
from src.ebx.ml.validation import leakage_report, validate_partition, validate_split_manifest  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    scope = audited_scope(root / "results/freeze/development_freeze.json")
    output = root / "results/ml"
    phase_start = time.perf_counter()
    selection_start = time.perf_counter()
    frozen_screen = load_frozen_feature_screen(root / "results/predictive/aggregate_ic.csv", scope)
    feature_set_path = output / "features/frozen_feature_set.csv"
    write_frozen_feature_set(frozen_screen, feature_set_path)
    selection_seconds = time.perf_counter() - selection_start

    target_start = time.perf_counter()
    profiles = build_target_profiles(
        processed_dir=root / "data/processed",
        scope=scope,
        horizons=TARGET_HORIZONS_SECONDS,
        output_path=output / "targets/target_profile.csv",
        recommendation_path=output / "targets/target_recommendation.json",
        frozen_screen=frozen_screen,
    )
    target_profile_seconds = time.perf_counter() - target_start
    recommendation = json.loads((output / "targets/target_recommendation.json").read_text())
    split = chronological_split(scope)
    validate_split_manifest(split, scope)
    write_split_manifest(split, output / "splits/split_manifest.json")
    build = build_model_dataset(
        processed_dir=root / "data/processed",
        output_root=output,
        scope=scope,
        split=split,
        frozen_screen=frozen_screen,
        target_horizon=int(recommendation["primary_horizon_seconds"]),
        frozen_paths={
            "development_freeze.json": root / "results/freeze/development_freeze.json",
            "aggregate_ic.csv": root / "results/predictive/aggregate_ic.csv",
            "config.yaml": root / "config/config.yaml",
        },
    )
    partition_reports = [
        validate_partition(report["path"], scope.available_development_days, build["feature_names"])
        for report in build["partition_reports"]
    ]
    report = leakage_report(
        scope=scope,
        split=split,
        target_alignment_checked=True,
        partition_reports=partition_reports,
        preprocessing_fit_days=build["preprocessing_fit_days"],
        source_days_loaded=build["source_days_loaded"],
    )
    write_json(report, output / "validation/leakage_report.json")
    phase_elapsed = time.perf_counter() - phase_start
    output_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        max_rss *= 1024
    write_json({
        **scope.as_dict(),
        "feature_selection_seconds": selection_seconds,
        "target_profile_seconds": target_profile_seconds,
        "dataset_build_seconds": phase_elapsed - selection_seconds - target_profile_seconds,
        "total_seconds": phase_elapsed,
        "processing_seconds_per_development_day": phase_elapsed / len(scope.available_development_days),
        "peak_rss_bytes_process_reported": int(max_rss),
        "output_size_bytes": int(output_bytes),
        "feature_selection_rows": int(len(frozen_screen)),
        "eligible_feature_horizon_rows": int(frozen_screen.eligible_for_ml.sum()),
        "eligible_unique_features": int(build["dataset_manifest"]["feature_count"]),
        "days_processed": list(scope.available_development_days),
        "holdout_days_processed": [],
        "model_training_performed": False,
    }, output / "validation/performance.json")
    print(json.dumps({
        "phase": "ML Phase 0",
        "model_training_performed": False,
        "expected_development_days": scope.expected_development_days,
        "available_development_days": len(scope.available_development_days),
        "missing_days": list(scope.missing_development_days),
        "target_profile_rows": len(profiles),
        "primary_horizon_seconds": recommendation["primary_horizon_seconds"],
        "feature_count": len(build["feature_names"]),
        "model_ready_rows": build["dataset_manifest"]["row_count"],
        "holdout_days_loaded": report["holdout_days_loaded"],
    }, indent=2))


if __name__ == "__main__":
    main()
