"""Phase 7: forensic taxonomy of the masked feature schema."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analytics.coverage import AVAILABLE_DEVELOPMENT_DAYS, available_days_from_manifest, coverage_metadata  # noqa: E402
from src.analytics.taxonomy import assemble_taxonomy  # noqa: E402


def _value_statistics(repo_root: Path, features: list[str]) -> pd.DataFrame:
    n = {feature: 0 for feature in features}
    mean = {feature: 0.0 for feature in features}
    m2 = {feature: 0.0 for feature in features}
    minimum = {feature: np.inf for feature in features}
    maximum = {feature: -np.inf for feature in features}
    for day in AVAILABLE_DEVELOPMENT_DAYS:
        parquet = pq.ParquetFile(repo_root / "data" / "processed" / f"day{day}.parquet")
        for batch in parquet.iter_batches(columns=features, batch_size=8192):
            frame = batch.to_pandas()
            for feature in features:
                values = frame[feature].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if not len(values):
                    continue
                count = len(values)
                batch_mean = float(np.mean(values))
                batch_m2 = float(np.sum((values - batch_mean) ** 2))
                old = n[feature]
                total = old + count
                delta = batch_mean - mean[feature]
                mean[feature] += delta * count / total
                m2[feature] += batch_m2 + delta * delta * old * count / total
                n[feature] = total
                minimum[feature] = min(minimum[feature], float(np.min(values)))
                maximum[feature] = max(maximum[feature], float(np.max(values)))
    rows = []
    for feature in features:
        variance = m2[feature] / (n[feature] - 1) if n[feature] > 1 else np.nan
        rows.append({
            "feature": feature,
            "valid_value_count": n[feature],
            "value_mean": mean[feature] if n[feature] else np.nan,
            "value_variance": variance,
            "scale_std_dev": np.sqrt(variance) if np.isfinite(variance) else np.nan,
            "value_min": minimum[feature] if n[feature] else np.nan,
            "value_max": maximum[feature] if n[feature] else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    available = available_days_from_manifest(repo_root)
    if available != AVAILABLE_DEVELOPMENT_DAYS:
        raise RuntimeError("available development universe changed")
    structural = pd.read_csv(repo_root / "results" / "missingness" / "structural_missingness.csv")
    warmup = pd.read_csv(repo_root / "results" / "missingness" / "cross_day_warmup.csv")
    if set(structural.day.unique()) != set(available) or len(structural) != len(available) * 691:
        raise RuntimeError("Phase 2 structural output does not cover the audited 70 days")
    features = sorted(structural.feature.unique())
    stats = _value_statistics(repo_root, features)
    taxonomy = assemble_taxonomy(structural, warmup, stats, coverage_metadata())
    output = repo_root / "results" / "features"
    output.mkdir(parents=True, exist_ok=True)
    taxonomy.to_csv(output / "feature_taxonomy.csv", index=False)
    family = taxonomy.groupby("family", as_index=False).agg(
        feature_count=("feature", "count"),
        nominal_window_available_count=("nominal_window_status", lambda values: int((values != "unavailable").sum())),
        actual_deviation_feature_count=("nominal_window_status", lambda values: int((values == "actual_deviations_retained").sum())),
        mean_valid_value_count=("valid_value_count", "mean"),
    )
    family.to_csv(output / "family_summary.csv", index=False)
    header = (repo_root / "data" / "processed" / "day1.parquet").read_bytes()
    scope = {
        **coverage_metadata(),
        "available_day_ids": list(available),
        "missing_day_ids": list(range(65, 80)),
        "holdout_day_ids": list(range(86, 109)),
        "holdout_processed": False,
        "feature_count": len(features),
        "source_outputs": ["results/missingness/structural_missingness.csv", "results/missingness/cross_day_warmup.csv"],
        "scale_definition": "pooled finite-value sample standard deviation across available development days",
        "semantic_policy": "family/subfamily/window are parsed hypotheses; identity is unconfirmed",
        "day_boundary_policy": "statistics reset per source day before pooled moment merge",
        "day1_processed_file_sha256": hashlib.sha256(header).hexdigest(),
    }
    (output / "phase7_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    print({"feature_count": len(features), "family_summary": family.to_dict(orient="records")})


if __name__ == "__main__":
    main()
