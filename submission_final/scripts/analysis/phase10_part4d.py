"""Phase 10: feature redundancy and PCA on available development days."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.decomposition import IncrementalPCA, PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analytics.coverage import AVAILABLE_DEVELOPMENT_DAYS, available_days_from_manifest, coverage_metadata  # noqa: E402
from src.analytics.redundancy import day_zscore, deterministic_rows  # noqa: E402


ROW_CAP = 512


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    available = available_days_from_manifest(repo_root)
    if available != AVAILABLE_DEVELOPMENT_DAYS:
        raise RuntimeError("available development universe changed")
    first = pq.read_table(repo_root / "data" / "processed" / "day1.parquet").to_pandas()
    features = [column for column in first.columns if column not in {"Time", "Price"}]
    feature_count = len(features)
    output = repo_root / "results" / "redundancy"
    output.mkdir(parents=True, exist_ok=True)
    pair_i, pair_j = np.triu_indices(feature_count, k=1)
    pearson_sum = np.zeros(len(pair_i), dtype=np.float64)
    spearman_sum = np.zeros(len(pair_i), dtype=np.float64)
    pair_days = np.zeros(len(pair_i), dtype=np.int16)
    pca_rows = []
    pooled_pca = IncrementalPCA(n_components=50, batch_size=ROW_CAP)
    pooled_fit_rows = 0
    for day in available:
        frame = pq.read_table(repo_root / "data" / "processed" / f"day{day}.parquet", columns=features).to_pandas()
        values = day_zscore(frame.to_numpy(dtype=float))
        complete = np.isfinite(values).all(axis=1)
        sample = deterministic_rows(values[complete], ROW_CAP)
        if len(sample) < 3:
            raise RuntimeError(f"day {day} has too few complete feature rows for PCA")
        pearson = np.corrcoef(sample, rowvar=False)
        ranked = pd.DataFrame(sample).rank(method="average").to_numpy(dtype=float)
        spearman = np.corrcoef(ranked, rowvar=False)
        pearson_sum += np.abs(pearson[pair_i, pair_j])
        spearman_sum += np.abs(spearman[pair_i, pair_j])
        pair_days += np.isfinite(pearson[pair_i, pair_j]) & np.isfinite(spearman[pair_i, pair_j])
        components = min(50, len(sample), feature_count)
        model = PCA(n_components=components, svd_solver="randomized", random_state=0)
        model.fit(sample)
        cumulative = np.cumsum(model.explained_variance_ratio_)
        pca_rows.append({
            "pca_type": "per_day", "day": day, "complete_rows": int(complete.sum()), "sample_rows": len(sample),
            "components_50pct": int(np.searchsorted(cumulative, .50) + 1) if cumulative[-1] >= .50 else np.nan,
            "components_80pct": int(np.searchsorted(cumulative, .80) + 1) if cumulative[-1] >= .80 else np.nan,
            "components_90pct": int(np.searchsorted(cumulative, .90) + 1) if cumulative[-1] >= .90 else np.nan,
            "variance_first_component": float(cumulative[0]), **coverage_metadata(),
        })
        # Incremental PCA cannot fit a batch smaller than n_components.
        pooled_pca.partial_fit(sample)
        pooled_fit_rows += len(sample)
        print(f"processed day {day}", flush=True)
    pairwise = pd.DataFrame({
        "feature_i": np.asarray(features, dtype=object)[pair_i], "feature_j": np.asarray(features, dtype=object)[pair_j],
        "mean_abs_pearson": pearson_sum / len(available), "mean_abs_spearman": spearman_sum / len(available),
        "days_scored": pair_days, **coverage_metadata(),
    })
    pairwise.to_csv(repo_root / "results" / "redundancy" / "pairwise_redundancy.csv", index=False)
    model = pooled_pca
    cumulative = np.cumsum(model.explained_variance_ratio_)
    pca_rows.append({
        "pca_type": "pooled_incremental", "day": "pooled", "complete_rows": "", "sample_rows": pooled_fit_rows,
        "components_50pct": int(np.searchsorted(cumulative, .50) + 1) if cumulative[-1] >= .50 else np.nan,
        "components_80pct": int(np.searchsorted(cumulative, .80) + 1) if cumulative[-1] >= .80 else np.nan,
        "components_90pct": int(np.searchsorted(cumulative, .90) + 1) if cumulative[-1] >= .90 else np.nan,
        "variance_first_component": float(cumulative[0]), **coverage_metadata(),
    })
    pd.DataFrame(pca_rows).to_csv(output / "pca_summary.csv", index=False)
    summary = {
        "pair_count": len(pairwise), "pairs_abs_pearson_ge_0_9": int((pairwise.mean_abs_pearson >= .9).sum()),
        "pairs_abs_spearman_ge_0_9": int((pairwise.mean_abs_spearman >= .9).sum()),
        "median_abs_pearson": float(pairwise.mean_abs_pearson.median()),
        "median_abs_spearman": float(pairwise.mean_abs_spearman.median()),
        **coverage_metadata(),
    }
    (output / "redundancy_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    scope = {
        **coverage_metadata(), "available_day_ids": list(available), "missing_day_ids": list(range(65, 80)),
        "holdout_day_ids": list(range(86, 109)), "holdout_processed": False,
        "row_cap_per_day": ROW_CAP, "row_policy": "deterministic evenly spaced complete feature rows after per-day z-scoring",
        "nan_policy": "no imputation; PCA/correlation use complete rows only after z-scoring valid values per day",
        "pca": "per-day randomized PCA and pooled IncrementalPCA, 50 components retained",
    }
    (output / "phase10_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    print({"pair_rows": len(pairwise), "pca_rows": len(pca_rows), **summary})


if __name__ == "__main__":
    main()
