"""Phase 9: forward-return predictive relevance with day-local FDR."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analytics.coverage import AVAILABLE_DEVELOPMENT_DAYS, available_days_from_manifest, coverage_metadata  # noqa: E402
from src.analytics.predictive import benjamini_hochberg, correlation_pvalue, forward_indices  # noqa: E402
from src.common.day_boundary import parse_time_seconds  # noqa: E402


HORIZONS = (1, 5, 30, 60, 300)
FDR_ALPHA = 0.05


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    available = available_days_from_manifest(repo_root)
    if available != AVAILABLE_DEVELOPMENT_DAYS:
        raise RuntimeError("available development universe changed")
    per_day = []
    for day in available:
        frame = pq.read_table(repo_root / "data" / "processed" / f"day{day}.parquet").to_pandas()
        features = [column for column in frame.columns if column not in {"Time", "Price"}]
        seconds = np.asarray([parse_time_seconds(value) for value in frame["Time"]], dtype=np.int64)
        # Ranking once per day preserves pairwise NaN handling while avoiding a
        # second rank operation for every horizon.
        ranked_features = frame[features].rank(method="average")
        price = frame["Price"].to_numpy(dtype=float)
        for horizon in HORIZONS:
            future = forward_indices(seconds, horizon)
            pair = future >= 0
            y = np.full(len(price), np.nan, dtype=float)
            valid_price = pair & np.isfinite(price) & (price > 0)
            future_price = np.full(len(price), np.nan, dtype=float)
            future_price[valid_price] = price[future[valid_price]]
            valid_price &= np.isfinite(future_price) & (future_price > 0)
            y[valid_price] = future_price[valid_price] / price[valid_price] - 1.0
            raw = frame[features]
            pearson = raw.corrwith(pd.Series(y, index=raw.index), method="pearson")
            spearman = ranked_features.corrwith(pd.Series(y, index=raw.index).rank(method="average"), method="pearson")
            for feature in features:
                mask = valid_price & raw[feature].notna().to_numpy() & np.isfinite(raw[feature].to_numpy(dtype=float))
                n = int(mask.sum())
                p = float(pearson[feature]) if np.isfinite(pearson[feature]) else np.nan
                s = float(spearman[feature]) if np.isfinite(spearman[feature]) else np.nan
                per_day.append({
                    "day": day, "feature": feature, "horizon_seconds": horizon,
                    "pair_count": n, "pearson_ic": p, "pearson_pvalue": correlation_pvalue(p, n),
                    "spearman_ic": s, "spearman_pvalue": correlation_pvalue(s, n),
                    **coverage_metadata(),
                })
        print(f"processed day {day}", flush=True)
    detail = pd.DataFrame(per_day)
    output = repo_root / "results" / "predictive"
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "per_day_ic.csv", index=False)
    grouped = []
    for (feature, horizon), group in detail.groupby(["feature", "horizon_seconds"], sort=True):
        pvalues = group.pearson_ic.dropna().to_numpy()
        svalues = group.spearman_ic.dropna().to_numpy()
        ptest = ttest_1samp(pvalues, 0.0) if len(pvalues) >= 2 else (np.nan, np.nan)
        stest = ttest_1samp(svalues, 0.0) if len(svalues) >= 2 else (np.nan, np.nan)
        grouped.append({
            "feature": feature, "horizon_seconds": horizon, "days_scored": len(pvalues),
            "mean_pearson_ic": np.mean(pvalues) if len(pvalues) else np.nan,
            "pearson_ic_std": np.std(pvalues, ddof=1) if len(pvalues) > 1 else np.nan,
            "pearson_pct_same_sign": np.mean(np.sign(pvalues) == np.sign(np.mean(pvalues))) if len(pvalues) else np.nan,
            "pearson_t_pvalue": float(ptest.pvalue) if hasattr(ptest, "pvalue") else float(ptest[1]),
            "mean_spearman_ic": np.mean(svalues) if len(svalues) else np.nan,
            "spearman_ic_std": np.std(svalues, ddof=1) if len(svalues) > 1 else np.nan,
            "spearman_pct_same_sign": np.mean(np.sign(svalues) == np.sign(np.mean(svalues))) if len(svalues) else np.nan,
            "spearman_t_pvalue": float(stest.pvalue) if hasattr(stest, "pvalue") else float(stest[1]),
            "mean_pair_count": group.pair_count.mean(),
            **coverage_metadata(),
        })
    aggregate = pd.DataFrame(grouped)
    reject, q = benjamini_hochberg(aggregate.pearson_t_pvalue.to_numpy(), FDR_ALPHA)
    aggregate["pearson_fdr_reject"] = reject
    aggregate["pearson_fdr_qvalue"] = q
    reject, q = benjamini_hochberg(aggregate.spearman_t_pvalue.to_numpy(), FDR_ALPHA)
    aggregate["spearman_fdr_reject"] = reject
    aggregate["spearman_fdr_qvalue"] = q
    aggregate.to_csv(output / "aggregate_ic.csv", index=False)
    scope = {
        **coverage_metadata(), "available_day_ids": list(available), "missing_day_ids": list(range(65, 80)),
        "holdout_day_ids": list(range(86, 109)), "holdout_processed": False,
        "horizons_seconds": list(HORIZONS), "alignment": "feature(t) paired only with exact timestamp t+h within the same day",
        "fdr_alpha": FDR_ALPHA, "fdr_method": "Benjamini-Hochberg separately for Pearson and Spearman aggregate tests",
        "missing_value_policy": "pairwise valid observations only; no imputation; pair counts emitted",
    }
    (output / "phase9_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    print({"per_day_rows": len(detail), "aggregate_rows": len(aggregate), "pearson_rejections": int(aggregate.pearson_fdr_reject.sum()), "spearman_rejections": int(aggregate.spearman_fdr_reject.sum())})


if __name__ == "__main__":
    main()
