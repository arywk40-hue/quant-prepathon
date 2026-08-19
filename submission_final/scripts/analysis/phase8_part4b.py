"""Phase 8: reverse-engineer masked features against fixed candidate formulas."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analytics.candidates import (  # noqa: E402
    PRICE_CANDIDATES,
    RETURN_CANDIDATES,
    VOLUME_CANDIDATES,
    candidate_series,
)
from src.analytics.coverage import AVAILABLE_DEVELOPMENT_DAYS, available_days_from_manifest, coverage_metadata  # noqa: E402


SAMPLE_STRIDE = 5


def _scores(target: np.ndarray, candidate: np.ndarray) -> dict[str, float] | None:
    valid = np.isfinite(target) & np.isfinite(candidate)
    x, y = candidate[valid], target[valid]
    if len(x) < 20 or np.std(x) == 0 or np.std(y) == 0:
        return None
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    beta = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
    fitted = np.mean(y) + beta * (x - np.mean(x))
    nrmse = float(np.sqrt(np.mean((y - fitted) ** 2)) / np.std(y))
    dx, dy = np.diff(x), np.diff(y)
    first_diff = float(np.corrcoef(dx, dy)[0, 1]) if np.std(dx) and np.std(dy) else np.nan
    sign_agreement = float(np.mean(np.sign(dx) == np.sign(dy))) if len(dx) else np.nan
    lagged = float(np.corrcoef(x[:-1], y[1:])[0, 1]) if len(x) > 2 and np.std(x[:-1]) and np.std(y[1:]) else np.nan
    return {
        "n": len(x), "pearson": pearson, "spearman": spearman,
        "normalized_rmse": nrmse, "first_difference_corr": first_diff,
        "sign_agreement": sign_agreement, "lag1_corr": lagged,
    }


def _tier(row: pd.Series) -> str:
    p, s, e, sign = abs(row.mean_pearson), abs(row.mean_spearman), row.mean_normalized_rmse, row.mean_sign_agreement
    if p >= .90 and s >= .90 and e <= .50 and sign >= .80 and row.days_scored >= 50:
        return "strong evidence"
    if p >= .70 and s >= .70 and e <= .80 and sign >= .65 and row.days_scored >= 35:
        return "moderate evidence"
    if p >= .40 and s >= .40 and row.days_scored >= 20:
        return "weak evidence"
    return "no convincing match"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    available = available_days_from_manifest(repo_root)
    if available != AVAILABLE_DEVELOPMENT_DAYS:
        raise RuntimeError("available development universe changed")
    taxonomy = pd.read_csv(repo_root / "results" / "features" / "feature_taxonomy.csv")
    price_features = taxonomy.loc[taxonomy.family.isin(["PB", "BB"]), "feature"].tolist()
    windows = dict(zip(taxonomy.feature, pd.to_numeric(taxonomy.nominal_window_seconds, errors="coerce")))
    keys = [(feature, candidate) for feature in price_features for candidate in (*PRICE_CANDIDATES, *RETURN_CANDIDATES)]
    stats = {key: {"n": 0, "days": 0, "sum": {k: 0.0 for k in ("pearson", "spearman", "normalized_rmse", "first_difference_corr", "sign_agreement", "lag1_corr")}, "sum2": {k: 0.0 for k in ("pearson", "spearman", "normalized_rmse", "first_difference_corr", "sign_agreement", "lag1_corr")}} for key in keys}
    daily_best = []
    for day in available:
        columns = ["Price"] + price_features
        frame = pq.read_table(repo_root / "data" / "processed" / f"day{day}.parquet", columns=columns).to_pandas()
        price = frame.Price.to_numpy(dtype=float)
        target_cache = {feature: frame[feature].to_numpy(dtype=float)[::SAMPLE_STRIDE] for feature in price_features}
        candidate_cache = {}
        day_score_cache = {}
        for feature in price_features:
            window = windows.get(feature)
            if not np.isfinite(window):
                continue
            window = int(window)
            for candidate in (*PRICE_CANDIDATES, *RETURN_CANDIDATES):
                key_cache = (candidate, window)
                if key_cache not in candidate_cache:
                    candidate_cache[key_cache] = candidate_series(price, candidate, window)[::SAMPLE_STRIDE]
                scores = _scores(target_cache[feature], candidate_cache[key_cache])
                if scores is None:
                    continue
                day_score_cache[(feature, candidate)] = scores
                key = (feature, candidate)
                stats[key]["n"] += scores["n"]
                stats[key]["days"] += 1
                for metric in stats[key]["sum"]:
                    value = scores[metric]
                    if np.isfinite(value):
                        stats[key]["sum"][metric] += value
                        stats[key]["sum2"][metric] += value * value
        # Select best by absolute Pearson/Spearman after this day for diagnostics.
        day_rows = []
        for feature in price_features:
            candidates = []
            for candidate in (*PRICE_CANDIDATES, *RETURN_CANDIDATES):
                window = windows.get(feature)
                if np.isfinite(window):
                    scored = day_score_cache.get((feature, candidate))
                    if scored is not None:
                        candidates.append((abs(scored["pearson"]) + abs(scored["spearman"]), candidate))
            if candidates:
                day_rows.append({"day": day, "feature": feature, "best_candidate": max(candidates)[1]})
        daily_best.extend(day_rows)
        print(f"processed day {day}", flush=True)
    rows = []
    metrics = ("pearson", "spearman", "normalized_rmse", "first_difference_corr", "sign_agreement", "lag1_corr")
    for (feature, candidate), value in stats.items():
        row = {"feature": feature, "candidate": candidate, "candidate_status": "scored", "days_scored": value["days"], "observations_scored": value["n"]}
        for metric in metrics:
            mean = value["sum"][metric] / value["days"] if value["days"] else np.nan
            row[f"mean_{metric}"] = mean
            row[f"std_{metric}"] = np.sqrt(max(value["sum2"][metric] / value["days"] - mean * mean, 0.0)) if value["days"] else np.nan
        rows.append(row)
    # Volume-dependent formulas are explicit unavailable records, never silently treated as non-matches.
    for feature in taxonomy.loc[~taxonomy.family.isin(["PB", "BB"]), "feature"]:
        for candidate in VOLUME_CANDIDATES:
            rows.append({"feature": feature, "candidate": candidate, "candidate_status": "unavailable_raw_volume", "days_scored": 0, "observations_scored": 0})
    scores = pd.DataFrame(rows)
    scores.to_csv(repo_root / "results" / "features" / "candidate_scores.csv", index=False)
    scored = scores[scores.candidate_status == "scored"].copy()
    scored["evidence_tier"] = scored.apply(_tier, axis=1)
    pd.DataFrame(daily_best).to_csv(repo_root / "results" / "features" / "daily_best_candidates.csv", index=False)
    best = scored.sort_values(["feature", "mean_spearman"], key=lambda col: col.abs() if col.name == "mean_spearman" else col, ascending=False).drop_duplicates("feature")
    best[["feature", "candidate", "mean_pearson", "mean_spearman", "mean_normalized_rmse", "days_scored", "evidence_tier"]].to_csv(repo_root / "results" / "features" / "candidate_best_matches.csv", index=False)
    scope = {
        **coverage_metadata(), "available_day_ids": list(available), "missing_day_ids": list(range(65, 80)),
        "holdout_day_ids": list(range(86, 109)), "holdout_processed": False,
        "candidate_library": {"price": list(PRICE_CANDIDATES), "return": list(RETURN_CANDIDATES), "volume": list(VOLUME_CANDIDATES)},
        "source_policy": "price and return candidates only where validated Price exists; volume candidates unavailable because raw volume semantics are not validated",
        "sample_stride_seconds": SAMPLE_STRIDE,
        "evidence_thresholds": {"strong": {"abs_corr": .90, "nrmse": .50, "sign": .80, "days": 50}, "moderate": {"abs_corr": .70, "nrmse": .80, "sign": .65, "days": 35}, "weak": {"abs_corr": .40, "days": 20}},
        "match_policy": "candidate matches are hypotheses supported by multiple metrics and cross-day stability; no identity is confirmed",
    }
    (repo_root / "results" / "features" / "phase8_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    print({"score_rows": len(scores), "best_rows": len(best), "daily_best_rows": len(daily_best), "tiers": scored.evidence_tier.value_counts().to_dict()})


if __name__ == "__main__":
    main()
