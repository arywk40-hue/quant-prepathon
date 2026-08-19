#!/usr/bin/env python3
"""PHASE 5 — Part 2 distribution and tail analysis."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import anderson, jarque_bera, norm, probplot

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analytics.coverage import available_days_from_manifest, coverage_metadata, load_price_day
from src.analytics.returns import clock_seconds, day_one_second_returns
from src.analytics.statistics import describe
from src.analytics.tails import hill_tail_index, sigma_probability


HORIZONS = {"1m": 60, "5m": 300}
SIGMA_LEVELS = (1, 2, 3)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def coverage(row: dict) -> dict:
    return {**coverage_metadata(), **row}


def horizon_records(prices: np.ndarray, times, horizon: int):
    seconds = clock_seconds(times)
    current = prices[horizon:]
    previous = prices[:-horizon]
    aligned = (seconds[horizon:] - seconds[:-horizon]) == horizon
    valid = aligned & np.isfinite(current) & np.isfinite(previous) & (current > 0) & (previous > 0)
    return seconds[horizon:][valid], previous[valid], current[valid], current[valid] / previous[valid] - 1.0


def normality_rows(scope: str, day: int | str, horizon: str, values: np.ndarray) -> list[dict]:
    values = values[np.isfinite(values)]
    stats = describe(values)
    result = []
    if len(values) >= 8:
        jb = jarque_bera(values)
        result.append(coverage({"scope": scope, "day": day, "horizon": horizon, "test": "jarque_bera", "statistic": float(jb.statistic), "p_value": float(jb.pvalue), "critical_value_5pct": np.nan, "reject_5pct": bool(jb.pvalue < 0.05), "n": len(values), "excess_kurtosis": stats["excess_kurtosis"]}))
        ad = anderson(values, dist="norm")
        critical = float(ad.critical_values[np.argmin(np.abs(np.asarray(ad.significance_level) - 5.0))])
        result.append(coverage({"scope": scope, "day": day, "horizon": horizon, "test": "anderson_darling", "statistic": float(ad.statistic), "p_value": np.nan, "critical_value_5pct": critical, "reject_5pct": bool(ad.statistic > critical), "n": len(values), "excess_kurtosis": stats["excess_kurtosis"], "significance_levels": "|".join(map(str, ad.significance_level))}))
    return result


def run(repo_root: Path) -> dict[str, object]:
    days = available_days_from_manifest(repo_root)
    distributions: dict[tuple[int, str], np.ndarray] = {}
    pooled: dict[str, list[np.ndarray]] = {h: [] for h in HORIZONS}
    normality: list[dict] = []
    sigma_rows: list[dict] = []
    hills: list[dict] = []
    extremes: list[dict] = []
    for day in days:
        frame = load_price_day(repo_root, day)
        prices = frame["Price"].to_numpy(dtype=float)
        times = frame["Time"].to_numpy()
        for horizon_name, horizon in HORIZONS.items():
            seconds, before, after, values = horizon_records(prices, times, horizon)
            distributions[(day, horizon_name)] = values
            pooled[horizon_name].append(values)
            normality.extend(normality_rows("day", day, horizon_name, values))
            stats = describe(values)
            std = float(stats["std"])
            center = float(stats["mean"])
            for level in SIGMA_LEVELS:
                empirical = float(np.mean(np.abs(values - center) > level * std)) if len(values) and std > 0 else np.nan
                theoretical = sigma_probability(level)
                sigma_rows.append(coverage({"scope": "day", "day": day, "horizon": horizon_name, "sigma_level": level, "theoretical_probability": theoretical, "empirical_probability": empirical, "empirical_theoretical_ratio": empirical / theoretical if theoretical else np.nan, "n": len(values), "fit_mean": center, "fit_std": std}))
            alpha, k, threshold = hill_tail_index(np.abs(values - center))
            hills.append(coverage({"scope": "day", "day": day, "horizon": horizon_name, "tail_variable": "absolute_centered_return", "hill_alpha": alpha, "k": k, "threshold": threshold, "n": len(values), "assumption_note": "descriptive estimator; iid and regular-variation assumptions not established"}))
            if horizon_name == "1m":
                one_second, _, one_seconds = day_one_second_returns(prices, times)
                one_vol = []
                for second in seconds:
                    idx = int(np.searchsorted(one_seconds, second, side="right"))
                    local = one_second[max(0, idx - 60):idx]
                    one_vol.append(float(np.std(local, ddof=1)) if len(local) > 1 else np.nan)
                for timestamp, p_before, p_after, value, vol in zip(seconds, before, after, values, one_vol):
                    extremes.append(coverage({"day": day, "timestamp_seconds": int(timestamp), "return": float(value), "abs_return": abs(float(value)), "price_before": float(p_before), "price_after": float(p_after), "rolling_1s_volatility_60": vol, "volume_context_status": "not_run_no_validated_volume_semantics"}))

    for horizon_name, arrays in pooled.items():
        values = np.concatenate(arrays)
        normality.extend(normality_rows("pooled_available_days", "pooled", horizon_name, values))
        stats = describe(values); std = float(stats["std"]); center = float(stats["mean"])
        for level in SIGMA_LEVELS:
            empirical = float(np.mean(np.abs(values - center) > level * std))
            theoretical = sigma_probability(level)
            sigma_rows.append(coverage({"scope": "pooled_available_days", "day": "pooled", "horizon": horizon_name, "sigma_level": level, "theoretical_probability": theoretical, "empirical_probability": empirical, "empirical_theoretical_ratio": empirical / theoretical, "n": len(values), "fit_mean": center, "fit_std": std}))
        alpha, k, threshold = hill_tail_index(np.abs(values - center))
        hills.append(coverage({"scope": "pooled_available_days", "day": "pooled", "horizon": horizon_name, "tail_variable": "absolute_centered_return", "hill_alpha": alpha, "k": k, "threshold": threshold, "n": len(values), "assumption_note": "descriptive estimator; iid and regular-variation assumptions not established"}))
        figure_dir = repo_root / "figures" / "part2"; figure_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4)); plt.hist(values, bins=200, density=True, alpha=0.6); x=np.linspace(np.quantile(values, .001), np.quantile(values, .999), 500); plt.plot(x, norm.pdf(x, center, std)); plt.title(f"{horizon_name} returns: empirical vs fitted normal — 70 days"); plt.tight_layout(); plt.savefig(figure_dir / f"{horizon_name}_histogram_normal.png", dpi=140); plt.close()
        plt.figure(figsize=(6, 6)); probplot(values, dist="norm", plot=plt); plt.title(f"{horizon_name} return QQ plot — 70 days"); plt.tight_layout(); plt.savefig(figure_dir / f"{horizon_name}_qq.png", dpi=140); plt.close()

    extremes = sorted(extremes, key=lambda row: row["abs_return"], reverse=True)[:20]
    out = repo_root / "results" / "distributions"
    write_csv(out / "normality_tests.csv", normality, list(normality[0]))
    write_csv(out / "sigma_events.csv", sigma_rows, list(sigma_rows[0]))
    write_csv(out / "tail_estimates.csv", hills, list(hills[0]))
    write_csv(out / "extreme_events.csv", extremes, list(extremes[0]))
    (out / "phase5_scope.txt").write_text("PHASE 5 SCOPE\nexpected_development_days=85\navailable_development_days=70\nmissing_development_days=65-79\nholdout_days=86-108; not opened or used\nnormality_tests=Jarque-Bera and Anderson-Darling\nvolume_context=not_run_without_validated_volume_semantics\n")
    return {"normality_rows": len(normality), "sigma_rows": len(sigma_rows), "hill_rows": len(hills), "extreme_rows": len(extremes)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2]); args=parser.parse_args(); print(run(args.repo_root.resolve())); return 0


if __name__ == "__main__":
    raise SystemExit(main())
