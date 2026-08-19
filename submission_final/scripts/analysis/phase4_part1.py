#!/usr/bin/env python3
"""PHASE 4 — Part 1 data hygiene on the available development days."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analytics.coverage import available_days_from_manifest, coverage_metadata, load_price_day
from src.analytics.returns import day_returns, day_one_second_returns
from src.analytics.statistics import acf_values, describe


HORIZONS = {"1s": 1, "1m": 60, "5m": 300}
ACF_MAX_LAG = 60
SEASONAL_BIN_SECONDS = 300


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def coverage_row(row: dict) -> dict:
    return {**coverage_metadata(), **row}


def run(repo_root: Path) -> dict[str, object]:
    days = available_days_from_manifest(repo_root)
    quality_dir = repo_root / "results" / "quality"
    diagnostics_dir = repo_root / "results" / "diagnostics"
    figures_dir = repo_root / "figures" / "part1"
    descriptive_rows: list[dict] = []
    acf_rows: list[dict] = []
    seasonality: dict[int, list[float]] = defaultdict(list)
    pooled: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    day_count = 0

    for day in days:
        frame = load_price_day(repo_root, day)
        times = frame["Time"].to_numpy()
        prices = frame["Price"].to_numpy(dtype=float)
        price_stats = describe(prices)
        pooled[("price", "level")].append(prices)
        descriptive_rows.append(coverage_row({"scope": "day", "day": day, "variable": "price", "horizon": "level", **price_stats}))
        for label, horizon in HORIZONS.items():
            simple, log_return = day_returns(prices, times, horizon)
            for return_name, values in (("simple_return", simple), ("log_return", log_return)):
                pooled[(return_name, label)].append(values)
                descriptive_rows.append(coverage_row({"scope": "day", "day": day, "variable": return_name, "horizon": label, **describe(values)}))
        one_second, _, return_seconds = day_one_second_returns(prices, times)
        acf = acf_values(one_second, ACF_MAX_LAG)
        for lag, value in enumerate(acf, start=1):
            acf_rows.append(coverage_row({"scope": "day", "day": day, "return_horizon": "1s", "lag_seconds": lag, "acf": value, "n_returns": len(one_second)}))
        if len(one_second):
            bins = return_seconds // SEASONAL_BIN_SECONDS
            for bin_id in sorted(set(bins)):
                values = one_second[bins == bin_id]
                if len(values):
                    seasonality[int(bin_id)].append(float(np.std(values, ddof=1)) if len(values) > 1 else np.nan)
        day_count += 1

    for (variable, horizon), arrays in sorted(pooled.items()):
        values = np.concatenate(arrays) if arrays else np.array([])
        descriptive_rows.append(coverage_row({"scope": "pooled_available_days", "day": "pooled", "variable": variable, "horizon": horizon, **describe(values)}))

    for lag in range(1, ACF_MAX_LAG + 1):
        values = np.array([row["acf"] for row in acf_rows if row["lag_seconds"] == lag], dtype=float)
        acf_rows.append(coverage_row({"scope": "mean_across_available_days", "day": "pooled", "return_horizon": "1s", "lag_seconds": lag, "acf": float(np.nanmean(values)) if np.isfinite(values).any() else np.nan, "acf_std_across_days": float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else np.nan, "n_returns": int(np.nansum([row["n_returns"] for row in acf_rows if row["lag_seconds"] == lag]))}))

    seasonality_rows = []
    for bin_id in sorted(seasonality):
        values = np.asarray(seasonality[bin_id], dtype=float)
        seasonality_rows.append(coverage_row({"scope": "pooled_available_days", "bin_seconds": bin_id * SEASONAL_BIN_SECONDS, "bin_end_seconds": (bin_id + 1) * SEASONAL_BIN_SECONDS - 1, "day_count": int(np.isfinite(values).sum()), "mean_realized_1s_volatility": float(np.nanmean(values)) if np.isfinite(values).any() else np.nan, "std_across_days": float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else np.nan}))

    write_csv(quality_dir / "descriptive_stats.csv", descriptive_rows, list(descriptive_rows[0]))
    write_csv(diagnostics_dir / "acf_returns.csv", acf_rows, list(acf_rows[0]))
    write_csv(diagnostics_dir / "volatility_seasonality.csv", seasonality_rows, list(seasonality_rows[0]))
    volume_status = [coverage_row({"status": "not_run", "reason": "no volume-like feature semantics validated before Part 1; no unsupported volume claim made"})]
    write_csv(diagnostics_dir / "volume_seasonality.csv", volume_status, list(volume_status[0]))
    x = [row["bin_seconds"] / 3600 for row in seasonality_rows]
    y = [row["mean_realized_1s_volatility"] for row in seasonality_rows]
    plt.figure(figsize=(10, 4))
    plt.plot(x, y)
    plt.xlabel("Seconds since day start / hours")
    plt.ylabel("Mean within-day 1s return volatility")
    plt.title("Part 1 volatility seasonality — 70 available development days")
    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "volatility_seasonality.png", dpi=140)
    plt.close()
    (quality_dir / "phase4_scope.txt").write_text(
        "PHASE 4 SCOPE\nexpected_development_days=85\navailable_development_days=70\nmissing_development_days=65-79\n"
        "holdout_days=86-108; not opened or used\ncalculations=within-day only; no cross-day lag/return operations\n"
        f"processed_days={day_count}\nvolume_analysis=not_run_without_validated_volume_semantics\n"
    )
    (quality_dir / "cleaning_policy.txt").write_text(
        "PHASE 4 CLEANING POLICY\n"
        "Raw CSVs remain untouched. Structural leading NaNs are preserved.\n"
        "Unexpected internal/trailing feature missingness is not imputed; feature-specific validity masks determine usable observations.\n"
        "Price observations are used only when finite and positive for return construction; invalid observations are excluded from that calculation and never silently replaced.\n"
        "Days with shorter sessions remain separate day records. No day concatenation is used for lagged calculations.\n"
        "Volume seasonality is not run because no volume-like feature semantics were validated before Part 1.\n"
    )
    return {"days": list(days), "descriptive_rows": len(descriptive_rows), "acf_rows": len(acf_rows), "seasonality_rows": len(seasonality_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(run(args.repo_root.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
