"""Training-only predictive screen and model-ready partition construction."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import t, ttest_1samp
from src.common.day_boundary import parse_time_seconds

from .cache import write_partition
from .feature_selection import FROZEN_SCREEN_RULE
from .preprocessing import TrainOnlyStandardizer, complete_case_mask
from .schemas import DevelopmentScope
from .targets import build_future_return_target


def _correlation_pvalue(correlation: float, n: int) -> float:
    if not np.isfinite(correlation) or n < 3 or abs(correlation) >= 1:
        return 0.0 if np.isfinite(correlation) and abs(correlation) == 1 else np.nan
    statistic = correlation * np.sqrt((n - 2) / (1 - correlation**2))
    return float(2 * t.sf(abs(statistic), n - 2))


def _benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Dependency-free Benjamini–Hochberg correction matching the project rule."""

    values = np.asarray(pvalues, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    reject = np.zeros(len(values), dtype=bool)
    corrected = np.full(len(values), np.nan, dtype=float)
    if not len(valid_positions):
        return reject, corrected
    valid = values[valid_positions]
    order = np.argsort(valid, kind="mergesort")
    sorted_values = valid[order]
    ranks = np.arange(1, len(sorted_values) + 1, dtype=float)
    threshold = alpha * ranks / len(sorted_values)
    passing = np.flatnonzero(sorted_values <= threshold)
    if len(passing):
        cutoff = sorted_values[passing[-1]]
        reject[valid_positions] = valid <= cutoff
    sorted_q = sorted_values * len(sorted_values) / ranks
    sorted_q = np.minimum.accumulate(sorted_q[::-1])[::-1]
    corrected[valid_positions[order]] = np.minimum(sorted_q, 1.0)
    return reject, corrected


def _required_ic_columns() -> set[str]:
    return {"day", "feature", "horizon_seconds", "pair_count", "pearson_ic", "spearman_ic"}


def _daily_ic_rows(
    frame: pd.DataFrame,
    *,
    day: int,
    horizons: tuple[int, ...],
    scope: DevelopmentScope,
) -> list[dict[str, object]]:
    if "Time" not in frame or "Price" not in frame:
        raise ValueError(f"day {day} is missing Time or Price")
    features = [column for column in frame.columns if column not in {"Time", "Price"}]
    if not features:
        raise ValueError(f"day {day} has no candidate features")
    raw = frame[features]
    ranked_features = raw.rank(method="average")
    price = pd.to_numeric(frame["Price"], errors="coerce").to_numpy(dtype=float)
    seconds = np.asarray([parse_time_seconds(value) for value in frame["Time"]], dtype=np.int64)
    if len(seconds) > 1 and np.any(np.diff(seconds) <= 0):
        raise ValueError(f"day {day} timestamps are not strictly increasing")
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        by_second = {int(second): index for index, second in enumerate(seconds)}
        future_price = np.full(len(price), np.nan, dtype=float)
        for index, second in enumerate(seconds):
            future_index = by_second.get(int(second + horizon))
            if future_index is not None:
                future_price[index] = price[future_index]
        valid_target = np.isfinite(price) & (price > 0) & np.isfinite(future_price) & (future_price > 0)
        target = np.full(len(price), np.nan, dtype=float)
        target[valid_target] = future_price[valid_target] / price[valid_target] - 1.0
        target_series = pd.Series(target, index=raw.index)
        ranked_target = target_series.rank(method="average")
        pearson = raw.corrwith(target_series, method="pearson")
        spearman = ranked_features.corrwith(ranked_target, method="pearson")
        for feature in features:
            values = pd.to_numeric(raw[feature], errors="coerce").to_numpy(dtype=float)
            pair = valid_target & np.isfinite(values)
            n = int(pair.sum())
            p = float(pearson[feature]) if np.isfinite(pearson[feature]) else np.nan
            s = float(spearman[feature]) if np.isfinite(spearman[feature]) else np.nan
            rows.append({
                "day": int(day),
                "feature": feature,
                "horizon_seconds": int(horizon),
                "pair_count": n,
                "pearson_ic": p,
                "pearson_pvalue": _correlation_pvalue(p, n),
                "spearman_ic": s,
                "spearman_pvalue": _correlation_pvalue(s, n),
                **scope.as_dict(),
            })
    return rows


def compute_training_daily_ic(
    *,
    processed_dir: str | Path,
    training_days: Iterable[int],
    horizons: Iterable[int],
    scope: DevelopmentScope,
) -> pd.DataFrame:
    """Compute candidate ICs by reading only the explicitly supplied train days."""

    days = tuple(int(day) for day in training_days)
    horizons_tuple = tuple(sorted({int(horizon) for horizon in horizons}))
    if not days or not horizons_tuple:
        raise ValueError("training days and horizons must be non-empty")
    scope.assert_development_days(days)
    if any(day in scope.missing_development_days for day in days):
        raise ValueError("missing development days cannot be used for selection")
    rows: list[dict[str, object]] = []
    expected_features: tuple[str, ...] | None = None
    for day in days:
        path = Path(processed_dir) / f"day{day}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pq.read_table(path).to_pandas()
        current_features = tuple(column for column in frame.columns if column not in {"Time", "Price"})
        if expected_features is None:
            expected_features = current_features
        elif current_features != expected_features:
            raise ValueError(f"candidate feature universe changed on day {day}")
        rows.extend(_daily_ic_rows(frame, day=day, horizons=horizons_tuple, scope=scope))
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("training-only IC computation produced no rows")
    return result.sort_values(["feature", "horizon_seconds", "day"], kind="stable").reset_index(drop=True)


def load_training_daily_ic(
    path: str | Path,
    *,
    training_days: Iterable[int],
    horizons: Iterable[int],
    scope: DevelopmentScope,
) -> pd.DataFrame:
    """Load only training-day rows from the audited day-level IC artifact.

    The file may also contain validation-day rows. They are parsed only to
    advance the CSV reader and are never retained or passed to selection.
    Holdout rows are rejected if present.
    """

    days = {int(day) for day in training_days}
    horizon_set = {int(horizon) for horizon in horizons}
    scope.assert_development_days(days)
    rows: list[dict[str, object]] = []
    required = _required_ic_columns()
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = required - set(reader.fieldnames or ())
            raise ValueError(f"daily IC artifact missing columns: {sorted(missing)}")
        for raw in reader:
            day = int(raw["day"])
            if day in scope.holdout_days:
                raise ValueError("holdout day present in daily IC artifact")
            if day not in days:
                continue
            horizon = int(raw["horizon_seconds"])
            if horizon not in horizon_set:
                continue
            rows.append({
                "day": day,
                "feature": raw["feature"],
                "horizon_seconds": horizon,
                "pair_count": int(raw["pair_count"]),
                "pearson_ic": float(raw["pearson_ic"]),
                "spearman_ic": float(raw["spearman_ic"]),
            })
    if not rows:
        raise ValueError("training-day IC artifact produced no rows")
    return pd.DataFrame(rows).sort_values(["feature", "horizon_seconds", "day"], kind="stable").reset_index(drop=True)


def fit_training_only_screen(
    daily_ic: pd.DataFrame,
    *,
    training_days: Iterable[int],
    target_horizon: int,
    scope: DevelopmentScope,
    fdr_alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit the existing screen rule using training-day ICs only.

    FDR is applied across every candidate feature-horizon hypothesis, matching
    the existing Part-4 aggregate-screen structure. The returned second frame
    is the selected target-horizon feature list.
    """

    missing = _required_ic_columns() - set(daily_ic.columns)
    if missing:
        raise ValueError(f"daily IC table missing columns: {sorted(missing)}")
    days = tuple(int(day) for day in training_days)
    if not days:
        raise ValueError("training days must be non-empty")
    observed_days = set(daily_ic["day"].astype(int))
    unexpected = observed_days - set(days)
    if unexpected:
        raise ValueError(f"selection received non-training days: {sorted(unexpected)}")
    if any(day in scope.holdout_days for day in observed_days):
        raise ValueError("holdout day entered training-only selection")
    if any(day in scope.missing_development_days for day in observed_days):
        raise ValueError("missing development day entered training-only selection")

    grouped: list[dict[str, object]] = []
    for (feature, horizon), group in daily_ic.groupby(["feature", "horizon_seconds"], sort=True):
        group_days = tuple(sorted(group["day"].astype(int).unique()))
        if group_days != tuple(sorted(days)):
            raise ValueError(f"incomplete training-day IC coverage for {feature}/{horizon}")
        pearson = group["pearson_ic"].to_numpy(dtype=float)
        spearman = group["spearman_ic"].to_numpy(dtype=float)
        if not np.isfinite(pearson).all() or not np.isfinite(spearman).all():
            raise ValueError(f"non-finite IC in training selection for {feature}/{horizon}")
        ptest = ttest_1samp(pearson, 0.0)
        stest = ttest_1samp(spearman, 0.0)
        mean_p = float(np.mean(pearson))
        mean_s = float(np.mean(spearman))
        grouped.append({
            "feature": feature,
            "horizon_seconds": int(horizon),
            "days_scored": int(len(pearson)),
            "mean_pearson_ic": mean_p,
            "pearson_ic_std": float(np.std(pearson, ddof=1)) if len(pearson) > 1 else np.nan,
            "pearson_pct_same_sign": float(np.mean(np.sign(pearson) == np.sign(mean_p))),
            "pearson_t_pvalue": float(ptest.pvalue),
            "mean_spearman_ic": mean_s,
            "spearman_ic_std": float(np.std(spearman, ddof=1)) if len(spearman) > 1 else np.nan,
            "spearman_pct_same_sign": float(np.mean(np.sign(spearman) == np.sign(mean_s))),
            "spearman_t_pvalue": float(stest.pvalue),
            "mean_pair_count": float(group["pair_count"].mean()),
            **scope.as_dict(),
        })
    aggregate = pd.DataFrame(grouped).sort_values(["feature", "horizon_seconds"], kind="stable").reset_index(drop=True)
    reject, qvalue = _benjamini_hochberg(aggregate["pearson_t_pvalue"].to_numpy(), fdr_alpha)
    aggregate["pearson_fdr_reject"] = reject
    aggregate["pearson_fdr_qvalue"] = qvalue
    reject, qvalue = _benjamini_hochberg(aggregate["spearman_t_pvalue"].to_numpy(), fdr_alpha)
    aggregate["spearman_fdr_reject"] = reject
    aggregate["spearman_fdr_qvalue"] = qvalue
    aggregate["eligible_for_ml"] = (
        aggregate["pearson_fdr_reject"]
        & (aggregate["pearson_pct_same_sign"] >= 0.70)
        & (aggregate["mean_pearson_ic"].abs() >= 0.05)
    )
    aggregate["screen_rule"] = FROZEN_SCREEN_RULE
    selected = aggregate[(aggregate["horizon_seconds"] == int(target_horizon)) & aggregate["eligible_for_ml"]].copy()
    selected = selected.sort_values("feature", kind="stable").reset_index(drop=True)
    return aggregate, selected


def _read_model_day(processed_dir: Path, day: int, features: tuple[str, ...], scope: DevelopmentScope) -> pd.DataFrame:
    scope.assert_development_days([day])
    path = processed_dir / f"day{day}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(path, columns=["Time", "Price", *features]).to_pandas()


def _counts(target_valid: np.ndarray, feature_valid: np.ndarray, complete: np.ndarray, rows: int) -> dict[str, int]:
    return {
        "source_rows": int(rows),
        "valid_target_count": int(target_valid.sum()),
        "valid_model_rows": int(complete.sum()),
        "excluded_row_count": int(rows - complete.sum()),
        "invalid_target_rows": int((~target_valid).sum()),
        "invalid_feature_rows": int((target_valid & ~feature_valid).sum()),
    }


def build_training_only_partitions(
    *,
    processed_dir: str | Path,
    output_root: str | Path,
    training_days: Iterable[int],
    validation_days: Iterable[int],
    features: tuple[str, ...],
    target_horizon: int,
    scope: DevelopmentScope,
) -> tuple[TrainOnlyStandardizer, list[dict[str, object]]]:
    """Fit preprocessing on train days and emit isolated train/validation data."""

    train = tuple(int(day) for day in training_days)
    validation = tuple(int(day) for day in validation_days)
    if set(train) & set(validation) or max(train) >= min(validation):
        raise ValueError("training and validation days must be disjoint and chronological")
    scope.assert_development_days((*train, *validation))
    processed = Path(processed_dir)
    root = Path(output_root)
    scaler = TrainOnlyStandardizer(features)
    for day in train:
        frame = _read_model_day(processed, day, features, scope)
        target = build_future_return_target(frame, target_horizon)
        _, _, complete = complete_case_mask(frame, features, target)
        scaler.update(frame.loc[complete, list(features)])
    scaler.finalize()

    reports: list[dict[str, object]] = []
    for day in (*train, *validation):
        frame = _read_model_day(processed, day, features, scope)
        target = build_future_return_target(frame, target_horizon)
        target_valid, feature_valid, complete = complete_case_mask(frame, features, target)
        transformed = scaler.transform(frame.loc[complete, list(features)])
        seconds = np.asarray([parse_time_seconds(value) for value in frame.loc[complete, "Time"]], dtype=np.int64)
        partition = pd.concat([
            pd.DataFrame({
                "day": np.full(len(transformed), day, dtype=np.int16),
                "timestamp": frame.loc[complete, "Time"].astype(str).to_numpy(),
                "timestamp_seconds": seconds,
                "target": target.loc[complete].to_numpy(dtype=np.float32),
            }),
            transformed.reset_index(drop=True),
        ], axis=1)
        split = "train" if day in train else "validation"
        path = root / "datasets" / split / f"day{day}.parquet"
        write_partition(partition, path)
        reports.append({"split": split, "day": day, "path": str(path), **_counts(target_valid, feature_valid, complete, len(frame))})
    return scaler, reports
