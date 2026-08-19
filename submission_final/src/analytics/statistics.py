"""Descriptive and day-local autocorrelation helpers."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def describe(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"count": 0, "mean": np.nan, "median": np.nan, "std": np.nan, "skew": np.nan, "excess_kurtosis": np.nan, "min": np.nan, "max": np.nan, **{f"q{int(q*100):02d}": np.nan for q in QUANTILES}}
    result: dict[str, float | int] = {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        "skew": float(skew(values, bias=False)) if len(values) > 2 else np.nan,
        "excess_kurtosis": float(kurtosis(values, fisher=True, bias=False)) if len(values) > 3 else np.nan,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    result.update({f"q{int(q*100):02d}": float(np.quantile(values, q)) for q in QUANTILES})
    return result


def acf_values(values: np.ndarray, max_lag: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return [np.nan] * max_lag
    centered = values - np.mean(values)
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return [np.nan] * max_lag
    return [float(np.dot(centered[:-lag], centered[lag:]) / denominator) for lag in range(1, max_lag + 1) if lag < len(values)] + [np.nan] * max(0, max_lag - len(values) + 1)
