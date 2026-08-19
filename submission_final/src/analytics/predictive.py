"""Exact day-local forward-return alignment and multiple-testing helpers."""

from __future__ import annotations

import numpy as np
from scipy.stats import t
from statsmodels.stats.multitest import multipletests


def forward_indices(seconds: np.ndarray, horizon: int) -> np.ndarray:
    """Map each timestamp to the exact future timestamp within one day."""

    values = np.asarray(seconds, dtype=np.int64)
    lookup = {int(value): index for index, value in enumerate(values)}
    result = np.full(len(values), -1, dtype=np.int64)
    for index, value in enumerate(values):
        result[index] = lookup.get(int(value) + horizon, -1)
    return result


def correlation_pvalue(correlation: float, n: int) -> float:
    if not np.isfinite(correlation) or n < 3 or abs(correlation) >= 1:
        return 0.0 if np.isfinite(correlation) and abs(correlation) == 1 else np.nan
    statistic = correlation * np.sqrt((n - 2) / (1 - correlation**2))
    return float(2 * t.sf(abs(statistic), n - 2))


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(values)
    reject = np.zeros(len(values), dtype=bool)
    corrected = np.full(len(values), np.nan, dtype=float)
    if valid.any():
        r, q, _, _ = multipletests(values[valid], alpha=alpha, method="fdr_bh")
        reject[valid] = r
        corrected[valid] = q
    return reject, corrected
