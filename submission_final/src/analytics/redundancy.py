"""Redundancy and PCA helpers with explicit complete-row handling."""

from __future__ import annotations

import numpy as np


def day_zscore(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    result = x.copy()
    for column in range(x.shape[1]):
        valid = np.isfinite(x[:, column])
        if valid.any():
            average = np.mean(x[valid, column])
            scale = np.std(x[valid, column], ddof=1)
            result[valid, column] = (x[valid, column] - average) / scale if scale > 0 else 0.0
    return result


def deterministic_rows(rows: np.ndarray, cap: int) -> np.ndarray:
    if len(rows) <= cap:
        return rows
    indices = np.linspace(0, len(rows) - 1, cap, dtype=int)
    return rows[indices]
