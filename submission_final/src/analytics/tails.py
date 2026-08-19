"""Distribution and tail diagnostics with transparent assumptions."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def sigma_probability(level: int) -> float:
    return float(2.0 * norm.sf(level))


def hill_tail_index(values: np.ndarray, k: int | None = None) -> tuple[float, int, float]:
    """Hill alpha estimate on positive magnitudes; descriptive, not a proof."""

    magnitudes = np.asarray(values, dtype=float)
    magnitudes = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0)]
    magnitudes = np.sort(magnitudes)[::-1]
    if k is None:
        k = max(10, min(1000, len(magnitudes) // 20))
    if len(magnitudes) <= k or k < 1:
        return np.nan, 0, np.nan
    threshold = magnitudes[k]
    logs = np.log(magnitudes[:k] / threshold)
    denominator = float(np.sum(logs))
    return (float(k / denominator) if denominator > 0 else np.nan, int(k), float(threshold))
