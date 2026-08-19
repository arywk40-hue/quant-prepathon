"""Configurable price/return candidate formulas for masked-feature forensics."""

from __future__ import annotations

import numpy as np
import pandas as pd


PRICE_CANDIDATES = (
    "rolling_mean", "rolling_median", "rolling_std", "rolling_variance",
    "rolling_min", "rolling_max", "price_minus_mean", "normalized_deviation",
    "z_score", "momentum", "ema", "distance_from_high", "distance_from_low",
)
RETURN_CANDIDATES = (
    "rolling_return_mean", "realized_variance", "realized_volatility",
    "absolute_return_mean", "downside_volatility", "upside_volatility",
)
VOLUME_CANDIDATES = (
    "rolling_volume_mean", "rolling_volume_std", "volume_z_score", "volume_change",
    "price_volume_covariance", "imbalance_proxy",
)


def candidate_series(price: np.ndarray, name: str, window: int) -> np.ndarray:
    """Compute one candidate within one day; invalid warm-up remains NaN."""

    p = pd.Series(np.asarray(price, dtype=float))
    if window <= 0:
        raise ValueError("candidate window must be positive")
    rolling = p.rolling(window=window, min_periods=window)
    mean = rolling.mean()
    log_return = np.log(p).diff()
    return_rolling = log_return.rolling(window=window, min_periods=window)
    if name == "rolling_mean":
        result = mean
    elif name == "rolling_median":
        result = rolling.median()
    elif name == "rolling_std":
        result = rolling.std(ddof=1)
    elif name == "rolling_variance":
        result = rolling.var(ddof=1)
    elif name == "rolling_min":
        result = rolling.min()
    elif name == "rolling_max":
        result = rolling.max()
    elif name == "price_minus_mean":
        result = p - mean
    elif name == "normalized_deviation":
        result = p / mean - 1.0
    elif name == "z_score":
        result = (p - mean) / rolling.std(ddof=1)
    elif name == "momentum":
        result = p / p.shift(window) - 1.0
    elif name == "ema":
        result = p.ewm(span=window, min_periods=window, adjust=False).mean()
    elif name == "distance_from_high":
        result = p / rolling.max() - 1.0
    elif name == "distance_from_low":
        result = p / rolling.min() - 1.0
    elif name == "rolling_return_mean":
        result = return_rolling.mean()
    elif name == "realized_variance":
        result = return_rolling.var(ddof=1)
    elif name == "realized_volatility":
        result = return_rolling.std(ddof=1)
    elif name == "absolute_return_mean":
        result = log_return.abs().rolling(window=window, min_periods=window).mean()
    elif name == "downside_volatility":
        result = log_return.where(log_return < 0, 0.0).rolling(window=window, min_periods=window).std(ddof=1)
    elif name == "upside_volatility":
        result = log_return.where(log_return > 0, 0.0).rolling(window=window, min_periods=window).std(ddof=1)
    else:
        raise ValueError(f"unknown candidate {name}")
    return result.to_numpy(dtype=float)
