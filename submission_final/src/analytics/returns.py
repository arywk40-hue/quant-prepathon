"""Day-local return construction with explicit timestamp alignment."""

from __future__ import annotations

import numpy as np


def clock_seconds(values) -> np.ndarray:
    result = []
    for value in values:
        text = str(value)
        result.append(int(text[:2]) * 3600 + int(text[3:5]) * 60 + int(text[6:]))
    return np.asarray(result, dtype=np.int64)


def day_returns(price, times, horizon_seconds: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (simple, log) returns for exact within-day horizons only."""

    prices = np.asarray(price, dtype=np.float64)
    seconds = clock_seconds(times)
    if len(prices) != len(seconds):
        raise ValueError("price and timestamp lengths differ")
    if horizon_seconds <= 0:
        raise ValueError("horizon must be positive")
    if len(prices) <= horizon_seconds:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    current = prices[horizon_seconds:]
    previous = prices[:-horizon_seconds]
    aligned = (seconds[horizon_seconds:] - seconds[:-horizon_seconds]) == horizon_seconds
    valid = aligned & np.isfinite(current) & np.isfinite(previous) & (current > 0) & (previous > 0)
    simple = current[valid] / previous[valid] - 1.0
    log_return = np.log(current[valid]) - np.log(previous[valid])
    return simple, log_return


def day_one_second_returns(price, times) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return values and their current timestamps for 1-second returns."""

    prices = np.asarray(price, dtype=np.float64)
    seconds = clock_seconds(times)
    current = prices[1:]
    previous = prices[:-1]
    valid = (
        ((seconds[1:] - seconds[:-1]) == 1)
        & np.isfinite(current)
        & np.isfinite(previous)
        & (current > 0)
        & (previous > 0)
    )
    return current[valid] / previous[valid] - 1.0, np.log(current[valid]) - np.log(previous[valid]), seconds[1:][valid]
