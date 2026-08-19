"""Day-local regime diagnostics and frozen classification rules."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


REGIME_THRESHOLDS = {
    "vr_q": 5,
    "vr_band": 0.05,
    "acf_abs": 0.02,
    "pvalue": 0.05,
    "hurst_band": 0.05,
}


def variance_ratio(values: np.ndarray, q: int = 5) -> tuple[float, float]:
    """Return the simple q-period variance ratio and a normal approximation p-value.

    Values must be one day of equally spaced returns. No observations are added or
    carried across a day boundary.
    """

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if q < 2 or len(x) <= q + 2:
        return float("nan"), float("nan")
    centered = x - np.mean(x)
    one_var = float(np.mean(centered**2))
    if one_var == 0:
        return float("nan"), float("nan")
    aggregate = np.convolve(x, np.ones(q), mode="valid")
    aggregate_var = float(np.var(aggregate, ddof=0))
    vr = aggregate_var / (q * one_var)
    # Homoskedastic asymptotic standard error for VR(q)-1.
    se = math.sqrt(2.0 * (2 * q - 1) * (q - 1) / (3 * q * len(x)))
    if se == 0 or not np.isfinite(se):
        return vr, float("nan")
    pvalue = float(2.0 * norm.sf(abs((vr - 1.0) / se)))
    return vr, pvalue


def return_acf(values: np.ndarray, lag: int = 1) -> tuple[float, float]:
    """Return lagged Pearson autocorrelation and a two-sided normal p-value."""

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if lag < 1 or len(x) <= lag + 2:
        return float("nan"), float("nan")
    left = x[:-lag]
    right = x[lag:]
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan"), float("nan")
    acf = float(np.corrcoef(left, right)[0, 1])
    se = 1.0 / math.sqrt(len(x))
    pvalue = float(2.0 * norm.sf(abs(acf / se)))
    return acf, pvalue


def hurst_rs(values: np.ndarray) -> float:
    """Estimate H using the rescaled-range slope over fixed within-day blocks."""

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 64:
        return float("nan")
    scales = np.asarray([8, 16, 32, 64], dtype=int)
    log_scales = []
    log_rs = []
    for scale in scales:
        if len(x) < scale:
            continue
        values_rs = []
        for block in np.array_split(x[: len(x) // scale * scale], len(x) // scale):
            if len(block) != scale:
                continue
            centered = block - np.mean(block)
            std = np.std(block, ddof=1)
            if std > 0 and np.isfinite(std):
                values_rs.append((np.max(np.cumsum(centered)) - np.min(np.cumsum(centered))) / std)
        if values_rs:
            mean_rs = float(np.mean(values_rs))
            if mean_rs > 0:
                log_scales.append(math.log(scale))
                log_rs.append(math.log(mean_rs))
    if len(log_scales) < 2:
        return float("nan")
    return float(np.polyfit(log_scales, log_rs, 1)[0])


def classify_regime(vr: float, vr_pvalue: float, hurst: float, acf: float,
                    acf_pvalue: float, adf_pvalue: float) -> tuple[str, str, str]:
    """Apply frozen rules; return regime, confidence, and evidence description."""

    evidence: dict[str, str] = {}
    if np.isfinite(vr) and np.isfinite(vr_pvalue) and vr_pvalue < REGIME_THRESHOLDS["pvalue"]:
        if vr < 1.0 - REGIME_THRESHOLDS["vr_band"]:
            evidence["VR"] = "mean-reverting"
        elif vr > 1.0 + REGIME_THRESHOLDS["vr_band"]:
            evidence["VR"] = "persistent"
    if np.isfinite(acf) and np.isfinite(acf_pvalue) and acf_pvalue < REGIME_THRESHOLDS["pvalue"]:
        if acf < -REGIME_THRESHOLDS["acf_abs"]:
            evidence["ACF"] = "mean-reverting"
        elif acf > REGIME_THRESHOLDS["acf_abs"]:
            evidence["ACF"] = "persistent"
    if np.isfinite(adf_pvalue) and adf_pvalue < REGIME_THRESHOLDS["pvalue"]:
        evidence["ADF"] = "mean-reverting"
    if np.isfinite(hurst):
        if hurst < 0.5 - REGIME_THRESHOLDS["hurst_band"]:
            evidence["Hurst"] = "mean-reverting"
        elif hurst > 0.5 + REGIME_THRESHOLDS["hurst_band"]:
            evidence["Hurst"] = "persistent"
    counts = {"mean-reverting": 0, "persistent": 0}
    for value in evidence.values():
        counts[value] += 1
    if counts["mean-reverting"] >= 2 and counts["persistent"] == 0:
        regime = "mean-reverting"
    elif counts["persistent"] >= 2 and counts["mean-reverting"] == 0:
        regime = "momentum / persistent"
    else:
        regime = "random-walk / inconclusive"
    if counts["mean-reverting"] >= 2 or counts["persistent"] >= 2:
        confidence = "high" if counts["mean-reverting"] == 0 or counts["persistent"] == 0 else "low"
    elif len(evidence) == 1:
        confidence = "low"
    else:
        confidence = "medium"
    detail = "; ".join(f"{name}={value}" for name, value in evidence.items()) or "no significant directional evidence"
    if counts["mean-reverting"] and counts["persistent"]:
        detail += "; conflict retained as inconclusive"
    return regime, confidence, detail
