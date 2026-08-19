"""Phase 6: day-local regime classification on available development days only."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analytics.coverage import (  # noqa: E402
    AVAILABLE_DEVELOPMENT_DAYS,
    EXPECTED_DEVELOPMENT_DAYS,
    MISSING_DEVELOPMENT_DAYS,
    available_days_from_manifest,
    coverage_metadata,
    load_price_day,
)
from src.analytics.regimes import (  # noqa: E402
    REGIME_THRESHOLDS,
    classify_regime,
    hurst_rs,
    return_acf,
    variance_ratio,
)
from src.analytics.returns import day_returns  # noqa: E402


def _adf(price: np.ndarray) -> tuple[float, float]:
    values = np.asarray(price, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 100 or np.std(values) == 0:
        return float("nan"), float("nan")
    try:
        result = adfuller(np.log(values), regression="c", autolag="AIC")
        return float(result[0]), float(result[1])
    except (ValueError, np.linalg.LinAlgError):
        return float("nan"), float("nan")


def _base_row(day: int, status: str, coverage: dict[str, object]) -> dict[str, object]:
    return {
        "day": day,
        "status": status,
        **coverage,
        "VR": np.nan,
        "VR_pvalue": np.nan,
        "Hurst": np.nan,
        "ACF": np.nan,
        "ACF_pvalue": np.nan,
        "ADF": np.nan,
        "ADF_pvalue": np.nan,
        "KPSS": np.nan,
        "KPSS_pvalue": np.nan,
        "regime": "missing source" if status != "available" else "",
        "confidence": "not assessed" if status != "available" else "",
        "evidence": "missing development source" if status != "available" else "",
        "n_returns_1m": np.nan,
    }


def _classify_day(repo_root: Path, day: int, coverage: dict[str, object]) -> dict[str, object]:
    frame = load_price_day(repo_root, day)
    price = frame["Price"].to_numpy(dtype=float)
    times = frame["Time"].astype(str).to_numpy()
    returns, _ = day_returns(price, times, 60)
    vr, vr_p = variance_ratio(returns, q=int(REGIME_THRESHOLDS["vr_q"]))
    acf, acf_p = return_acf(returns, lag=1)
    hurst = hurst_rs(returns)
    adf, adf_p = _adf(price)
    regime, confidence, evidence = classify_regime(vr, vr_p, hurst, acf, acf_p, adf_p)
    row = _base_row(day, "available", coverage)
    row.update({
        "VR": vr,
        "VR_pvalue": vr_p,
        "Hurst": hurst,
        "ACF": acf,
        "ACF_pvalue": acf_p,
        "ADF": adf,
        "ADF_pvalue": adf_p,
        "regime": regime,
        "confidence": confidence,
        "evidence": evidence,
        "n_returns_1m": len(returns),
    })
    return row


def _transitions(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = table[table["status"] == "available"].sort_values("day")
    records = []
    for first, second in zip(valid.iloc[:-1].itertuples(), valid.iloc[1:].itertuples()):
        if int(second.day) != int(first.day) + 1:
            continue
        records.append({
            "from_day": int(first.day),
            "to_day": int(second.day),
            "from_regime": first.regime,
            "to_regime": second.regime,
            "is_persistent": bool(first.regime == second.regime),
        })
    transitions = pd.DataFrame(records)
    regimes = ["mean-reverting", "momentum / persistent", "random-walk / inconclusive"]
    matrix = pd.DataFrame(0, index=regimes, columns=regimes, dtype=int)
    for row in records:
        matrix.loc[row["from_regime"], row["to_regime"]] += 1
    matrix.index.name = "from_regime"
    matrix = matrix.reset_index().melt(id_vars="from_regime", var_name="to_regime", value_name="count")

    durations = []
    for regime in regimes:
        ids = [int(day) for day in valid.loc[valid["regime"] == regime, "day"]]
        runs = []
        current = []
        for day in ids:
            if not current or day == current[-1] + 1:
                current.append(day)
            else:
                runs.append(current)
                current = [day]
        if current:
            runs.append(current)
        lengths = [len(run) for run in runs]
        durations.append({
            "regime": regime,
            "run_count": len(lengths),
            "average_duration_days": float(np.mean(lengths)) if lengths else np.nan,
            "max_duration_days": max(lengths) if lengths else 0,
        })
    return matrix, pd.DataFrame(durations), transitions


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    available = available_days_from_manifest(repo_root)
    if available != AVAILABLE_DEVELOPMENT_DAYS:
        raise RuntimeError("available development universe changed")
    coverage = coverage_metadata()
    output = repo_root / "results" / "regimes"
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for day in range(1, EXPECTED_DEVELOPMENT_DAYS + 1):
        if day in MISSING_DEVELOPMENT_DAYS:
            rows.append(_base_row(day, "missing_source", coverage))
        else:
            rows.append(_classify_day(repo_root, day, coverage))
    table = pd.DataFrame(rows)
    table.to_csv(output / "regime_table.csv", index=False)

    matrix, durations, transitions = _transitions(table)
    matrix.to_csv(output / "transition_matrix.csv", index=False)
    durations.to_csv(output / "regime_durations.csv", index=False)
    transitions.to_csv(output / "regime_transitions.csv", index=False)

    available_table = table[table["status"] == "available"]
    counts = available_table["regime"].value_counts().rename_axis("regime").reset_index(name="count")
    counts["proportion"] = counts["count"] / len(available_table)
    counts.to_csv(output / "regime_summary.csv", index=False)
    scope = {
        **coverage,
        "available_day_ids": list(available),
        "missing_day_ids": list(MISSING_DEVELOPMENT_DAYS),
        "holdout_day_ids": list(range(86, 109)),
        "holdout_processed": False,
        "returns_horizon_seconds": 60,
        "thresholds": REGIME_THRESHOLDS,
        "methods": ["variance_ratio_q5", "hurst_rescaled_range", "return_acf_lag1", "ADF_log_price"],
        "kpss": "not run; four independent/partly independent diagnostics satisfy the two-method requirement",
        "transition_rule": "only adjacent available day IDs; no transition bridges missing Days 65-79",
    }
    (output / "phase6_scope.json").write_text(json.dumps(scope, indent=2) + "\n")
    print({
        "table_rows": len(table),
        "available_rows": len(available_table),
        "missing_rows": int((table["status"] == "missing_source").sum()),
        "adjacent_transitions": len(transitions),
        "regimes": counts.to_dict(orient="records"),
    })


if __name__ == "__main__":
    main()
