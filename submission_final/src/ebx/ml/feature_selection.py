"""Consumption of the frozen Part 4 predictive-relevance screen."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.common.features import parse_feature
from .schemas import DevelopmentScope, TARGET_HORIZONS_SECONDS


FROZEN_SCREEN_RULE = (
    "development pearson_fdr_reject AND pearson_pct_same_sign >= 0.70 "
    "AND abs(mean_pearson_ic) >= 0.05"
)


def load_frozen_feature_screen(path: str | Path, scope: DevelopmentScope) -> pd.DataFrame:
    """Load the frozen aggregate IC table without recomputing eligibility."""

    table = pd.read_csv(path)
    required = {
        "feature", "horizon_seconds", "days_scored", "mean_pearson_ic",
        "pearson_pct_same_sign", "pearson_t_pvalue", "pearson_fdr_reject",
        "pearson_fdr_qvalue", "expected_development_days",
        "available_development_days", "missing_development_days",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"frozen predictive artifact missing columns: {sorted(missing)}")
    coverage = table[["expected_development_days", "available_development_days", "missing_development_days"]].drop_duplicates()
    expected = (scope.expected_development_days, len(scope.available_development_days), len(scope.missing_development_days))
    if {tuple(row) for row in coverage.to_numpy()} != {expected}:
        raise ValueError("frozen predictive artifact has unexpected development coverage")
    if set(table["horizon_seconds"].astype(int)) - set(TARGET_HORIZONS_SECONDS):
        raise ValueError("unexpected target horizon in frozen predictive artifact")
    table["pearson_fdr_reject"] = table["pearson_fdr_reject"].astype(str).str.lower().eq("true")
    table["eligible_for_ml"] = (
        table["pearson_fdr_reject"]
        & (table["pearson_pct_same_sign"] >= 0.70)
        & (table["mean_pearson_ic"].abs() >= 0.05)
    )
    metas = [parse_feature(feature) for feature in table["feature"]]
    table.insert(1, "family", [meta.family for meta in metas])
    table.insert(2, "horizon_if_applicable", table["horizon_seconds"].astype(int))
    table.insert(3, "development_evidence", [
        json.dumps({
            "mean_pearson_ic": float(row.mean_pearson_ic),
            "pearson_pct_same_sign": float(row.pearson_pct_same_sign),
            "pearson_fdr_reject": bool(row.pearson_fdr_reject),
            "pearson_fdr_qvalue": float(row.pearson_fdr_qvalue),
        }, sort_keys=True)
        for row in table.itertuples()
    ])
    table.insert(4, "development_screen_status", table["eligible_for_ml"].map({True: "eligible", False: "excluded"}))
    table.insert(5, "reason", table.apply(_reason, axis=1))
    return table.sort_values(["horizon_seconds", "feature"], kind="stable").reset_index(drop=True)


def _reason(row: pd.Series) -> str:
    if bool(row["eligible_for_ml"]):
        return FROZEN_SCREEN_RULE
    failures = []
    if not bool(row["pearson_fdr_reject"]):
        failures.append("pearson_fdr_reject=false")
    if float(row["pearson_pct_same_sign"]) < 0.70:
        failures.append("pearson_pct_same_sign<0.70")
    if abs(float(row["mean_pearson_ic"])) < 0.05:
        failures.append("abs(mean_pearson_ic)<0.05")
    return "; ".join(failures)


def write_frozen_feature_set(table: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "feature", "family", "horizon_if_applicable", "development_evidence",
        "development_screen_status", "eligible_for_ml", "reason",
        "mean_pearson_ic", "pearson_pct_same_sign", "pearson_fdr_reject",
        "pearson_fdr_qvalue", "days_scored",
    ]
    table[output_columns].to_csv(path, index=False)
