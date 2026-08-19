"""Lossless structural-missingness classification and validity masks."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from src.common.day_boundary import format_time_seconds
from src.common.features import OTHER_LADDER, PB_LADDER, FeatureMeta, parse_feature


def _is_numeric_type(type_name: str) -> bool:
    return any(token in type_name for token in ("int", "float", "decimal"))


def _validity_values(array: Any, expected_type: str | None = None) -> list[bool]:
    if expected_type is not None and _is_numeric_type(expected_type) and not _is_numeric_type(str(array.type)):
        return [False] * len(array)
    values = array.to_pylist()
    valid: list[bool] = []
    for value in values:
        if value is None:
            valid.append(False)
            continue
        try:
            valid.append(math.isfinite(float(value)))
        except (TypeError, ValueError):
            valid.append(False)
    return valid


def build_validity_mask(
    table: Any,
    reference_schema: list[dict[str, str]] | None = None,
) -> Any:
    """Return Time plus one boolean validity column per data feature."""

    import pyarrow as pa

    expected = {item["name"]: item["type"] for item in (reference_schema or [])}
    arrays = [table["Time"]]
    names = ["Time"]
    for name in table.column_names:
        if name == "Time":
            continue
        arrays.append(pa.array(_validity_values(table[name], expected.get(name)), type=pa.bool_()))
        names.append(name)
    return pa.table(arrays, names=names)


def classify_structural_missingness(
    table: Any,
    day: int,
    timestamp_seconds: list[int | None],
    reference_schema: list[dict[str, str]] | None = None,
    pb_ladder: tuple[int, ...] = PB_LADDER,
    other_ladder: tuple[int, ...] = OTHER_LADDER,
) -> list[dict[str, Any]]:
    expected = {item["name"]: item["type"] for item in (reference_schema or [])}
    rows: list[dict[str, Any]] = []
    for name in table.column_names:
        if name in {"Time", "Price"}:
            continue
        meta: FeatureMeta = parse_feature(name, pb_ladder, other_ladder)
        values = table[name].to_pylist()
        valid = _validity_values(table[name], expected.get(name))
        valid_positions = [index for index, flag in enumerate(valid) if flag]
        first = valid_positions[0] if valid_positions else None
        last = valid_positions[-1] if valid_positions else None
        leading = first if first is not None else len(values)
        trailing = (len(values) - last - 1) if last is not None else 0
        internal = sum(not flag for flag in valid[first : last + 1]) if first is not None else 0
        nan_count = sum(value is None for value in values)
        inf_count = 0
        for value in values:
            try:
                if value is not None and math.isinf(float(value)):
                    inf_count += 1
            except (TypeError, ValueError):
                pass
        first_seconds = timestamp_seconds[first] if first is not None and first < len(timestamp_seconds) else None
        start_seconds = next((value for value in timestamp_seconds if value is not None), None)
        rows.append(
            {
                "day": day,
                "feature": name,
                "family": meta.family,
                "subfamily": meta.subfamily,
                "suffix": meta.suffix,
                "nominal_window_seconds": meta.nominal_window_seconds if meta.nominal_window_seconds is not None else "",
                "dtype": str(table[name].type),
                "first_valid_index": (first + 1) if first is not None else "",
                "first_valid_timestamp": format_time_seconds(first_seconds),
                "last_valid_index": (last + 1) if last is not None else "",
                "leading_nan_count": leading,
                "internal_nan_count": internal,
                "trailing_nan_count": trailing,
                "total_nan_count": nan_count,
                "total_inf_count": inf_count,
                "missing_fraction": (sum(not flag for flag in valid) / len(valid)) if valid else "",
                "actual_warmup_seconds": (first_seconds - start_seconds) if first_seconds is not None and start_seconds is not None else "",
                "stability_class": (
                    "all_nan" if first is None else
                    "leading_only" if internal == 0 and trailing == 0 and leading > 0 else
                    "no_missingness" if leading == 0 and internal == 0 and trailing == 0 else
                    "internal_or_trailing_missing"
                ),
                "unexpected_internal_nan": internal > 0,
            }
        )
    return rows


def aggregate_cross_day_warmup(
    rows: list[dict[str, Any]],
    expected_days: tuple[int, ...],
) -> list[dict[str, Any]]:
    by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_feature[row["feature"]].append(row)
    present_days = {row["day"] for row in rows}
    missing_days = sorted(set(expected_days) - present_days)
    output: list[dict[str, Any]] = []
    for feature in sorted(by_feature):
        feature_rows = by_feature[feature]
        warmups = [row["actual_warmup_seconds"] for row in feature_rows if row["actual_warmup_seconds"] != ""]
        nominal = feature_rows[0]["nominal_window_seconds"]
        internal_days = sum(bool(row["unexpected_internal_nan"]) for row in feature_rows)
        output.append(
            {
                "feature": feature,
                "family": feature_rows[0]["family"],
                "subfamily": feature_rows[0]["subfamily"],
                "suffix": feature_rows[0]["suffix"],
                "nominal_window_seconds": nominal,
                "days_expected": len(expected_days),
                "days_present": len(present_days),
                "days_with_feature": len(feature_rows),
                "missing_days": "|".join(map(str, missing_days)),
                "mean_warmup_sec": sum(warmups) / len(warmups) if warmups else "",
                "median_warmup_sec": sorted(warmups)[len(warmups) // 2] if warmups else "",
                "std_warmup_sec": _std(warmups),
                "min_warmup_sec": min(warmups) if warmups else "",
                "max_warmup_sec": max(warmups) if warmups else "",
                "days_matching_nominal": sum(value == nominal for value in warmups) if nominal != "" else "",
                "days_deviating": sum(value != nominal for value in warmups) if nominal != "" else "",
                "internal_nan_days": internal_days,
                "stability_class": "stable" if len(set(warmups)) <= 1 and internal_days == 0 else "variable_or_internal_missingness",
            }
        )
    return output


def _std(values: list[float]) -> float | str:
    if len(values) < 2:
        return 0.0 if values else ""
    average = sum(values) / len(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))
