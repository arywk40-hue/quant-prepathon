"""Schema, timestamp, and price validation for one loaded day."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from src.common.day_boundary import format_time_seconds, parse_time_seconds


def _column_values(table: Any, name: str) -> list[Any]:
    return table[name].to_pylist()


def validate_schema(table: Any, reference_schema: list[dict[str, str]] | None) -> dict[str, Any]:
    actual = [{"name": field.name, "type": str(field.type)} for field in table.schema]
    actual_names = [item["name"] for item in actual]
    result: dict[str, Any] = {
        "column_count": len(actual_names),
        "reference_column_count": len(reference_schema or actual),
        "missing_columns": "",
        "unexpected_columns": "",
        "dtype_mismatches": "",
        "same_order": True,
        "same_set": True,
        "status": "reference" if reference_schema is None else "valid",
    }
    if reference_schema is None:
        return result
    expected_names = [item["name"] for item in reference_schema]
    missing = sorted(set(expected_names) - set(actual_names))
    unexpected = sorted(set(actual_names) - set(expected_names))
    actual_by_name = {item["name"]: item["type"] for item in actual}
    mismatches = [
        f"{name}:{actual_by_name.get(name)}!={expected['type']}"
        for expected in reference_schema
        for name in [expected["name"]]
        if name in actual_by_name and actual_by_name[name] != expected["type"]
    ]
    result.update(
        {
            "missing_columns": "|".join(missing),
            "unexpected_columns": "|".join(unexpected),
            "dtype_mismatches": "|".join(mismatches),
            "same_order": actual_names == expected_names,
            "same_set": set(actual_names) == set(expected_names),
            "status": "valid" if not (missing or unexpected or mismatches) else "invalid",
        }
    )
    return result


def validate_timestamps(table: Any) -> dict[str, Any]:
    values = _column_values(table, "Time")
    seconds = [parse_time_seconds(value) for value in values]
    valid_seconds = [value for value in seconds if value is not None]
    malformed = sum(value is None for value in seconds)
    duplicates = len(valid_seconds) - len(set(valid_seconds))
    # Do not bridge over malformed timestamps. Such a bridge would invent an
    # interval between observations that are not adjacent in the source.
    intervals: Counter[int] = Counter()
    previous: int | None = None
    skipped_due_to_malformed = 0
    for current in seconds:
        if current is None:
            if previous is not None:
                skipped_due_to_malformed += 1
            previous = None
            continue
        if previous is not None:
            intervals[current - previous] += 1
        previous = current
    out_of_order = sum(count for delta, count in intervals.items() if delta <= 0)
    non_one = sum(count for delta, count in intervals.items() if delta != 1)
    missing_seconds = sum(max(delta - 1, 0) * count for delta, count in intervals.items() if delta > 1)
    mode = intervals.most_common(1)[0][0] if intervals else ""
    start = valid_seconds[0] if valid_seconds else None
    end = valid_seconds[-1] if valid_seconds else None
    expected_rows = end - start + 1 if start is not None and end is not None and end >= start else ""
    return {
        "rows": len(values),
        "start_time": format_time_seconds(start),
        "end_time": format_time_seconds(end),
        "expected_rows": expected_rows,
        "frequency_mode": mode,
        "frequency_mode_seconds": mode,
        "non_one_second_intervals": non_one,
        "missing_seconds": missing_seconds,
        "duplicate_timestamps": duplicates,
        "out_of_order": out_of_order,
        "malformed_time_rows": malformed,
        "intervals_skipped_due_to_malformed_time": skipped_due_to_malformed,
        "interval_distribution": "|".join(f"{key}:{value}" for key, value in sorted(intervals.items())),
        "status": "valid" if malformed == 0 and duplicates == 0 and out_of_order == 0 and non_one == 0 else "warning",
        "_seconds": seconds,
    }


def validate_price(table: Any) -> dict[str, Any]:
    column = table["Price"]
    values = _column_values(table, "Price")
    kind = str(column.type)
    numeric_values: list[float] = []
    nan_count = 0
    inf_count = 0
    non_numeric_count = 0
    for value in values:
        if value is None:
            nan_count += 1
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            non_numeric_count += 1
            continue
        if math.isnan(number):
            nan_count += 1
        elif math.isinf(number):
            inf_count += 1
        else:
            numeric_values.append(number)
    zero_count = sum(value == 0 for value in numeric_values)
    negative_count = sum(value < 0 for value in numeric_values)
    # A missing/invalid price breaks adjacency; do not manufacture a return
    # across an invalid observation.
    zero_return_count = 0
    previous_valid: float | None = None
    for value in values:
        try:
            number = float(value) if value is not None else None
        except (TypeError, ValueError):
            number = None
        if number is None or not math.isfinite(number):
            previous_valid = None
            continue
        if previous_valid == number:
            zero_return_count += 1
        previous_valid = number
    flags = []
    if nan_count:
        flags.append("nan")
    if inf_count:
        flags.append("inf")
    if non_numeric_count:
        flags.append("non_numeric")
    if zero_count:
        flags.append("zero")
    if negative_count:
        flags.append("negative")
    return {
        "price_dtype": kind,
        "price_min": min(numeric_values) if numeric_values else "",
        "price_max": max(numeric_values) if numeric_values else "",
        "price_mean": sum(numeric_values) / len(numeric_values) if numeric_values else "",
        "price_std": _sample_std(numeric_values),
        "valid_count": len(numeric_values),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "non_numeric_count": non_numeric_count,
        "zero_count": zero_count,
        "negative_count": negative_count,
        "zero_return_count": zero_return_count,
        "price_flags": "|".join(flags) or "none",
        "status": "valid" if not flags else "invalid",
    }


def _sample_std(values: list[float]) -> float | str:
    if len(values) < 2:
        return ""
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
