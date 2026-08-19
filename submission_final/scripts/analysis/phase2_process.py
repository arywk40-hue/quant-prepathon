#!/usr/bin/env python3
"""Run PHASE 2 on available development days only.

The runner never opens Days 86-108. Missing development days are retained in
the manifest as ``missing_source`` and are never synthesized or imputed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cleaning.missingness import aggregate_cross_day_warmup, build_validity_mask, classify_structural_missingness
from src.ingestion.discovery import discover_days
from src.ingestion.loader import load_day, schema_record, write_parquet
from src.ingestion.validation import validate_price, validate_schema, validate_timestamps


EXPECTED_DEVELOPMENT_DAYS = tuple(range(1, 86))


def load_config(config_path: Path) -> dict[str, object]:
    """Read the scalar/list settings used by the Phase 2 runner."""

    settings: dict[str, object] = {}
    if not config_path.is_file():
        return settings
    for raw_line in config_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, raw_value = (part.strip() for part in line.split(":", 1))
        if raw_value.startswith("[") and raw_value.endswith("]"):
            settings[key] = ast.literal_eval(raw_value)
        else:
            settings[key] = raw_value.strip("\"'")
    return settings


def integer_tuple_setting(config: dict[str, object], key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = config.get(key, list(default))
    if not isinstance(value, list) or not value or not all(isinstance(item, int) for item in value):
        raise ValueError(f"{key} must be a non-empty integer list")
    return tuple(value)


def configured_development_days(config: dict[str, object]) -> tuple[int, ...]:
    value = config.get("development_days", [1, 85])
    if not isinstance(value, list) or len(value) != 2 or not all(isinstance(item, int) for item in value):
        raise ValueError("development_days must be a two-item integer list")
    start, end = value
    if start < 1 or end < start or end > 85:
        raise ValueError("Phase 2 development_days must be wholly within Days 1-85")
    return tuple(range(start, end + 1))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def process(repo_root: Path, config_path: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path or repo_root / "config" / "config.yaml")
    raw_data_dir = Path(str(config.get("raw_data_dir", "data")))
    validated_data_dir = Path(str(config.get("validated_data_dir", "data/validated")))
    processed_data_dir = Path(str(config.get("processed_data_dir", "data/processed")))
    input_dir = raw_data_dir if raw_data_dir.is_absolute() else repo_root / raw_data_dir
    validated_dir = validated_data_dir if validated_data_dir.is_absolute() else repo_root / validated_data_dir
    processed_dir = processed_data_dir if processed_data_dir.is_absolute() else repo_root / processed_data_dir
    results_dir = repo_root / "results" / "missingness"
    expected_development_days = configured_development_days(config)
    parquet_compression = str(config.get("parquet_compression", "zstd"))
    pb_ladder = integer_tuple_setting(config, "pb_nominal_windows_seconds", (15, 30, 90, 180, 270, 360, 900, 1800, 2700, 4500, 5400, 10800))
    other_ladder = integer_tuple_setting(config, "other_nominal_windows_seconds", (5, 10, 30, 60, 90, 120, 300, 600, 900, 1500, 1800, 3600))
    discovery = discover_days(input_dir, expected_development_days)
    if discovery.duplicate_ids:
        raise RuntimeError(f"duplicate development IDs: {discovery.duplicate_ids}")
    if discovery.malformed_names:
        raise RuntimeError(f"malformed day-like filenames: {discovery.malformed_names}")

    reference_schema: list[dict[str, str]] | None = None
    reference_day: int | None = None
    schema_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []
    price_rows: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []
    processed_days: list[int] = []
    parquet_rows: list[dict[str, Any]] = []
    for day in sorted(discovery.files):
        day_file = discovery.files[day]
        loaded = load_day(day_file.path, day)
        if reference_schema is None:
            reference_schema = schema_record(loaded.table)
            reference_day = day
            schema_result = validate_schema(loaded.table, None)
        else:
            schema_result = validate_schema(loaded.table, reference_schema)
        timestamps = validate_timestamps(loaded.table)
        prices = validate_price(loaded.table)
        integrity_status = "valid"
        if schema_result["status"] == "invalid" or prices["status"] == "invalid":
            integrity_status = "invalid"
        elif timestamps["status"] != "valid":
            integrity_status = "warning"
        schema_rows.append({"day": day, "source_path": str(day_file.path), **schema_result})
        integrity_rows.append(
            {
                "day": day,
                "source_path": str(day_file.path),
                "rows": loaded.rows,
                "start_time": timestamps["start_time"],
                "end_time": timestamps["end_time"],
                "time_range": f"{timestamps['start_time']}-{timestamps['end_time']}",
                "expected_rows": timestamps["expected_rows"],
                "frequency_mode": timestamps["frequency_mode"],
                "frequency_mode_seconds": timestamps["frequency_mode_seconds"],
                "non_one_second_intervals": timestamps["non_one_second_intervals"],
                "missing_seconds": timestamps["missing_seconds"],
                "gaps": timestamps["missing_seconds"],
                "duplicate_timestamps": timestamps["duplicate_timestamps"],
                "duplicates": timestamps["duplicate_timestamps"],
                "out_of_order": timestamps["out_of_order"],
                "malformed_time_rows": timestamps["malformed_time_rows"],
                "intervals_skipped_due_to_malformed_time": timestamps["intervals_skipped_due_to_malformed_time"],
                "price_flags": prices["price_flags"],
                "status": integrity_status,
            }
        )
        price_rows.append({"day": day, "source_path": str(day_file.path), **{key: value for key, value in prices.items() if key != "status"}, "status": prices["status"]})
        mask = build_validity_mask(loaded.table, reference_schema)
        missingness = classify_structural_missingness(
            loaded.table,
            day,
            timestamps["_seconds"],
            reference_schema,
            pb_ladder,
            other_ladder,
        )
        structural_rows.extend(missingness)
        processed_path = processed_dir / f"day{day}.parquet"
        mask_path = processed_dir / f"day{day}_validity_mask.parquet"
        roundtrip = write_parquet(loaded.table, processed_path, compression=parquet_compression)
        mask_roundtrip = write_parquet(mask, mask_path, compression=parquet_compression)
        parquet_rows.append(
            {
                "day": day,
                "processed_path": str(processed_path),
                "mask_path": str(mask_path),
                "rows": loaded.rows,
                "columns": len(loaded.columns),
                "processed_schema_equal": roundtrip["schema_equal"],
                "processed_values_equal": roundtrip["values_equal"],
                "mask_rows": mask.num_rows,
                "mask_columns": len(mask.column_names),
                "mask_schema_equal": mask_roundtrip["schema_equal"],
                "mask_values_equal": mask_roundtrip["values_equal"],
            }
        )
        processed_days.append(day)

    if reference_schema is None:
        raise RuntimeError("no available development CSVs found")
    missing_rows = [
        {
            "day": day,
            "source_path": "",
            "rows": "",
            "start_time": "",
            "end_time": "",
            "time_range": "",
            "expected_rows": "",
            "frequency_mode": "",
            "frequency_mode_seconds": "",
            "non_one_second_intervals": "",
            "missing_seconds": "",
            "gaps": "",
            "duplicate_timestamps": "",
            "duplicates": "",
            "out_of_order": "",
            "malformed_time_rows": "",
            "intervals_skipped_due_to_malformed_time": "",
            "price_flags": "",
            "status": "missing_source",
        }
        for day in discovery.missing_days
    ]
    manifest_rows = sorted(integrity_rows + missing_rows, key=lambda row: row["day"])
    schema_rows = sorted(schema_rows, key=lambda row: row["day"])
    price_rows = sorted(price_rows, key=lambda row: row["day"])
    cross_day = aggregate_cross_day_warmup(structural_rows, expected_development_days)
    ladder_rows = []
    for row in cross_day:
        nominal = row["nominal_window_seconds"]
        ladder_rows.append(
            {
                "feature": row["feature"],
                "family": row["family"],
                "suffix": row["suffix"],
                "nominal_window_seconds": nominal,
                "days_present": row["days_present"],
                "days_with_feature": row["days_with_feature"],
                "observed_warmup_min_seconds": row["min_warmup_sec"],
                "observed_warmup_median_seconds": row["median_warmup_sec"],
                "observed_warmup_mean_seconds": row["mean_warmup_sec"],
                "observed_warmup_max_seconds": row["max_warmup_sec"],
                "exact_nominal_matches": row["days_matching_nominal"],
                "days_deviating": row["days_deviating"],
                "note": "nominal window is a hypothesis; actual warm-up is reported, not forced",
            }
        )

    validated_dir.mkdir(parents=True, exist_ok=True)
    write_csv(validated_dir / "manifest.csv", manifest_rows, list(manifest_rows[0]))
    write_csv(validated_dir / "schema_validation.csv", schema_rows, list(schema_rows[0]))
    write_csv(validated_dir / "day_integrity.csv", manifest_rows, list(manifest_rows[0]))
    write_csv(validated_dir / "price_validation.csv", price_rows, list(price_rows[0]))
    write_csv(validated_dir / "parquet_roundtrip.csv", parquet_rows, list(parquet_rows[0]))
    (validated_dir / "reference_schema.json").write_text(json.dumps({"reference_day": reference_day, "schema": reference_schema}, indent=2) + "\n")
    write_csv(results_dir / "structural_missingness.csv", structural_rows, list(structural_rows[0]))
    write_csv(results_dir / "cross_day_warmup.csv", cross_day, list(cross_day[0]))
    write_csv(results_dir / "window_ladder_validation.csv", ladder_rows, list(ladder_rows[0]))
    (results_dir / "phase2_scope.txt").write_text(
        "PHASE 2 SCOPE\n"
        f"expected_development_days={expected_development_days[0]}-{expected_development_days[-1]}\n"
        f"real_data_days_processed={len(processed_days)}\n"
        f"processed_day_ids={','.join(map(str, processed_days))}\n"
        f"missing_days={','.join(map(str, discovery.missing_days))}\n"
        "holdout_days=86-108; not opened or processed\n"
        "out_of_scope_ids_detected=" + ",".join(map(str, discovery.out_of_scope_ids)) + "\n"
        "raw_csv_policy=read-only\n"
        "cleaning_policy=structural NaNs preserved; no imputation; unexpected missingness flagged\n"
    )
    return {
        "expected_days": len(expected_development_days),
        "processed_days": processed_days,
        "missing_days": list(discovery.missing_days),
        "reference_day": reference_day,
        "reference_columns": len(reference_schema),
        "structural_rows": len(structural_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    summary = process(args.repo_root.resolve(), args.config.resolve() if args.config else None)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
