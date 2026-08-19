"""Coverage metadata and safe iteration over processed development days."""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow.parquet as pq


EXPECTED_DEVELOPMENT_DAYS = 85
MISSING_DEVELOPMENT_DAYS = tuple(range(65, 80))
AVAILABLE_DEVELOPMENT_DAYS = tuple(list(range(1, 65)) + list(range(80, 86)))


def coverage_metadata() -> dict[str, object]:
    return {
        "expected_development_days": EXPECTED_DEVELOPMENT_DAYS,
        "available_development_days": len(AVAILABLE_DEVELOPMENT_DAYS),
        "missing_development_days": len(MISSING_DEVELOPMENT_DAYS),
    }


def available_days_from_manifest(repo_root: Path) -> tuple[int, ...]:
    manifest_path = repo_root / "data" / "validated" / "manifest.csv"
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    available = tuple(
        sorted(int(row["day"]) for row in rows if row["status"] != "missing_source")
    )
    if set(available) != set(AVAILABLE_DEVELOPMENT_DAYS):
        raise RuntimeError(
            "validated manifest does not match the audited 70-day development universe"
        )
    return available


def load_price_day(repo_root: Path, day: int):
    """Load only one processed development day's Time and Price columns."""

    if day not in AVAILABLE_DEVELOPMENT_DAYS:
        raise ValueError(f"day {day} is not an available development day")
    path = repo_root / "data" / "processed" / f"day{day}.parquet"
    table = pq.read_table(path, columns=["Time", "Price"])
    frame = table.to_pandas()
    return frame
