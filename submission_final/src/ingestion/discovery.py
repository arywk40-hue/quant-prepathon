"""Safe day-file discovery with explicit missing-day reporting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


DAY_RE = re.compile(r"^day(?P<day>[1-9]\d*)\.csv$")


@dataclass(frozen=True)
class DayFile:
    day: int
    path: Path
    size_bytes: int


@dataclass(frozen=True)
class DiscoveryResult:
    expected_days: tuple[int, ...]
    files: dict[int, DayFile]
    missing_days: tuple[int, ...]
    malformed_names: tuple[str, ...]
    duplicate_ids: tuple[int, ...]
    out_of_scope_ids: tuple[int, ...]


def parse_day_filename(name: str) -> int | None:
    match = DAY_RE.fullmatch(name)
    return int(match.group("day")) if match else None


def discover_days(input_dir: Path, expected_days: range | tuple[int, ...]) -> DiscoveryResult:
    """Discover only the requested day universe; never reads CSV contents.

    Files outside ``expected_days`` are reported as out-of-scope IDs but are
    not stat'ed or opened. This makes the Phase 2 runner unable to process
    holdout days by accident.
    """

    expected = tuple(sorted(set(expected_days)))
    expected_set = set(expected)
    files: dict[int, DayFile] = {}
    malformed: list[str] = []
    duplicates: list[int] = []
    out_of_scope: list[int] = []
    for path in sorted(input_dir.iterdir()):
        if not path.name.startswith("day"):
            continue
        day = parse_day_filename(path.name)
        if day is None:
            malformed.append(path.name)
            continue
        if day not in expected_set:
            out_of_scope.append(day)
            continue
        if not path.is_file():
            continue
        if day in files:
            duplicates.append(day)
            continue
        files[day] = DayFile(day=day, path=path, size_bytes=path.stat().st_size)
    missing = tuple(day for day in expected if day not in files)
    return DiscoveryResult(
        expected_days=expected,
        files=files,
        missing_days=missing,
        malformed_names=tuple(sorted(malformed)),
        duplicate_ids=tuple(sorted(set(duplicates))),
        out_of_scope_ids=tuple(sorted(set(out_of_scope))),
    )
