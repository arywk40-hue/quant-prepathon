"""Timestamp helpers used by every day-local operation."""

from __future__ import annotations

import re


TIME_RE = re.compile(r"^(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})$")


def parse_time_seconds(value: str | None) -> int | None:
    if value is None:
        return None
    match = TIME_RE.fullmatch(str(value).strip())
    if match is None:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second"))
    if minute >= 60 or second >= 60:
        return None
    return hour * 3600 + minute * 60 + second


def format_time_seconds(seconds: int | None) -> str:
    if seconds is None:
        return ""
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
