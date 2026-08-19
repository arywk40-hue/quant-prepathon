"""Masked-feature parsing and nominal window hypotheses."""

from __future__ import annotations

import re
from dataclasses import dataclass


PB_LADDER = (15, 30, 90, 180, 270, 360, 900, 1800, 2700, 4500, 5400, 10800)
OTHER_LADDER = (5, 10, 30, 60, 90, 120, 300, 600, 900, 1500, 1800, 3600)
FEATURE_RE = re.compile(
    r"^(?P<family>PB|VB|PV|BB|V)(?P<body>.*?)(?:_T(?P<suffix>\d+))?$"
)


@dataclass(frozen=True)
class FeatureMeta:
    feature: str
    family: str
    subfamily: str
    suffix: str
    nominal_window_seconds: int | None


def parse_feature(
    name: str,
    pb_ladder: tuple[int, ...] = PB_LADDER,
    other_ladder: tuple[int, ...] = OTHER_LADDER,
) -> FeatureMeta:
    match = FEATURE_RE.fullmatch(name)
    if match is None:
        return FeatureMeta(name, "unknown", "", "", None)
    family = match.group("family")
    body = match.group("body")
    suffix = match.group("suffix") or ""
    subfamily = f"{family}{body}" if body else family
    nominal = None
    if suffix:
        index = int(suffix)
        ladder = pb_ladder if family == "PB" else other_ladder
        if 1 <= index <= len(ladder):
            nominal = ladder[index - 1]
    return FeatureMeta(name, family, subfamily, suffix, nominal)
