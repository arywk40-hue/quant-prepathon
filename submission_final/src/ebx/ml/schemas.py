"""Shared schemas and immutable data-scope definitions for ML Phase 0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable


EXPECTED_DEVELOPMENT_DAYS = 85
AVAILABLE_DEVELOPMENT_DAYS = tuple(range(1, 65)) + tuple(range(80, 86))
MISSING_DEVELOPMENT_DAYS = tuple(range(65, 80))
HOLDOUT_DAYS = tuple(range(86, 109))
TARGET_HORIZONS_SECONDS = (1, 5, 30, 60, 300)


@dataclass(frozen=True)
class DevelopmentScope:
    expected_development_days: int
    available_development_days: tuple[int, ...]
    missing_development_days: tuple[int, ...]
    holdout_days: tuple[int, ...]

    @classmethod
    def from_freeze(cls, freeze_path: str | Path) -> "DevelopmentScope":
        payload = json.loads(Path(freeze_path).read_text())
        scope = payload["scope"]
        result = cls(
            expected_development_days=int(scope["expected_development_days"]),
            available_development_days=tuple(int(day) for day in scope["available_day_ids"]),
            missing_development_days=tuple(int(day) for day in scope["missing_day_ids"]),
            holdout_days=tuple(int(day) for day in scope["holdout_day_ids"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        expected = set(range(1, self.expected_development_days + 1))
        available = set(self.available_development_days)
        missing = set(self.missing_development_days)
        if available & missing or available | missing != expected:
            raise ValueError("development scope is not a complete disjoint partition")
        if len(available) != 70 or len(missing) != 15:
            raise ValueError("ML Phase 0 requires the audited 70/85 development scope")
        if set(self.holdout_days) != set(HOLDOUT_DAYS):
            raise ValueError("holdout scope changed")

    def assert_development_days(self, days: Iterable[int]) -> None:
        observed = {int(day) for day in days}
        unexpected = observed - set(self.available_development_days)
        if unexpected:
            raise ValueError(f"non-development days requested: {sorted(unexpected)}")

    def as_dict(self) -> dict[str, object]:
        return {
            "expected_development_days": self.expected_development_days,
            "available_development_days": list(self.available_development_days),
            "missing_development_days": list(self.missing_development_days),
            "holdout_days": list(self.holdout_days),
        }


def audited_scope(freeze_path: str | Path) -> DevelopmentScope:
    """Load and validate the immutable development/holdout boundary."""

    return DevelopmentScope.from_freeze(freeze_path)


def dataclass_dict(value: object) -> dict[str, object]:
    """Convert a dataclass to JSON-friendly primitives."""

    return asdict(value)  # type: ignore[arg-type]
