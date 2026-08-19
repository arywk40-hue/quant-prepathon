"""Chronological, whole-day development splits."""

from __future__ import annotations

from pathlib import Path
import json

from .schemas import DevelopmentScope


def chronological_split(scope: DevelopmentScope, validation_start_day: int | None = None) -> dict[str, object]:
    """Use earlier available days for training and later available days for validation."""

    available = list(scope.available_development_days)
    if validation_start_day is None:
        validation_start_day = max(scope.missing_development_days) + 1 if scope.missing_development_days else available[len(available) // 2]
    training = [day for day in available if day < validation_start_day]
    validation = [day for day in available if day >= validation_start_day]
    if not training or not validation or set(training) & set(validation):
        raise ValueError("chronological split must have disjoint non-empty day sets")
    if max(training) >= min(validation):
        raise ValueError("training days must precede validation days")
    return {
        "split_type": "chronological_whole_day",
        "validation_start_day": int(validation_start_day),
        "expected_development_days": scope.expected_development_days,
        "available_development_days": available,
        "training_days": training,
        "validation_days": validation,
        "missing_days": list(scope.missing_development_days),
        "holdout_days_excluded": list(scope.holdout_days),
    }


def write_split_manifest(split: dict[str, object], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(split, indent=2) + "\n")
