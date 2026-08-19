import pandas as pd
import pytest

from src.ebx.ml.schemas import DevelopmentScope
from src.ebx.ml.temporal_robustness import (
    TEMPORAL_WINDOWS,
    feature_jaccard,
    feature_overlap_matrix,
    validate_temporal_windows,
)


def _scope():
    return DevelopmentScope(85, tuple(range(1, 65)) + tuple(range(80, 86)), tuple(range(65, 80)), tuple(range(86, 109)))


def test_fixed_windows_match_the_approved_boundaries():
    validate_temporal_windows(TEMPORAL_WINDOWS, _scope())
    assert TEMPORAL_WINDOWS["W1"]["training_days"] == tuple(range(1, 45))
    assert TEMPORAL_WINDOWS["W1"]["validation_days"] == tuple(range(45, 55))
    assert TEMPORAL_WINDOWS["W2"]["training_days"] == tuple(range(1, 55))
    assert TEMPORAL_WINDOWS["W2"]["validation_days"] == tuple(range(55, 65))
    assert TEMPORAL_WINDOWS["W3"]["training_days"] == tuple(range(1, 65))
    assert TEMPORAL_WINDOWS["W3"]["validation_days"] == tuple(range(80, 86))


def test_windows_reject_overlap_missing_days_and_holdout():
    bad = {name: dict(window) for name, window in TEMPORAL_WINDOWS.items()}
    bad["W1"] = {"training_days": (1, 2), "validation_days": (2, 3)}
    with pytest.raises(ValueError, match="overlap"):
        validate_temporal_windows(bad, _scope())
    bad["W1"] = {"training_days": (1, 65), "validation_days": (80,)}
    with pytest.raises(ValueError, match="non-development"):
        validate_temporal_windows(bad, _scope())
    bad["W1"] = {"training_days": (1, 86), "validation_days": (87,)}
    with pytest.raises(ValueError, match="non-development"):
        validate_temporal_windows(bad, _scope())


def test_feature_jaccard_and_matrix_are_exact():
    assert feature_jaccard(("a", "b"), ("b", "c")) == pytest.approx(1 / 3)
    matrix = feature_overlap_matrix({"W2": ("a", "b"), "W1": ("b", "c")})
    assert list(matrix.index) == ["W1", "W2"]
    assert matrix.loc["W1", "W1"] == 1.0
    assert matrix.loc["W1", "W2"] == pytest.approx(1 / 3)
