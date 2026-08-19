import numpy as np
import pandas as pd
import pytest

from src.ebx.ml.baseline import RidgeBaseline, validate_baseline_scope, validation_metrics
from src.ebx.ml.schemas import DevelopmentScope


def _scope():
    return DevelopmentScope(85, (1, 2, 80, 81), tuple(range(3, 80)), tuple(range(86, 109)))


def _partition(path, day, offset):
    pd.DataFrame({
        "day": [day] * 3,
        "timestamp": ["00:00:00", "00:00:01", "00:00:02"],
        "timestamp_seconds": [0, 1, 2],
        "target": [offset, offset + 1.0, offset + 2.0],
        "f1": [1.0, 2.0, 3.0],
        "f2": [3.0, 2.0, 1.0],
    }).to_parquet(path, index=False)


def test_ridge_fit_and_prediction_are_deterministic(tmp_path):
    train_path = tmp_path / "train.parquet"
    _partition(train_path, 1, 0.0)
    first = RidgeBaseline(("f1", "f2"), alpha=1.0).fit_partition_paths([train_path])
    second = RidgeBaseline(("f1", "f2"), alpha=1.0).fit_partition_paths([train_path])
    np.testing.assert_array_equal(first.coef_, second.coef_)
    assert first.intercept_ == second.intercept_
    frame = pd.read_parquet(train_path)
    np.testing.assert_array_equal(first.predict(frame), second.predict(frame))


def test_scope_rejects_wrong_horizon_feature_count_and_holdout():
    scope = _scope()
    with pytest.raises(ValueError, match="300 seconds"):
        validate_baseline_scope(scope, [1, 2], [80, 81], ("f1",) * 197, 60)
    with pytest.raises(ValueError, match="non-development"):
        validate_baseline_scope(scope, [1, 2], [86, 80], tuple(f"f{i}" for i in range(197)), 300)


def test_daily_metrics_preserve_alignment_and_calculate_requested_fields():
    predictions = pd.DataFrame({
        "day": [80, 80, 81, 81],
        "target": [1.0, -1.0, 2.0, -2.0],
        "prediction": [0.5, -0.5, 1.0, -1.0],
    })
    pooled, daily = validation_metrics(predictions)
    assert len(daily) == 2
    assert pooled["validation_observations"] == 4
    assert pooled["directional_accuracy"] == 1.0
    assert pooled["mae"] == 0.75
    assert pooled["rmse"] == (2.5 / 4) ** 0.5
    assert pooled["mean_daily_pearson_ic"] == pytest.approx(1.0)
