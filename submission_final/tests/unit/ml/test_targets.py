import numpy as np
import pandas as pd
import pytest

from src.ebx.ml.targets import build_future_return_target


def test_future_return_is_exact_and_last_h_rows_are_invalid():
    day = pd.DataFrame({"Time": ["00:00:00", "00:00:01", "00:00:02"], "Price": [100.0, 110.0, 121.0]})
    target = build_future_return_target(day, 1)
    np.testing.assert_allclose(target.iloc[:2], [0.10, 0.10])
    assert np.isnan(target.iloc[2])


def test_missing_timestamp_does_not_create_a_target():
    day = pd.DataFrame({"Time": ["00:00:00", "00:00:02", "00:00:04"], "Price": [100.0, 110.0, 121.0]})
    target = build_future_return_target(day, 1)
    assert target.isna().all()


def test_target_input_must_be_one_ordered_day():
    concatenated = pd.DataFrame({"Time": ["00:00:00", "00:00:01", "00:00:00"], "Price": [100.0, 110.0, 90.0]})
    with pytest.raises(ValueError, match="strictly increasing"):
        build_future_return_target(concatenated, 1)
