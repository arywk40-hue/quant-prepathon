import numpy as np
import pandas as pd

from src.ebx.ml.preprocessing import TrainOnlyStandardizer, complete_case_mask
from src.ebx.ml.schemas import DevelopmentScope
from src.ebx.ml.splits import chronological_split


def _scope():
    return DevelopmentScope(85, (1, 2, 80, 81), tuple(range(3, 80)), tuple(range(86, 109)))


def test_split_is_whole_day_chronological_and_retains_missing_days():
    split = chronological_split(_scope())
    assert split["training_days"] == [1, 2]
    assert split["validation_days"] == [80, 81]
    assert split["missing_days"] == list(range(3, 80))


def test_standardizer_uses_training_parameters_only():
    train = pd.DataFrame({"x": [1.0, 3.0], "y": [10.0, 14.0]})
    validation = pd.DataFrame({"x": [100.0], "y": [100.0]})
    scaler = TrainOnlyStandardizer(("x", "y"))
    scaler.update(train)
    scaler.finalize()
    transformed = scaler.transform(validation)
    assert transformed.iloc[0, 0] > 50
    assert scaler.count == 2


def test_validity_mask_does_not_impute_nan():
    frame = pd.DataFrame({"x": [1.0, np.nan], "y": [2.0, 3.0]})
    target = pd.Series([0.1, 0.2])
    target_valid, feature_valid, complete = complete_case_mask(frame, ("x", "y"), target)
    assert target_valid.tolist() == [True, True]
    assert feature_valid.tolist() == [True, False]
    assert complete.tolist() == [True, False]
