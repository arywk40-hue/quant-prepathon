import pandas as pd

from src.ebx.ml.schemas import DevelopmentScope
from src.ebx.ml.targets import build_future_return_target
from src.ebx.ml.validation import leakage_report, validate_target_alignment


def test_alignment_and_leakage_report_explicitly_exclude_holdout():
    scope = DevelopmentScope(85, (1, 2), tuple(range(3, 80)), tuple(range(86, 109)))
    day = pd.DataFrame({"Time": ["00:00:00", "00:00:01"], "Price": [100.0, 101.0]})
    target = build_future_return_target(day, 1)
    assert validate_target_alignment(day, target, 1)
    report = leakage_report(
        scope=scope,
        split={"training_days": [1], "validation_days": [2], "holdout_days_excluded": list(scope.holdout_days)},
        target_alignment_checked=True,
        partition_reports=[],
        preprocessing_fit_days=[1],
        source_days_loaded=[1, 2],
    )
    assert report["holdout_days_loaded"] == []
    assert report["holdout_exclusion_passed"] is True
    assert report["preprocessing_leakage_free"] is True
