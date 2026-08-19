import pandas as pd

from src.ebx.ml.feature_selection import FROZEN_SCREEN_RULE, load_frozen_feature_screen
from src.ebx.ml.schemas import DevelopmentScope


def _scope():
    return DevelopmentScope(85, tuple(range(1, 65)) + tuple(range(80, 86)), tuple(range(65, 80)), tuple(range(86, 109)))


def test_frozen_screen_rule_is_consumed_without_rediscovery(tmp_path):
    source = pd.DataFrame([
        {"feature": "PB1_T1", "horizon_seconds": 300, "days_scored": 70, "mean_pearson_ic": -0.08, "pearson_pct_same_sign": 0.8, "pearson_t_pvalue": 0.001, "pearson_fdr_reject": True, "pearson_fdr_qvalue": 0.01, "expected_development_days": 85, "available_development_days": 70, "missing_development_days": 15},
        {"feature": "BB1_T1", "horizon_seconds": 300, "days_scored": 70, "mean_pearson_ic": -0.02, "pearson_pct_same_sign": 0.9, "pearson_t_pvalue": 0.001, "pearson_fdr_reject": True, "pearson_fdr_qvalue": 0.01, "expected_development_days": 85, "available_development_days": 70, "missing_development_days": 15},
    ])
    path = tmp_path / "aggregate.csv"
    source.to_csv(path, index=False)
    result = load_frozen_feature_screen(path, _scope())
    assert result.loc[result.feature == "PB1_T1", "eligible_for_ml"].item()
    assert not result.loc[result.feature == "BB1_T1", "eligible_for_ml"].item()
    assert FROZEN_SCREEN_RULE in result.loc[result.feature == "PB1_T1", "reason"].item()
