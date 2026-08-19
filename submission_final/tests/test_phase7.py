import unittest

import pandas as pd

from src.analytics.taxonomy import assemble_taxonomy


class Phase7TaxonomyTests(unittest.TestCase):
    def test_taxonomy_keeps_nominal_deviation_and_coverage(self):
        structural = pd.DataFrame([{
            "feature": "PB1_T1", "day": 1, "family": "PB", "subfamily": "PB1", "suffix": 1,
            "nominal_window_seconds": 15, "leading_nan_count": 5, "internal_nan_count": 0,
            "trailing_nan_count": 0, "total_nan_count": 5, "total_inf_count": 0,
            "missing_fraction": .5, "stability_class": "leading_only",
        }])
        warmup = pd.DataFrame([{
            "feature": "PB1_T1", "family": "PB", "subfamily": "PB1", "suffix": 1,
            "nominal_window_seconds": 15, "days_expected": 85, "days_present": 1,
            "days_with_feature": 1, "missing_days": "65|66", "mean_warmup_sec": 5,
            "median_warmup_sec": 5, "std_warmup_sec": 0, "min_warmup_sec": 5,
            "max_warmup_sec": 5, "days_matching_nominal": 0, "days_deviating": 1,
            "internal_nan_days": 0, "stability_class": "variable_or_internal_missingness",
        }])
        stats = pd.DataFrame([{"feature": "PB1_T1", "valid_value_count": 1, "value_variance": 0.0, "scale_std_dev": 0.0}])
        result = assemble_taxonomy(structural, warmup, stats, {"available_development_days": 70})
        self.assertEqual(result.loc[0, "nominal_window_status"], "actual_deviations_retained")
        self.assertEqual(result.loc[0, "available_development_days"], 70)


if __name__ == "__main__":
    unittest.main()
