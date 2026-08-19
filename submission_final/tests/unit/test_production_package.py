import unittest
from pathlib import Path

from ebx.config import ProjectConfig, load_yaml_subset
from ebx.forensics.predictive import forward_indices
from ebx.features.parser import parse_feature


class ProductionPackageTests(unittest.TestCase):
    def test_config_is_single_checked_in_source(self):
        root = Path(__file__).resolve().parents[2]
        config = ProjectConfig.from_root(root)
        self.assertEqual(config.development_days[0], 1)
        self.assertEqual(config.development_days[-1], 85)
        self.assertEqual(config.holdout_days, tuple(range(86, 109)))
        self.assertEqual(len(load_yaml_subset(root / "config/config.yaml")), 13)

    def test_facade_exposes_day_safe_feature_and_forward_helpers(self):
        self.assertEqual(parse_feature("PB18_T12").nominal_window_seconds, 10800)
        self.assertEqual(forward_indices([0, 1, 3], 1).tolist(), [1, -1, -1])


if __name__ == "__main__":
    unittest.main()
