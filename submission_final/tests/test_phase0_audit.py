import tempfile
import unittest
from pathlib import Path

from scripts.analysis.phase0_audit import discover_days, parse_day_filename, scope_for_day
from scripts.analysis.phase1_reconnaissance import (
    OTHER_LADDER,
    PB_LADDER,
    load_config_lists,
    parse_feature,
    parse_time_seconds,
)


class Phase0AuditTests(unittest.TestCase):
    def test_parse_day_filename_requires_exact_shape(self):
        self.assertEqual(parse_day_filename("day1.csv"), 1)
        self.assertIsNone(parse_day_filename("day01.csv"))
        self.assertIsNone(parse_day_filename("day1.txt"))
        self.assertIsNone(parse_day_filename("day.csv"))

    def test_discover_days_is_sorted_and_reports_malformed_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "day2.csv").write_bytes(b"two")
            (root / "day1.csv").write_bytes(b"one")
            (root / "day3.txt").write_bytes(b"bad")
            files, malformed = discover_days(root)
            self.assertEqual([item.day for item in files], [1, 2])
            self.assertEqual(malformed, ["day3.txt"])
            self.assertEqual([item.size_bytes for item in files], [3, 3])

    def test_scope_boundaries(self):
        self.assertEqual(scope_for_day(1), "development")
        self.assertEqual(scope_for_day(85), "development")
        self.assertEqual(scope_for_day(86), "holdout")
        self.assertEqual(scope_for_day(108), "holdout")
        self.assertEqual(scope_for_day(109), "out_of_scope")

    def test_feature_parser_preserves_compound_subfamilies(self):
        pb = parse_feature("PB18_T12")
        self.assertEqual((pb.family, pb.subfamily, pb.suffix), ("PB", "PB18", "12"))
        self.assertEqual(pb.nominal_window_seconds, PB_LADDER[11])

        compound = parse_feature("PV3_B1_T4")
        self.assertEqual((compound.family, compound.subfamily, compound.suffix), ("PV", "PV3_B1", "4"))
        self.assertEqual(compound.nominal_window_seconds, OTHER_LADDER[3])

        pairwise = parse_feature("V8_T1_T2")
        self.assertEqual((pairwise.family, pairwise.subfamily, pairwise.suffix), ("V", "V8_T1", "2"))

    def test_time_parser_rejects_invalid_clock_values(self):
        self.assertEqual(parse_time_seconds("06:28:59"), 23339)
        self.assertIsNone(parse_time_seconds("06:60:00"))
        self.assertIsNone(parse_time_seconds("not-a-time"))

    def test_phase1_settings_are_read_from_config(self):
        settings = load_config_lists(Path("config/config.yaml"))
        self.assertEqual(settings["phase1_sample_days"], [1, 2, 21, 40, 60, 64, 81, 85])
        self.assertEqual(settings["pb_nominal_windows_seconds"], list(PB_LADDER))


if __name__ == "__main__":
    unittest.main()
