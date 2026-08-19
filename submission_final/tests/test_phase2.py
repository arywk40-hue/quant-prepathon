import csv
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa

from src.cleaning.missingness import (
    aggregate_cross_day_warmup,
    build_validity_mask,
    classify_structural_missingness,
)
from src.ingestion.discovery import discover_days
from src.ingestion.loader import load_day, schema_record, write_parquet
from src.ingestion.validation import validate_price, validate_schema, validate_timestamps
from scripts.analysis.phase2_process import configured_development_days, load_config


def write_fixture(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)


class Phase2Tests(unittest.TestCase):
    def test_discovery_reports_missing_and_ignores_holdout_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "day1.csv").write_text("Time,Price\n00:00:00,100\n")
            (root / "day3.csv").write_text("Time,Price\n00:00:00,100\n")
            (root / "day86.csv").write_text("this holdout must not be opened\n")
            (root / "dayX.csv").write_text("malformed\n")
            result = discover_days(root, range(1, 4))
            self.assertEqual(tuple(result.files), (1, 3))
            self.assertEqual(result.missing_days, (2,))
            self.assertEqual(result.out_of_scope_ids, (86,))
            self.assertEqual(result.malformed_names, ("dayX.csv",))

    def test_loader_preserves_order_and_validates_day_local_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "day1.csv"
            write_fixture(
                path,
                [
                    ["Time", "Price", "PB1_T1", "VB1_T1"],
                    ["00:00:00", "100", "", ""],
                    ["00:00:01", "101", "10", "5"],
                    ["00:00:03", "0", "", "6"],
                    ["00:00:04", "-1", "12", ""],
                ],
            )
            loaded = load_day(path, 1)
            self.assertEqual(loaded.rows, 4)
            self.assertEqual(loaded.columns, ["Time", "Price", "PB1_T1", "VB1_T1"])
            timestamp = validate_timestamps(loaded.table)
            price = validate_price(loaded.table)
            self.assertEqual(timestamp["non_one_second_intervals"], 1)
            self.assertEqual(timestamp["missing_seconds"], 1)
            self.assertEqual(timestamp["status"], "warning")
            self.assertIn("zero", price["price_flags"])
            self.assertIn("negative", price["price_flags"])
            self.assertEqual(price["status"], "invalid")

    def test_loader_rejects_day_id_filename_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "day86.csv"
            path.write_text("Time,Price\n00:00:00,100\n")
            with self.assertRaises(ValueError):
                load_day(path, 1)

    def test_malformed_timestamp_does_not_create_a_fake_gap(self):
        table = pa.table(
            {
                "Time": ["00:00:00", "not-a-time", "00:00:03"],
                "Price": [100.0, 101.0, 102.0],
            }
        )
        result = validate_timestamps(table)
        self.assertEqual(result["malformed_time_rows"], 1)
        self.assertEqual(result["missing_seconds"], 0)
        self.assertEqual(result["intervals_skipped_due_to_malformed_time"], 1)

    def test_invalid_price_breaks_zero_return_adjacency(self):
        table = pa.table(
            {
                "Time": ["00:00:00", "00:00:01", "00:00:02"],
                "Price": [100.0, None, 100.0],
            }
        )
        result = validate_price(table)
        self.assertEqual(result["zero_return_count"], 0)

    def test_schema_and_structural_missingness_are_explicit(self):
        table = pa.table(
            {
                "Time": ["00:00:00", "00:00:01", "00:00:02", "00:00:03"],
                "Price": [100.0, 101.0, 102.0, 103.0],
                "PB1_T1": [None, 10.0, None, 12.0],
                "VB1_T1": [None, 5.0, 6.0, None],
            }
        )
        reference = schema_record(table)
        self.assertEqual(validate_schema(table, None)["status"], "reference")
        self.assertEqual(validate_schema(table, reference)["status"], "valid")
        missing = classify_structural_missingness(
            table, 1, [0, 1, 2, 3], reference
        )
        pb = next(row for row in missing if row["feature"] == "PB1_T1")
        vb = next(row for row in missing if row["feature"] == "VB1_T1")
        self.assertEqual((pb["leading_nan_count"], pb["internal_nan_count"], pb["trailing_nan_count"]), (1, 1, 0))
        self.assertEqual((vb["leading_nan_count"], vb["internal_nan_count"], vb["trailing_nan_count"]), (1, 0, 1))
        self.assertTrue(pb["unexpected_internal_nan"])
        self.assertFalse(vb["unexpected_internal_nan"])

        mask = build_validity_mask(table, reference)
        self.assertEqual(mask["PB1_T1"].to_pylist(), [False, True, False, True])
        self.assertEqual(mask["VB1_T1"].to_pylist(), [False, True, True, False])

    def test_cross_day_aggregation_and_window_ladder_do_not_fill_missing_days(self):
        table = pa.table(
            {
                "Time": ["00:00:00", "00:00:01"],
                "Price": [100.0, 101.0],
                "PB1_T1": [None, 10.0],
            }
        )
        rows = classify_structural_missingness(table, 1, [0, 1], schema_record(table))
        rows += classify_structural_missingness(table, 3, [0, 1], schema_record(table))
        aggregate = aggregate_cross_day_warmup(rows, (1, 2, 3))
        self.assertEqual(len(aggregate), 1)
        self.assertEqual(aggregate[0]["days_expected"], 3)
        self.assertEqual(aggregate[0]["days_present"], 2)
        self.assertEqual(aggregate[0]["missing_days"], "2")
        self.assertEqual(aggregate[0]["days_matching_nominal"], 0)

    def test_parquet_roundtrip_preserves_rows_schema_and_values(self):
        table = pa.table({"Time": ["00:00:00"], "Price": [100.0], "PB1_T1": [None]})
        with tempfile.TemporaryDirectory() as directory:
            result = write_parquet(table, Path(directory) / "day1.parquet")
            self.assertEqual(result["rows"], 1)
            self.assertEqual(result["columns"], ["Time", "Price", "PB1_T1"])
            self.assertTrue(result["schema_equal"])
            self.assertTrue(result["values_equal"])

    def test_phase2_runner_uses_configured_paths_and_range(self):
        settings = load_config(Path("config/config.yaml"))
        self.assertEqual(settings["raw_data_dir"], "data")
        self.assertEqual(settings["validated_data_dir"], "data/validated")
        self.assertEqual(settings["processed_data_dir"], "data/processed")
        self.assertEqual(settings["development_days"], [1, 85])
        self.assertEqual(settings["parquet_compression"], "zstd")
        self.assertEqual(configured_development_days(settings), tuple(range(1, 86)))
        with self.assertRaises(ValueError):
            configured_development_days({"development_days": [86, 108]})


if __name__ == "__main__":
    unittest.main()
