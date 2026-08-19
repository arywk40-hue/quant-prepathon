import unittest

import numpy as np

from src.analytics.returns import day_returns
from src.analytics.statistics import acf_values, describe


class Phase4Tests(unittest.TestCase):
    def test_returns_do_not_cross_timestamp_gaps(self):
        prices = np.array([100.0, 101.0, 102.0, 103.0])
        times = np.array(["00:00:00", "00:00:01", "00:00:03", "00:00:04"])
        simple, log_return = day_returns(prices, times, 1)
        self.assertEqual(len(simple), 2)
        np.testing.assert_allclose(simple, [0.01, 103.0 / 102.0 - 1.0])
        self.assertEqual(len(log_return), 2)

    def test_describe_reports_excess_kurtosis_and_quantiles(self):
        result = describe(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        self.assertEqual(result["count"], 5)
        self.assertIn("excess_kurtosis", result)
        self.assertAlmostEqual(result["median"], 3.0)
        self.assertIn("q01", result)

    def test_acf_is_day_local_and_lag_bounded(self):
        values = np.arange(10, dtype=float)
        result = acf_values(values, 4)
        self.assertEqual(len(result), 4)
        self.assertGreater(result[0], 0.5)


if __name__ == "__main__":
    unittest.main()
