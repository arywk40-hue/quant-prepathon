import unittest

import numpy as np

from src.analytics.tails import hill_tail_index, sigma_probability


class Phase5Tests(unittest.TestCase):
    def test_sigma_probability_matches_gaussian_reference(self):
        self.assertAlmostEqual(sigma_probability(1), 0.31731050786291415, places=12)
        self.assertAlmostEqual(sigma_probability(3), 0.002699796063260207, places=12)

    def test_hill_estimator_returns_valid_descriptive_output(self):
        values = np.linspace(1.0, 100.0, 1000)
        alpha, k, threshold = hill_tail_index(values, k=100)
        self.assertEqual(k, 100)
        self.assertTrue(np.isfinite(alpha))
        self.assertGreater(threshold, 0)


if __name__ == "__main__":
    unittest.main()
