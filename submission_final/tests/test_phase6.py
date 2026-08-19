import unittest

import numpy as np

from src.analytics.regimes import classify_regime, hurst_rs, return_acf, variance_ratio


class Phase6RegimeTests(unittest.TestCase):
    def test_variance_ratio_and_acf_are_day_local(self):
        first = np.ones(100) * 0.01
        second = np.ones(100) * -0.01
        vr, _ = variance_ratio(np.concatenate([first, second]), q=5)
        acf, _ = return_acf(np.concatenate([first, second]), lag=1)
        self.assertTrue(np.isfinite(vr))
        self.assertTrue(np.isfinite(acf))

    def test_hurst_returns_finite_for_nonconstant_series(self):
        rng = np.random.default_rng(7)
        self.assertTrue(np.isfinite(hurst_rs(rng.normal(size=512))))

    def test_conflicting_evidence_is_inconclusive(self):
        regime, confidence, evidence = classify_regime(
            vr=0.8, vr_pvalue=0.001, hurst=0.7, acf=0.1, acf_pvalue=0.001, adf_pvalue=0.8
        )
        self.assertEqual(regime, "random-walk / inconclusive")
        self.assertIn("conflict", evidence)
        self.assertEqual(confidence, "low")


if __name__ == "__main__":
    unittest.main()
