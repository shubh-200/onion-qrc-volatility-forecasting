import unittest

import numpy as np

from gic.prototype.readout import (
    IQPQuantumKernel,
    QuantumKernelClassifier,
    VolQRCReadout,
    block_bootstrap_confidence_interval,
    compute_metrics,
    compute_regime_metrics,
    diebold_mariano,
    mincer_zarnowitz,
    model_confidence_set,
)


class ReadoutTests(unittest.TestCase):
    def test_default_gating_is_causal_and_oracle_is_explicit(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(36, 4))
        regimes = (X[:, 0] > 0).astype(int)
        y = rng.normal(size=len(X))

        changed = regimes.copy()
        changed[14] = 1 - changed[14]
        causal_a = VolQRCReadout(regime_cv_splits=3).fit(X, y, regimes)
        causal_b = VolQRCReadout(regime_cv_splits=3).fit(X, y, changed)

        # The gate at t=14 is trained only on observations strictly before its fold.
        self.assertEqual(causal_a.gating_mode, "cross_fitted")
        self.assertIsNotNone(causal_a.training_regime_predictions_)
        self.assertIsNotNone(causal_b.training_regime_predictions_)
        self.assertEqual(
            causal_a.training_regime_predictions_[14],
            causal_b.training_regime_predictions_[14],
        )
        self.assertTrue(any(value is None for value in causal_a.training_regime_predictions_))
        self.assertFalse(np.array_equal(causal_a.training_regime_predictions_, regimes))

        oracle = VolQRCReadout(gating_mode="oracle").fit(X, y, regimes)
        np.testing.assert_array_equal(oracle.training_regime_predictions_, regimes)

    def test_iqp_and_rbf_gram_matrices_are_symmetric_psd(self):
        rng = np.random.default_rng(8)
        X = rng.normal(size=(9, 3))
        kernels = [
            IQPQuantumKernel(n_qubits=3, scale=0.7).compute_kernel_matrix(X),
            QuantumKernelClassifier(gamma=0.4)._compute_kernel(X),
        ]
        for K in kernels:
            np.testing.assert_allclose(K, K.T, atol=1e-12)
            np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)
            self.assertGreaterEqual(np.linalg.eigvalsh(K).min(), -1e-10)

    def test_classifier_accepts_pca_reduced_inputs(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(30, 8))
        y = np.repeat(["calm", "stress"], 15)
        X[15:] += 2.0
        classifier = QuantumKernelClassifier(pca_components=3).fit(X, y)
        self.assertIsNotNone(classifier._support)
        self.assertEqual(classifier._support.shape, (30, 3))
        self.assertEqual(classifier.predict(X[:4]).shape, (4,))

    def test_forecast_and_regime_metrics(self):
        y_true = np.array([0.0, np.log(2.0), np.log(4.0)])
        y_pred = np.array([0.0, np.log(2.0), np.log(2.0)])
        metrics = compute_metrics(y_true, y_pred, is_log_rv=True)
        errors = y_true - y_pred
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(np.mean(errors ** 2)))
        self.assertAlmostEqual(metrics["mae"], np.mean(np.abs(errors)))
        self.assertAlmostEqual(metrics["qlike"], (2.0 - np.log(2.0) - 1.0) / 3.0)
        self.assertEqual(metrics["n_obs"], 3)

        regimes = compute_regime_metrics(
            np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2])
        )
        self.assertAlmostEqual(regimes["accuracy"], 0.75)
        self.assertAlmostEqual(regimes["balanced_accuracy"], (0.5 + 1.0 + 1.0) / 3.0)
        self.assertEqual(regimes["confusion_matrix"].shape, (3, 3))

    def test_hac_mz_and_dm_statistics_are_well_formed(self):
        rng = np.random.default_rng(19)
        y = rng.normal(size=120)
        good = y + rng.normal(scale=0.05, size=len(y))
        bad = y + rng.normal(scale=0.8, size=len(y))

        mz = mincer_zarnowitz(y, good, hac_lags=3)
        self.assertEqual(mz["hac_lags"], 3)
        self.assertTrue(0.0 <= mz["joint_pvalue"] <= 1.0)
        dm = diebold_mariano(y, good, bad, loss="squared", hac_lags=2)
        self.assertLess(dm["mean_loss_difference"], 0.0)
        self.assertTrue(0.0 <= dm["pvalue"] <= 1.0)

    def test_block_bootstrap_is_deterministic_with_seed(self):
        y = np.linspace(-1.0, 1.0, 40)
        pred = y + 0.2 * np.sin(np.arange(40))
        first = block_bootstrap_confidence_interval(
            y, pred, metric="rmse", block_size=5,
            n_bootstrap=200, random_state=123,
        )
        second = block_bootstrap_confidence_interval(
            y, pred, metric="rmse", block_size=5,
            n_bootstrap=200, random_state=123,
        )
        self.assertEqual(first, second)
        self.assertTrue(first["lower"] <= first["estimate"] <= first["upper"])


if __name__ == "__main__":
    unittest.main()


class MCSTests(unittest.TestCase):
    def test_mcs_eliminates_clearly_worse_model(self):
        """A model with much larger losses should be excluded from the MCS."""
        rng = np.random.default_rng(7)
        n = 80
        good = rng.standard_normal(n) ** 2 * 0.1
        bad = rng.standard_normal(n) ** 2 * 5.0  # much higher losses
        result = model_confidence_set(
            {"good": good, "bad": bad},
            alpha=0.10,
            n_bootstrap=100,
            block_size=4,
            random_state=99,
        )
        self.assertIn("good", result["included"])
        self.assertIn("bad", result["included"])
        self.assertTrue(result["included"]["good"])
        self.assertFalse(result["included"]["bad"])
        self.assertEqual(result["ranking"][0], "good")
        self.assertEqual(result["n_obs"], n)

    def test_mcs_keeps_equivalent_models(self):
        """Two models with losses drawn from the same distribution should both stay in the MCS."""
        rng = np.random.default_rng(13)
        n = 100
        # Both models have identical expected loss — only differ by rounding noise
        base = rng.standard_normal(n) ** 2
        # Identical loss arrays: MCS cannot distinguish them, both must be included
        result = model_confidence_set(
            {"a": base.copy(), "b": base.copy()},
            alpha=0.10,
            n_bootstrap=100,
            block_size=5,
            random_state=77,
        )
        self.assertTrue(result["included"]["a"])
        self.assertTrue(result["included"]["b"])
        self.assertEqual(result["alpha"], 0.10)

    def test_mcs_raises_on_single_model_and_non_finite(self):
        with self.assertRaises(ValueError, msg="MCS requires at least two models"):
            model_confidence_set({"only": np.ones(10)})
        with self.assertRaises(ValueError, msg="non-finite"):
            model_confidence_set({"a": np.array([1.0, np.nan]), "b": np.array([1.0, 2.0])})
