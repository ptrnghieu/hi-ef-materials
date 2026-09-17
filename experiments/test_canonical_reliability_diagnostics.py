import unittest

import numpy as np

from canonical_reliability_diagnostics import (
    cross_fit,
    fit_alpha,
    fit_temperature,
    mean_nll,
    oracle_logits,
)


class CanonicalReliabilityDiagnosticTests(unittest.TestCase):
    def test_temperature_search_never_worsens_fit_objective(self) -> None:
        labels = np.asarray([0, 1, 1, 0, 0, 1])
        logits = np.asarray([
            [4, 0], [0, 4], [3, 0], [0, 3], [2, 0], [0, 2],
        ], dtype=float)
        temperature, selected_nll = fit_temperature(logits, labels)
        self.assertGreaterEqual(temperature, 0.05)
        self.assertLessEqual(temperature, 20.0)
        self.assertLessEqual(selected_nll, mean_nll(logits, labels) + 1e-12)

    def test_alpha_prefers_context_when_delta_is_harmful(self) -> None:
        labels = np.asarray([0, 1, 0, 1])
        context = np.asarray([[3, 0], [0, 3], [3, 0], [0, 3]], dtype=float)
        delta = -2.0 * context
        alpha, _ = fit_alpha(context, delta, labels)
        self.assertEqual(alpha, 0.0)

    def test_cross_fit_never_uses_held_out_folder_for_fitting(self) -> None:
        labels = np.asarray([0, 1, 0, 1])
        context = np.asarray([[2, 0], [0, 2], [2, 0], [0, 2]], dtype=float)
        delta = np.zeros_like(context)
        final = context + delta
        folders = np.asarray(["a", "a", "b", "b"])
        outputs, rows = cross_fit(context, delta, final, labels, folders)
        self.assertEqual(set(outputs), {"temperature", "shrinkage"})
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["fit_samples"] == 2 for row in rows))

    def test_oracles_choose_the_better_available_prediction(self) -> None:
        labels = np.asarray([0, 1])
        context = np.asarray([[3, 0], [3, 0]], dtype=float)
        final = np.asarray([[2, 0], [0, 3]], dtype=float)
        outputs, choices = oracle_logits(context, final, labels)
        self.assertFalse(choices["oracle_accuracy_choose_final"][0])
        self.assertTrue(choices["oracle_accuracy_choose_final"][1])
        self.assertEqual(outputs["oracle_accuracy"].argmax(axis=1).tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
