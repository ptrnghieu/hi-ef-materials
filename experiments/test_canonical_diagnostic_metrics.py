import unittest

import numpy as np

from canonical_diagnostic_metrics import (
    calibration_metrics,
    rank_correlation,
    sample_residual_diagnostics,
)


class CanonicalDiagnosticMetricTests(unittest.TestCase):
    def test_perfect_predictions_have_perfect_recall_and_low_nll(self) -> None:
        labels = np.tile(np.arange(7), 3)
        logits = np.full((len(labels), 7), -5.0)
        logits[np.arange(len(labels)), labels] = 5.0
        metrics, per_class = calibration_metrics(logits, labels)
        self.assertAlmostEqual(metrics["uar"], 1.0)
        self.assertAlmostEqual(metrics["war"], 1.0)
        self.assertLess(metrics["micro_nll"], 0.01)
        self.assertLess(metrics["macro_nll"], 0.01)
        self.assertEqual(len(per_class), 7)

    def test_residual_flip_categories(self) -> None:
        labels = np.asarray([0, 1, 2, 3])
        context = np.asarray([
            [2, 0, 0, 0, 0, 0, 0],
            [2, 1, 0, 0, 0, 0, 0],
            [0, 2, 1, 0, 0, 0, 0],
            [0, 2, 0, 1, 0, 0, 0],
        ], dtype=float)
        final = np.asarray([
            [3, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0],
            [0, 0, 2, 1, 0, 0, 0],
        ], dtype=float)
        result = sample_residual_diagnostics(context, final - context, final, labels)
        self.assertEqual(
            result["flip_category"].tolist(),
            ["unchanged", "beneficial", "unchanged", "wrong_to_wrong"],
        )

    def test_rank_correlation_detects_monotonic_order(self) -> None:
        first = np.asarray([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(rank_correlation(first, first * 10), 1.0)
        self.assertAlmostEqual(rank_correlation(first, -first), -1.0)
        tied = np.asarray([1.0, 1.0, 2.0, 2.0])
        self.assertAlmostEqual(rank_correlation(tied, tied), 1.0)


if __name__ == "__main__":
    unittest.main()
