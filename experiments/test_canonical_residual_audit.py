import unittest

import numpy as np

from analyze_canonical_residual_matrix import (
    MODEL_SEEDS,
    exact_two_sided_sign_p,
    hierarchical_pair_bootstrap,
)


class CanonicalResidualAuditTests(unittest.TestCase):
    def synthetic_predictions(self):
        labels = np.tile(np.arange(7), 8)
        folders = np.repeat([f"{index:02d}" for index in range(8)], 7)
        better = np.full((len(labels), 7), -2.0)
        better[np.arange(len(labels)), labels] = 2.0
        reference = np.zeros((len(labels), 7))
        return labels, folders, better, reference

    def test_bootstrap_is_deterministic_and_detects_positive_pair(self) -> None:
        labels, folders, better, reference = self.synthetic_predictions()
        better_by_seed = {seed: better.copy() for seed in MODEL_SEEDS}
        reference_by_seed = {seed: reference.copy() for seed in MODEL_SEEDS}
        first = hierarchical_pair_bootstrap(
            "synthetic", better_by_seed, reference_by_seed,
            labels, folders, replicates=100, seed=5901,
        )
        second = hierarchical_pair_bootstrap(
            "synthetic", better_by_seed, reference_by_seed,
            labels, folders, replicates=100, seed=5901,
        )
        self.assertEqual(first, second)
        self.assertGreater(first["uar_delta"]["ci95_low"], 0)
        self.assertGreater(first["nll_delta"]["ci95_low"], 0)

    def test_exact_sign_test_minimum_with_five_positive_seeds(self) -> None:
        self.assertEqual(exact_two_sided_sign_p(np.ones(5)), 0.0625)


if __name__ == "__main__":
    unittest.main()
