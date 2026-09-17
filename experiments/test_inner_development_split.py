import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class InnerDevelopmentSplitTests(unittest.TestCase):
    def test_committed_inner_split_is_sealed_and_disjoint(self) -> None:
        manifest = ROOT / "experiments/manifests/inner_development_seed8042.csv"
        audit_path = ROOT / "experiments/manifests/inner_development_seed8042_audit.json"
        rows = pd.read_csv(manifest, dtype={"source_folder": str})
        audit = json.loads(audit_path.read_text())
        self.assertEqual(len(rows), 1993)
        self.assertEqual(set(rows["original_split"]), {"train"})
        self.assertEqual(set(rows["split"]), {"inner_train", "inner_development"})
        train = rows[rows["split"] == "inner_train"]
        development = rows[rows["split"] == "inner_development"]
        self.assertEqual(len(train), 1557)
        self.assertEqual(len(development), 436)
        self.assertEqual(train["source_folder"].nunique(), 29)
        self.assertEqual(development["source_folder"].nunique(), 8)
        self.assertFalse(set(train["source_folder"]) & set(development["source_folder"]))
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["original_validation_rows_included"], 0)
        self.assertEqual(audit["original_test_rows_included"], 0)
        self.assertTrue(all(
            value == 0 for value in audit["pairwise_disjointness"].values()
        ))


if __name__ == "__main__":
    unittest.main()
