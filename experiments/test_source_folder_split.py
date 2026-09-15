import csv
import tempfile
import unittest
from pathlib import Path

import build_source_folder_split as splitter


class SourceFolderSplitTests(unittest.TestCase):
    def test_largest_remainder_counts_for_released_folder_count(self):
        counts = splitter.largest_remainder_counts(
            53, {"train": 0.70, "val": 0.15, "test": 0.15}
        )
        self.assertEqual(counts, {"train": 37, "val": 8, "test": 8})

    def test_cross_folder_mcis_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = root / "annotation.csv"
            sample = root / "sample.csv"
            annotation.write_text(
                "01/a,text,,,,,,happy,1\n"
                "01/b,text,,,,,,neutral,1\n"
                "02/c,text,,,,,,sad,1\n"
                "02/d,text,,,,,,angry,1\n",
                encoding="utf-8",
            )
            sample.write_text(
                "sample_name,clip1,clip2,clip3,clip4\n"
                "s1,01/a,01/b,02/c,02/d\n",
                encoding="utf-8",
            )
            annotations = splitter.parse_annotations(annotation)
            with self.assertRaisesRegex(ValueError, "cross source folders"):
                splitter.parse_samples(sample, annotations)

    def test_audit_detects_no_cross_split_folder_or_clip_overlap(self):
        samples = [
            splitter.Sample(
                sample_id=f"s{i}",
                clips=(f"{folder}/a", f"{folder}/b", f"{folder}/c", f"{folder}/d"),
                source_folder=folder,
                clip3_emotion="happy",
                clip4_emotion="neutral",
            )
            for i, folder in enumerate(("01", "02", "03"), start=1)
        ]
        assignment = {"01": "train", "02": "val", "03": "test"}
        audit = splitter.audit_split(
            samples, assignment, {"train": 1 / 3, "val": 1 / 3, "test": 1 / 3}
        )
        self.assertTrue(audit["passed"])
        for result in audit["pairwise_leakage"].values():
            self.assertEqual(result["shared_source_folders"], 0)
            self.assertEqual(result["shared_clips"], 0)
            self.assertEqual(result["shared_adjacent_pairs"], 0)

    def test_manifest_uses_one_split_per_folder(self):
        samples = [
            splitter.Sample("s1", ("01/a", "01/b", "01/c", "01/d"), "01", "sad", "sad"),
            splitter.Sample("s2", ("01/e", "01/f", "01/g", "01/h"), "01", "happy", "happy"),
            splitter.Sample("s3", ("02/a", "02/b", "02/c", "02/d"), "02", "fear", "fear"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            splitter.write_manifest(path, samples, {"01": "train", "02": "test"})
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        folder_splits: dict[str, set[str]] = {}
        for row in rows:
            folder_splits.setdefault(row["source_folder"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in folder_splits.values()))


if __name__ == "__main__":
    unittest.main()
