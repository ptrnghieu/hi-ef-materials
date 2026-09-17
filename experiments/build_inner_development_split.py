#!/usr/bin/env python3
"""Build the frozen source-folder inner split from original training rows only."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random

import pandas as pd


INNER_SPLITS = ("inner_train", "inner_development")
EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_cost(rows: pd.DataFrame, development_folders: set[str]) -> float:
    total = len(rows)
    dev = rows[rows["source_folder"].isin(development_folders)]
    target_ratio = 8 / 37
    size_cost = ((len(dev) / total - target_ratio) / target_ratio) ** 2
    total_counts = Counter(rows["clip4_emotion"])
    dev_counts = Counter(dev["clip4_emotion"])
    emotion_cost = sum(
        ((dev_counts[e] - total_counts[e] * target_ratio) /
         max(total_counts[e] * target_ratio, 1.0)) ** 2
        for e in EMOTIONS
    )
    return 100.0 * size_cost + 2.0 * emotion_cost


def choose_development_folders(
    rows: pd.DataFrame, seed: int, restarts: int
) -> tuple[set[str], float]:
    folders = sorted(rows["source_folder"].unique())
    if len(folders) != 37:
        raise RuntimeError(f"Expected 37 original training folders, found {len(folders)}")
    rng = random.Random(seed)
    best: tuple[float, tuple[str, ...]] | None = None
    for _ in range(restarts):
        candidate = tuple(sorted(rng.sample(folders, 8)))
        cost = split_cost(rows, set(candidate))
        if best is None or (cost, candidate) < best:
            best = (cost, candidate)
    assert best is not None
    return set(best[1]), best[0]


def clip_set(rows: pd.DataFrame) -> set[str]:
    return set().union(*(set(rows[f"clip{i}"]) for i in range(1, 5)))


def adjacent_pairs(rows: pd.DataFrame) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for row in rows.itertuples(index=False):
        clips = [getattr(row, f"clip{i}") for i in range(1, 5)]
        pairs.update(zip(clips, clips[1:]))
    return pairs


def audit(rows: pd.DataFrame, source_manifest: Path, seed: int, cost: float) -> dict[str, object]:
    train = rows[rows["split"] == "inner_train"]
    dev = rows[rows["split"] == "inner_development"]
    overlap = {
        "shared_source_folders": len(set(train["source_folder"]) & set(dev["source_folder"])),
        "shared_clips": len(clip_set(train) & clip_set(dev)),
        "shared_adjacent_pairs": len(adjacent_pairs(train) & adjacent_pairs(dev)),
        "shared_sample_ids": len(set(train["sample_id"]) & set(dev["sample_id"])),
    }
    return {
        "protocol": "original-train-source-folder-inner-development-v1",
        "seed": seed,
        "optimizer_restarts": 50000,
        "allocation_cost": cost,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "eligible_original_split": "train",
        "original_validation_rows_included": 0,
        "original_test_rows_included": 0,
        "sample_counts": {name: int((rows["split"] == name).sum()) for name in INNER_SPLITS},
        "source_folder_counts": {
            name: int(rows.loc[rows["split"] == name, "source_folder"].nunique())
            for name in INNER_SPLITS
        },
        "source_folders": {
            name: sorted(rows.loc[rows["split"] == name, "source_folder"].unique())
            for name in INNER_SPLITS
        },
        "clip4_emotion_counts": {
            name: dict(sorted(Counter(
                rows.loc[rows["split"] == name, "clip4_emotion"]
            ).items()))
            for name in INNER_SPLITS
        },
        "pairwise_disjointness": overlap,
        "passed": all(value == 0 for value in overlap.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=8042)
    parser.add_argument("--restarts", type=int, default=50000)
    args = parser.parse_args()

    source = pd.read_csv(args.manifest, dtype={"source_folder": str})
    eligible = source[source["split"] == "train"].copy()
    if len(eligible) != 1993 or set(eligible["split"]) != {"train"}:
        raise RuntimeError("Frozen source manifest does not contain the expected 1,993 train rows")
    development, cost = choose_development_folders(eligible, args.seed, args.restarts)
    eligible.insert(2, "original_split", "train")
    eligible["split"] = eligible["source_folder"].map(
        lambda folder: "inner_development" if folder in development else "inner_train"
    )
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(args.output_manifest, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    result = audit(eligible, args.manifest, args.seed, cost)
    result["optimizer_restarts"] = args.restarts
    if not result["passed"]:
        raise RuntimeError(f"Inner split audit failed: {result['pairwise_disjointness']}")
    args.output_audit.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
