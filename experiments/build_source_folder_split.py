#!/usr/bin/env python3
"""Build a deterministic source-folder-held-out split for Hi-EF.

The numerical prefix in a clip id (for example ``01`` in ``01/00059``)
is treated as an anonymized source-folder identifier.  Every MCIS from a
source folder is assigned to the same partition.  The script balances sample
counts and clip-III/clip-IV emotion distributions while preserving an exact
37/8/8 allocation of the 53 released source folders.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    clips: tuple[str, str, str, str]
    source_folder: str
    clip3_emotion: str
    clip4_emotion: str


@dataclass(frozen=True)
class FolderGroup:
    folder_id: str
    sample_indices: tuple[int, ...]
    clip3_counts: Counter[str]
    clip4_counts: Counter[str]

    @property
    def size(self) -> int:
        return len(self.sample_indices)


def parse_annotations(path: Path) -> dict[str, str]:
    annotations: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) != 9:
                raise ValueError(
                    f"{path}:{line_number}: expected 9 columns, found {len(row)}"
                )
            clip_id = row[0].strip()
            emotion = row[7].strip().lower()
            if emotion and emotion not in EMOTIONS:
                raise ValueError(
                    f"{path}:{line_number}: unknown emotion {emotion!r} for {clip_id}"
                )
            if clip_id in annotations:
                raise ValueError(f"Duplicate annotation for clip {clip_id}")
            annotations[clip_id] = emotion
    return annotations


def clip_folder(clip_id: str) -> str:
    if "/" not in clip_id:
        raise ValueError(f"Clip id has no source-folder prefix: {clip_id!r}")
    folder, filename = clip_id.split("/", 1)
    if not folder or not filename:
        raise ValueError(f"Malformed clip id: {clip_id!r}")
    return folder


def parse_samples(path: Path, annotations: dict[str, str]) -> list[Sample]:
    samples: list[Sample] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ["sample_name", "clip1", "clip2", "clip3", "clip4"]
        if reader.fieldnames != expected:
            raise ValueError(
                f"Unexpected sample schema: {reader.fieldnames}; expected {expected}"
            )
        for line_number, row in enumerate(reader, start=2):
            sample_id = row["sample_name"].strip()
            clips = tuple(row[f"clip{i}"].strip() for i in range(1, 5))
            if not sample_id or any(not clip for clip in clips):
                raise ValueError(f"{path}:{line_number}: blank sample or clip identifier")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample id {sample_id}")
            seen_ids.add(sample_id)

            folders = {clip_folder(clip) for clip in clips}
            if len(folders) != 1:
                raise ValueError(
                    f"{sample_id}: clips cross source folders: {sorted(folders)}"
                )
            clip3_emotion = annotations.get(clips[2], "")
            clip4_emotion = annotations.get(clips[3], "")
            if not clip3_emotion or not clip4_emotion:
                raise ValueError(
                    f"{sample_id}: missing clip III/IV emotion: "
                    f"{clips[2]}={clip3_emotion!r}, {clips[3]}={clip4_emotion!r}"
                )
            samples.append(
                Sample(
                    sample_id=sample_id,
                    clips=clips,  # type: ignore[arg-type]
                    source_folder=next(iter(folders)),
                    clip3_emotion=clip3_emotion,
                    clip4_emotion=clip4_emotion,
                )
            )
    return samples


def build_folder_groups(samples: list[Sample]) -> list[FolderGroup]:
    members: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        members.setdefault(sample.source_folder, []).append(index)
    return [
        FolderGroup(
            folder_id=folder,
            sample_indices=tuple(indices),
            clip3_counts=Counter(samples[i].clip3_emotion for i in indices),
            clip4_counts=Counter(samples[i].clip4_emotion for i in indices),
        )
        for folder, indices in sorted(members.items())
    ]


def largest_remainder_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {split: total * ratios[split] for split in SPLITS}
    counts = {split: math.floor(raw[split]) for split in SPLITS}
    remainder = total - sum(counts.values())
    order = sorted(SPLITS, key=lambda split: (-(raw[split] - counts[split]), split))
    for split in order[:remainder]:
        counts[split] += 1
    return counts


def summarize_assignment(
    groups: list[FolderGroup], assignment: dict[str, str]
) -> tuple[dict[str, int], dict[str, Counter[str]], dict[str, Counter[str]]]:
    sizes = {split: 0 for split in SPLITS}
    clip3 = {split: Counter() for split in SPLITS}
    clip4 = {split: Counter() for split in SPLITS}
    for group in groups:
        split = assignment[group.folder_id]
        sizes[split] += group.size
        clip3[split].update(group.clip3_counts)
        clip4[split].update(group.clip4_counts)
    return sizes, clip3, clip4


def allocation_cost(
    groups: list[FolderGroup],
    assignment: dict[str, str],
    ratios: dict[str, float],
) -> float:
    sizes, clip3_counts, clip4_counts = summarize_assignment(groups, assignment)
    total_size = sum(group.size for group in groups)
    total_clip3 = sum((group.clip3_counts for group in groups), Counter())
    total_clip4 = sum((group.clip4_counts for group in groups), Counter())

    size_cost = sum(
        ((sizes[split] - total_size * ratios[split]) / max(total_size * ratios[split], 1)) ** 2
        for split in SPLITS
    )
    clip3_cost = sum(
        (
            (clip3_counts[split][emotion] - total_clip3[emotion] * ratios[split])
            / max(total_clip3[emotion] * ratios[split], 1)
        ) ** 2
        for split in SPLITS
        for emotion in EMOTIONS
    )
    clip4_cost = sum(
        (
            (clip4_counts[split][emotion] - total_clip4[emotion] * ratios[split])
            / max(total_clip4[emotion] * ratios[split], 1)
        ) ** 2
        for split in SPLITS
        for emotion in EMOTIONS
    )
    # Folder groups vary considerably in size, so sample-count balance needs a
    # stronger weight than it would in an item-level split. Clip IV is the EF
    # target and therefore receives more weight than the diagnostic clip-III
    # distribution.
    return 100.0 * size_cost + 0.5 * clip3_cost + 2.0 * clip4_cost


def optimize_assignment(
    groups: list[FolderGroup],
    ratios: dict[str, float],
    seed: int,
    restarts: int,
) -> dict[str, str]:
    """Search deterministic random allocations with fixed folder counts."""
    if restarts < 1:
        raise ValueError("restarts must be at least 1")
    folder_counts = largest_remainder_counts(len(groups), ratios)
    slots = [split for split in SPLITS for _ in range(folder_counts[split])]
    folder_ids = [group.folder_id for group in groups]
    rng = random.Random(seed)
    best_cost = math.inf
    best_key: tuple[str, ...] | None = None
    best_assignment: dict[str, str] | None = None

    for _ in range(restarts):
        shuffled_slots = slots.copy()
        rng.shuffle(shuffled_slots)
        assignment = dict(zip(folder_ids, shuffled_slots, strict=True))
        cost = allocation_cost(groups, assignment, ratios)
        key = tuple(assignment[folder] for folder in folder_ids)
        if cost < best_cost or (math.isclose(cost, best_cost) and key < (best_key or key)):
            best_cost = cost
            best_key = key
            best_assignment = assignment

    if best_assignment is None:
        raise RuntimeError("No source-folder assignment was produced")
    return best_assignment


def adjacent_pairs(sample: Sample) -> set[tuple[str, str]]:
    return {(sample.clips[i], sample.clips[i + 1]) for i in range(3)}


def audit_split(
    samples: list[Sample], assignment: dict[str, str], ratios: dict[str, float]
) -> dict[str, object]:
    by_split = {
        split: [sample for sample in samples if assignment[sample.source_folder] == split]
        for split in SPLITS
    }
    folders = {
        split: {sample.source_folder for sample in split_samples}
        for split, split_samples in by_split.items()
    }
    clips = {
        split: {clip for sample in split_samples for clip in sample.clips}
        for split, split_samples in by_split.items()
    }
    pairs = {
        split: {pair for sample in split_samples for pair in adjacent_pairs(sample)}
        for split, split_samples in by_split.items()
    }

    pairwise: dict[str, dict[str, int]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        pairwise[f"{left}-{right}"] = {
            "shared_source_folders": len(folders[left] & folders[right]),
            "shared_clips": len(clips[left] & clips[right]),
            "shared_adjacent_pairs": len(pairs[left] & pairs[right]),
        }

    sample_counts = {split: len(split_samples) for split, split_samples in by_split.items()}
    total_samples = len(samples)
    actual_ratios = {
        split: sample_counts[split] / total_samples for split in SPLITS
    }
    passed = all(all(value == 0 for value in result.values()) for result in pairwise.values())
    return {
        "protocol": "source-folder-held-out",
        "source_folder_semantics": "anonymized; likely episode-level proxy, awaiting author confirmation",
        "ratios": ratios,
        "sample_counts": sample_counts,
        "actual_sample_ratios": actual_ratios,
        "source_folder_counts": {split: len(folders[split]) for split in SPLITS},
        "source_folders": {split: sorted(folders[split]) for split in SPLITS},
        "clip3_emotion_counts": {
            split: dict(sorted(Counter(s.clip3_emotion for s in split_samples).items()))
            for split, split_samples in by_split.items()
        },
        "clip4_emotion_counts": {
            split: dict(sorted(Counter(s.clip4_emotion for s in split_samples).items()))
            for split, split_samples in by_split.items()
        },
        "pairwise_leakage": pairwise,
        "passed": passed,
    }


def write_manifest(
    path: Path, samples: list[Sample], assignment: dict[str, str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "sample_id", "split", "source_folder", "clip1", "clip2", "clip3", "clip4",
            "clip3_emotion", "clip4_emotion",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample.sample_id,
                    "split": assignment[sample.source_folder],
                    "source_folder": sample.source_folder,
                    "clip1": sample.clips[0],
                    "clip2": sample.clips[1],
                    "clip3": sample.clips[2],
                    "clip4": sample.clips[3],
                    "clip3_emotion": sample.clip3_emotion,
                    "clip4_emotion": sample.clip4_emotion,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-csv", type=Path, default=Path("sample.csv"))
    parser.add_argument("--annotation-csv", type=Path, default=Path("annotation.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/manifests"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restarts", type=int, default=20_000)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    annotations = parse_annotations(args.annotation_csv)
    samples = parse_samples(args.sample_csv, annotations)
    groups = build_folder_groups(samples)
    assignment = optimize_assignment(groups, ratios, args.seed, args.restarts)

    manifest_path = args.output_dir / f"source_folder_split_seed{args.seed}.csv"
    audit_path = args.output_dir / f"source_folder_split_seed{args.seed}_audit.json"
    write_manifest(manifest_path, samples, assignment)
    audit = audit_split(samples, assignment, ratios)
    audit["seed"] = args.seed
    audit["search_restarts"] = args.restarts
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not audit["passed"]:
        raise RuntimeError(f"Leakage audit failed; inspect {audit_path}")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"Manifest: {manifest_path}")
    print(f"Audit: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
