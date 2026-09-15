#!/usr/bin/env python3
"""Audit all frozen Hi-EF input features before training."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch


REQUIRED_SHAPES = {
    "ori_features": (16, 512),
    "face_features": (16, 512),
    "text_feature": (512,),
    "audio_feature": (527,),
}


def feature_path(features_dir: Path, clip_id: str) -> Path:
    return features_dir / f"{clip_id.replace('/', '_')}.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working/feature_preflight.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"source_folder": str})

    usage: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in manifest.itertuples(index=False):
        for position in ("clip1", "clip2", "clip3"):
            usage[getattr(row, position)].add((row.split, position))

    diagnostics = {
        "missing_files": [],
        "load_errors": [],
        "clip_id_mismatches": [],
        "wrong_shapes": [],
        "nonfinite_tensors": [],
        "invalid_face_masks": [],
    }
    clip_stats: dict[str, dict[str, object]] = {}

    for index, clip_id in enumerate(sorted(usage), start=1):
        path = feature_path(args.features_dir, clip_id)
        if not path.exists():
            diagnostics["missing_files"].append(clip_id)
            continue
        try:
            item = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:  # noqa: BLE001 - report every corrupt input
            diagnostics["load_errors"].append({"clip_id": clip_id, "error": repr(error)})
            continue

        if item.get("clip_id") != clip_id:
            diagnostics["clip_id_mismatches"].append(
                {"expected": clip_id, "found": item.get("clip_id")}
            )

        for key, expected_shape in REQUIRED_SHAPES.items():
            value = item.get(key)
            found_shape = tuple(value.shape) if torch.is_tensor(value) else None
            if found_shape != expected_shape:
                diagnostics["wrong_shapes"].append(
                    {
                        "clip_id": clip_id,
                        "key": key,
                        "expected": expected_shape,
                        "found": found_shape,
                    }
                )
            elif not bool(torch.isfinite(value).all()):
                diagnostics["nonfinite_tensors"].append(
                    {"clip_id": clip_id, "key": key}
                )

        mask = item.get("face_valid_mask")
        mask_valid = isinstance(mask, (list, tuple)) and len(mask) == 16
        if not mask_valid:
            diagnostics["invalid_face_masks"].append(clip_id)
            valid_face_frames = None
        else:
            valid_face_frames = sum(bool(value) for value in mask)

        face = item.get("face_features")
        audio = item.get("audio_feature")
        clip_stats[clip_id] = {
            "valid_face_frames": valid_face_frames,
            "face_all_zero": bool(torch.is_tensor(face) and torch.count_nonzero(face) == 0),
            "audio_found": bool(item.get("audio_found", False)),
            "audio_all_zero": bool(torch.is_tensor(audio) and torch.count_nonzero(audio) == 0),
        }
        if index % 500 == 0 or index == len(usage):
            print(f"Audited {index}/{len(usage)} unique input clips")

    def aggregate(predicate) -> dict[str, int | float]:
        clip_ids = [clip_id for clip_id in usage if predicate(clip_id) and clip_id in clip_stats]
        total = len(clip_ids)
        no_face = sum(clip_stats[c]["valid_face_frames"] == 0 for c in clip_ids)
        fewer_than_half = sum(
            isinstance(clip_stats[c]["valid_face_frames"], int)
            and clip_stats[c]["valid_face_frames"] < 8
            for c in clip_ids
        )
        return {
            "unique_clips": total,
            "zero_valid_face_clips": no_face,
            "zero_valid_face_rate": no_face / total if total else 0.0,
            "fewer_than_8_valid_face_clips": fewer_than_half,
            "fewer_than_8_valid_face_rate": fewer_than_half / total if total else 0.0,
            "all_zero_face_features": sum(clip_stats[c]["face_all_zero"] for c in clip_ids),
            "audio_not_found": sum(not clip_stats[c]["audio_found"] for c in clip_ids),
            "all_zero_audio_features": sum(clip_stats[c]["audio_all_zero"] for c in clip_ids),
        }

    by_position = {
        position: aggregate(
            lambda clip_id, p=position: any(used_position == p for _, used_position in usage[clip_id])
        )
        for position in ("clip1", "clip2", "clip3")
    }
    by_split = {
        split: aggregate(
            lambda clip_id, s=split: any(used_split == s for used_split, _ in usage[clip_id])
        )
        for split in ("train", "val", "test")
    }

    fatal_keys = (
        "missing_files",
        "load_errors",
        "clip_id_mismatches",
        "wrong_shapes",
        "nonfinite_tensors",
        "invalid_face_masks",
    )
    result = {
        "passed": not any(diagnostics[key] for key in fatal_keys),
        "manifest_rows": len(manifest),
        "unique_input_clips": len(usage),
        "by_position": by_position,
        "by_split": by_split,
        "diagnostic_counts": {key: len(value) for key, value in diagnostics.items()},
        "diagnostic_examples": {key: value[:20] for key, value in diagnostics.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Report: {args.output}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
