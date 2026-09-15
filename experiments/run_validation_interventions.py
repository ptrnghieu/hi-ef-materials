#!/usr/bin/env python3
"""Evaluate fixed clip-III interventions on validation using frozen checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_baselines import (
    FrozenFeatureBaseline,
    HiEFFrozenDataset,
    collate_batch,
    evaluate,
    public_metrics,
    save_predictions,
    seed_everything,
    seed_worker,
)


REPLACEMENT_CONDITIONS = (
    "global_shuffle",
    "same_source_wrong_emotion",
    "same_emotion_different_source",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024]
    )
    parser.add_argument("--replacement-seed-start", type=int, default=1701)
    parser.add_argument("--replacement-replicates", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


class InterventionDataset(HiEFFrozenDataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        features_dir: Path,
        condition: str,
        replacements: dict[str, str] | None = None,
    ) -> None:
        super().__init__(rows, features_dir, num_clips=3)
        self.condition = condition
        self.replacements = replacements or {}

    def __getitem__(self, index: int) -> dict[str, object]:
        item = super().__getitem__(index)
        sample_id = str(item["sample_id"])
        if self.condition == "true":
            return item
        if self.condition == "zero":
            clip = item["clip3"]
            item["clip3"] = {
                key: torch.zeros_like(value)
                for key, value in clip.items()
            }
            return item
        replacement = self.replacements.get(sample_id)
        if replacement is None:
            raise KeyError(f"No replacement for {sample_id} under {self.condition}")
        item["clip3"] = self._load_clip(replacement)
        return item


def build_intervention_loader(
    rows: pd.DataFrame,
    features_dir: Path,
    condition: str,
    replacements: dict[str, str] | None,
    batch_size: int,
    workers: int,
    seed: int,
):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        InterventionDataset(rows, features_dir, condition, replacements),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=collate_batch,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def clip_pool(rows: pd.DataFrame) -> pd.DataFrame:
    columns = ["clip3", "clip3_emotion", "source_folder"]
    pool = rows[columns].drop_duplicates().reset_index(drop=True)
    conflicts = pool.groupby("clip3").agg(
        emotions=("clip3_emotion", "nunique"),
        folders=("source_folder", "nunique"),
    )
    if (conflicts > 1).any().any():
        raise ValueError("A clip III has inconsistent emotion or source-folder metadata")
    return pool.drop_duplicates("clip3").reset_index(drop=True)


def global_derangement(rows: pd.DataFrame, rng: random.Random) -> list[str]:
    indices = list(range(len(rows)))
    forbidden = [
        {str(row[f"clip{i}"]) for i in range(1, 5)}
        for _, row in rows.iterrows()
    ]
    for _ in range(10000):
        permutation = indices.copy()
        rng.shuffle(permutation)
        replacements = [str(rows.iloc[j]["clip3"]) for j in permutation]
        if all(replacements[i] not in forbidden[i] for i in indices):
            return replacements
    raise RuntimeError("Could not construct a valid global clip-III derangement")


def matched_replacements(
    rows: pd.DataFrame, pool: pd.DataFrame, condition: str, rng: random.Random
) -> list[str]:
    replacements = []
    for _, row in rows.iterrows():
        forbidden = {str(row[f"clip{i}"]) for i in range(1, 5)}
        if condition == "same_source_wrong_emotion":
            candidates = pool[
                (pool["source_folder"] == row["source_folder"])
                & (pool["clip3_emotion"] != row["clip3_emotion"])
                & (~pool["clip3"].isin(forbidden))
            ]
        elif condition == "same_emotion_different_source":
            candidates = pool[
                (pool["clip3_emotion"] == row["clip3_emotion"])
                & (pool["source_folder"] != row["source_folder"])
                & (~pool["clip3"].isin(forbidden))
            ]
        else:
            raise ValueError(f"Unknown matched condition: {condition}")
        if candidates.empty:
            raise RuntimeError(
                f"No candidate for {row['sample_id']} under condition {condition}"
            )
        replacements.append(str(candidates.iloc[rng.randrange(len(candidates))]["clip3"]))
    return replacements


def manifest_frame(
    rows: pd.DataFrame, pool: pd.DataFrame, condition: str, replicate_seed: int
) -> pd.DataFrame:
    rng = random.Random(replicate_seed)
    if condition == "global_shuffle":
        replacement_ids = global_derangement(rows, rng)
    else:
        replacement_ids = matched_replacements(rows, pool, condition, rng)
    metadata = pool.set_index("clip3").to_dict("index")
    records = []
    for (_, row), replacement in zip(rows.iterrows(), replacement_ids, strict=True):
        replacement_meta = metadata[replacement]
        records.append(
            {
                "sample_id": row["sample_id"],
                "condition": condition,
                "replicate_seed": replicate_seed,
                "true_clip3": row["clip3"],
                "replacement_clip3": replacement,
                "true_emotion": row["clip3_emotion"],
                "replacement_emotion": replacement_meta["clip3_emotion"],
                "true_source_folder": row["source_folder"],
                "replacement_source_folder": replacement_meta["source_folder"],
            }
        )
    frame = pd.DataFrame(records)
    if (frame["true_clip3"] == frame["replacement_clip3"]).any():
        raise AssertionError("Replacement manifest contains an unchanged clip III")
    if condition == "same_source_wrong_emotion":
        if not (frame["true_source_folder"] == frame["replacement_source_folder"]).all():
            raise AssertionError("Same-source invariant failed")
        if not (frame["true_emotion"] != frame["replacement_emotion"]).all():
            raise AssertionError("Wrong-emotion invariant failed")
    if condition == "same_emotion_different_source":
        if not (frame["true_emotion"] == frame["replacement_emotion"]).all():
            raise AssertionError("Same-emotion invariant failed")
        if not (frame["true_source_folder"] != frame["replacement_source_folder"]).all():
            raise AssertionError("Different-source invariant failed")
    return frame


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if config["model"] != "full" or config["test_evaluation_requested"]:
        raise ValueError(f"Unexpected checkpoint config in {checkpoint_path}")
    model = FrozenFeatureBaseline(
        num_clips=3,
        d_model=config["d_model"],
        temporal_layers=config["temporal_layers"],
        inter_layers=config["inter_layers"],
        dropout=config["dropout"],
        face_pooling=config["face_pooling"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, config


def evaluate_condition(
    model: nn.Module,
    rows: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    criterion: nn.Module,
    condition: str,
    replacement_frame: pd.DataFrame | None,
    loader_seed: int,
    predictions_path: Path,
) -> dict[str, object]:
    replacements = None
    if replacement_frame is not None:
        replacements = dict(
            zip(
                replacement_frame["sample_id"],
                replacement_frame["replacement_clip3"],
                strict=True,
            )
        )
    loader = build_intervention_loader(
        rows, args.features_dir, condition, replacements,
        args.batch_size, args.workers, loader_seed,
    )
    result = evaluate(model, loader, device, criterion)
    save_predictions(predictions_path, result)
    return public_metrics(result)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = args.output_dir / "replacement_manifests"
    predictions_dir = args.output_dir / "predictions"
    manifests_dir.mkdir(exist_ok=True)
    predictions_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    baseline_summary_path = args.checkpoints_dir / "validation_matrix_summary.json"
    baseline_summary = json.loads(baseline_summary_path.read_text())
    if baseline_summary["test_evaluated"] is not False:
        raise RuntimeError("Baseline matrix must be validation-only")
    if baseline_summary["seeds"] != args.seeds:
        raise RuntimeError("Requested seeds do not match the frozen baseline matrix")
    if baseline_summary["manifest_sha256"] != sha256(args.manifest):
        raise RuntimeError("Baseline and intervention manifests differ")
    trainer_path = Path(__file__).with_name("train_baselines.py")
    if baseline_summary["trainer_sha256"] != sha256(trainer_path):
        raise RuntimeError("Current model code differs from baseline training code")

    rows = pd.read_csv(args.manifest, dtype={"source_folder": str})
    rows = rows[rows["split"] == "val"].reset_index(drop=True)
    pool = clip_pool(rows)
    manifest_lookup: dict[tuple[str, int], pd.DataFrame] = {}
    manifest_audit = []
    for condition in REPLACEMENT_CONDITIONS:
        for replicate in range(args.replacement_replicates):
            replicate_seed = args.replacement_seed_start + replicate
            frame = manifest_frame(rows, pool, condition, replicate_seed)
            path = manifests_dir / f"{condition}_seed{replicate_seed}.csv"
            frame.to_csv(path, index=False, lineterminator="\n")
            manifest_lookup[(condition, replicate_seed)] = frame
            manifest_audit.append(
                {
                    "condition": condition,
                    "replicate_seed": replicate_seed,
                    "rows": len(frame),
                    "unique_replacement_clips": int(frame["replacement_clip3"].nunique()),
                    "same_emotion_rate": float(
                        (frame["true_emotion"] == frame["replacement_emotion"]).mean()
                    ),
                    "same_source_rate": float(
                        (
                            frame["true_source_folder"]
                            == frame["replacement_source_folder"]
                        ).mean()
                    ),
                    "sha256": sha256(path),
                }
            )

    records = []
    true_uars: dict[int, float] = {}
    for seed in args.seeds:
        seed_everything(seed)
        checkpoint_path = args.checkpoints_dir / f"full_seed{seed}" / "best.pt"
        metrics_path = args.checkpoints_dir / f"full_seed{seed}" / "metrics.json"
        baseline_metrics = json.loads(metrics_path.read_text())
        if baseline_metrics["test"] is not None:
            raise RuntimeError(f"Test metrics found beside {checkpoint_path}")
        model, config = load_model(checkpoint_path, device)
        if config["seed"] != seed:
            raise RuntimeError(f"Checkpoint seed mismatch at {checkpoint_path}")

        for condition in ("true", "zero"):
            metrics = evaluate_condition(
                model, rows, args, device, criterion, condition, None, seed,
                predictions_dir / f"full_seed{seed}_{condition}.npz",
            )
            records.append(
                {"model_seed": seed, "condition": condition, "replicate_seed": None, **metrics}
            )
            if condition == "true":
                true_uars[seed] = float(metrics["uar"])
                expected_uar = baseline_metrics["validation"]["uar"]
                if not np.isclose(true_uars[seed], expected_uar, atol=1e-12):
                    raise RuntimeError(
                        f"True-condition UAR {true_uars[seed]} does not reproduce "
                        f"baseline UAR {expected_uar} for seed {seed}"
                    )

        for condition in REPLACEMENT_CONDITIONS:
            for replicate in range(args.replacement_replicates):
                replicate_seed = args.replacement_seed_start + replicate
                metrics = evaluate_condition(
                    model, rows, args, device, criterion, condition,
                    manifest_lookup[(condition, replicate_seed)], seed + replicate_seed,
                    predictions_dir
                    / f"full_seed{seed}_{condition}_seed{replicate_seed}.npz",
                )
                records.append(
                    {
                        "model_seed": seed,
                        "condition": condition,
                        "replicate_seed": replicate_seed,
                        **metrics,
                    }
                )

    flat_records = []
    for record in records:
        flat_records.append(
            {
                "model_seed": record["model_seed"],
                "condition": record["condition"],
                "replicate_seed": record["replicate_seed"],
                "loss": record["loss"],
                "uar": record["uar"],
                "war": record["war"],
                "delta_uar_from_true": true_uars[record["model_seed"]] - record["uar"],
            }
        )
    results_frame = pd.DataFrame(flat_records)
    results_path = args.output_dir / "intervention_results.csv"
    results_frame.to_csv(results_path, index=False, lineterminator="\n")

    aggregate = {}
    for condition in ("zero",) + REPLACEMENT_CONDITIONS:
        condition_rows = results_frame[results_frame["condition"] == condition]
        per_seed = condition_rows.groupby("model_seed")["delta_uar_from_true"].mean()
        aggregate[condition] = {
            "mean_control_uar": float(condition_rows["uar"].mean()),
            "per_seed_mean_delta_uar": {
                str(seed): float(value) for seed, value in per_seed.items()
            },
            "mean_delta_uar": float(per_seed.mean()),
            "std_delta_uar": float(per_seed.std(ddof=1)),
            "positive_delta_seeds": int((per_seed > 0).sum()),
            "num_model_seeds": len(per_seed),
            "replacement_replicates": (
                args.replacement_replicates if condition in REPLACEMENT_CONDITIONS else 1
            ),
        }

    summary = {
        "protocol": "full-checkpoint-validation-interventions-v1",
        "partition": "val",
        "test_evaluated": False,
        "model_seeds": args.seeds,
        "replacement_seeds": list(
            range(
                args.replacement_seed_start,
                args.replacement_seed_start + args.replacement_replicates,
            )
        ),
        "conditions": ["true", "zero", *REPLACEMENT_CONDITIONS],
        "baseline_summary_sha256": sha256(baseline_summary_path),
        "manifest_sha256": sha256(args.manifest),
        "trainer_sha256": sha256(trainer_path),
        "intervention_runner_sha256": sha256(Path(__file__)),
        "true_uar_by_seed": {str(seed): value for seed, value in true_uars.items()},
        "aggregate": aggregate,
        "manifest_audit": manifest_audit,
    }
    summary_path = args.output_dir / "intervention_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "intervention_protocol.json").open("w") as handle:
        json.dump(vars(args) | {"device": str(device)}, handle, indent=2, default=str)
        handle.write("\n")
    print(json.dumps({"aggregate": aggregate, "test_evaluated": False}, indent=2))
    print(f"Summary: {summary_path}")
    print(f"Results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
