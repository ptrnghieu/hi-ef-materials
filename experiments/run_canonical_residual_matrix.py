#!/usr/bin/env python3
"""Run the frozen validation-only canonical residual matrix.

This is the sole entry point for the 4 variants x 5 model seeds experiment.
The scientific configuration is deliberately fixed here and in
RESEARCH_SPEC_v0.4.md.  The test partition is never loaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


VARIANTS = ("context", "affect", "interaction", "both")
SEEDS = (42, 123, 456, 789, 1024)
FROZEN_CONFIG: dict[str, object] = {
    "epochs": 50,
    "batch_size": 32,
    "workers": 2,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "patience": 8,
    "d_model": 512,
    "temporal_layers": 2,
    "context_layers": 2,
    "dropout": 0.1,
    "face_pooling": "masked",
    "context_weight": 1.0,
    "emotion_weight": 0.5,
    "contrastive_weight": 0.1,
    "null_weight": 1.0,
    "nuisance_weight": 0.05,
    "contrastive_temperature": 0.1,
    "null_divergence": "context-to-null",
    "gradient_reversal_scale": 1.0,
    "limit_train_batches": None,
    "limit_val_batches": None,
    "test_evaluation_requested": False,
}


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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def expected_config(seed: int, variant: str) -> dict[str, object]:
    return {**FROZEN_CONFIG, "seed": seed, "variant": variant}


def validate_existing(run_dir: Path, expected: dict[str, object]) -> bool:
    required = (
        run_dir / "metrics.json",
        run_dir / "config.json",
        run_dir / "history.csv",
        run_dir / "best.pt",
        run_dir / "val_predictions.npz",
    )
    if not all(path.exists() for path in required):
        return False
    metrics = json.loads((run_dir / "metrics.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    if metrics.get("test") is not None:
        raise RuntimeError(f"Test results unexpectedly present in {run_dir}")
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Existing run has incompatible config: {mismatches}")
    predictions = np.load(run_dir / "val_predictions.npz")
    np.testing.assert_allclose(
        predictions["final_logits"],
        predictions["context_logits"] + predictions["delta_logits"],
        rtol=1e-6,
        atol=1e-6,
    )
    if expected["variant"] == "context":
        np.testing.assert_array_equal(
            predictions["delta_logits"],
            np.zeros_like(predictions["delta_logits"]),
        )
    return True


def train_command(
    trainer: Path,
    args: argparse.Namespace,
    seed: int,
    variant: str,
    run_dir: Path,
) -> list[str]:
    config = expected_config(seed, variant)
    command = [
        sys.executable,
        str(trainer),
        "--manifest", str(args.manifest),
        "--features-dir", str(args.features_dir),
        "--output-dir", str(run_dir),
        "--variant", variant,
        "--seed", str(seed),
    ]
    for name, flag in (
        ("epochs", "--epochs"),
        ("batch_size", "--batch-size"),
        ("workers", "--workers"),
        ("learning_rate", "--learning-rate"),
        ("weight_decay", "--weight-decay"),
        ("patience", "--patience"),
        ("d_model", "--d-model"),
        ("temporal_layers", "--temporal-layers"),
        ("context_layers", "--context-layers"),
        ("dropout", "--dropout"),
        ("face_pooling", "--face-pooling"),
        ("context_weight", "--context-weight"),
        ("emotion_weight", "--emotion-weight"),
        ("contrastive_weight", "--contrastive-weight"),
        ("null_weight", "--null-weight"),
        ("nuisance_weight", "--nuisance-weight"),
        ("contrastive_temperature", "--contrastive-temperature"),
        ("null_divergence", "--null-divergence"),
        ("gradient_reversal_scale", "--gradient-reversal-scale"),
    ):
        command.extend([flag, str(config[name])])
    return command


def git_commit(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def sample_std(values: np.ndarray) -> float:
    return float(values.std(ddof=1 if len(values) > 1 else 0))


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    experiments = root / "experiments"
    trainer = experiments / "train_contextual_affective_residual.py"
    test_file = experiments / "test_contextual_affective_residual.py"
    spec = experiments / "RESEARCH_SPEC_v0.4.md"
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.features_dir.is_dir():
        raise FileNotFoundError(args.features_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(experiments)
    subprocess.run(
        [sys.executable, "-m", "unittest", str(test_file.relative_to(root))],
        cwd=root,
        env=environment,
        check=True,
    )

    for seed in SEEDS:
        for variant in VARIANTS:
            run_dir = args.output_dir / f"{variant}_seed{seed}"
            expected = expected_config(seed, variant)
            if validate_existing(run_dir, expected):
                print(f"Validated existing run: {variant}, seed={seed}", flush=True)
                continue
            print(f"Training {variant}, seed={seed} -> {run_dir}", flush=True)
            subprocess.run(
                train_command(trainer, args, seed, variant, run_dir),
                cwd=root,
                env=environment,
                check=True,
            )
            if not validate_existing(run_dir, expected):
                raise RuntimeError(f"Incomplete run artifacts: {run_dir}")

    records: list[dict[str, object]] = []
    runs: dict[str, object] = {}
    for seed in SEEDS:
        seed_runs: dict[str, object] = {}
        context_metrics = json.loads(
            (args.output_dir / f"context_seed{seed}" / "metrics.json").read_text()
        )["validation"]
        context_baseline_uar = float(context_metrics["final_uar"])
        for variant in VARIANTS:
            run_dir = args.output_dir / f"{variant}_seed{seed}"
            metrics = json.loads((run_dir / "metrics.json").read_text())
            config = json.loads((run_dir / "config.json").read_text())
            validation = metrics["validation"]
            final_uar = float(validation["final_uar"])
            own_context_uar = float(validation["context_uar"])
            record = {
                "seed": seed,
                "variant": variant,
                "best_epoch": metrics["best_epoch"],
                "final_uar": final_uar,
                "final_war": validation["final_war"],
                "final_loss": validation["final_loss"],
                "context_uar": own_context_uar,
                "within_model_delta_uar": final_uar - own_context_uar,
                "delta_uar_vs_context_run": final_uar - context_baseline_uar,
                "delta_l2_mean": validation["delta_l2_mean"],
                "affect_uar": validation["affect_uar"],
                "affect_unweighted_loss": validation["affect_unweighted_loss"],
                "num_parameters": config["num_parameters"],
            }
            records.append(record)
            seed_runs[variant] = {
                "best_epoch": metrics["best_epoch"],
                "validation": validation,
                "num_parameters": config["num_parameters"],
            }
        runs[str(seed)] = seed_runs

    aggregates: dict[str, object] = {}
    for variant in VARIANTS:
        subset = [row for row in records if row["variant"] == variant]
        final = np.asarray([row["final_uar"] for row in subset], dtype=float)
        within = np.asarray([row["within_model_delta_uar"] for row in subset], dtype=float)
        versus = np.asarray([row["delta_uar_vs_context_run"] for row in subset], dtype=float)
        aggregates[variant] = {
            "final_uar_mean": float(final.mean()),
            "final_uar_std": sample_std(final),
            "within_model_delta_uar_mean": float(within.mean()),
            "within_model_delta_uar_std": sample_std(within),
            "positive_within_model_delta_seeds": int((within > 0).sum()),
            "delta_uar_vs_context_run_mean": float(versus.mean()),
            "delta_uar_vs_context_run_std": sample_std(versus),
            "positive_delta_vs_context_run_seeds": int((versus > 0).sum()),
        }

    summary = {
        "protocol": "canonical-contextual-affective-residual-validation-matrix-v1",
        "research_spec": "RESEARCH_SPEC_v0.4.md",
        "variants": list(VARIANTS),
        "model_seeds": list(SEEDS),
        "fixed_config": FROZEN_CONFIG,
        "training_regime": "joint-end-to-end",
        "affect_class_weighting": "training-only-mean-one-inverse-frequency",
        "partitions_touched": ["train", "validation"],
        "test_evaluated": False,
        "method_selection_permitted": False,
        "selection_pending": "frozen hierarchical validation audit",
        "git_commit": git_commit(root),
        "manifest_sha256": sha256(args.manifest),
        "trainer_sha256": sha256(trainer),
        "test_sha256": sha256(test_file),
        "research_spec_sha256": sha256(spec),
        "runs": runs,
        "aggregates": aggregates,
    }
    table_path = args.output_dir / "canonical_residual_matrix.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summary_path = args.output_dir / "canonical_residual_matrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Summary: {summary_path}")
    print(f"Table:   {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
