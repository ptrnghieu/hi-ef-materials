#!/usr/bin/env python3
"""Run and audit the fixed Hi-EF validation-only multi-seed matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


MODELS = ("context", "full")


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
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024]
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--inter-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--face-pooling", choices=("masked", "unmasked"), default="masked")
    return parser.parse_args()


def expected_config(args: argparse.Namespace, seed: int, model: str) -> dict[str, object]:
    return {
        "model": model,
        "face_pooling": args.face_pooling,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "d_model": args.d_model,
        "temporal_layers": args.temporal_layers,
        "inter_layers": args.inter_layers,
        "dropout": args.dropout,
        "limit_train_batches": None,
        "limit_val_batches": None,
        "test_evaluation_requested": False,
    }


def validate_existing(run_dir: Path, expected: dict[str, object]) -> bool:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    required = (
        metrics_path,
        config_path,
        run_dir / "history.csv",
        run_dir / "best.pt",
        run_dir / "val_predictions.npz",
    )
    if not all(path.exists() for path in required):
        return False
    metrics = json.loads(metrics_path.read_text())
    config = json.loads(config_path.read_text())
    if metrics.get("test") is not None:
        raise RuntimeError(f"Test results unexpectedly present in {metrics_path}")
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Existing run has incompatible config: {mismatches}")
    return True


def train_command(
    trainer: Path, args: argparse.Namespace, seed: int, model: str, run_dir: Path
) -> list[str]:
    return [
        sys.executable,
        str(trainer),
        "--manifest", str(args.manifest),
        "--features-dir", str(args.features_dir),
        "--output-dir", str(run_dir),
        "--model", model,
        "--face-pooling", args.face_pooling,
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--patience", str(args.patience),
        "--d-model", str(args.d_model),
        "--temporal-layers", str(args.temporal_layers),
        "--inter-layers", str(args.inter_layers),
        "--dropout", str(args.dropout),
    ]


def git_commit(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Path(__file__).with_name("train_baselines.py")

    for seed in args.seeds:
        for model in MODELS:
            run_dir = args.output_dir / f"{model}_seed{seed}"
            expected = expected_config(args, seed, model)
            if validate_existing(run_dir, expected):
                print(f"Validated existing run: {model}, seed={seed}", flush=True)
                continue
            print(f"Training {model}, seed={seed} -> {run_dir}", flush=True)
            subprocess.run(
                train_command(trainer, args, seed, model, run_dir), check=True
            )
            if not validate_existing(run_dir, expected):
                raise RuntimeError(f"Run did not produce complete artifacts: {run_dir}")

    records = []
    detailed_runs: dict[str, object] = {}
    for seed in args.seeds:
        seed_runs = {}
        row: dict[str, object] = {"seed": seed}
        for model in MODELS:
            run_dir = args.output_dir / f"{model}_seed{seed}"
            metrics = json.loads((run_dir / "metrics.json").read_text())
            config = json.loads((run_dir / "config.json").read_text())
            validation = metrics["validation"]
            seed_runs[model] = {
                "best_epoch": metrics["best_epoch"],
                "validation": validation,
                "num_parameters": config["num_parameters"],
            }
            row[f"{model}_best_epoch"] = metrics["best_epoch"]
            row[f"{model}_uar"] = validation["uar"]
            row[f"{model}_war"] = validation["war"]
            row[f"{model}_loss"] = validation["loss"]
        row["delta_uar"] = row["full_uar"] - row["context_uar"]
        row["delta_war"] = row["full_war"] - row["context_war"]
        records.append(row)
        detailed_runs[str(seed)] = seed_runs

    deltas = np.asarray([row["delta_uar"] for row in records], dtype=float)
    context_uars = np.asarray([row["context_uar"] for row in records], dtype=float)
    full_uars = np.asarray([row["full_uar"] for row in records], dtype=float)
    ddof = 1 if len(records) > 1 else 0
    summary = {
        "protocol": "source-folder-held-out-validation-only-multiseed-v1",
        "seeds": args.seeds,
        "models": list(MODELS),
        "test_evaluated": False,
        "git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "trainer_sha256": sha256(trainer),
        "fixed_config": expected_config(args, args.seeds[0], "context") | {
            "seed": args.seeds,
            "model": list(MODELS),
        },
        "runs": detailed_runs,
        "aggregate": {
            "context_uar_mean": float(context_uars.mean()),
            "context_uar_std": float(context_uars.std(ddof=ddof)),
            "full_uar_mean": float(full_uars.mean()),
            "full_uar_std": float(full_uars.std(ddof=ddof)),
            "delta_uar_mean": float(deltas.mean()),
            "delta_uar_std": float(deltas.std(ddof=ddof)),
            "positive_delta_seeds": int((deltas > 0).sum()),
            "num_seeds": len(records),
        },
    }

    csv_path = args.output_dir / "validation_matrix.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summary_path = args.output_dir / "validation_matrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Summary: {summary_path}")
    print(f"Table:   {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
