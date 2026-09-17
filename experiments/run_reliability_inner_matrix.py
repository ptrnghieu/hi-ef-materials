#!/usr/bin/env python3
"""Run the frozen v0.8 2x2 reliability inner-development matrix."""

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


VARIANTS = ("ungated", "gate", "counterfactual", "gate_counterfactual")
SEEDS = (42, 123, 456)
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
    "counterfactual_weight": 1.0,
    "invalid_gate_weight": 0.1,
    "contrastive_temperature": 0.1,
    "null_divergence": "context-to-null",
    "gradient_reversal_scale": 1.0,
    "limit_train_batches": None,
    "limit_val_batches": None,
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
        run_dir / "metrics.json", run_dir / "config.json", run_dir / "history.csv",
        run_dir / "best.pt", run_dir / "inner_development_predictions.npz",
    )
    if not all(path.exists() for path in required):
        return False
    metrics = json.loads((run_dir / "metrics.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    if metrics.get("original_validation") is not None or metrics.get("test") is not None:
        raise RuntimeError(f"Sealed partition result found in {run_dir}")
    mismatches = {
        key: (config.get(key), value) for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Existing run has incompatible config: {mismatches}")
    predictions = np.load(run_dir / "inner_development_predictions.npz")
    np.testing.assert_allclose(
        predictions["delta_logits"],
        predictions["reliability_gate"] * predictions["raw_delta_logits"],
        rtol=1e-6, atol=1e-6,
    )
    np.testing.assert_allclose(
        predictions["final_logits"],
        predictions["context_logits"] + predictions["delta_logits"],
        rtol=1e-6, atol=1e-6,
    )
    if expected["variant"] in {"ungated", "counterfactual"}:
        np.testing.assert_array_equal(
            predictions["reliability_gate"],
            np.ones_like(predictions["reliability_gate"]),
        )
    return True


def train_command(
    trainer: Path, args: argparse.Namespace, seed: int, variant: str, run_dir: Path
) -> list[str]:
    config = expected_config(seed, variant)
    command = [
        sys.executable, str(trainer),
        "--manifest", str(args.manifest),
        "--features-dir", str(args.features_dir),
        "--output-dir", str(run_dir),
        "--variant", variant,
        "--seed", str(seed),
    ]
    for name, flag in (
        ("epochs", "--epochs"), ("batch_size", "--batch-size"),
        ("workers", "--workers"), ("learning_rate", "--learning-rate"),
        ("weight_decay", "--weight-decay"), ("patience", "--patience"),
        ("d_model", "--d-model"), ("temporal_layers", "--temporal-layers"),
        ("context_layers", "--context-layers"), ("dropout", "--dropout"),
        ("face_pooling", "--face-pooling"), ("context_weight", "--context-weight"),
        ("emotion_weight", "--emotion-weight"),
        ("contrastive_weight", "--contrastive-weight"),
        ("null_weight", "--null-weight"), ("nuisance_weight", "--nuisance-weight"),
        ("counterfactual_weight", "--counterfactual-weight"),
        ("invalid_gate_weight", "--invalid-gate-weight"),
        ("contrastive_temperature", "--contrastive-temperature"),
        ("null_divergence", "--null-divergence"),
        ("gradient_reversal_scale", "--gradient-reversal-scale"),
    ):
        command.extend([flag, str(config[name])])
    return command


def git_commit(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
        text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def sample_std(values: np.ndarray) -> float:
    return float(values.std(ddof=1 if len(values) > 1 else 0))


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    experiments = root / "experiments"
    trainer = experiments / "train_reliability_gated_residual.py"
    tests = experiments / "test_reliability_gated_residual.py"
    spec = experiments / "RESEARCH_SPEC_v0.8.md"
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.features_dir.is_dir():
        raise FileNotFoundError(args.features_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(experiments)
    subprocess.run(
        [sys.executable, "-m", "unittest", str(tests.relative_to(root))],
        cwd=root, env=environment, check=True,
    )

    for seed in SEEDS:
        for variant in VARIANTS:
            run_dir = args.output_dir / f"{variant}_seed{seed}"
            expected = expected_config(seed, variant)
            if validate_existing(run_dir, expected):
                print(f"Validated existing run: {variant}, seed={seed}", flush=True)
                continue
            subprocess.run(
                train_command(trainer, args, seed, variant, run_dir),
                cwd=root, env=environment, check=True,
            )
            if not validate_existing(run_dir, expected):
                raise RuntimeError(f"Incomplete run artifacts: {run_dir}")

    records: list[dict[str, object]] = []
    runs: dict[str, object] = {}
    for seed in SEEDS:
        seed_runs: dict[str, object] = {}
        for variant in VARIANTS:
            run_dir = args.output_dir / f"{variant}_seed{seed}"
            metrics = json.loads((run_dir / "metrics.json").read_text())
            config = json.loads((run_dir / "config.json").read_text())
            validation = metrics["inner_development"]
            row = {
                "seed": seed,
                "variant": variant,
                "best_epoch": metrics["best_epoch"],
                "final_uar": validation["final_uar"],
                "final_war": validation["final_war"],
                "final_nll": validation["final_loss"],
                "context_uar": validation["context_uar"],
                "within_model_delta_uar": validation["final_uar"] - validation["context_uar"],
                "raw_delta_l2_mean": validation["raw_delta_l2_mean"],
                "delta_l2_mean": validation["delta_l2_mean"],
                "gate_mean": validation["gate_mean"],
                "prediction_flip_rate": validation["prediction_flip_rate"],
                "beneficial_flip_rate": validation["beneficial_flip_rate"],
                "harmful_flip_rate": validation["harmful_flip_rate"],
                "num_parameters": config["num_parameters"],
            }
            records.append(row)
            seed_runs[variant] = {
                "best_epoch": metrics["best_epoch"],
                "inner_development": validation,
                "num_parameters": config["num_parameters"],
            }
        runs[str(seed)] = seed_runs

    aggregates: dict[str, object] = {}
    for variant in VARIANTS:
        subset = [row for row in records if row["variant"] == variant]
        aggregates[variant] = {}
        for key in (
            "final_uar", "final_nll", "within_model_delta_uar", "gate_mean",
            "harmful_flip_rate", "beneficial_flip_rate",
        ):
            values = np.asarray([row[key] for row in subset], dtype=float)
            aggregates[variant][f"{key}_mean"] = float(values.mean())
            aggregates[variant][f"{key}_std"] = sample_std(values)
        aggregates[variant]["positive_within_model_delta_seeds"] = int(sum(
            float(row["within_model_delta_uar"]) > 0 for row in subset
        ))

    candidate = aggregates["gate_counterfactual"]
    baseline = aggregates["ungated"]
    criteria = {
        "mean_nll_lower_than_ungated": candidate["final_nll_mean"] < baseline["final_nll_mean"],
        "mean_uar_noninferior_margin_minus_0_005": (
            candidate["final_uar_mean"] - baseline["final_uar_mean"] >= -0.005
        ),
        "own_context_improved_at_least_two_of_three_seeds": (
            candidate["positive_within_model_delta_seeds"] >= 2
        ),
        "mean_harmful_flip_not_above_ungated": (
            candidate["harmful_flip_rate_mean"] <= baseline["harmful_flip_rate_mean"]
        ),
    }
    advancement = {
        "candidate": "gate_counterfactual",
        "rule": "all four RESEARCH_SPEC_v0.8.md inner-development criteria",
        "criteria": criteria,
        "advance_to_separately_frozen_original_validation_audit": all(criteria.values()),
        "original_validation_evaluated": False,
        "test_evaluated": False,
    }
    summary = {
        "protocol": "reliability-gated-residual-inner-development-matrix-v1",
        "research_spec": spec.name,
        "variants": list(VARIANTS),
        "model_seeds": list(SEEDS),
        "fixed_config": FROZEN_CONFIG,
        "partitions_touched": ["original-train/inner-train", "original-train/inner-development"],
        "original_validation_evaluated": False,
        "test_evaluated": False,
        "temperature_scaling_role": "calibration-only; not an architectural contribution",
        "manifest_sha256": sha256(args.manifest),
        "trainer_sha256": sha256(trainer),
        "test_sha256": sha256(tests),
        "research_spec_sha256": sha256(spec),
        "git_commit": git_commit(root),
        "runs": runs,
        "aggregates": aggregates,
        "advancement_gate": advancement,
    }
    table_path = args.output_dir / "reliability_inner_matrix.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summary_path = args.output_dir / "reliability_inner_matrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
