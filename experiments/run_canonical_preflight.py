#!/usr/bin/env python3
"""Run validation-only contract tests and tiny smoke runs for the canonical model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


VARIANTS = ("context", "affect", "interaction", "both")
SMOKE_WEIGHTS = {
    "context": 1.0,
    "emotion": 1.0,
    "contrastive": 1.0,
    "null": 1.0,
    "nuisance": 1.0,
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


def run(command: list[str], cwd: Path, environment: dict[str, str] | None = None) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    experiments = root / "experiments"
    trainer = experiments / "train_contextual_affective_residual.py"
    test_file = experiments / "test_contextual_affective_residual.py"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if not args.features_dir.is_dir():
        raise FileNotFoundError(args.features_dir)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(experiments)
    run(
        [sys.executable, "-m", "unittest", str(test_file.relative_to(root))],
        cwd=root,
        environment=environment,
    )

    summaries: dict[str, object] = {}
    for variant in VARIANTS:
        variant_dir = args.output_dir / variant
        command = [
            sys.executable,
            str(trainer),
            "--manifest", str(args.manifest),
            "--features-dir", str(args.features_dir),
            "--output-dir", str(variant_dir),
            "--variant", variant,
            "--seed", "42",
            "--epochs", "1",
            "--batch-size", "8",
            "--workers", "0",
            "--d-model", "64",
            "--temporal-layers", "1",
            "--context-layers", "1",
            "--patience", "1",
            "--limit-train-batches", "2",
            "--limit-val-batches", "1",
            "--context-weight", str(SMOKE_WEIGHTS["context"]),
            "--emotion-weight", str(SMOKE_WEIGHTS["emotion"]),
            "--contrastive-weight", str(SMOKE_WEIGHTS["contrastive"]),
            "--null-weight", str(SMOKE_WEIGHTS["null"]),
            "--nuisance-weight", str(SMOKE_WEIGHTS["nuisance"]),
            "--contrastive-temperature", "0.1",
            "--null-divergence", "context-to-null",
        ]
        run(command, cwd=root, environment=environment)
        metrics = json.loads((variant_dir / "metrics.json").read_text())
        config = json.loads((variant_dir / "config.json").read_text())
        predictions = np.load(variant_dir / "val_predictions.npz")
        np.testing.assert_allclose(
            predictions["final_logits"],
            predictions["context_logits"] + predictions["delta_logits"],
            rtol=1e-6,
            atol=1e-6,
        )
        if variant == "context":
            np.testing.assert_allclose(
                predictions["delta_logits"],
                np.zeros_like(predictions["delta_logits"]),
                rtol=0.0,
                atol=0.0,
            )
        if metrics.get("test") is not None:
            raise RuntimeError(f"Test output unexpectedly present for {variant}")
        if config.get("test_evaluation_requested") is not False:
            raise RuntimeError(f"Test evaluation unexpectedly requested for {variant}")
        summaries[variant] = {
            "best_epoch": metrics["best_epoch"],
            "validation": metrics["validation"],
            "num_parameters": config["num_parameters"],
            "invariant_passed": True,
            "test_evaluated": False,
        }

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    summary = {
        "protocol": "canonical-contextual-affective-residual-preflight-v1",
        "smoke_only": True,
        "research_interpretation_permitted": False,
        "partitions_touched": ["train", "validation"],
        "test_evaluated": False,
        "git_commit": commit,
        "manifest_sha256": sha256(args.manifest),
        "trainer_sha256": sha256(trainer),
        "test_sha256": sha256(test_file),
        "smoke_weights": SMOKE_WEIGHTS,
        "contrastive_temperature": 0.1,
        "null_divergence": "context-to-null",
        "variants": summaries,
    }
    summary_path = args.output_dir / "canonical_preflight_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
