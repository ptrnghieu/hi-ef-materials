#!/usr/bin/env python3
"""Hierarchical validation audit for the canonical residual matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
MODEL_SEEDS = (42, 123, 456, 789, 1024)
VARIANTS = ("context", "affect", "interaction", "both")
COMPARISONS = (
    ("both_vs_context_run", "both", "final_logits", "context", "final_logits"),
    ("both_final_vs_own_context", "both", "final_logits", "both", "context_logits"),
    ("affect_vs_context_run", "affect", "final_logits", "context", "final_logits"),
    ("interaction_vs_context_run", "interaction", "final_logits", "context", "final_logits"),
    ("affect_final_vs_own_context", "affect", "final_logits", "affect", "context_logits"),
    ("interaction_final_vs_own_context", "interaction", "final_logits", "interaction", "context_logits"),
    ("both_vs_affect", "both", "final_logits", "affect", "final_logits"),
    ("both_vs_interaction", "both", "final_logits", "interaction", "final_logits"),
    ("interaction_vs_affect", "interaction", "final_logits", "affect", "final_logits"),
)
BOOTSTRAP_SEEDS = {name: 5901 + index for index, (name, *_) in enumerate(COMPARISONS)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    return parser.parse_args()


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def prediction_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    target = np.arange(len(labels))
    predictions = logits.argmax(axis=1)
    recalls = np.asarray([
        np.mean(predictions[labels == class_id] == class_id)
        for class_id in range(len(EMOTIONS))
    ])
    return {
        "uar": float(recalls.mean()),
        "nll": float(np.mean(-log_softmax(logits)[target, labels])),
        "correct_probability": float(np.mean(np.exp(log_softmax(logits)[target, labels]))),
        "predictions": predictions,
        "recalls": recalls,
    }


def load_matrix_predictions(
    matrix_dir: Path,
) -> tuple[dict[int, dict[str, dict[str, np.ndarray]]], dict[str, np.ndarray]]:
    data: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    reference: dict[str, np.ndarray] | None = None
    required = (
        "context_logits", "delta_logits", "final_logits", "labels",
        "sample_ids", "source_folders",
    )
    for model_seed in MODEL_SEEDS:
        data[model_seed] = {}
        for variant in VARIANTS:
            path = matrix_dir / f"{variant}_seed{model_seed}" / "val_predictions.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path) as archive:
                missing = set(required) - set(archive.files)
                if missing:
                    raise RuntimeError(f"Missing prediction arrays in {path}: {missing}")
                current = {key: archive[key] for key in required}
            np.testing.assert_allclose(
                current["final_logits"],
                current["context_logits"] + current["delta_logits"],
                rtol=1e-6,
                atol=1e-6,
            )
            identity = {
                key: current[key] for key in ("labels", "sample_ids", "source_folders")
            }
            if reference is None:
                reference = identity
            else:
                for key, values in identity.items():
                    if not np.array_equal(reference[key].astype(str), values.astype(str)):
                        raise RuntimeError(f"Prediction alignment mismatch: {path}, {key}")
            data[model_seed][variant] = current
    assert reference is not None
    return data, reference


def folder_statistics(
    logits: np.ndarray,
    labels: np.ndarray,
    folders: np.ndarray,
    unique_folders: np.ndarray,
) -> dict[str, np.ndarray]:
    target = np.arange(len(labels))
    predictions = logits.argmax(axis=1)
    log_probability = log_softmax(logits)[target, labels]
    return {
        "correct": np.asarray([
            [
                np.sum((folders == folder) & (labels == class_id) & (predictions == class_id))
                for class_id in range(len(EMOTIONS))
            ]
            for folder in unique_folders
        ], dtype=float),
        "nll_sum": np.asarray([
            np.sum(-log_probability[folders == folder]) for folder in unique_folders
        ]),
    }


def bootstrap_folder_multiplicities(
    rng: np.random.Generator,
    folder_class_counts: np.ndarray,
) -> np.ndarray:
    num_folders = folder_class_counts.shape[0]
    for _ in range(10000):
        sampled = rng.integers(0, num_folders, size=num_folders)
        multiplicities = np.bincount(sampled, minlength=num_folders)
        if np.all(multiplicities @ folder_class_counts > 0):
            return multiplicities
    raise RuntimeError("Could not sample source folders containing every class")


def interval(values: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "bootstrap_mean": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "probability_above_zero": float(np.mean(values > 0)),
    }


def hierarchical_pair_bootstrap(
    name: str,
    better_logits: dict[int, np.ndarray],
    reference_logits: dict[int, np.ndarray],
    labels: np.ndarray,
    folders: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    unique_folders = np.unique(folders)
    folder_class_counts = np.asarray([
        [np.sum((folders == folder) & (labels == class_id)) for class_id in range(len(EMOTIONS))]
        for folder in unique_folders
    ], dtype=float)
    folder_sample_counts = folder_class_counts.sum(axis=1)
    better_values = {
        model_seed: folder_statistics(logits, labels, folders, unique_folders)
        for model_seed, logits in better_logits.items()
    }
    reference_values = {
        model_seed: folder_statistics(logits, labels, folders, unique_folders)
        for model_seed, logits in reference_logits.items()
    }
    uar_deltas = np.empty(replicates)
    nll_deltas = np.empty(replicates)
    for bootstrap_index in range(replicates):
        multiplicities = bootstrap_folder_multiplicities(rng, folder_class_counts)
        class_denominators = multiplicities @ folder_class_counts
        sample_denominator = float(multiplicities @ folder_sample_counts)
        sampled_seeds = rng.choice(MODEL_SEEDS, size=len(MODEL_SEEDS), replace=True)
        seed_uar_deltas = []
        seed_nll_deltas = []
        for sampled_seed in sampled_seeds:
            model_seed = int(sampled_seed)
            better = better_values[model_seed]
            reference = reference_values[model_seed]
            better_uar = float(np.mean((multiplicities @ better["correct"]) / class_denominators))
            reference_uar = float(np.mean((multiplicities @ reference["correct"]) / class_denominators))
            better_nll = float(multiplicities @ better["nll_sum"] / sample_denominator)
            reference_nll = float(multiplicities @ reference["nll_sum"] / sample_denominator)
            seed_uar_deltas.append(better_uar - reference_uar)
            seed_nll_deltas.append(reference_nll - better_nll)
        uar_deltas[bootstrap_index] = np.mean(seed_uar_deltas)
        nll_deltas[bootstrap_index] = np.mean(seed_nll_deltas)
    return {
        "comparison": name,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "resampling_units": ["model_seed", "source_folder"],
        "uar_delta": interval(uar_deltas),
        "nll_delta": interval(nll_deltas),
    }


def observed_pair(
    name: str,
    better_logits: dict[int, np.ndarray],
    reference_logits: dict[int, np.ndarray],
    labels: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    for model_seed in MODEL_SEEDS:
        better = prediction_metrics(better_logits[model_seed], labels)
        reference = prediction_metrics(reference_logits[model_seed], labels)
        better_prediction = better["predictions"]
        reference_prediction = reference["predictions"]
        better_correct = better_prediction == labels
        reference_correct = reference_prediction == labels
        rows.append({
            "comparison": name,
            "model_seed": model_seed,
            "uar_delta": better["uar"] - reference["uar"],
            "nll_delta": reference["nll"] - better["nll"],
            "correct_probability_delta": (
                better["correct_probability"] - reference["correct_probability"]
            ),
            "prediction_flip_rate": float(np.mean(better_prediction != reference_prediction)),
            "harmful_flip_rate": float(np.mean(reference_correct & ~better_correct)),
            "beneficial_flip_rate": float(np.mean(~reference_correct & better_correct)),
        })
        for class_id, emotion in enumerate(EMOTIONS):
            class_rows.append({
                "comparison": name,
                "model_seed": model_seed,
                "emotion": emotion,
                "recall_delta": better["recalls"][class_id] - reference["recalls"][class_id],
            })
    return rows, class_rows


def exact_two_sided_sign_p(values: np.ndarray) -> float:
    nonzero = values[values != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positives = int(np.sum(nonzero > 0))
    extreme = min(positives, n - positives)
    probability = sum(math.comb(n, count) for count in range(extreme + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * probability))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.matrix_dir / "canonical_residual_matrix_summary.json"
    matrix_summary = json.loads(summary_path.read_text())
    if matrix_summary.get("protocol") != "canonical-contextual-affective-residual-validation-matrix-v1":
        raise RuntimeError("Unexpected matrix protocol")
    if matrix_summary.get("test_evaluated") is not False:
        raise RuntimeError("Input matrix is not validation-only")
    if matrix_summary.get("model_seeds") != list(MODEL_SEEDS):
        raise RuntimeError("Model seeds differ from the frozen audit")
    if matrix_summary.get("variants") != list(VARIANTS):
        raise RuntimeError("Variants differ from the frozen audit")
    if matrix_summary.get("manifest_sha256") != sha256(args.manifest):
        raise RuntimeError("Matrix and audit manifests differ")

    data, reference = load_matrix_predictions(args.matrix_dir)
    labels = reference["labels"].astype(int)
    folders = reference["source_folders"].astype(str)
    sample_ids = reference["sample_ids"].astype(str)
    manifest = pd.read_csv(args.manifest, dtype={"source_folder": str})
    manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    if not np.array_equal(sample_ids, manifest["sample_id"].to_numpy(str)):
        raise RuntimeError("Predictions differ from frozen validation sample order")
    if not np.array_equal(folders, manifest["source_folder"].to_numpy(str)):
        raise RuntimeError("Predictions differ from frozen validation folders")

    bootstrap: dict[str, object] = {}
    observed_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    for name, better_variant, better_key, reference_variant, reference_key in COMPARISONS:
        better_logits = {
            seed: data[seed][better_variant][better_key] for seed in MODEL_SEEDS
        }
        reference_logits = {
            seed: data[seed][reference_variant][reference_key] for seed in MODEL_SEEDS
        }
        rows, per_class = observed_pair(name, better_logits, reference_logits, labels)
        observed_rows.extend(rows)
        class_rows.extend(per_class)
        bootstrap[name] = hierarchical_pair_bootstrap(
            name, better_logits, reference_logits, labels, folders,
            args.bootstrap_replicates, BOOTSTRAP_SEEDS[name],
        )

    aggregates: dict[str, object] = {}
    observed_frame = pd.DataFrame(observed_rows)
    for name, group in observed_frame.groupby("comparison", sort=False):
        values = group["uar_delta"].to_numpy(float)
        aggregates[name] = {
            metric: float(group[metric].mean())
            for metric in (
                "uar_delta", "nll_delta", "correct_probability_delta",
                "prediction_flip_rate", "harmful_flip_rate", "beneficial_flip_rate",
            )
        }
        aggregates[name].update({
            "uar_delta_std_across_model_seeds": float(values.std(ddof=1)),
            "positive_uar_delta_model_seeds": int(np.sum(values > 0)),
            "two_sided_exact_sign_p": exact_two_sided_sign_p(values),
        })

    primary_run = bootstrap["both_vs_context_run"]["uar_delta"]
    primary_within = bootstrap["both_final_vs_own_context"]["uar_delta"]
    advancement_passed = (
        primary_run["ci95_low"] > 0 and primary_within["ci95_low"] > 0
    )
    affect_increment = bootstrap["both_vs_interaction"]["uar_delta"]["ci95_low"] > 0
    interaction_increment = bootstrap["both_vs_affect"]["uar_delta"]["ci95_low"] > 0

    observed_path = args.output_dir / "canonical_audit_effects_by_seed.csv"
    class_path = args.output_dir / "canonical_audit_per_class_effects.csv"
    observed_frame.to_csv(observed_path, index=False, lineterminator="\n")
    pd.DataFrame(class_rows).to_csv(class_path, index=False, lineterminator="\n")
    audit_spec = Path(__file__).with_name("RESEARCH_SPEC_v0.5.md")
    summary = {
        "protocol": "canonical-residual-hierarchical-validation-audit-v1",
        "partition": "val",
        "num_samples": int(len(labels)),
        "num_source_folders": int(len(np.unique(folders))),
        "model_seeds": list(MODEL_SEEDS),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seeds": BOOTSTRAP_SEEDS,
        "comparisons": [name for name, *_ in COMPARISONS],
        "aggregates": aggregates,
        "hierarchical_bootstrap": bootstrap,
        "advancement_gate": {
            "candidate": "both",
            "rule": "ci95_low > 0 for both primary UAR comparisons",
            "both_vs_context_run_passed": primary_run["ci95_low"] > 0,
            "both_final_vs_own_context_passed": primary_within["ci95_low"] > 0,
            "advance_both_to_single_test_evaluation": advancement_passed,
        },
        "claim_scope": {
            "incremental_affect_beyond_interaction_supported": affect_increment,
            "incremental_interaction_beyond_affect_supported": interaction_increment,
        },
        "inference_note": (
            "Validation uncertainty diagnostic only: checkpoints were selected on validation, "
            "and the audit specification was frozen after descriptive matrix means were observed."
        ),
        "partitions_touched": ["validation"],
        "test_evaluated": False,
        "matrix_summary_sha256": sha256(summary_path),
        "manifest_sha256": sha256(args.manifest),
        "analysis_runner_sha256": sha256(Path(__file__)),
        "audit_spec": audit_spec.name,
        "audit_spec_sha256": sha256(audit_spec),
    }
    output_path = args.output_dir / "canonical_residual_audit_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Summary:   {output_path}")
    print(f"By seed:   {observed_path}")
    print(f"Per class: {class_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
