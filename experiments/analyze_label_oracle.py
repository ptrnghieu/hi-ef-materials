#!/usr/bin/env python3
"""Statistical audit of the Hi-EF Party-A emotion-label oracle experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_validation_interventions import (
    EMOTIONS,
    bootstrap_folder_multiplicities,
    exact_two_sided_sign_p,
    load_prediction,
    log_softmax,
    uar,
    validate_alignment,
)


MODEL_COMPARISONS = {
    "raw_vs_context": ("raw", "context"),
    "oracle_vs_context": ("oracle", "context"),
    "oracle_vs_raw": ("oracle", "raw"),
}
LABEL_CONTROLS = ("global_label_shuffle", "wrong_emotion_permutation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--label-oracle-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=3901)
    return parser.parse_args()


def baseline_prediction_path(root: Path, model_seed: int, model: str) -> Path:
    directory = "context" if model == "context" else "full"
    return root / f"{directory}_seed{model_seed}" / "val_predictions.npz"


def oracle_prediction_path(root: Path, model_seed: int) -> Path:
    return root / "runs" / f"oracle_label_seed{model_seed}" / "val_predictions.npz"


def control_prediction_path(
    root: Path, model_seed: int, condition: str, replacement_seed: int
) -> Path:
    return (
        root
        / "control_predictions"
        / f"oracle_label_seed{model_seed}_{condition}_seed{replacement_seed}.npz"
    )


def load_inputs(
    baseline_dir: Path,
    oracle_dir: Path,
    model_seeds: list[int],
    replacement_seeds: list[int],
) -> tuple[
    dict[str, dict[int, dict[str, np.ndarray]]],
    dict[str, dict[int, np.ndarray]],
    dict[str, np.ndarray],
]:
    models: dict[str, dict[int, dict[str, np.ndarray]]] = {
        "context": {}, "raw": {}, "oracle": {}
    }
    controls: dict[str, dict[int, np.ndarray]] = {
        condition: {} for condition in LABEL_CONTROLS
    }
    reference = None
    for model_seed in model_seeds:
        paths = {
            "context": baseline_prediction_path(baseline_dir, model_seed, "context"),
            "raw": baseline_prediction_path(baseline_dir, model_seed, "raw"),
            "oracle": oracle_prediction_path(oracle_dir, model_seed),
        }
        for model_name, path in paths.items():
            prediction = load_prediction(path)
            if reference is None:
                reference = prediction
            else:
                validate_alignment(reference, prediction, path)
            models[model_name][model_seed] = prediction
        for condition in LABEL_CONTROLS:
            logits = []
            for replacement_seed in replacement_seeds:
                path = control_prediction_path(
                    oracle_dir, model_seed, condition, replacement_seed
                )
                prediction = load_prediction(path)
                validate_alignment(reference, prediction, path)
                logits.append(prediction["logits"])
            controls[condition][model_seed] = np.stack(logits)
    assert reference is not None
    return models, controls, reference


def paired_metrics(
    better_logits: np.ndarray,
    reference_logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    target = np.arange(len(labels))
    better_pred = better_logits.argmax(axis=1)
    reference_pred = reference_logits.argmax(axis=1)
    better_log_prob = log_softmax(better_logits)[target, labels]
    reference_log_prob = log_softmax(reference_logits)[target, labels]
    return {
        "uar_delta": uar(better_pred, labels) - uar(reference_pred, labels),
        "nll_delta": float(np.mean(-reference_log_prob + better_log_prob)),
        "correct_probability_delta": float(
            np.mean(np.exp(better_log_prob) - np.exp(reference_log_prob))
        ),
        "prediction_flip_rate": float(np.mean(better_pred != reference_pred)),
        "beneficial_flip_rate": float(
            np.mean((reference_pred != labels) & (better_pred == labels))
        ),
        "harmful_flip_rate": float(
            np.mean((reference_pred == labels) & (better_pred != labels))
        ),
    }


def per_class_delta(
    better_logits: np.ndarray,
    reference_logits: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    better_pred = better_logits.argmax(axis=1)
    reference_pred = reference_logits.argmax(axis=1)
    result = {}
    for class_id, emotion in enumerate(EMOTIONS):
        mask = labels == class_id
        result[emotion] = float(
            (better_pred[mask] == class_id).mean()
            - (reference_pred[mask] == class_id).mean()
        )
    return result


def observed_effects(
    model_seeds: list[int],
    models: dict[str, dict[int, dict[str, np.ndarray]]],
    controls: dict[str, dict[int, np.ndarray]],
    labels: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    comparison_rows = []
    control_rows = []
    class_rows = []
    for comparison, (better, reference) in MODEL_COMPARISONS.items():
        for model_seed in model_seeds:
            better_logits = models[better][model_seed]["logits"]
            reference_logits = models[reference][model_seed]["logits"]
            comparison_rows.append(
                {
                    "comparison": comparison,
                    "model_seed": model_seed,
                    **paired_metrics(better_logits, reference_logits, labels),
                }
            )
            for emotion, delta in per_class_delta(
                better_logits, reference_logits, labels
            ).items():
                class_rows.append(
                    {
                        "family": "model_comparison",
                        "comparison": comparison,
                        "model_seed": model_seed,
                        "emotion": emotion,
                        "mean_recall_delta": delta,
                    }
                )

    for condition in LABEL_CONTROLS:
        for model_seed in model_seeds:
            true_logits = models["oracle"][model_seed]["logits"]
            replicate_metrics = []
            replicate_class = []
            for control_logits in controls[condition][model_seed]:
                replicate_metrics.append(
                    paired_metrics(true_logits, control_logits, labels)
                )
                replicate_class.append(
                    per_class_delta(true_logits, control_logits, labels)
                )
            frame = pd.DataFrame(replicate_metrics)
            control_rows.append(
                {
                    "condition": condition,
                    "model_seed": model_seed,
                    **{
                        f"mean_{column}": float(frame[column].mean())
                        for column in frame.columns
                    },
                    "replacement_replicates": len(frame),
                }
            )
            for emotion in EMOTIONS:
                class_rows.append(
                    {
                        "family": "label_control",
                        "comparison": condition,
                        "model_seed": model_seed,
                        "emotion": emotion,
                        "mean_recall_delta": float(
                            np.mean([item[emotion] for item in replicate_class])
                        ),
                    }
                )
    return comparison_rows, control_rows, class_rows


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
        "correct": np.asarray(
            [
                [
                    np.sum(
                        (folders == folder)
                        & (labels == class_id)
                        & (predictions == class_id)
                    )
                    for class_id in range(len(EMOTIONS))
                ]
                for folder in unique_folders
            ],
            dtype=float,
        ),
        "nll_sum": np.asarray(
            [np.sum(-log_probability[folders == folder]) for folder in unique_folders]
        ),
    }


def bootstrap_pair(
    name: str,
    better_logits: dict[int, np.ndarray],
    reference_logits: dict[int, np.ndarray],
    labels: np.ndarray,
    folders: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    return hierarchical_pair_bootstrap(
        name, better_logits, reference_logits, labels, folders, replicates, seed,
        sample_replacement_manifests=False,
    )


def hierarchical_pair_bootstrap(
    name: str,
    better_logits: dict[int, np.ndarray],
    reference_logits: dict[int, np.ndarray],
    labels: np.ndarray,
    folders: np.ndarray,
    replicates: int,
    seed: int,
    sample_replacement_manifests: bool,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    model_seeds = list(better_logits)
    unique_folders = np.unique(folders)
    folder_class_counts = np.asarray(
        [
            [
                np.sum((folders == folder) & (labels == class_id))
                for class_id in range(len(EMOTIONS))
            ]
            for folder in unique_folders
        ],
        dtype=float,
    )
    folder_sample_counts = folder_class_counts.sum(axis=1)
    def prediction_replicates(logits: np.ndarray) -> np.ndarray:
        if logits.ndim == 2:
            return logits[None, :, :]
        if logits.ndim == 3:
            return logits
        raise ValueError(f"Unexpected prediction tensor shape: {logits.shape}")

    better_values = {
        model_seed: [
            folder_statistics(logits, labels, folders, unique_folders)
            for logits in prediction_replicates(better_logits[model_seed])
        ]
        for model_seed in model_seeds
    }
    reference_values = {
        model_seed: [
            folder_statistics(logits, labels, folders, unique_folders)
            for logits in prediction_replicates(reference_logits[model_seed])
        ]
        for model_seed in model_seeds
    }
    num_manifests = len(reference_values[model_seeds[0]])
    uar_deltas = np.empty(replicates)
    nll_deltas = np.empty(replicates)
    for bootstrap_index in range(replicates):
        multiplicities = bootstrap_folder_multiplicities(rng, folder_class_counts)
        class_denominators = multiplicities @ folder_class_counts
        sample_denominator = float(multiplicities @ folder_sample_counts)
        sampled_model_seeds = rng.choice(
            model_seeds, size=len(model_seeds), replace=True
        )
        if sample_replacement_manifests:
            manifest_indices = rng.choice(
                num_manifests, size=num_manifests, replace=True
            )
        else:
            manifest_indices = np.asarray([0])
        current_uar = []
        current_nll = []
        for sampled_seed in sampled_model_seeds:
            model_seed = int(sampled_seed)
            better = better_values[model_seed][0]
            better_uar = float(
                np.mean((multiplicities @ better["correct"]) / class_denominators)
            )
            better_nll = float(
                multiplicities @ better["nll_sum"] / sample_denominator
            )
            for manifest_index in manifest_indices:
                reference = reference_values[model_seed][int(manifest_index)]
                reference_uar = float(
                    np.mean(
                        (multiplicities @ reference["correct"]) / class_denominators
                    )
                )
                reference_nll = float(
                    multiplicities @ reference["nll_sum"] / sample_denominator
                )
                current_uar.append(better_uar - reference_uar)
                current_nll.append(reference_nll - better_nll)
        uar_deltas[bootstrap_index] = np.mean(current_uar)
        nll_deltas[bootstrap_index] = np.mean(current_nll)

    def interval(values: np.ndarray) -> dict[str, float]:
        low, high = np.quantile(values, [0.025, 0.975])
        return {
            "bootstrap_mean": float(values.mean()),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "probability_above_zero": float(np.mean(values > 0)),
        }

    units = ["model_seed", "source_folder"]
    if sample_replacement_manifests:
        units.append("replacement_manifest")
    return {
        "comparison": name,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "resampling_units": units,
        "uar_delta": interval(uar_deltas),
        "nll_delta": interval(nll_deltas),
    }


def aggregate_rows(
    frame: pd.DataFrame, name_column: str, metric_prefix: str = ""
) -> dict[str, dict[str, float | int]]:
    aggregate = {}
    metrics = (
        "uar_delta", "nll_delta", "correct_probability_delta",
        "prediction_flip_rate", "beneficial_flip_rate", "harmful_flip_rate",
    )
    for name, group in frame.groupby(name_column, sort=False):
        values = group[f"{metric_prefix}uar_delta"].to_numpy()
        aggregate[str(name)] = {
            metric: float(group[f"{metric_prefix}{metric}"].mean())
            for metric in metrics
        }
        aggregate[str(name)].update(
            {
                "uar_delta_std_across_model_seeds": float(values.std(ddof=1)),
                "positive_uar_delta_model_seeds": int((values > 0).sum()),
                "two_sided_exact_sign_p": exact_two_sided_sign_p(
                    int((values > 0).sum()), len(values)
                ),
            }
        )
    return aggregate


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_summary_path = args.baseline_dir / "validation_matrix_summary.json"
    oracle_summary_path = args.label_oracle_dir / "label_oracle_summary.json"
    baseline_summary = json.loads(baseline_summary_path.read_text())
    oracle_summary = json.loads(oracle_summary_path.read_text())
    manifest_hash = sha256(args.manifest)
    if baseline_summary.get("test_evaluated") is not False:
        raise RuntimeError("Baseline input is not validation-only")
    if oracle_summary.get("test_evaluated") is not False:
        raise RuntimeError("Oracle input is not validation-only")
    if oracle_summary.get("partition") != "val":
        raise RuntimeError("This audit accepts only validation predictions")
    if baseline_summary.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("Baseline and frozen manifests differ")
    if oracle_summary.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("Oracle and frozen manifests differ")
    if oracle_summary.get("baseline_summary_sha256") != sha256(baseline_summary_path):
        raise RuntimeError("Oracle run was not built from this baseline summary")

    model_seeds = [int(seed) for seed in oracle_summary["model_seeds"]]
    replacement_seeds = [int(seed) for seed in oracle_summary["replacement_seeds"]]
    if model_seeds != [int(seed) for seed in baseline_summary["seeds"]]:
        raise RuntimeError("Baseline and oracle model seeds differ")
    models, controls, reference = load_inputs(
        args.baseline_dir, args.label_oracle_dir, model_seeds, replacement_seeds
    )
    labels = reference["labels"].astype(int)
    folders = reference["source_folders"].astype(str)
    manifest = pd.read_csv(args.manifest, dtype={"source_folder": str})
    manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    if not np.array_equal(
        reference["sample_ids"].astype(str), manifest["sample_id"].to_numpy(str)
    ):
        raise RuntimeError("Prediction sample order differs from validation manifest")
    if not np.array_equal(folders, manifest["source_folder"].to_numpy(str)):
        raise RuntimeError("Prediction folders differ from validation manifest")

    comparison_rows, control_rows, class_rows = observed_effects(
        model_seeds, models, controls, labels
    )
    comparison_frame = pd.DataFrame(comparison_rows)
    control_frame = pd.DataFrame(control_rows)
    comparison_path = args.output_dir / "model_comparison_effects_by_seed.csv"
    control_path = args.output_dir / "label_control_effects_by_seed.csv"
    class_path = args.output_dir / "per_class_effects.csv"
    comparison_frame.to_csv(comparison_path, index=False, lineterminator="\n")
    control_frame.to_csv(control_path, index=False, lineterminator="\n")
    pd.DataFrame(class_rows).to_csv(class_path, index=False, lineterminator="\n")

    model_bootstrap = {}
    for index, (comparison, (better, reference_model)) in enumerate(
        MODEL_COMPARISONS.items()
    ):
        model_bootstrap[comparison] = bootstrap_pair(
            comparison,
            {seed: models[better][seed]["logits"] for seed in model_seeds},
            {seed: models[reference_model][seed]["logits"] for seed in model_seeds},
            labels, folders, args.bootstrap_replicates, args.bootstrap_seed + index,
        )
    control_bootstrap = {}
    for index, condition in enumerate(LABEL_CONTROLS):
        control_bootstrap[condition] = hierarchical_pair_bootstrap(
            condition,
            {seed: models["oracle"][seed]["logits"] for seed in model_seeds},
            controls[condition], labels, folders, args.bootstrap_replicates,
            args.bootstrap_seed + len(MODEL_COMPARISONS) + index,
            sample_replacement_manifests=True,
        )

    summary = {
        "protocol": "label-oracle-validation-statistical-audit-v1",
        "partition": "val",
        "test_evaluated": False,
        "model_seeds": model_seeds,
        "replacement_seeds": replacement_seeds,
        "num_samples": len(labels),
        "num_source_folders": int(len(np.unique(folders))),
        "class_counts": {
            emotion: int((labels == class_id).sum())
            for class_id, emotion in enumerate(EMOTIONS)
        },
        "baseline_summary_sha256": sha256(baseline_summary_path),
        "label_oracle_summary_sha256": sha256(oracle_summary_path),
        "manifest_sha256": manifest_hash,
        "analysis_runner_sha256": sha256(Path(__file__)),
        "model_comparisons": aggregate_rows(comparison_frame, "comparison"),
        "label_controls": aggregate_rows(control_frame, "condition", "mean_"),
        "hierarchical_bootstrap": {
            "model_comparisons": model_bootstrap,
            "label_controls": control_bootstrap,
        },
        "inference_note": (
            "Intervals are validation diagnostics, not confirmatory test estimates. "
            "Only five model seeds and eight validation source folders are available, "
            "and validation was used for checkpoint selection."
        ),
    }
    summary_path = args.output_dir / "oracle_statistical_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    protocol = {
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "model_comparison_resampling_units": ["model_seed", "source_folder"],
        "label_control_resampling_units": [
            "model_seed", "source_folder", "replacement_manifest"
        ],
        "require_all_classes_in_each_bootstrap_draw": True,
        "test_evaluated": False,
    }
    (args.output_dir / "statistical_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "model_comparisons": summary["model_comparisons"],
                "label_controls": summary["label_controls"],
                "hierarchical_bootstrap": summary["hierarchical_bootstrap"],
                "test_evaluated": False,
            },
            indent=2,
        )
    )
    print(f"Summary:          {summary_path}")
    print(f"Model effects:    {comparison_path}")
    print(f"Control effects:  {control_path}")
    print(f"Per-class effects: {class_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
