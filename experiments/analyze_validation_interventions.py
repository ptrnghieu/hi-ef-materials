#!/usr/bin/env python3
"""Statistical audit of frozen Hi-EF validation intervention predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
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
    parser.add_argument("--interventions-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2901)
    return parser.parse_args()


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = ("logits", "labels", "predictions", "sample_ids", "source_folders")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")
        result = {key: np.asarray(data[key]) for key in required}
    if result["logits"].ndim != 2 or result["logits"].shape[1] != len(EMOTIONS):
        raise ValueError(f"Unexpected logits shape in {path}: {result['logits'].shape}")
    return result


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def uar(predictions: np.ndarray, labels: np.ndarray, indices: np.ndarray | None = None) -> float:
    if indices is not None:
        predictions = predictions[indices]
        labels = labels[indices]
    recalls = []
    for class_id in range(len(EMOTIONS)):
        mask = labels == class_id
        if not mask.any():
            raise ValueError(f"UAR sample has no observations for class {class_id}")
        recalls.append(float((predictions[mask] == class_id).mean()))
    return float(np.mean(recalls))


def exact_two_sided_sign_p(positive: int, total: int) -> float:
    from math import comb

    tail = sum(comb(total, k) for k in range(0, min(positive, total - positive) + 1))
    return min(1.0, 2.0 * tail / (2**total))


def prediction_path(
    root: Path, model_seed: int, condition: str, replacement_seed: int | None = None
) -> Path:
    name = f"full_seed{model_seed}_{condition}"
    if replacement_seed is not None:
        name += f"_seed{replacement_seed}"
    return root / "predictions" / f"{name}.npz"


def validate_alignment(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], path: Path) -> None:
    for key in ("labels", "sample_ids", "source_folders"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"{key} alignment differs in {path}")
    recomputed = candidate["logits"].argmax(axis=1)
    if not np.array_equal(recomputed, candidate["predictions"]):
        raise ValueError(f"Stored predictions do not match logits in {path}")


def load_all_predictions(
    root: Path, model_seeds: list[int], replacement_seeds: list[int]
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, dict[int, np.ndarray]], dict[str, np.ndarray]]:
    true_data: dict[int, dict[str, np.ndarray]] = {}
    condition_logits: dict[str, dict[int, np.ndarray]] = {
        "zero": {}, **{condition: {} for condition in REPLACEMENT_CONDITIONS}
    }
    reference = None
    for model_seed in model_seeds:
        true_path = prediction_path(root, model_seed, "true")
        current_true = load_prediction(true_path)
        if reference is None:
            reference = current_true
        else:
            validate_alignment(reference, current_true, true_path)
        true_data[model_seed] = current_true

        zero_path = prediction_path(root, model_seed, "zero")
        zero = load_prediction(zero_path)
        validate_alignment(reference, zero, zero_path)
        condition_logits["zero"][model_seed] = zero["logits"][None, :, :]

        for condition in REPLACEMENT_CONDITIONS:
            logits = []
            for replacement_seed in replacement_seeds:
                path = prediction_path(root, model_seed, condition, replacement_seed)
                prediction = load_prediction(path)
                validate_alignment(reference, prediction, path)
                logits.append(prediction["logits"])
            condition_logits[condition][model_seed] = np.stack(logits)
    assert reference is not None
    return true_data, condition_logits, reference


def observed_effects(
    model_seeds: list[int],
    true_data: dict[int, dict[str, np.ndarray]],
    condition_logits: dict[str, dict[int, np.ndarray]],
    labels: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    rows = []
    class_rows = []
    seed_mean_uar_deltas: dict[str, np.ndarray] = {}
    target = np.arange(len(labels))

    for condition, by_seed in condition_logits.items():
        per_seed_deltas = []
        for model_seed in model_seeds:
            true_logits = true_data[model_seed]["logits"]
            true_pred = true_logits.argmax(axis=1)
            true_log_prob = log_softmax(true_logits)[target, labels]
            true_prob = np.exp(true_log_prob)
            true_uar = uar(true_pred, labels)
            replicate_rows = []
            class_deltas = []
            for replicate_index, control_logits in enumerate(by_seed[model_seed]):
                control_pred = control_logits.argmax(axis=1)
                control_log_prob = log_softmax(control_logits)[target, labels]
                control_prob = np.exp(control_log_prob)
                replicate_row = {
                    "condition": condition,
                    "model_seed": model_seed,
                    "replicate_index": replicate_index,
                    "uar_delta": true_uar - uar(control_pred, labels),
                    "nll_delta": float(np.mean(control_log_prob * -1.0 + true_log_prob)),
                    "correct_probability_delta": float(np.mean(true_prob - control_prob)),
                    "prediction_flip_rate": float(np.mean(true_pred != control_pred)),
                    "harmful_flip_rate": float(
                        np.mean((true_pred == labels) & (control_pred != labels))
                    ),
                    "beneficial_flip_rate": float(
                        np.mean((true_pred != labels) & (control_pred == labels))
                    ),
                }
                replicate_rows.append(replicate_row)
                for class_id, emotion in enumerate(EMOTIONS):
                    mask = labels == class_id
                    class_deltas.append(
                        {
                            "condition": condition,
                            "model_seed": model_seed,
                            "replicate_index": replicate_index,
                            "emotion": emotion,
                            "recall_delta": float(
                                (true_pred[mask] == class_id).mean()
                                - (control_pred[mask] == class_id).mean()
                            ),
                        }
                    )
            frame = pd.DataFrame(replicate_rows)
            rows.append(
                {
                    "condition": condition,
                    "model_seed": model_seed,
                    **{
                        f"mean_{column}": float(frame[column].mean())
                        for column in (
                            "uar_delta",
                            "nll_delta",
                            "correct_probability_delta",
                            "prediction_flip_rate",
                            "harmful_flip_rate",
                            "beneficial_flip_rate",
                        )
                    },
                    "replacement_replicates": len(frame),
                }
            )
            per_seed_deltas.append(float(frame["uar_delta"].mean()))
            class_frame = pd.DataFrame(class_deltas)
            for emotion, group in class_frame.groupby("emotion"):
                class_rows.append(
                    {
                        "condition": condition,
                        "model_seed": model_seed,
                        "emotion": emotion,
                        "mean_recall_delta": float(group["recall_delta"].mean()),
                    }
                )
        seed_mean_uar_deltas[condition] = np.asarray(per_seed_deltas)
    return rows, class_rows, seed_mean_uar_deltas


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
    raise RuntimeError("Could not draw a cluster bootstrap sample containing all classes")


def hierarchical_bootstrap(
    condition: str,
    model_seeds: list[int],
    true_data: dict[int, dict[str, np.ndarray]],
    condition_logits: dict[int, np.ndarray],
    labels: np.ndarray,
    folders: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    target = np.arange(len(labels))
    unique_folders = np.unique(folders)
    folder_class_counts = np.asarray(
        [
            [np.sum((folders == folder) & (labels == class_id)) for class_id in range(len(EMOTIONS))]
            for folder in unique_folders
        ],
        dtype=float,
    )
    folder_sample_counts = folder_class_counts.sum(axis=1)
    true_values = {}
    control_values = {}
    for model_seed in model_seeds:
        true_logits = true_data[model_seed]["logits"]
        true_log_prob = log_softmax(true_logits)[target, labels]
        true_pred = true_logits.argmax(axis=1)
        true_values[model_seed] = {
            "correct": np.asarray(
                [
                    [
                        np.sum(
                            (folders == folder)
                            & (labels == class_id)
                            & (true_pred == class_id)
                        )
                        for class_id in range(len(EMOTIONS))
                    ]
                    for folder in unique_folders
                ],
                dtype=float,
            ),
            "nll_sum": np.asarray(
                [np.sum(-true_log_prob[folders == folder]) for folder in unique_folders]
            ),
        }
        controls = []
        for logits in condition_logits[model_seed]:
            log_prob = log_softmax(logits)[target, labels]
            pred = logits.argmax(axis=1)
            controls.append(
                {
                    "correct": np.asarray(
                        [
                            [
                                np.sum(
                                    (folders == folder)
                                    & (labels == class_id)
                                    & (pred == class_id)
                                )
                                for class_id in range(len(EMOTIONS))
                            ]
                            for folder in unique_folders
                        ],
                        dtype=float,
                    ),
                    "nll_sum": np.asarray(
                        [np.sum(-log_prob[folders == folder]) for folder in unique_folders]
                    ),
                }
            )
        control_values[model_seed] = controls

    uar_deltas = np.empty(replicates)
    nll_deltas = np.empty(replicates)
    for bootstrap_index in range(replicates):
        multiplicities = bootstrap_folder_multiplicities(rng, folder_class_counts)
        class_denominators = multiplicities @ folder_class_counts
        sample_denominator = float(multiplicities @ folder_sample_counts)
        sampled_model_seeds = rng.choice(model_seeds, size=len(model_seeds), replace=True)
        num_control_replicates = len(control_values[model_seeds[0]])
        sampled_replicates = rng.choice(
            num_control_replicates, size=num_control_replicates, replace=True
        )
        current_uar = []
        current_nll = []
        for model_seed in sampled_model_seeds:
            true = true_values[int(model_seed)]
            true_uar = float(
                np.mean((multiplicities @ true["correct"]) / class_denominators)
            )
            true_nll = float(multiplicities @ true["nll_sum"] / sample_denominator)
            for replicate_index in sampled_replicates:
                control = control_values[int(model_seed)][int(replicate_index)]
                control_uar = float(
                    np.mean(
                        (multiplicities @ control["correct"]) / class_denominators
                    )
                )
                control_nll = float(
                    multiplicities @ control["nll_sum"] / sample_denominator
                )
                current_uar.append(true_uar - control_uar)
                current_nll.append(control_nll - true_nll)
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

    return {
        "condition": condition,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "resampling_units": ["model_seed", "source_folder", "replacement_manifest"],
        "uar_delta": interval(uar_deltas),
        "nll_delta": interval(nll_deltas),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    intervention_summary_path = args.interventions_dir / "intervention_summary.json"
    intervention_summary = json.loads(intervention_summary_path.read_text())
    if intervention_summary["test_evaluated"] is not False:
        raise RuntimeError("Input intervention run is not validation-only")
    if intervention_summary["partition"] != "val":
        raise RuntimeError("This audit accepts only validation predictions")
    if intervention_summary["manifest_sha256"] != sha256(args.manifest):
        raise RuntimeError("Statistical audit and intervention manifests differ")

    model_seeds = [int(seed) for seed in intervention_summary["model_seeds"]]
    replacement_seeds = [int(seed) for seed in intervention_summary["replacement_seeds"]]
    true_data, condition_logits, reference = load_all_predictions(
        args.interventions_dir, model_seeds, replacement_seeds
    )
    labels = reference["labels"].astype(int)
    folders = reference["source_folders"].astype(str)

    manifest = pd.read_csv(args.manifest, dtype={"source_folder": str})
    manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    if not np.array_equal(reference["sample_ids"].astype(str), manifest["sample_id"].to_numpy(str)):
        raise RuntimeError("Prediction sample order differs from the frozen validation manifest")
    if not np.array_equal(folders, manifest["source_folder"].to_numpy(str)):
        raise RuntimeError("Prediction folder ids differ from the frozen validation manifest")

    effect_rows, class_rows, seed_deltas = observed_effects(
        model_seeds, true_data, condition_logits, labels
    )
    effects_path = args.output_dir / "condition_effects_by_model_seed.csv"
    pd.DataFrame(effect_rows).to_csv(effects_path, index=False, lineterminator="\n")
    class_path = args.output_dir / "per_class_effects.csv"
    pd.DataFrame(class_rows).to_csv(class_path, index=False, lineterminator="\n")

    aggregate = {}
    for condition in condition_logits:
        rows = [row for row in effect_rows if row["condition"] == condition]
        frame = pd.DataFrame(rows)
        deltas = seed_deltas[condition]
        aggregate[condition] = {
            column: float(frame[column].mean())
            for column in (
                "mean_uar_delta",
                "mean_nll_delta",
                "mean_correct_probability_delta",
                "mean_prediction_flip_rate",
                "mean_harmful_flip_rate",
                "mean_beneficial_flip_rate",
            )
        }
        aggregate[condition].update(
            {
                "uar_delta_std_across_model_seeds": float(deltas.std(ddof=1)),
                "positive_uar_delta_model_seeds": int((deltas > 0).sum()),
                "two_sided_exact_sign_p": exact_two_sided_sign_p(
                    int((deltas > 0).sum()), len(deltas)
                ),
            }
        )

    bootstrap = {}
    for condition_index, condition in enumerate(condition_logits):
        bootstrap[condition] = hierarchical_bootstrap(
            condition=condition,
            model_seeds=model_seeds,
            true_data=true_data,
            condition_logits=condition_logits[condition],
            labels=labels,
            folders=folders,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + condition_index,
        )

    summary = {
        "protocol": "validation-intervention-statistical-audit-v1",
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
        "intervention_summary_sha256": sha256(intervention_summary_path),
        "manifest_sha256": sha256(args.manifest),
        "analysis_runner_sha256": sha256(Path(__file__)),
        "aggregate": aggregate,
        "hierarchical_bootstrap": bootstrap,
        "inference_note": (
            "Bootstrap intervals are uncertainty diagnostics: only five model seeds "
            "and eight validation source folders are available, and validation was "
            "used for checkpoint selection."
        ),
    }
    summary_path = args.output_dir / "statistical_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "statistical_protocol.json").write_text(
        json.dumps(
            {
                "bootstrap_replicates": args.bootstrap_replicates,
                "bootstrap_seed": args.bootstrap_seed,
                "resampling_units": [
                    "model_seed", "source_folder", "replacement_manifest"
                ],
                "require_all_classes_in_each_bootstrap_draw": True,
                "test_evaluated": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Summary:          {summary_path}")
    print(f"Condition table:  {effects_path}")
    print(f"Per-class table:  {class_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
