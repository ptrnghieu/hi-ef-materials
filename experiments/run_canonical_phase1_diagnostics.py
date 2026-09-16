#!/usr/bin/env python3
"""Run frozen post-audit calibration, residual, and branch diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analyze_canonical_residual_matrix import (
    MODEL_SEEDS,
    VARIANTS,
    hierarchical_pair_bootstrap,
    load_matrix_predictions,
    observed_pair,
)
from canonical_diagnostic_metrics import (
    EMOTIONS,
    calibration_metrics,
    rank_correlation,
    sample_residual_diagnostics,
)
from train_baselines import seed_everything, to_device
from train_contextual_affective_residual import (
    ContextualAffectiveResidual,
    build_loader,
)


INTERVENTION_COMPARISONS = (
    ("affect_only_vs_context", "affect_only", "context"),
    ("interaction_only_vs_context", "interaction_only", "context"),
    ("both_vs_context", "both", "context"),
    ("both_vs_affect_only", "both", "affect_only"),
    ("both_vs_interaction_only", "both", "interaction_only"),
)
BOOTSTRAP_SEEDS = {
    name: 6901 + index
    for index, (name, _, _) in enumerate(INTERVENTION_COMPARISONS)
}


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
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    return parser.parse_args()


def public_aggregate(frame: pd.DataFrame, group_columns: list[str]) -> list[dict[str, object]]:
    numeric = [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column not in {"model_seed", "label"}
    ]
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_columns, keys)}
        for column in numeric:
            values = group[column].to_numpy(float)
            row[f"{column}_mean"] = float(np.nanmean(values))
            row[f"{column}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        row["rows"] = int(len(group))
        rows.append(row)
    return rows


@torch.no_grad()
def run_both_interventions(
    args: argparse.Namespace,
    validation_rows: pd.DataFrame,
    expected_sample_ids: np.ndarray,
    expected_labels: np.ndarray,
    expected_folders: np.ndarray,
) -> tuple[dict[int, dict[str, np.ndarray]], pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_logits: dict[int, dict[str, np.ndarray]] = {}
    metadata_reference: pd.DataFrame | None = None
    for model_seed in MODEL_SEEDS:
        run_dir = args.matrix_dir / f"both_seed{model_seed}"
        config = json.loads((run_dir / "config.json").read_text())
        if config.get("variant") != "both" or int(config.get("seed", -1)) != model_seed:
            raise RuntimeError(f"Unexpected checkpoint configuration in {run_dir}")
        if config.get("manifest_sha256") != sha256(args.manifest):
            raise RuntimeError(f"Checkpoint manifest differs in {run_dir}")
        seed_everything(model_seed)
        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model = ContextualAffectiveResidual(
            variant="both",
            num_source_folders=int(config["num_training_source_folders"]),
            d_model=int(config["d_model"]),
            temporal_layers=int(config["temporal_layers"]),
            context_layers=int(config["context_layers"]),
            dropout=float(config["dropout"]),
            face_pooling=str(config["face_pooling"]),
            gradient_reversal_scale=float(config["gradient_reversal_scale"]),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        loader = build_loader(
            validation_rows, args.features_dir, args.batch_size, args.workers,
            False, model_seed + 1,
        )
        arrays = {condition: [] for condition in ("context", "affect_only", "interaction_only", "both")}
        sample_ids: list[str] = []
        folders: list[str] = []
        labels: list[np.ndarray] = []
        face_counts: list[np.ndarray] = []
        audio_found: list[np.ndarray] = []
        for batch in loader:
            device_batch = to_device(batch, device)
            logits = model.intervention_logits(device_batch)
            for condition, values in logits.items():
                arrays[condition].append(values.cpu().numpy())
            sample_ids.extend(batch["sample_id"])
            folders.extend(batch["source_folder"])
            labels.append(batch["target"].numpy())
            face_counts.append(batch["clip3"]["face_mask"].sum(dim=1).numpy())
            audio_found.append(batch["clip3"]["audio_found"].numpy())
        merged = {condition: np.concatenate(values) for condition, values in arrays.items()}
        current_ids = np.asarray(sample_ids).astype(str)
        current_folders = np.asarray(folders).astype(str)
        current_labels = np.concatenate(labels).astype(int)
        if not np.array_equal(current_ids, expected_sample_ids):
            raise RuntimeError(f"Intervention sample order mismatch for seed {model_seed}")
        if not np.array_equal(current_folders, expected_folders):
            raise RuntimeError(f"Intervention folder order mismatch for seed {model_seed}")
        if not np.array_equal(current_labels, expected_labels):
            raise RuntimeError(f"Intervention labels mismatch for seed {model_seed}")
        with np.load(run_dir / "val_predictions.npz") as original:
            np.testing.assert_allclose(
                merged["context"], original["context_logits"], rtol=1e-5, atol=1e-5
            )
            np.testing.assert_allclose(
                merged["both"], original["final_logits"], rtol=1e-5, atol=1e-5
            )
        output_path = args.output_dir / f"both_seed{model_seed}_branch_interventions.npz"
        np.savez_compressed(
            output_path,
            **merged,
            labels=current_labels,
            sample_ids=current_ids,
            source_folders=current_folders,
        )
        all_logits[model_seed] = merged
        metadata = pd.DataFrame({
            "sample_id": current_ids,
            "source_folder": current_folders,
            "label": current_labels,
            "party_a_valid_face_frames": np.concatenate(face_counts).astype(int),
            "party_a_audio_found": np.concatenate(audio_found).astype(bool),
        })
        if metadata_reference is None:
            metadata_reference = metadata
        elif not metadata_reference.equals(metadata):
            raise RuntimeError("Feature-availability metadata changed across seeds")
    assert metadata_reference is not None
    return all_logits, metadata_reference


def face_availability_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count < 8:
        return "1-7"
    if count < 16:
        return "8-15"
    return "16"


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.matrix_dir / "canonical_residual_matrix_summary.json"
    matrix_summary = json.loads(summary_path.read_text())
    if matrix_summary.get("protocol") != "canonical-contextual-affective-residual-validation-matrix-v1":
        raise RuntimeError("Unexpected matrix protocol")
    if matrix_summary.get("test_evaluated") is not False:
        raise RuntimeError("Matrix input is not validation-only")
    if matrix_summary.get("model_seeds") != list(MODEL_SEEDS):
        raise RuntimeError("Unexpected model seeds")
    if matrix_summary.get("variants") != list(VARIANTS):
        raise RuntimeError("Unexpected matrix variants")
    if matrix_summary.get("manifest_sha256") != sha256(args.manifest):
        raise RuntimeError("Matrix and diagnostic manifests differ")

    data, reference = load_matrix_predictions(args.matrix_dir)
    labels = reference["labels"].astype(int)
    sample_ids = reference["sample_ids"].astype(str)
    folders = reference["source_folders"].astype(str)
    manifest = pd.read_csv(args.manifest, dtype={"source_folder": str})
    validation_rows = manifest[manifest["split"] == "val"].reset_index(drop=True)
    if not np.array_equal(sample_ids, validation_rows["sample_id"].to_numpy(str)):
        raise RuntimeError("Prediction order differs from validation manifest")
    if not np.array_equal(folders, validation_rows["source_folder"].to_numpy(str)):
        raise RuntimeError("Prediction folders differ from validation manifest")
    expected_labels = validation_rows["clip4_emotion"].map(
        {emotion: index for index, emotion in enumerate(EMOTIONS)}
    ).to_numpy(int)
    if not np.array_equal(labels, expected_labels):
        raise RuntimeError("Prediction labels differ from validation manifest")

    calibration_rows: list[dict[str, object]] = []
    calibration_class_rows: list[dict[str, object]] = []
    for model_seed in MODEL_SEEDS:
        for variant in VARIANTS:
            prediction_sets = {"final": data[model_seed][variant]["final_logits"]}
            if variant != "context":
                prediction_sets["context"] = data[model_seed][variant]["context_logits"]
            for prediction_type, logits in prediction_sets.items():
                metrics, per_class = calibration_metrics(logits, labels)
                calibration_rows.append({
                    "model_seed": model_seed,
                    "variant": variant,
                    "prediction_type": prediction_type,
                    **metrics,
                })
                for row in per_class:
                    calibration_class_rows.append({
                        "model_seed": model_seed,
                        "variant": variant,
                        "prediction_type": prediction_type,
                        **row,
                    })
    calibration_frame = pd.DataFrame(calibration_rows)
    calibration_class_frame = pd.DataFrame(calibration_class_rows)

    intervention_logits, metadata = run_both_interventions(
        args, validation_rows, sample_ids, labels, folders
    )
    intervention_calibration_rows: list[dict[str, object]] = []
    intervention_class_calibration_rows: list[dict[str, object]] = []
    for model_seed in MODEL_SEEDS:
        for condition, logits in intervention_logits[model_seed].items():
            metrics, per_class = calibration_metrics(logits, labels)
            intervention_calibration_rows.append({
                "model_seed": model_seed, "condition": condition, **metrics,
            })
            for row in per_class:
                intervention_class_calibration_rows.append({
                    "model_seed": model_seed, "condition": condition, **row,
                })

    residual_rows: list[dict[str, object]] = []
    for model_seed in MODEL_SEEDS:
        for variant in ("affect", "interaction", "both"):
            current = data[model_seed][variant]
            diagnostic = sample_residual_diagnostics(
                current["context_logits"], current["delta_logits"],
                current["final_logits"], labels,
            )
            for index, sample_id in enumerate(sample_ids):
                residual_rows.append({
                    "model_seed": model_seed,
                    "variant": variant,
                    "sample_id": sample_id,
                    "source_folder": folders[index],
                    "label": int(labels[index]),
                    "emotion": EMOTIONS[int(labels[index])],
                    "party_a_valid_face_frames": int(metadata.iloc[index]["party_a_valid_face_frames"]),
                    "party_a_face_bucket": face_availability_bucket(
                        int(metadata.iloc[index]["party_a_valid_face_frames"])
                    ),
                    "party_a_audio_found": bool(metadata.iloc[index]["party_a_audio_found"]),
                    **{key: values[index].item() for key, values in diagnostic.items()},
                })
    residual_frame = pd.DataFrame(residual_rows)

    residual_aggregates: dict[str, object] = {}
    for variant, group in residual_frame.groupby("variant", sort=False):
        by_flip = {
            str(category): {
                "count": int(len(values)),
                "delta_l2_mean": float(values["delta_l2"].mean()),
                "nll_change_mean": float(values["nll_change"].mean()),
            }
            for category, values in group.groupby("flip_category", sort=False)
        }
        residual_aggregates[str(variant)] = {
            "delta_l2_mean": float(group["delta_l2"].mean()),
            "delta_l2_std": float(group["delta_l2"].std(ddof=1)),
            "nll_change_mean": float(group["nll_change"].mean()),
            "confidence_delta_mean": float(group["confidence_delta"].mean()),
            "entropy_delta_mean": float(group["entropy_delta"].mean()),
            "residual_norm_vs_nll_change_rank_correlation": rank_correlation(
                group["delta_l2"].to_numpy(float), group["nll_change"].to_numpy(float)
            ),
            "residual_norm_vs_confidence_delta_rank_correlation": rank_correlation(
                group["delta_l2"].to_numpy(float), group["confidence_delta"].to_numpy(float)
            ),
            "by_flip_category": by_flip,
        }

    group_rows = []
    for grouping in (
        "model_seed", "party_a_face_bucket", "party_a_audio_found",
        "emotion", "source_folder",
    ):
        for (variant, value), group in residual_frame.groupby(["variant", grouping], sort=False):
            group_rows.append({
                "grouping": grouping,
                "variant": variant,
                "value": str(value),
                "rows": int(len(group)),
                "delta_l2_mean": float(group["delta_l2"].mean()),
                "nll_change_mean": float(group["nll_change"].mean()),
                "beneficial_flip_rate": float(np.mean(group["flip_category"] == "beneficial")),
                "harmful_flip_rate": float(np.mean(group["flip_category"] == "harmful")),
            })

    bootstrap: dict[str, object] = {}
    intervention_effect_rows: list[dict[str, object]] = []
    intervention_effect_class_rows: list[dict[str, object]] = []
    for name, better_condition, reference_condition in INTERVENTION_COMPARISONS:
        better = {seed: intervention_logits[seed][better_condition] for seed in MODEL_SEEDS}
        reference_condition_logits = {
            seed: intervention_logits[seed][reference_condition] for seed in MODEL_SEEDS
        }
        rows, class_rows = observed_pair(name, better, reference_condition_logits, labels)
        intervention_effect_rows.extend(rows)
        intervention_effect_class_rows.extend(class_rows)
        bootstrap[name] = hierarchical_pair_bootstrap(
            name, better, reference_condition_logits, labels, folders,
            args.bootstrap_replicates, BOOTSTRAP_SEEDS[name],
        )

    calibration_path = args.output_dir / "calibration_by_seed.csv"
    calibration_class_path = args.output_dir / "calibration_per_class.csv"
    residual_path = args.output_dir / "residual_sample_diagnostics.csv"
    residual_group_path = args.output_dir / "residual_group_diagnostics.csv"
    intervention_calibration_path = args.output_dir / "branch_intervention_calibration.csv"
    intervention_class_calibration_path = args.output_dir / "branch_intervention_per_class.csv"
    intervention_effect_path = args.output_dir / "branch_intervention_effects_by_seed.csv"
    intervention_effect_class_path = args.output_dir / "branch_intervention_per_class_effects.csv"
    calibration_frame.to_csv(calibration_path, index=False, lineterminator="\n")
    calibration_class_frame.to_csv(calibration_class_path, index=False, lineterminator="\n")
    residual_frame.to_csv(residual_path, index=False, lineterminator="\n")
    pd.DataFrame(group_rows).to_csv(residual_group_path, index=False, lineterminator="\n")
    pd.DataFrame(intervention_calibration_rows).to_csv(
        intervention_calibration_path, index=False, lineterminator="\n"
    )
    pd.DataFrame(intervention_class_calibration_rows).to_csv(
        intervention_class_calibration_path, index=False, lineterminator="\n"
    )
    pd.DataFrame(intervention_effect_rows).to_csv(
        intervention_effect_path, index=False, lineterminator="\n"
    )
    pd.DataFrame(intervention_effect_class_rows).to_csv(
        intervention_effect_class_path, index=False, lineterminator="\n"
    )

    spec_path = Path(__file__).with_name("RESEARCH_SPEC_v0.6.md")
    summary = {
        "protocol": "canonical-residual-post-audit-diagnostics-v1",
        "diagnostic_only": True,
        "research_interpretation_permitted": False,
        "model_selection_permitted": False,
        "partitions_touched": ["validation"],
        "test_evaluated": False,
        "num_samples": int(len(labels)),
        "num_source_folders": int(len(np.unique(folders))),
        "model_seeds": list(MODEL_SEEDS),
        "calibration_aggregates": public_aggregate(
            calibration_frame, ["variant", "prediction_type"]
        ),
        "residual_aggregates": residual_aggregates,
        "branch_intervention_calibration_aggregates": public_aggregate(
            pd.DataFrame(intervention_calibration_rows), ["condition"]
        ),
        "branch_intervention_hierarchical_bootstrap": bootstrap,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seeds": BOOTSTRAP_SEEDS,
        "manifest_sha256": sha256(args.manifest),
        "matrix_summary_sha256": sha256(summary_path),
        "both_checkpoint_sha256": {
            str(model_seed): sha256(args.matrix_dir / f"both_seed{model_seed}" / "best.pt")
            for model_seed in MODEL_SEEDS
        },
        "diagnostic_runner_sha256": sha256(Path(__file__)),
        "diagnostic_spec": spec_path.name,
        "diagnostic_spec_sha256": sha256(spec_path),
        "output_files": [
            path.name for path in (
                calibration_path, calibration_class_path, residual_path,
                residual_group_path, intervention_calibration_path,
                intervention_class_calibration_path, intervention_effect_path,
                intervention_effect_class_path,
            )
        ] + [
            f"both_seed{model_seed}_branch_interventions.npz"
            for model_seed in MODEL_SEEDS
        ],
    }
    output_path = args.output_dir / "canonical_phase1_diagnostic_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Summary: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
