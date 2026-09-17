#!/usr/bin/env python3
"""Run frozen cross-fitted calibration, shrinkage, and oracle diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_canonical_residual_matrix import (
    MODEL_SEEDS,
    hierarchical_pair_bootstrap,
    load_matrix_predictions,
)
from canonical_diagnostic_metrics import calibration_metrics
from canonical_reliability_diagnostics import cross_fit, oracle_logits


CONDITIONS = (
    "context", "final", "temperature", "shrinkage", "oracle_nll", "oracle_accuracy"
)
COMPARISONS = (
    ("temperature_vs_final", "temperature", "final"),
    ("shrinkage_vs_final", "shrinkage", "final"),
    ("shrinkage_vs_context", "shrinkage", "context"),
    ("oracle_nll_vs_final", "oracle_nll", "final"),
    ("oracle_nll_vs_shrinkage", "oracle_nll", "shrinkage"),
    ("oracle_accuracy_vs_final", "oracle_accuracy", "final"),
    ("oracle_accuracy_vs_shrinkage", "oracle_accuracy", "shrinkage"),
)
BOOTSTRAP_SEEDS = {
    name: 7901 + index for index, (name, _, _) in enumerate(COMPARISONS)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    return parser.parse_args()


def aggregate(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    numeric = [
        column for column in frame.select_dtypes(include=[np.number]).columns
        if column != "model_seed"
    ]
    for condition, group in frame.groupby("condition", sort=False):
        row: dict[str, object] = {"condition": condition, "model_seeds": len(group)}
        for column in numeric:
            values = group[column].to_numpy(float)
            row[f"{column}_mean"] = float(np.nanmean(values))
            row[f"{column}_std"] = float(np.nanstd(values, ddof=1))
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_summary_path = args.matrix_dir / "canonical_residual_matrix_summary.json"
    matrix_summary = json.loads(matrix_summary_path.read_text())
    if matrix_summary.get("protocol") != "canonical-contextual-affective-residual-validation-matrix-v1":
        raise RuntimeError("Unexpected matrix protocol")
    if matrix_summary.get("test_evaluated") is not False:
        raise RuntimeError("Matrix input is not validation-only")
    if matrix_summary.get("model_seeds") != list(MODEL_SEEDS):
        raise RuntimeError("Unexpected model seeds")

    matrix, reference = load_matrix_predictions(args.matrix_dir)
    labels = reference["labels"].astype(int)
    folders = reference["source_folders"].astype(str)
    sample_ids = reference["sample_ids"].astype(str)

    logits_by_seed: dict[int, dict[str, np.ndarray]] = {}
    parameter_rows: list[dict[str, object]] = []
    choice_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []

    for model_seed in MODEL_SEEDS:
        values = matrix[model_seed]["both"]
        context = values["context_logits"]
        delta = values["delta_logits"]
        final = values["final_logits"]
        np.testing.assert_allclose(final, context + delta, rtol=1e-6, atol=1e-6)

        cross_fitted, fold_rows = cross_fit(context, delta, final, labels, folders)
        oracles, choices = oracle_logits(context, final, labels)
        conditions = {
            "context": context,
            "final": final,
            **cross_fitted,
            **oracles,
        }
        logits_by_seed[model_seed] = conditions
        for row in fold_rows:
            parameter_rows.append({"model_seed": model_seed, **row})
        choice_rows.append({
            "model_seed": model_seed,
            "oracle_nll_choose_final_rate": float(
                choices["oracle_nll_choose_final"].mean()
            ),
            "oracle_accuracy_choose_final_rate": float(
                choices["oracle_accuracy_choose_final"].mean()
            ),
            "context_only_correct_rate": float(np.mean(
                choices["context_correct"] & ~choices["final_correct"]
            )),
            "final_only_correct_rate": float(np.mean(
                ~choices["context_correct"] & choices["final_correct"]
            )),
            "both_correct_rate": float(np.mean(
                choices["context_correct"] & choices["final_correct"]
            )),
            "both_wrong_rate": float(np.mean(
                ~choices["context_correct"] & ~choices["final_correct"]
            )),
        })
        for condition in CONDITIONS:
            metrics, per_class = calibration_metrics(conditions[condition], labels)
            metric_rows.append({
                "model_seed": model_seed,
                "condition": condition,
                **metrics,
            })
            for row in per_class:
                class_rows.append({
                    "model_seed": model_seed,
                    "condition": condition,
                    **row,
                })
        np.savez_compressed(
            args.output_dir / f"both_seed{model_seed}_reliability_predictions.npz",
            labels=labels,
            sample_ids=sample_ids,
            source_folders=folders,
            **conditions,
        )

    bootstrap = {}
    for name, better_name, reference_name in COMPARISONS:
        better = {
            seed: logits_by_seed[seed][better_name] for seed in MODEL_SEEDS
        }
        baseline = {
            seed: logits_by_seed[seed][reference_name] for seed in MODEL_SEEDS
        }
        bootstrap[name] = hierarchical_pair_bootstrap(
            name,
            better,
            baseline,
            labels,
            folders,
            args.bootstrap_replicates,
            BOOTSTRAP_SEEDS[name],
        )

    parameters = pd.DataFrame(parameter_rows)
    choices = pd.DataFrame(choice_rows)
    metrics = pd.DataFrame(metric_rows)
    per_class = pd.DataFrame(class_rows)
    parameters_path = args.output_dir / "reliability_crossfit_parameters.csv"
    choices_path = args.output_dir / "reliability_oracle_choice_rates.csv"
    metrics_path = args.output_dir / "reliability_metrics_by_seed.csv"
    classes_path = args.output_dir / "reliability_metrics_per_class.csv"
    parameters.to_csv(parameters_path, index=False, lineterminator="\n")
    choices.to_csv(choices_path, index=False, lineterminator="\n")
    metrics.to_csv(metrics_path, index=False, lineterminator="\n")
    per_class.to_csv(classes_path, index=False, lineterminator="\n")

    alpha_vs_final = bootstrap["shrinkage_vs_final"]
    oracle_vs_alpha = bootstrap["oracle_accuracy_vs_shrinkage"]
    indicators = {
        "temperature_nll_improvement_ci_above_zero": (
            bootstrap["temperature_vs_final"]["nll_delta"]["ci95_low"] > 0
        ),
        "shrinkage_nll_improvement_ci_above_zero": (
            alpha_vs_final["nll_delta"]["ci95_low"] > 0
        ),
        "shrinkage_uar_noninferior_margin_minus_0_005": (
            alpha_vs_final["uar_delta"]["ci95_low"] > -0.005
        ),
        "accuracy_oracle_uar_headroom_beyond_shrinkage": (
            oracle_vs_alpha["uar_delta"]["ci95_low"] > 0
        ),
        "nll_oracle_headroom_beyond_shrinkage": (
            bootstrap["oracle_nll_vs_shrinkage"]["nll_delta"]["ci95_low"] > 0
        ),
    }

    spec_path = Path(__file__).with_name("RESEARCH_SPEC_v0.7.md")
    summary = {
        "protocol": "canonical-residual-reliability-mechanism-diagnostic-v1",
        "diagnostic_only": True,
        "research_interpretation_permitted": False,
        "model_selection_permitted": False,
        "partitions_touched": ["validation"],
        "test_evaluated": False,
        "model_seeds": list(MODEL_SEEDS),
        "num_samples": int(len(labels)),
        "num_source_folders": int(len(np.unique(folders))),
        "crossfit_unit": "source_folder",
        "conditions": list(CONDITIONS),
        "comparisons": [name for name, _, _ in COMPARISONS],
        "metric_aggregates": aggregate(metrics),
        "hierarchical_bootstrap": bootstrap,
        "decision_indicators": indicators,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seeds": BOOTSTRAP_SEEDS,
        "matrix_summary_sha256": sha256(matrix_summary_path),
        "diagnostic_spec": spec_path.name,
        "diagnostic_spec_sha256": sha256(spec_path),
        "diagnostic_runner_sha256": sha256(Path(__file__)),
        "inference_note": (
            "Cross-fitted validation mechanism diagnostic only. Bootstrap uses "
            "fixed cross-fitted predictions and does not refit fold parameters."
        ),
        "output_files": [
            parameters_path.name,
            choices_path.name,
            metrics_path.name,
            classes_path.name,
        ] + [
            f"both_seed{seed}_reliability_predictions.npz"
            for seed in MODEL_SEEDS
        ],
    }
    summary_path = args.output_dir / "canonical_reliability_diagnostic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
