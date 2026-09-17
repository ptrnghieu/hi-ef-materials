#!/usr/bin/env python3
"""Run the frozen, train-only, cross-fitted Party-A attribution audit.

This program intentionally refuses validation/test evaluation.  It fits fixed-
epoch context and direct-fusion baselines inside source-folder folds of the
original train split, then compares true clip III with matched replacements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from run_validation_interventions import (
    REPLACEMENT_CONDITIONS,
    build_intervention_loader,
    clip_pool,
    manifest_frame,
)
from train_baselines import (
    EMOTIONS,
    FrozenFeatureBaseline,
    build_loader,
    evaluate,
    seed_everything,
    to_device,
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8,
                        help="Fixed pre-registered epochs; no early stopping.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--inter-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--face-pooling", choices=("masked", "unmasked"), default="masked")
    parser.add_argument("--replacement-seed-start", type=int, default=7101)
    parser.add_argument("--replacement-replicates", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=7201)
    return parser.parse_args()


def make_folds(rows: pd.DataFrame, num_folds: int) -> dict[int, set[str]]:
    folders = sorted(rows["source_folder"].astype(str).unique())
    if num_folds < 2 or num_folds > len(folders):
        raise ValueError("--inner-folds must be in [2, number of train source folders]")
    folds: dict[int, set[str]] = {index: set() for index in range(num_folds)}
    # Deterministic round-robin, so each fold has several folders for matching.
    for index, folder in enumerate(folders):
        folds[index % num_folds].add(folder)
    if any(len(value) < 2 for value in folds.values()):
        raise ValueError("Each fold needs at least two source folders for matched controls")
    return folds


def fit_fixed_model(
    fit_rows: pd.DataFrame, args: argparse.Namespace, seed: int, num_clips: int,
    device: torch.device,
) -> FrozenFeatureBaseline:
    seed_everything(seed)
    loader = build_loader(
        fit_rows, args.features_dir, num_clips, args.batch_size, args.workers, True, seed
    )
    model = FrozenFeatureBaseline(
        num_clips=num_clips, d_model=args.d_model,
        temporal_layers=args.temporal_layers, inter_layers=args.inter_layers,
        dropout=args.dropout, face_pooling=args.face_pooling,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    for _ in range(args.epochs):
        model.train()
        for batch in loader:
            device_batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = criterion(model(device_batch), device_batch["target"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
    return model


def probabilities_and_nll(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, -np.log(probabilities[np.arange(len(labels)), labels].clip(1e-12))


def per_row_frame(
    result: dict[str, object], rows: pd.DataFrame, model_seed: int, fold: int,
    condition: str, replicate: int | None, context_nll: np.ndarray,
    context_pred: np.ndarray, context_confidence: np.ndarray,
) -> pd.DataFrame:
    sample_order = pd.Series(result["sample_ids"], name="sample_id").astype(str)
    metadata = rows.set_index("sample_id").loc[sample_order].reset_index()
    logits = np.asarray(result["logits"])
    labels = np.asarray(result["labels"], dtype=int)
    probabilities, nll = probabilities_and_nll(logits, labels)
    prediction = logits.argmax(axis=1)
    changed = prediction != context_pred
    return pd.DataFrame({
        "sample_id": sample_order,
        "model_seed": model_seed,
        "inner_fold": fold,
        "condition": condition,
        "replicate_seed": replicate,
        "source_folder": metadata["source_folder"].astype(str).to_numpy(),
        "party_a_emotion": metadata["clip3_emotion"].to_numpy(),
        "target_emotion": metadata["clip4_emotion"].to_numpy(),
        "label": labels,
        "context_prediction": context_pred,
        "prediction": prediction,
        "context_nll": context_nll,
        "nll": nll,
        "utility": context_nll - nll,
        "context_confidence": context_confidence,
        "prediction_flip": changed,
        "beneficial_flip": changed & (context_pred != labels) & (prediction == labels),
        "harmful_flip": changed & (context_pred == labels) & (prediction != labels),
        **{f"prob_{emotion}": probabilities[:, index] for index, emotion in enumerate(EMOTIONS)},
    })


def assign_context_strata(frame: pd.DataFrame) -> pd.DataFrame:
    # The thresholds use only cross-fitted, input-derived context confidence.
    low, high = frame["context_confidence"].quantile([1 / 3, 2 / 3]).to_list()
    frame = frame.copy()
    frame["context_confidence_stratum"] = np.select(
        [frame["context_confidence"] <= low, frame["context_confidence"] <= high],
        ["low", "medium"], default="high",
    )
    return frame


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    recalls = []
    for label in range(len(EMOTIONS)):
        subset = frame[frame["label"] == label]
        if len(subset):
            recalls.append(float((subset["prediction"] == label).mean()))
    return {
        "rows": len(frame), "uar": float(np.mean(recalls)),
        "war": float((frame["prediction"] == frame["label"]).mean()),
        "nll": float(frame["nll"].mean()), "utility": float(frame["utility"].mean()),
        "prediction_flip_rate": float(frame["prediction_flip"].mean()),
        "beneficial_flip_rate": float(frame["beneficial_flip"].mean()),
        "harmful_flip_rate": float(frame["harmful_flip"].mean()),
    }


def stratified_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for grouping in ("target_emotion", "party_a_emotion", "source_folder", "context_confidence_stratum"):
        for (condition, value), subset in frame.groupby(["condition", grouping], dropna=False):
            records.append({"grouping": grouping, "condition": condition, "value": value, **metrics(subset)})
    return pd.DataFrame(records)


def hierarchical_ci(rank: pd.DataFrame, replicates: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    seeds = rank["model_seed"].unique()
    folders = rank["source_folder"].unique()
    values = []
    for _ in range(replicates):
        picked_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        picked_folders = rng.choice(folders, size=len(folders), replace=True)
        pieces = []
        for model_seed in picked_seeds:
            for folder in picked_folders:
                subset = rank[(rank["model_seed"] == model_seed) & (rank["source_folder"] == folder)]
                if len(subset):
                    pieces.append(subset["true_beats_matched"].to_numpy())
        values.append(float(np.concatenate(pieces).mean()))
    return {"mean": float(np.mean(values)), "ci95_low": float(np.quantile(values, .025)),
            "ci95_high": float(np.quantile(values, .975)),
            "probability_above_chance": float(np.mean(np.asarray(values) > .5))}


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_all = pd.read_csv(args.manifest, dtype={"source_folder": str})
    if set(rows_all["split"].unique()) - {"train", "val", "test"}:
        raise ValueError("Unexpected original split label")
    rows = rows_all.loc[rows_all["split"] == "train"].copy().reset_index(drop=True)
    if rows.empty:
        raise ValueError("No original train rows")
    folds = make_folds(rows, args.inner_folds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    predictions: list[pd.DataFrame] = []
    replacement_audit: list[dict[str, object]] = []

    for fold, held_folders in folds.items():
        held = rows[rows["source_folder"].isin(held_folders)].reset_index(drop=True)
        fit = rows[~rows["source_folder"].isin(held_folders)].reset_index(drop=True)
        pool = clip_pool(held)
        for model_seed in args.seeds:
            # Fixed epochs, with no held-out metric read until both fits finish.
            context_model = fit_fixed_model(fit, args, model_seed + 10000 * fold, 2, device)
            full_model = fit_fixed_model(fit, args, model_seed + 10000 * fold + 1, 3, device)
            context_result = evaluate(context_model, build_loader(
                held, args.features_dir, 2, args.batch_size, args.workers, False, model_seed
            ), device, criterion)
            context_logits = np.asarray(context_result["logits"])
            context_probs, context_nll = probabilities_and_nll(context_logits, np.asarray(context_result["labels"], dtype=int))
            context_pred = context_logits.argmax(axis=1)
            context_conf = context_probs.max(axis=1)
            for condition in ("true", "zero"):
                result = evaluate(full_model, build_intervention_loader(
                    held, args.features_dir, condition, None, args.batch_size, args.workers,
                    model_seed + fold,
                ), device, criterion)
                predictions.append(per_row_frame(result, held, model_seed, fold, condition, None,
                                                 context_nll, context_pred, context_conf))
            for condition in REPLACEMENT_CONDITIONS:
                for offset in range(args.replacement_replicates):
                    replacement_seed = args.replacement_seed_start + offset
                    replacement = manifest_frame(held, pool, condition, replacement_seed)
                    result = evaluate(full_model, build_intervention_loader(
                        held, args.features_dir, condition,
                        dict(zip(replacement["sample_id"], replacement["replacement_clip3"], strict=True)),
                        args.batch_size, args.workers, model_seed + replacement_seed,
                    ), device, criterion)
                    predictions.append(per_row_frame(result, held, model_seed, fold, condition,
                                                     replacement_seed, context_nll, context_pred,
                                                     context_conf))
                    replacement_audit.append({
                        "inner_fold": fold, "condition": condition,
                        "replicate_seed": replacement_seed, "rows": len(replacement),
                        "sha256": hashlib.sha256(replacement.to_csv(index=False).encode()).hexdigest(),
                    })

    frame = assign_context_strata(pd.concat(predictions, ignore_index=True))
    frame.to_csv(args.output_dir / "crossfit_candidate_predictions.csv", index=False, lineterminator="\n")
    overall = pd.DataFrame([{"condition": condition, **metrics(part)}
                            for condition, part in frame.groupby("condition")])
    overall.to_csv(args.output_dir / "overall_metrics.csv", index=False, lineterminator="\n")
    stratified_metrics(frame).to_csv(args.output_dir / "stratified_metrics.csv", index=False, lineterminator="\n")

    true_rows = frame[frame["condition"] == "true"].set_index(["model_seed", "inner_fold", "sample_id"])
    matched = frame[frame["condition"].isin(REPLACEMENT_CONDITIONS)].copy()
    matched_mean = matched.groupby(["model_seed", "inner_fold", "sample_id"], as_index=True)["utility"].mean()
    rank = true_rows[["source_folder", "party_a_emotion", "target_emotion", "context_confidence_stratum", "utility"]].join(
        matched_mean.rename("matched_utility"), how="inner"
    ).reset_index()
    rank["true_beats_matched"] = np.where(rank["utility"] > rank["matched_utility"], 1.0,
                                            np.where(rank["utility"] < rank["matched_utility"], 0.0, .5))
    rank.to_csv(args.output_dir / "true_vs_matched_ranking.csv", index=False, lineterminator="\n")
    ci = hierarchical_ci(rank, args.bootstrap_replicates, args.bootstrap_seed)
    ranking_strata = []
    for grouping in ("target_emotion", "party_a_emotion", "source_folder", "context_confidence_stratum"):
        for value, subset in rank.groupby(grouping):
            ranking_strata.append({"grouping": grouping, "value": value, "rows": len(subset),
                                   "true_beats_matched_rate": float(subset["true_beats_matched"].mean())})
    pd.DataFrame(ranking_strata).to_csv(args.output_dir / "ranking_strata.csv", index=False, lineterminator="\n")

    summary = {
        "protocol": "train-only-conditional-attribution-audit-v1",
        "research_spec": "RESEARCH_SPEC_v1.0.md",
        "manifest_sha256": sha256(args.manifest),
        "original_partition_read": "train", "validation_evaluated": False, "test_evaluated": False,
        "model_seeds": args.seeds, "inner_folds": {str(key): sorted(value) for key, value in folds.items()},
        "fixed_epochs": args.epochs, "replacement_replicates": args.replacement_replicates,
        "replacement_manifest_audit": replacement_audit,
        "actual_A_vs_matched_ranking": ci,
        "h3_advancement_gate": {
            "rule": "hierarchical CI lower bound for actual-A ranking advantage over chance (.5) > 0",
            "passed": ci["ci95_low"] > .5,
        },
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
