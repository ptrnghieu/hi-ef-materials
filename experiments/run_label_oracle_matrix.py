#!/usr/bin/env python3
"""Train and audit the validation-only Hi-EF context-plus-A-label oracle matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_baselines import (
    EMOTIONS,
    EMOTION_TO_ID,
    ClipEncoder,
    HiEFFrozenDataset,
    collate_batch,
    evaluate,
    public_metrics,
    save_predictions,
    seed_everything,
    seed_worker,
    to_device,
)


CONTROL_CONDITIONS = ("global_label_shuffle", "wrong_emotion_permutation")


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
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024]
    )
    parser.add_argument("--replacement-seed-start", type=int, default=1701)
    parser.add_argument("--replacement-replicates", type=int, default=20)
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


class LabelOracleDataset(HiEFFrozenDataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        features_dir: Path,
        label_overrides: dict[str, int] | None = None,
    ) -> None:
        super().__init__(rows, features_dir, num_clips=2)
        self.label_overrides = label_overrides or {}

    def __getitem__(self, index: int) -> dict[str, object]:
        item = super().__getitem__(index)
        row = self.rows.iloc[index]
        sample_id = str(item["sample_id"])
        default_label = EMOTION_TO_ID[str(row["clip3_emotion"])]
        item["party_a_emotion"] = self.label_overrides.get(sample_id, default_label)
        return item


def collate_label_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    result = collate_batch(batch)
    result["party_a_emotion"] = torch.tensor(
        [item["party_a_emotion"] for item in batch], dtype=torch.long
    )
    return result


def build_label_loader(
    rows: pd.DataFrame,
    features_dir: Path,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    label_overrides: dict[str, int] | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        LabelOracleDataset(rows, features_dir, label_overrides),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=collate_label_batch,
        worker_init_fn=seed_worker,
        generator=generator,
    )


class OracleLabelBaseline(nn.Module):
    """Forecast B from context clips I--II plus the oracle emotion label of A."""

    def __init__(
        self,
        d_model: int,
        temporal_layers: int,
        inter_layers: int,
        dropout: float,
        face_pooling: str,
    ) -> None:
        super().__init__()
        self.clip_encoder = ClipEncoder(d_model, temporal_layers, dropout, face_pooling)
        self.emotion_embedding = nn.Embedding(len(EMOTIONS), d_model)
        self.token_position = nn.Parameter(torch.randn(1, 3, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.inter_encoder = nn.TransformerEncoder(
            layer, num_layers=inter_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, len(EMOTIONS)),
        )

    def forward(self, batch: dict[str, object]) -> torch.Tensor:
        context = [self.clip_encoder(batch[f"clip{position}"]) for position in (1, 2)]
        emotion = self.emotion_embedding(batch["party_a_emotion"])
        sequence = torch.stack([*context, emotion], dim=1) + self.token_position
        return self.head(self.inter_encoder(sequence).mean(dim=1))


def model_from_config(config: dict[str, object], device: torch.device) -> nn.Module:
    return OracleLabelBaseline(
        d_model=int(config["d_model"]),
        temporal_layers=int(config["temporal_layers"]),
        inter_layers=int(config["inter_layers"]),
        dropout=float(config["dropout"]),
        face_pooling=str(config["face_pooling"]),
    ).to(device)


def training_config(args: argparse.Namespace, seed: int, run_dir: Path) -> dict[str, object]:
    return {
        "manifest": str(args.manifest),
        "features_dir": str(args.features_dir),
        "output_dir": str(run_dir),
        "model": "oracle_label",
        "input": "clips1-2_plus_clip3_emotion_label",
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


def complete_existing_run(run_dir: Path, expected: dict[str, object]) -> bool:
    required = [
        run_dir / "best.pt",
        run_dir / "config.json",
        run_dir / "history.csv",
        run_dir / "metrics.json",
        run_dir / "val_predictions.npz",
    ]
    if not all(path.exists() for path in required):
        return False
    config = json.loads((run_dir / "config.json").read_text())
    metrics = json.loads((run_dir / "metrics.json").read_text())
    if metrics.get("test") is not None:
        raise RuntimeError(f"Test metrics unexpectedly present in {run_dir}")
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Existing label run has incompatible config: {mismatches}")
    return True


def train_one(
    args: argparse.Namespace,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    seed: int,
    run_dir: Path,
    device: torch.device,
) -> None:
    expected = training_config(args, seed, run_dir)
    if complete_existing_run(run_dir, expected):
        print(f"Validated existing oracle-label run: seed={seed}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    train_loader = build_label_loader(
        train_rows, args.features_dir, args.batch_size, args.workers, True, seed
    )
    val_loader = build_label_loader(
        val_rows, args.features_dir, args.batch_size, args.workers, False, seed + 1
    )
    model = model_from_config(expected, device)
    expected["device"] = str(device)
    expected["num_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    (run_dir / "config.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(json.dumps(expected, indent=2), flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_uar, best_loss, stale_epochs = -math.inf, math.inf, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, seen = 0.0, 0
        for batch in train_loader:
            device_batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(device_batch)
                loss = criterion(logits, device_batch["target"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            current_batch = len(device_batch["target"])
            train_loss += float(loss.detach()) * current_batch
            seen += current_batch

        validation = evaluate(model, val_loader, device, criterion)
        scheduler.step(validation["loss"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss / seen,
            "val_loss": validation["loss"],
            "val_war": validation["war"],
            "val_uar": validation["uar"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps({"seed": seed, **row}), flush=True)
        improved = validation["uar"] > best_uar or (
            math.isclose(validation["uar"], best_uar) and validation["loss"] < best_loss
        )
        if improved:
            best_uar, best_loss, stale_epochs = validation["uar"], validation["loss"], 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_uar": best_uar,
                    "val_loss": best_loss,
                    "config": expected,
                },
                run_dir / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping seed {seed} after epoch {epoch}", flush=True)
                break

    with (run_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation = evaluate(model, val_loader, device, criterion)
    save_predictions(run_dir / "val_predictions.npz", validation)
    metrics = {
        "best_epoch": checkpoint["epoch"],
        "validation": public_metrics(validation),
        "test": None,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )


def runtime_preflight(
    args: argparse.Namespace, train_rows: pd.DataFrame, device: torch.device
) -> None:
    """Fail quickly before launching the five full training runs."""
    seed_everything(args.seeds[0])
    loader = build_label_loader(
        train_rows.head(2), args.features_dir, 2, 0, False, args.seeds[0]
    )
    batch = to_device(next(iter(loader)), device)
    config = training_config(args, args.seeds[0], Path("<preflight>"))
    model = model_from_config(config, device)
    logits = model(batch)
    loss = nn.CrossEntropyLoss()(logits, batch["target"])
    if logits.shape != (2, len(EMOTIONS)) or not torch.isfinite(loss):
        raise RuntimeError(
            f"Runtime preflight failed: logits={tuple(logits.shape)}, loss={loss}"
        )
    loss.backward()
    print(
        json.dumps(
            {
                "runtime_preflight": "passed",
                "batch_size": 2,
                "logits_shape": list(logits.shape),
                "device": str(device),
            }
        ),
        flush=True,
    )
    del model, batch, logits, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()


def global_label_shuffle(rows: pd.DataFrame, rng: random.Random) -> np.ndarray:
    indices = list(range(len(rows)))
    for _ in range(10000):
        permutation = indices.copy()
        rng.shuffle(permutation)
        if all(index != replacement for index, replacement in enumerate(permutation)):
            return rows.iloc[permutation]["clip3_emotion"].map(EMOTION_TO_ID).to_numpy()
    raise RuntimeError("Could not generate a sample-level label derangement")


def wrong_emotion_permutation(rows: pd.DataFrame, rng: random.Random) -> np.ndarray:
    labels = rows["clip3_emotion"].map(EMOTION_TO_ID).to_numpy()
    class_order = list(range(len(EMOTIONS)))
    rng.shuffle(class_order)
    ordered_indices = []
    for class_id in class_order:
        indices = np.flatnonzero(labels == class_id).tolist()
        rng.shuffle(indices)
        ordered_indices.extend(indices)
    ordered_labels = labels[ordered_indices]
    max_count = int(np.bincount(labels, minlength=len(EMOTIONS)).max())
    shifted = np.roll(ordered_labels, max_count)
    replacements = np.empty_like(labels)
    replacements[np.asarray(ordered_indices)] = shifted
    if np.any(replacements == labels):
        raise RuntimeError("Wrong-emotion permutation retained at least one true label")
    if not np.array_equal(np.bincount(replacements), np.bincount(labels)):
        raise RuntimeError("Wrong-emotion permutation changed the label distribution")
    return replacements


def build_label_manifests(
    rows: pd.DataFrame, output_dir: Path, seed_start: int, replicates: int
) -> tuple[dict[tuple[str, int], dict[str, int]], list[dict[str, object]]]:
    manifests = {}
    audit = []
    true_labels = rows["clip3_emotion"].map(EMOTION_TO_ID).to_numpy()
    for condition in CONTROL_CONDITIONS:
        for replicate in range(replicates):
            replacement_seed = seed_start + replicate
            rng = random.Random(replacement_seed)
            if condition == "global_label_shuffle":
                replacements = global_label_shuffle(rows, rng)
            else:
                replacements = wrong_emotion_permutation(rows, rng)
            frame = pd.DataFrame(
                {
                    "sample_id": rows["sample_id"],
                    "condition": condition,
                    "replacement_seed": replacement_seed,
                    "true_emotion": [EMOTIONS[value] for value in true_labels],
                    "replacement_emotion": [EMOTIONS[value] for value in replacements],
                    "true_emotion_id": true_labels,
                    "replacement_emotion_id": replacements,
                }
            )
            path = output_dir / f"{condition}_seed{replacement_seed}.csv"
            frame.to_csv(path, index=False, lineterminator="\n")
            manifests[(condition, replacement_seed)] = dict(
                zip(frame["sample_id"], frame["replacement_emotion_id"], strict=True)
            )
            audit.append(
                {
                    "condition": condition,
                    "replacement_seed": replacement_seed,
                    "same_emotion_rate": float(
                        (frame["true_emotion_id"] == frame["replacement_emotion_id"]).mean()
                    ),
                    "label_counts_preserved": bool(
                        np.array_equal(
                            np.bincount(frame["true_emotion_id"], minlength=len(EMOTIONS)),
                            np.bincount(
                                frame["replacement_emotion_id"], minlength=len(EMOTIONS)
                            ),
                        )
                    ),
                    "sha256": sha256(path),
                }
            )
    return manifests, audit


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    predictions_dir = args.output_dir / "control_predictions"
    manifests_dir = args.output_dir / "label_manifests"
    for path in (runs_dir, predictions_dir, manifests_dir):
        path.mkdir(exist_ok=True)
    baseline = json.loads(args.baseline_summary.read_text())
    if baseline["test_evaluated"] is not False or baseline["seeds"] != args.seeds:
        raise RuntimeError("Baseline summary is not the frozen validation-only five-seed run")
    if baseline["manifest_sha256"] != sha256(args.manifest):
        raise RuntimeError("Baseline and oracle-label manifests differ")
    trainer = Path(__file__).with_name("train_baselines.py")
    if baseline["trainer_sha256"] != sha256(trainer):
        raise RuntimeError("Shared encoder code differs from the frozen baseline code")

    manifest = pd.read_csv(args.manifest, dtype={"source_folder": str})
    split_rows = {
        split: manifest[manifest["split"] == split].reset_index(drop=True)
        for split in ("train", "val")
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime_preflight(args, split_rows["train"], device)
    for seed in args.seeds:
        train_one(
            args, split_rows["train"], split_rows["val"], seed,
            runs_dir / f"oracle_label_seed{seed}", device,
        )

    manifests, manifest_audit = build_label_manifests(
        split_rows["val"], manifests_dir,
        args.replacement_seed_start, args.replacement_replicates,
    )
    criterion = nn.CrossEntropyLoss()
    records = []
    for seed in args.seeds:
        run_dir = runs_dir / f"oracle_label_seed{seed}"
        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
        model = model_from_config(checkpoint["config"], device)
        model.load_state_dict(checkpoint["model_state"])
        true_metrics = json.loads((run_dir / "metrics.json").read_text())
        if true_metrics["test"] is not None:
            raise RuntimeError("Oracle-label run unexpectedly contains test metrics")
        true_uar = float(true_metrics["validation"]["uar"])
        records.append(
            {
                "model_seed": seed,
                "condition": "true_label",
                "replacement_seed": None,
                "uar": true_uar,
                "war": true_metrics["validation"]["war"],
                "loss": true_metrics["validation"]["loss"],
                "delta_uar_from_true": 0.0,
            }
        )
        for condition in CONTROL_CONDITIONS:
            for replicate in range(args.replacement_replicates):
                replacement_seed = args.replacement_seed_start + replicate
                loader = build_label_loader(
                    split_rows["val"], args.features_dir, args.batch_size,
                    args.workers, False, seed + replacement_seed,
                    manifests[(condition, replacement_seed)],
                )
                result = evaluate(model, loader, device, criterion)
                save_predictions(
                    predictions_dir
                    / f"oracle_label_seed{seed}_{condition}_seed{replacement_seed}.npz",
                    result,
                )
                metrics = public_metrics(result)
                records.append(
                    {
                        "model_seed": seed,
                        "condition": condition,
                        "replacement_seed": replacement_seed,
                        "uar": metrics["uar"],
                        "war": metrics["war"],
                        "loss": metrics["loss"],
                        "delta_uar_from_true": true_uar - metrics["uar"],
                    }
                )

    results = pd.DataFrame(records)
    results_path = args.output_dir / "label_oracle_results.csv"
    results.to_csv(results_path, index=False, lineterminator="\n")
    combined_rows = []
    for seed in args.seeds:
        baseline_seed = baseline["runs"][str(seed)]
        label_metrics = json.loads(
            (runs_dir / f"oracle_label_seed{seed}" / "metrics.json").read_text()
        )["validation"]
        row = {
            "seed": seed,
            "context_uar": baseline_seed["context"]["validation"]["uar"],
            "full_raw_a_uar": baseline_seed["full"]["validation"]["uar"],
            "oracle_label_uar": label_metrics["uar"],
        }
        row["raw_minus_context"] = row["full_raw_a_uar"] - row["context_uar"]
        row["label_minus_context"] = row["oracle_label_uar"] - row["context_uar"]
        row["label_minus_raw"] = row["oracle_label_uar"] - row["full_raw_a_uar"]
        combined_rows.append(row)
    combined = pd.DataFrame(combined_rows)
    combined_path = args.output_dir / "context_raw_label_comparison.csv"
    combined.to_csv(combined_path, index=False, lineterminator="\n")

    controls = {}
    for condition in CONTROL_CONDITIONS:
        current = results[results["condition"] == condition]
        per_seed = current.groupby("model_seed")["delta_uar_from_true"].mean()
        controls[condition] = {
            "mean_control_uar": float(current["uar"].mean()),
            "mean_delta_uar": float(per_seed.mean()),
            "std_delta_uar": float(per_seed.std(ddof=1)),
            "positive_delta_seeds": int((per_seed > 0).sum()),
            "per_seed_mean_delta_uar": {
                str(seed): float(value) for seed, value in per_seed.items()
            },
        }
    summary = {
        "protocol": "context-plus-party-a-oracle-emotion-validation-v1",
        "partition": "val",
        "test_evaluated": False,
        "model_seeds": args.seeds,
        "replacement_seeds": list(
            range(
                args.replacement_seed_start,
                args.replacement_seed_start + args.replacement_replicates,
            )
        ),
        "fixed_config": training_config(args, args.seeds[0], Path("<per-seed>"))
        | {"seed": args.seeds, "output_dir": "<per-seed>"},
        "baseline_summary_sha256": sha256(args.baseline_summary),
        "manifest_sha256": sha256(args.manifest),
        "shared_encoder_sha256": sha256(trainer),
        "label_runner_sha256": sha256(Path(__file__)),
        "num_parameters": json.loads(
            (runs_dir / f"oracle_label_seed{args.seeds[0]}" / "config.json").read_text()
        )["num_parameters"],
        "comparison": {
            column: {
                "mean": float(combined[column].mean()),
                "std": float(combined[column].std(ddof=1)),
                "positive_seeds": int((combined[column] > 0).sum()),
            }
            for column in (
                "context_uar", "full_raw_a_uar", "oracle_label_uar",
                "raw_minus_context", "label_minus_context", "label_minus_raw",
            )
        },
        "label_controls": controls,
        "manifest_audit": manifest_audit,
    }
    summary_path = args.output_dir / "label_oracle_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(
        {"comparison": summary["comparison"], "label_controls": controls, "test_evaluated": False},
        indent=2,
    ))
    print(f"Summary:    {summary_path}")
    print(f"Comparison: {combined_path}")
    print(f"Controls:   {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
