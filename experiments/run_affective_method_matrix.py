#!/usr/bin/env python3
"""Run the frozen validation-only Hi-EF affective method matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_baselines import (
    EMOTIONS,
    EMOTION_TO_ID,
    ClipEncoder,
    HiEFFrozenDataset,
    collate_batch,
    metrics_from_logits,
    seed_everything,
    seed_worker,
    to_device,
)


VARIANTS = ("full_aux", "affective_bottleneck", "hybrid_residual")


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
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
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
    parser.add_argument("--affect-weight", type=float, default=0.5)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--orthogonality-weight", type=float, default=0.05)
    return parser.parse_args()


class AffectiveDataset(HiEFFrozenDataset):
    def __init__(self, rows: pd.DataFrame, features_dir: Path) -> None:
        super().__init__(rows, features_dir, num_clips=3)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = super().__getitem__(index)
        item["party_a_target"] = EMOTION_TO_ID[
            str(self.rows.iloc[index]["clip3_emotion"])
        ]
        return item


def collate_affective_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    result = collate_batch(batch)
    result["party_a_target"] = torch.tensor(
        [item["party_a_target"] for item in batch], dtype=torch.long
    )
    return result


def build_loader(
    rows: pd.DataFrame,
    features_dir: Path,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        AffectiveDataset(rows, features_dir),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=collate_affective_batch,
        worker_init_fn=seed_worker,
        generator=generator,
    )


class AffectiveForecastModel(nn.Module):
    def __init__(
        self,
        variant: str,
        d_model: int,
        temporal_layers: int,
        inter_layers: int,
        dropout: float,
        face_pooling: str,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant
        self.clip_encoder = ClipEncoder(d_model, temporal_layers, dropout, face_pooling)
        self.affect_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, len(EMOTIONS)),
        )
        self.emotion_embedding = (
            nn.Parameter(torch.randn(len(EMOTIONS), d_model) * 0.02)
            if variant != "full_aux" else None
        )
        if variant == "hybrid_residual":
            self.context_predictor = nn.Linear(d_model, d_model)
            self.residual_projection = nn.Linear(d_model, d_model)
            self.residual_norm = nn.LayerNorm(d_model)
        else:
            self.context_predictor = None
            self.residual_projection = None
            self.residual_norm = None
        num_tokens = 4 if variant == "hybrid_residual" else 3
        self.token_position = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)
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
        self.future_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, len(EMOTIONS)),
        )

    def forward(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        h1, h2, h_a = [
            self.clip_encoder(batch[f"clip{position}"]) for position in (1, 2, 3)
        ]
        affect_logits = self.affect_head(h_a)
        affect_probabilities = affect_logits.softmax(dim=-1)
        affect_token = (
            affect_probabilities @ self.emotion_embedding
            if self.emotion_embedding is not None else None
        )
        zero = h_a.new_zeros(())
        reconstruction_loss = zero
        orthogonality_loss = zero
        if self.variant == "full_aux":
            tokens = [h1, h2, h_a]
        elif self.variant == "affective_bottleneck":
            assert affect_token is not None
            tokens = [h1, h2, affect_token]
        else:
            assert affect_token is not None
            assert self.context_predictor is not None
            assert self.residual_projection is not None
            assert self.residual_norm is not None
            context = (h1 + h2) / 2.0
            predicted_a = self.context_predictor(context)
            residual = self.residual_norm(
                self.residual_projection(h_a) - predicted_a
            )
            reconstruction_loss = F.mse_loss(predicted_a, h_a.detach())
            orthogonality_loss = F.cosine_similarity(
                residual, context.detach(), dim=-1
            ).square().mean()
            tokens = [h1, h2, affect_token, residual]
        sequence = torch.stack(tokens, dim=1) + self.token_position
        future_logits = self.future_head(self.inter_encoder(sequence).mean(dim=1))
        return {
            "future_logits": future_logits,
            "affect_logits": affect_logits,
            "reconstruction_loss": reconstruction_loss,
            "orthogonality_loss": orthogonality_loss,
        }


def model_from_config(config: dict[str, object], device: torch.device) -> nn.Module:
    return AffectiveForecastModel(
        variant=str(config["variant"]),
        d_model=int(config["d_model"]),
        temporal_layers=int(config["temporal_layers"]),
        inter_layers=int(config["inter_layers"]),
        dropout=float(config["dropout"]),
        face_pooling=str(config["face_pooling"]),
    ).to(device)


def balanced_affect_weights(train_rows: pd.DataFrame) -> np.ndarray:
    counts = (
        train_rows["clip3_emotion"]
        .value_counts()
        .reindex(EMOTIONS, fill_value=0)
        .to_numpy(dtype=float)
    )
    if np.any(counts == 0):
        raise RuntimeError(f"Training partition lacks an A-emotion class: {counts}")
    weights = len(train_rows) / (len(EMOTIONS) * counts)
    return weights / weights.mean()


def loss_terms(
    output: dict[str, torch.Tensor],
    batch: dict[str, object],
    future_criterion: nn.Module,
    affect_criterion: nn.Module,
    affect_weight: float,
    reconstruction_weight: float,
    orthogonality_weight: float,
) -> dict[str, torch.Tensor]:
    future = future_criterion(output["future_logits"], batch["target"])
    affect = affect_criterion(output["affect_logits"], batch["party_a_target"])
    total = future + affect_weight * affect
    total = total + reconstruction_weight * output["reconstruction_loss"]
    total = total + orthogonality_weight * output["orthogonality_loss"]
    return {
        "total": total,
        "future": future,
        "affect": affect,
        "reconstruction": output["reconstruction_loss"],
        "orthogonality": output["orthogonality_loss"],
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    future_logits, affect_logits = [], []
    future_labels, affect_labels = [], []
    sample_ids, source_folders = [], []
    future_loss, affect_loss, seen = 0.0, 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        device_batch = to_device(batch, device)
        output = model(device_batch)
        current = len(device_batch["target"])
        future_loss += float(
            criterion(output["future_logits"], device_batch["target"])
        ) * current
        affect_loss += float(
            criterion(output["affect_logits"], device_batch["party_a_target"])
        ) * current
        seen += current
        future_logits.append(output["future_logits"].cpu().numpy())
        affect_logits.append(output["affect_logits"].cpu().numpy())
        future_labels.append(device_batch["target"].cpu().numpy())
        affect_labels.append(device_batch["party_a_target"].cpu().numpy())
        sample_ids.extend(batch["sample_id"])
        source_folders.extend(batch["source_folder"])
    merged_future = np.concatenate(future_logits)
    merged_affect = np.concatenate(affect_logits)
    merged_future_labels = np.concatenate(future_labels)
    merged_affect_labels = np.concatenate(affect_labels)
    result = {
        "future_loss": future_loss / seen,
        "affect_loss": affect_loss / seen,
        "future_logits": merged_future,
        "affect_logits": merged_affect,
        "future_labels": merged_future_labels,
        "affect_labels": merged_affect_labels,
        "sample_ids": np.asarray(sample_ids),
        "source_folders": np.asarray(source_folders),
    }
    result.update(
        {f"future_{key}": value for key, value in metrics_from_logits(
            merged_future, merged_future_labels
        ).items()}
    )
    result.update(
        {f"affect_{key}": value for key, value in metrics_from_logits(
            merged_affect, merged_affect_labels
        ).items()}
    )
    return result


def public_metrics(result: dict[str, object]) -> dict[str, object]:
    hidden = {
        "future_logits", "affect_logits", "future_labels", "affect_labels",
        "sample_ids", "source_folders",
    }
    return {key: value for key, value in result.items() if key not in hidden}


def save_predictions(path: Path, result: dict[str, object]) -> None:
    np.savez_compressed(
        path,
        future_logits=result["future_logits"],
        future_labels=result["future_labels"],
        future_predictions=result["future_logits"].argmax(axis=1),
        affect_logits=result["affect_logits"],
        affect_labels=result["affect_labels"],
        affect_predictions=result["affect_logits"].argmax(axis=1),
        sample_ids=result["sample_ids"],
        source_folders=result["source_folders"],
    )


def run_config(
    args: argparse.Namespace, variant: str, seed: int, run_dir: Path
) -> dict[str, object]:
    return {
        "manifest": str(args.manifest),
        "features_dir": str(args.features_dir),
        "output_dir": str(run_dir),
        "variant": variant,
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
        "face_pooling": args.face_pooling,
        "affect_weight": args.affect_weight,
        "reconstruction_weight": (
            args.reconstruction_weight if variant == "hybrid_residual" else 0.0
        ),
        "orthogonality_weight": (
            args.orthogonality_weight if variant == "hybrid_residual" else 0.0
        ),
        "test_evaluation_requested": False,
    }


def complete_existing_run(run_dir: Path, expected: dict[str, object]) -> bool:
    required = [
        run_dir / "best.pt", run_dir / "config.json", run_dir / "history.csv",
        run_dir / "metrics.json", run_dir / "val_predictions.npz",
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
        raise RuntimeError(f"Existing method run has incompatible config: {mismatches}")
    return True


def train_one(
    args: argparse.Namespace,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    class_weights: torch.Tensor,
    variant: str,
    seed: int,
    run_dir: Path,
    device: torch.device,
) -> None:
    config = run_config(args, variant, seed, run_dir)
    if complete_existing_run(run_dir, config):
        print(f"Validated existing run: variant={variant}, seed={seed}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    train_loader = build_loader(
        train_rows, args.features_dir, args.batch_size, args.workers, True, seed
    )
    val_loader = build_loader(
        val_rows, args.features_dir, args.batch_size, args.workers, False, seed + 1
    )
    model = model_from_config(config, device)
    config["device"] = str(device)
    config["num_parameters"] = sum(p.numel() for p in model.parameters())
    config["affect_class_weights"] = class_weights.cpu().tolist()
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2), flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )
    future_criterion = nn.CrossEntropyLoss()
    affect_criterion = nn.CrossEntropyLoss(weight=class_weights)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_uar, best_loss, stale = -math.inf, math.inf, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {key: 0.0 for key in (
            "total", "future", "affect", "reconstruction", "orthogonality"
        )}
        seen = 0
        for batch in train_loader:
            device_batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                output = model(device_batch)
                losses = loss_terms(
                    output, device_batch, future_criterion, affect_criterion,
                    args.affect_weight,
                    config["reconstruction_weight"],
                    config["orthogonality_weight"],
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            current = len(device_batch["target"])
            for key in sums:
                sums[key] += float(losses[key].detach()) * current
            seen += current
        validation = evaluate(model, val_loader, device)
        scheduler.step(validation["future_loss"])
        row = {
            "epoch": epoch,
            **{f"train_{key}_loss": value / seen for key, value in sums.items()},
            "val_future_loss": validation["future_loss"],
            "val_future_war": validation["future_war"],
            "val_future_uar": validation["future_uar"],
            "val_affect_loss": validation["affect_loss"],
            "val_affect_war": validation["affect_war"],
            "val_affect_uar": validation["affect_uar"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps({"variant": variant, "seed": seed, **row}), flush=True)
        improved = validation["future_uar"] > best_uar or (
            math.isclose(validation["future_uar"], best_uar)
            and validation["future_loss"] < best_loss
        )
        if improved:
            best_uar = validation["future_uar"]
            best_loss = validation["future_loss"]
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(), "epoch": epoch,
                    "val_uar": best_uar, "val_loss": best_loss, "config": config,
                },
                run_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(
                    f"Early stopping {variant} seed {seed} after epoch {epoch}",
                    flush=True,
                )
                break
    with (run_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation = evaluate(model, val_loader, device)
    save_predictions(run_dir / "val_predictions.npz", validation)
    metrics = {
        "best_epoch": checkpoint["epoch"],
        "validation": public_metrics(validation),
        "test": None,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    del model, train_loader, val_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()


def runtime_preflight(
    args: argparse.Namespace,
    train_rows: pd.DataFrame,
    class_weights: torch.Tensor,
    device: torch.device,
) -> None:
    seed_everything(args.seeds[0])
    loader = build_loader(train_rows.head(2), args.features_dir, 2, 0, False, args.seeds[0])
    batch = to_device(next(iter(loader)), device)
    for variant in args.variants:
        config = run_config(args, variant, args.seeds[0], Path("<preflight>"))
        model = model_from_config(config, device)
        output = model(batch)
        losses = loss_terms(
            output, batch, nn.CrossEntropyLoss(),
            nn.CrossEntropyLoss(weight=class_weights), args.affect_weight,
            config["reconstruction_weight"], config["orthogonality_weight"],
        )
        if output["future_logits"].shape != (2, len(EMOTIONS)):
            raise RuntimeError(f"Preflight shape failure for {variant}")
        if not torch.isfinite(losses["total"]):
            raise RuntimeError(f"Preflight non-finite loss for {variant}")
        losses["total"].backward()
        print(json.dumps({
            "runtime_preflight": "passed", "variant": variant,
            "future_logits_shape": list(output["future_logits"].shape),
            "device": str(device),
        }), flush=True)
        del model, output, losses
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    baseline = json.loads(args.baseline_summary.read_text())
    if baseline.get("test_evaluated") is not False:
        raise RuntimeError("Baseline summary is not validation-only")
    if [int(seed) for seed in baseline["seeds"]] != args.seeds:
        raise RuntimeError("Baseline and method seeds differ")
    if baseline["manifest_sha256"] != sha256(args.manifest):
        raise RuntimeError("Baseline and method manifests differ")
    trainer = Path(__file__).with_name("train_baselines.py")
    if baseline["trainer_sha256"] != sha256(trainer):
        raise RuntimeError("Frozen encoder code differs from the baseline run")
    manifest = pd.read_csv(args.manifest, dtype={"source_folder": str})
    train_rows = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_rows = manifest[manifest["split"] == "val"].reset_index(drop=True)
    weights_array = balanced_affect_weights(train_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor(weights_array, dtype=torch.float32, device=device)
    runtime_preflight(args, train_rows, class_weights, device)
    for variant in args.variants:
        for seed in args.seeds:
            train_one(
                args, train_rows, val_rows, class_weights, variant, seed,
                runs_dir / f"{variant}_seed{seed}", device,
            )

    rows = []
    for variant in args.variants:
        for seed in args.seeds:
            run_dir = runs_dir / f"{variant}_seed{seed}"
            metrics = json.loads((run_dir / "metrics.json").read_text())
            config = json.loads((run_dir / "config.json").read_text())
            if metrics.get("test") is not None:
                raise RuntimeError(f"Test metrics unexpectedly present in {run_dir}")
            raw = baseline["runs"][str(seed)]["full"]["validation"]
            validation = metrics["validation"]
            rows.append({
                "variant": variant,
                "seed": seed,
                "best_epoch": metrics["best_epoch"],
                "future_uar": validation["future_uar"],
                "future_war": validation["future_war"],
                "future_loss": validation["future_loss"],
                "affect_uar": validation["affect_uar"],
                "affect_war": validation["affect_war"],
                "affect_loss": validation["affect_loss"],
                "raw_uar": raw["uar"],
                "raw_loss": raw["loss"],
                "delta_uar_vs_raw": validation["future_uar"] - raw["uar"],
                "delta_nll_vs_raw": raw["loss"] - validation["future_loss"],
                "num_parameters": config["num_parameters"],
            })
    results = pd.DataFrame(rows)
    results_path = args.output_dir / "affective_method_matrix.csv"
    results.to_csv(results_path, index=False, lineterminator="\n")
    aggregates = {}
    for variant, group in results.groupby("variant", sort=False):
        aggregates[variant] = {
            "future_uar_mean": float(group["future_uar"].mean()),
            "future_uar_std": float(group["future_uar"].std(ddof=1)),
            "future_war_mean": float(group["future_war"].mean()),
            "future_nll_mean": float(group["future_loss"].mean()),
            "affect_uar_mean": float(group["affect_uar"].mean()),
            "affect_uar_std": float(group["affect_uar"].std(ddof=1)),
            "delta_uar_vs_raw_mean": float(group["delta_uar_vs_raw"].mean()),
            "delta_nll_vs_raw_mean": float(group["delta_nll_vs_raw"].mean()),
            "positive_uar_delta_seeds": int((group["delta_uar_vs_raw"] > 0).sum()),
            "num_parameters": int(group["num_parameters"].iloc[0]),
        }
    best_uar = max(item["future_uar_mean"] for item in aggregates.values())
    finalists = [
        variant for variant, item in aggregates.items()
        if best_uar - item["future_uar_mean"] < 0.005
    ]
    selected = min(
        finalists,
        key=lambda variant: (
            aggregates[variant]["future_nll_mean"],
            aggregates[variant]["num_parameters"],
        ),
    )
    uar_gate = (
        aggregates[selected]["delta_uar_vs_raw_mean"] >= 0.01
        and aggregates[selected]["positive_uar_delta_seeds"] >= 4
    )
    summary = {
        "protocol": "affective-method-validation-matrix-v1",
        "partition": "val",
        "test_evaluated": False,
        "variants": args.variants,
        "model_seeds": args.seeds,
        "manifest_sha256": sha256(args.manifest),
        "baseline_summary_sha256": sha256(args.baseline_summary),
        "shared_encoder_sha256": sha256(trainer),
        "method_runner_sha256": sha256(Path(__file__)),
        "fixed_loss_weights": {
            "affect": args.affect_weight,
            "reconstruction": args.reconstruction_weight,
            "orthogonality": args.orthogonality_weight,
        },
        "affect_class_weights": {
            emotion: float(weight) for emotion, weight in zip(EMOTIONS, weights_array)
        },
        "aggregates": aggregates,
        "provisional_selection": {
            "variant": selected,
            "rule": (
                "highest mean validation UAR; within 0.5 points prefer lower "
                "mean validation NLL, then fewer parameters"
            ),
            "uar_advancement_gate_passed": uar_gate,
            "final_advancement_pending_hierarchical_audit": True,
        },
        "inference_note": (
            "This is a validation-only exploratory method matrix. The selected "
            "candidate is provisional until the frozen hierarchical audit."
        ),
    }
    summary_path = args.output_dir / "affective_method_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "aggregates": aggregates,
        "provisional_selection": summary["provisional_selection"],
        "test_evaluated": False,
    }, indent=2))
    print(f"Summary: {summary_path}")
    print(f"Matrix:  {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
