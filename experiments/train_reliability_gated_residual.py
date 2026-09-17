#!/usr/bin/env python3
"""Train the frozen v0.8 reliability-gated residual inner-development model."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_baselines import EMOTIONS, metrics_from_logits, seed_everything, to_device
from train_contextual_affective_residual import (
    ContextualAffectiveResidual,
    balanced_affect_weights,
    build_loader,
    compute_losses,
    sha256,
)


VARIANTS = ("ungated", "gate", "counterfactual", "gate_counterfactual")


def uses_gate(variant: str) -> bool:
    return variant in {"gate", "gate_counterfactual"}


def uses_counterfactual(variant: str) -> bool:
    return variant in {"counterfactual", "gate_counterfactual"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--context-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--face-pooling", choices=("masked",), default="masked")
    parser.add_argument("--context-weight", type=float, required=True)
    parser.add_argument("--emotion-weight", type=float, required=True)
    parser.add_argument("--contrastive-weight", type=float, required=True)
    parser.add_argument("--null-weight", type=float, required=True)
    parser.add_argument("--nuisance-weight", type=float, required=True)
    parser.add_argument("--counterfactual-weight", type=float, required=True)
    parser.add_argument("--invalid-gate-weight", type=float, required=True)
    parser.add_argument("--contrastive-temperature", type=float, required=True)
    parser.add_argument("--null-divergence", choices=("context-to-null",), required=True)
    parser.add_argument("--gradient-reversal-scale", type=float, default=1.0)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    return parser.parse_args()


class ReliabilityGatedResidual(ContextualAffectiveResidual):
    """Canonical `both` residual with an optional scalar sample gate."""

    def __init__(
        self,
        variant: str,
        num_source_folders: int,
        d_model: int,
        temporal_layers: int,
        context_layers: int,
        dropout: float,
        face_pooling: str,
        gradient_reversal_scale: float = 1.0,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown reliability variant: {variant}")
        super().__init__(
            variant="both",
            num_source_folders=num_source_folders,
            d_model=d_model,
            temporal_layers=temporal_layers,
            context_layers=context_layers,
            dropout=dropout,
            face_pooling=face_pooling,
            gradient_reversal_scale=gradient_reversal_scale,
        )
        self.variant = variant
        self.gated = uses_gate(variant)
        self.counterfactual_training = uses_counterfactual(variant)
        self.reliability_head = (
            nn.Sequential(
                nn.LayerNorm(3 * d_model),
                nn.Linear(3 * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            if self.gated else None
        )
        if self.reliability_head is not None:
            nn.init.zeros_(self.reliability_head[-1].weight)
            nn.init.zeros_(self.reliability_head[-1].bias)

    def reliability(
        self,
        context: torch.Tensor,
        affect: torch.Tensor,
        interaction: torch.Tensor,
    ) -> torch.Tensor:
        if not self.gated:
            return context.new_ones((context.size(0), 1))
        return torch.sigmoid(self.reliability_logit(context, affect, interaction))

    def reliability_logit(
        self,
        context: torch.Tensor,
        affect: torch.Tensor,
        interaction: torch.Tensor,
    ) -> torch.Tensor:
        if not self.gated:
            return context.new_zeros((context.size(0), 1))
        assert self.reliability_head is not None
        return self.reliability_head(
            torch.cat([context, affect, interaction], dim=-1)
        )

    def forward(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        output = super().forward(batch)
        context = output["context_representation"]
        affect = output["affect_representation"]
        interaction = output["interaction_representation"]
        raw_delta = output["delta_logits"]
        gate = self.reliability(context, affect, interaction)
        effective_delta = gate * raw_delta

        zero = torch.zeros_like(affect)
        null_raw_delta = self._residual(context, zero, zero)
        null_gate = self.reliability(context, zero, zero)
        null_effective_delta = null_gate * null_raw_delta
        output.update({
            "raw_delta_logits": raw_delta,
            "reliability_gate": gate,
            "delta_logits": effective_delta,
            "final_logits": output["context_logits"] + effective_delta,
            "null_raw_delta_logits": null_raw_delta,
            "null_reliability_gate": null_gate,
            "null_delta_logits": null_effective_delta,
            "null_logits": output["context_logits"] + null_effective_delta,
        })
        return output

    def invalid_party_a(
        self, output: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Pair each context with a different in-batch Party-A representation."""
        context = output["context_representation"]
        affect = output["affect_representation"]
        interaction = output["interaction_representation"]
        if context.size(0) > 1:
            affect_invalid = affect.roll(1, dims=0)
            interaction_invalid = interaction.roll(1, dims=0)
        else:
            affect_invalid = torch.zeros_like(affect)
            interaction_invalid = torch.zeros_like(interaction)
        raw_delta = self._residual(context, affect_invalid, interaction_invalid)
        gate_logit = self.reliability_logit(
            context, affect_invalid, interaction_invalid
        )
        gate = (
            torch.sigmoid(gate_logit)
            if self.gated
            else context.new_ones((context.size(0), 1))
        )
        effective_delta = gate * raw_delta
        return {
            "invalid_raw_delta_logits": raw_delta,
            "invalid_reliability_logit": gate_logit,
            "invalid_reliability_gate": gate,
            "invalid_delta_logits": effective_delta,
            "invalid_logits": output["context_logits"].detach() + effective_delta,
        }


def context_consistency_loss(
    context_logits: torch.Tensor, alternative_logits: torch.Tensor
) -> torch.Tensor:
    context = context_logits.detach()
    probability = F.softmax(context, dim=-1)
    return F.kl_div(F.log_softmax(alternative_logits, dim=-1), probability, reduction="batchmean")


def invalid_gate_zero_loss(gate_logit: torch.Tensor) -> torch.Tensor:
    """AMP-safe binary loss for the pre-sigmoid invalid-A gate logit."""
    return F.binary_cross_entropy_with_logits(
        gate_logit, torch.zeros_like(gate_logit)
    )


def compute_reliability_losses(
    model: ReliabilityGatedResidual,
    output: dict[str, torch.Tensor],
    batch: dict[str, object],
    source_mapping: dict[str, int],
    weights: dict[str, float],
    affect_class_weights: torch.Tensor,
    temperature: float,
    null_divergence: str,
) -> dict[str, torch.Tensor]:
    base = compute_losses(
        output,
        batch,
        "both",
        source_mapping,
        weights,
        affect_class_weights,
        temperature,
        null_divergence,
    )
    zero = output["final_logits"].sum() * 0.0
    counterfactual = zero
    invalid_gate = zero
    if model.counterfactual_training:
        invalid = model.invalid_party_a(output)
        counterfactual = context_consistency_loss(
            output["context_logits"], invalid["invalid_logits"]
        )
        if model.gated:
            invalid_gate = invalid_gate_zero_loss(
                invalid["invalid_reliability_logit"]
            )
    total = (
        base["total"]
        + weights["counterfactual"] * counterfactual
        + weights["invalid_gate"] * invalid_gate
    )
    return {**base, "counterfactual": counterfactual, "invalid_gate": invalid_gate, "total": total}


@torch.no_grad()
def evaluate(
    model: ReliabilityGatedResidual,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    limit_batches: int | None = None,
) -> dict[str, object]:
    model.eval()
    keys = (
        "context_logits", "raw_delta_logits", "delta_logits", "final_logits",
        "reliability_gate", "affect_logits",
    )
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    arrays.update({"labels": [], "party_a_labels": []})
    sample_ids: list[str] = []
    source_folders: list[str] = []
    for batch_index, batch in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        device_batch = to_device(batch, device)
        output = model(device_batch)
        for key in keys:
            arrays[key].append(output[key].cpu().numpy())
        arrays["labels"].append(device_batch["target"].cpu().numpy())
        arrays["party_a_labels"].append(device_batch["party_a_target"].cpu().numpy())
        sample_ids.extend(batch["sample_id"])
        source_folders.extend(batch["source_folder"])
    merged = {key: np.concatenate(value) for key, value in arrays.items()}
    labels = merged["labels"]
    context_metrics = metrics_from_logits(merged["context_logits"], labels)
    final_metrics = metrics_from_logits(merged["final_logits"], labels)
    context_prediction = merged["context_logits"].argmax(axis=1)
    final_prediction = merged["final_logits"].argmax(axis=1)
    context_correct = context_prediction == labels
    final_correct = final_prediction == labels
    gate = merged["reliability_gate"].reshape(-1)
    result: dict[str, object] = {
        **merged,
        "sample_ids": np.asarray(sample_ids),
        "source_folders": np.asarray(source_folders),
        "final_loss": float(F.cross_entropy(torch.from_numpy(merged["final_logits"]), torch.from_numpy(labels))),
        "context_loss": float(F.cross_entropy(torch.from_numpy(merged["context_logits"]), torch.from_numpy(labels))),
        "raw_delta_l2_mean": float(np.linalg.norm(merged["raw_delta_logits"], axis=1).mean()),
        "delta_l2_mean": float(np.linalg.norm(merged["delta_logits"], axis=1).mean()),
        "gate_mean": float(gate.mean()),
        "gate_std": float(gate.std()),
        "gate_p10": float(np.quantile(gate, 0.10)),
        "gate_p50": float(np.quantile(gate, 0.50)),
        "gate_p90": float(np.quantile(gate, 0.90)),
        "prediction_flip_rate": float(np.mean(context_prediction != final_prediction)),
        "beneficial_flip_rate": float(np.mean(~context_correct & final_correct)),
        "harmful_flip_rate": float(np.mean(context_correct & ~final_correct)),
    }
    result.update({f"final_{key}": value for key, value in final_metrics.items()})
    result.update({f"context_{key}": value for key, value in context_metrics.items()})
    affect_metrics = metrics_from_logits(merged["affect_logits"], merged["party_a_labels"])
    result["affect_unweighted_loss"] = float(F.cross_entropy(
        torch.from_numpy(merged["affect_logits"]), torch.from_numpy(merged["party_a_labels"])
    ))
    result.update({f"affect_{key}": value for key, value in affect_metrics.items()})
    return result


def public_metrics(result: dict[str, object]) -> dict[str, object]:
    hidden = {
        "context_logits", "raw_delta_logits", "delta_logits", "final_logits",
        "reliability_gate", "affect_logits", "labels", "party_a_labels",
        "sample_ids", "source_folders",
    }
    return {key: value for key, value in result.items() if key not in hidden}


def save_predictions(path: Path, result: dict[str, object]) -> None:
    np.savez_compressed(path, **{
        key: result[key] for key in (
            "context_logits", "raw_delta_logits", "delta_logits", "final_logits",
            "reliability_gate", "affect_logits", "labels", "party_a_labels",
            "sample_ids", "source_folders",
        )
    })


def main() -> int:
    args = parse_args()
    nonnegative = (
        "context_weight", "emotion_weight", "contrastive_weight", "null_weight",
        "nuisance_weight", "counterfactual_weight", "invalid_gate_weight",
    )
    for name in nonnegative:
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    manifest = pd.read_csv(args.manifest, dtype={"source_folder": str})
    if set(manifest["split"]) != {"inner_train", "inner_development"}:
        raise RuntimeError("v0.8 trainer accepts only the frozen inner-development manifest")
    if "original_split" not in manifest or set(manifest["original_split"]) != {"train"}:
        raise RuntimeError("Inner manifest must prove every row came from original train")
    train_rows = manifest[manifest["split"] == "inner_train"].reset_index(drop=True)
    val_rows = manifest[manifest["split"] == "inner_development"].reset_index(drop=True)
    if set(train_rows["source_folder"]) & set(val_rows["source_folder"]):
        raise RuntimeError("Inner train/development source folders overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_mapping = {
        folder: index for index, folder in enumerate(sorted(train_rows["source_folder"].unique()))
    }
    affect_weights_array = balanced_affect_weights(train_rows)
    affect_weights = torch.tensor(affect_weights_array, dtype=torch.float32, device=device)
    train_loader = build_loader(train_rows, args.features_dir, args.batch_size, args.workers, True, args.seed)
    val_loader = build_loader(val_rows, args.features_dir, args.batch_size, args.workers, False, args.seed + 1)
    model = ReliabilityGatedResidual(
        args.variant, len(source_mapping), args.d_model, args.temporal_layers,
        args.context_layers, args.dropout, args.face_pooling,
        args.gradient_reversal_scale,
    ).to(device)
    weights = {
        "context": args.context_weight,
        "emotion": args.emotion_weight,
        "contrastive": args.contrastive_weight,
        "null": args.null_weight,
        "nuisance": args.nuisance_weight,
        "counterfactual": args.counterfactual_weight,
        "invalid_gate": args.invalid_gate_weight,
    }
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "num_training_source_folders": len(source_mapping),
        "manifest_sha256": sha256(args.manifest),
        "partitions_touched": ["original-train/inner-train", "original-train/inner-development"],
        "original_validation_evaluated": False,
        "test_evaluated": False,
        "architecture_invariant": "final_logits=context_logits+reliability_gate*raw_delta_logits",
        "invalid_party_a": "cyclic-one-position-roll; singleton-zero",
        "positive_gate_targets_used": False,
        "affect_class_weights": {
            emotion: float(weight) for emotion, weight in zip(EMOTIONS, affect_weights_array)
        },
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_names = (
        "total", "final", "context", "emotion", "contrastive", "null", "nuisance",
        "counterfactual", "invalid_gate", "contrastive_valid_anchor_rate",
    )
    history: list[dict[str, object]] = []
    best_uar, best_loss, stale = -math.inf, math.inf, 0
    checkpoint_path = args.output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {name: 0.0 for name in loss_names}
        seen = 0
        for batch_index, batch in enumerate(train_loader):
            if args.limit_train_batches is not None and batch_index >= args.limit_train_batches:
                break
            device_batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(device_batch)
                losses = compute_reliability_losses(
                    model, output, device_batch, source_mapping, weights, affect_weights,
                    args.contrastive_temperature, args.null_divergence,
                )
            if not torch.isfinite(losses["total"]):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            current = len(device_batch["target"])
            for name in loss_names:
                sums[name] += float(losses[name].detach()) * current
            seen += current
        validation = evaluate(model, val_loader, device, args.limit_val_batches)
        scheduler.step(validation["final_loss"])
        row = {
            "epoch": epoch,
            **{f"train_{name}": value / seen for name, value in sums.items()},
            **{f"val_{name}": validation[name] for name in (
                "final_loss", "final_uar", "final_war", "context_loss", "context_uar",
                "raw_delta_l2_mean", "delta_l2_mean", "gate_mean", "harmful_flip_rate",
                "beneficial_flip_rate", "affect_uar",
            )},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        improved = validation["final_uar"] > best_uar or (
            math.isclose(validation["final_uar"], best_uar) and validation["final_loss"] < best_loss
        )
        if improved:
            best_uar, best_loss, stale = float(validation["final_uar"]), float(validation["final_loss"]), 0
            torch.save({
                "model_state": model.state_dict(), "epoch": epoch, "val_uar": best_uar,
                "val_loss": best_loss, "config": config, "source_mapping": source_mapping,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                break

    with (args.output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation = evaluate(model, val_loader, device, args.limit_val_batches)
    save_predictions(args.output_dir / "inner_development_predictions.npz", validation)
    metrics = {
        "protocol": "reliability-gated-residual-inner-development-v1",
        "variant": args.variant,
        "best_epoch": checkpoint["epoch"],
        "inner_development": public_metrics(validation),
        "original_validation": None,
        "test": None,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
