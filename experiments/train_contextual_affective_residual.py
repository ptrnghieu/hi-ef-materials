#!/usr/bin/env python3
"""Train the canonical context-prior plus Party-A logit-residual model.

This runner is validation-only.  It implements the agreed architecture

    final_logits = context_logits + delta_A(context, Party-A representations)

and the four required ablations: context, affect, interaction, and both.
Loss weights are intentionally required command-line arguments: they were not
fixed by the research discussion and must not be selected silently in code.
"""

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


VARIANTS = ("context", "affect", "interaction", "both")


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
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--face-pooling", choices=("masked", "unmasked"), default="masked")

    # No defaults: these values are scientifically consequential and remain
    # to be frozen in the experiment specification.
    parser.add_argument("--context-weight", type=float, required=True)
    parser.add_argument("--emotion-weight", type=float, required=True)
    parser.add_argument("--contrastive-weight", type=float, required=True)
    parser.add_argument("--null-weight", type=float, required=True)
    parser.add_argument("--nuisance-weight", type=float, required=True)
    parser.add_argument("--contrastive-temperature", type=float, required=True)
    parser.add_argument(
        "--null-divergence",
        choices=("context-to-null", "null-to-context", "symmetric"),
        required=True,
        help="KL direction; required because the discussion notes contain both directions.",
    )
    parser.add_argument("--gradient-reversal-scale", type=float, default=1.0)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    return parser.parse_args()


class ResidualDataset(HiEFFrozenDataset):
    def __init__(self, rows: pd.DataFrame, features_dir: Path) -> None:
        super().__init__(rows, features_dir, num_clips=3)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = super().__getitem__(index)
        item["party_a_target"] = EMOTION_TO_ID[
            str(self.rows.iloc[index]["clip3_emotion"])
        ]
        return item


def collate_residual_batch(batch: list[dict[str, object]]) -> dict[str, object]:
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
        ResidualDataset(rows, features_dir),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=collate_residual_batch,
        worker_init_fn=seed_worker,
        generator=generator,
    )


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return value.view_as(value)

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


def reverse_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
    return GradientReversal.apply(value, scale)


class ContextualAffectiveResidual(nn.Module):
    """Context prior with context-conditioned Party-A correction in logit space."""

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
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        if d_model % 8:
            raise ValueError("d_model must be divisible by 8")
        self.variant = variant
        self.d_model = d_model
        self.gradient_reversal_scale = gradient_reversal_scale
        self.clip_encoder = ClipEncoder(d_model, temporal_layers, dropout, face_pooling)

        self.context_position = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            layer, num_layers=context_layers, enable_nested_tensor=False
        )
        self.context_norm = nn.LayerNorm(d_model)
        self.context_head = self._classification_head(d_model, dropout)

        self.affect_encoder = self._representation_head(d_model, dropout)
        self.interaction_encoder = self._representation_head(d_model, dropout)
        self.affect_head = self._classification_head(d_model, dropout)
        self.nuisance_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_source_folders),
        )

        residual_inputs = {
            "context": 0,
            "affect": 2,
            "interaction": 2,
            "both": 3,
        }[variant]
        self.residual_head = (
            nn.Sequential(
                nn.LayerNorm(residual_inputs * d_model),
                nn.Linear(residual_inputs * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, len(EMOTIONS)),
            )
            if residual_inputs else None
        )

    @staticmethod
    def _representation_head(d_model: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    @staticmethod
    def _classification_head(d_model: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, len(EMOTIONS)),
        )

    def _residual(
        self,
        context: torch.Tensor,
        affect: torch.Tensor,
        interaction: torch.Tensor,
    ) -> torch.Tensor:
        if self.variant == "context":
            return context.new_zeros((context.size(0), len(EMOTIONS)))
        assert self.residual_head is not None
        if self.variant == "affect":
            residual_input = torch.cat([context, affect], dim=-1)
        elif self.variant == "interaction":
            residual_input = torch.cat([context, interaction], dim=-1)
        else:
            residual_input = torch.cat([context, affect, interaction], dim=-1)
        return self.residual_head(residual_input)

    def forward(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        h1 = self.clip_encoder(batch["clip1"])
        h2 = self.clip_encoder(batch["clip2"])
        context_sequence = torch.stack([h1, h2], dim=1) + self.context_position
        context = self.context_norm(self.context_encoder(context_sequence).mean(dim=1))
        context_logits = self.context_head(context)

        if self.variant == "context":
            h_a = context.new_zeros(context.shape)
        else:
            h_a = self.clip_encoder(batch["clip3"])
        affect = self.affect_encoder(h_a)
        interaction = self.interaction_encoder(h_a)
        delta = self._residual(context, affect, interaction)
        final_logits = context_logits + delta

        null_representation = torch.zeros_like(context)
        null_delta = self._residual(context, null_representation, null_representation)
        null_logits = context_logits + null_delta
        affect_logits = self.affect_head(affect)
        nuisance_logits = self.nuisance_head(
            reverse_gradient(affect, self.gradient_reversal_scale)
        )
        return {
            "context_representation": context,
            "affect_representation": affect,
            "interaction_representation": interaction,
            "context_logits": context_logits,
            "delta_logits": delta,
            "final_logits": final_logits,
            "null_delta_logits": null_delta,
            "null_logits": null_logits,
            "affect_logits": affect_logits,
            "nuisance_logits": nuisance_logits,
        }


def source_labels(source_folders: list[str], mapping: dict[str, int], device: torch.device) -> torch.Tensor:
    try:
        labels = [mapping[str(folder)] for folder in source_folders]
    except KeyError as error:
        raise ValueError(f"Unknown training source folder: {error.args[0]}") from error
    return torch.tensor(labels, dtype=torch.long, device=device)


def conditional_supervised_contrastive_loss(
    representations: torch.Tensor,
    emotion_labels: torch.Tensor,
    source_folders: list[str],
    temperature: float,
) -> torch.Tensor:
    """Contrast affect using cross-source positives and matched-source negatives.

    Positive: same Party-A emotion, different source folder.
    Negative: different Party-A emotion, same source folder.
    Anchors lacking either kind of pair do not contribute.
    """
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    batch_size = representations.size(0)
    if batch_size < 2:
        return representations.sum() * 0.0
    source_codes = {folder: index for index, folder in enumerate(sorted(set(source_folders)))}
    source = torch.tensor(
        [source_codes[folder] for folder in source_folders],
        dtype=torch.long,
        device=representations.device,
    )
    identity = torch.eye(batch_size, dtype=torch.bool, device=representations.device)
    same_emotion = emotion_labels[:, None].eq(emotion_labels[None, :])
    same_source = source[:, None].eq(source[None, :])
    positives = same_emotion & ~same_source & ~identity
    negatives = ~same_emotion & same_source & ~identity
    candidates = positives | negatives
    valid = positives.any(dim=1) & negatives.any(dim=1)
    if not valid.any():
        return representations.sum() * 0.0
    normalized = F.normalize(representations, dim=-1)
    logits = normalized @ normalized.T / temperature
    logits = logits.masked_fill(~candidates, -torch.inf)
    log_denominator = torch.logsumexp(logits, dim=1)
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    log_numerator = torch.logsumexp(positive_logits, dim=1)
    return -(log_numerator[valid] - log_denominator[valid]).mean()


def null_consistency_loss(
    context_logits: torch.Tensor,
    null_logits: torch.Tensor,
    direction: str,
) -> torch.Tensor:
    context_log = F.log_softmax(context_logits, dim=-1)
    null_log = F.log_softmax(null_logits, dim=-1)
    context_probability = context_log.exp()
    null_probability = null_log.exp()
    context_to_null = F.kl_div(null_log, context_probability, reduction="batchmean")
    null_to_context = F.kl_div(context_log, null_probability, reduction="batchmean")
    if direction == "context-to-null":
        return context_to_null
    if direction == "null-to-context":
        return null_to_context
    if direction == "symmetric":
        return 0.5 * (context_to_null + null_to_context)
    raise ValueError(f"Unknown null divergence: {direction}")


def compute_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, object],
    variant: str,
    source_mapping: dict[str, int],
    weights: dict[str, float],
    temperature: float,
    null_divergence: str,
) -> dict[str, torch.Tensor]:
    target = batch["target"]
    zero = output["final_logits"].sum() * 0.0
    final = F.cross_entropy(output["final_logits"], target)
    context = F.cross_entropy(output["context_logits"], target)
    emotion = zero
    contrastive = zero
    nuisance = zero
    null = zero
    if variant in {"affect", "both"}:
        emotion = F.cross_entropy(output["affect_logits"], batch["party_a_target"])
        contrastive = conditional_supervised_contrastive_loss(
            output["affect_representation"],
            batch["party_a_target"],
            batch["source_folder"],
            temperature,
        )
        nuisance_target = source_labels(
            batch["source_folder"], source_mapping, output["final_logits"].device
        )
        nuisance = F.cross_entropy(output["nuisance_logits"], nuisance_target)
    if variant != "context":
        null = null_consistency_loss(
            output["context_logits"], output["null_logits"], null_divergence
        )
    total = (
        final
        + weights["context"] * context
        + weights["emotion"] * emotion
        + weights["contrastive"] * contrastive
        + weights["null"] * null
        + weights["nuisance"] * nuisance
    )
    return {
        "total": total,
        "final": final,
        "context": context,
        "emotion": emotion,
        "contrastive": contrastive,
        "null": null,
        "nuisance": nuisance,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit_batches: int | None = None,
) -> dict[str, object]:
    model.eval()
    arrays: dict[str, list[np.ndarray]] = {
        "context_logits": [], "delta_logits": [], "final_logits": [],
        "affect_logits": [], "labels": [], "party_a_labels": [],
    }
    sample_ids: list[str] = []
    source_folders: list[str] = []
    for batch_index, batch in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        device_batch = to_device(batch, device)
        output = model(device_batch)
        for key in ("context_logits", "delta_logits", "final_logits", "affect_logits"):
            arrays[key].append(output[key].cpu().numpy())
        arrays["labels"].append(device_batch["target"].cpu().numpy())
        arrays["party_a_labels"].append(device_batch["party_a_target"].cpu().numpy())
        sample_ids.extend(batch["sample_id"])
        source_folders.extend(batch["source_folder"])
    merged = {key: np.concatenate(value) for key, value in arrays.items()}
    final_metrics = metrics_from_logits(merged["final_logits"], merged["labels"])
    context_metrics = metrics_from_logits(merged["context_logits"], merged["labels"])
    result: dict[str, object] = {
        **merged,
        "sample_ids": np.asarray(sample_ids),
        "source_folders": np.asarray(source_folders),
        "final_loss": float(F.cross_entropy(
            torch.from_numpy(merged["final_logits"]),
            torch.from_numpy(merged["labels"]),
        )),
        "context_loss": float(F.cross_entropy(
            torch.from_numpy(merged["context_logits"]),
            torch.from_numpy(merged["labels"]),
        )),
        "delta_l2_mean": float(np.linalg.norm(merged["delta_logits"], axis=1).mean()),
    }
    result.update({f"final_{key}": value for key, value in final_metrics.items()})
    result.update({f"context_{key}": value for key, value in context_metrics.items()})
    return result


def public_metrics(result: dict[str, object]) -> dict[str, object]:
    hidden = {
        "context_logits", "delta_logits", "final_logits", "affect_logits",
        "labels", "party_a_labels", "sample_ids", "source_folders",
    }
    return {key: value for key, value in result.items() if key not in hidden}


def save_predictions(path: Path, result: dict[str, object]) -> None:
    np.savez_compressed(
        path,
        context_logits=result["context_logits"],
        delta_logits=result["delta_logits"],
        final_logits=result["final_logits"],
        affect_logits=result["affect_logits"],
        labels=result["labels"],
        party_a_labels=result["party_a_labels"],
        sample_ids=result["sample_ids"],
        source_folders=result["source_folders"],
    )


def main() -> int:
    args = parse_args()
    for name in (
        "context_weight", "emotion_weight", "contrastive_weight",
        "null_weight", "nuisance_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(args.manifest, dtype={"source_folder": str})
    train_rows = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_rows = manifest[manifest["split"] == "val"].reset_index(drop=True)
    source_mapping = {
        folder: index
        for index, folder in enumerate(sorted(train_rows["source_folder"].unique()))
    }
    train_loader = build_loader(
        train_rows, args.features_dir, args.batch_size, args.workers, True, args.seed
    )
    val_loader = build_loader(
        val_rows, args.features_dir, args.batch_size, args.workers, False, args.seed + 1
    )
    model = ContextualAffectiveResidual(
        variant=args.variant,
        num_source_folders=len(source_mapping),
        d_model=args.d_model,
        temporal_layers=args.temporal_layers,
        context_layers=args.context_layers,
        dropout=args.dropout,
        face_pooling=args.face_pooling,
        gradient_reversal_scale=args.gradient_reversal_scale,
    ).to(device)
    weights = {
        "context": args.context_weight,
        "emotion": args.emotion_weight,
        "contrastive": args.contrastive_weight,
        "null": args.null_weight,
        "nuisance": args.nuisance_weight,
    }
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "num_training_source_folders": len(source_mapping),
        "manifest_sha256": sha256(args.manifest),
        "test_evaluation_requested": False,
        "architecture_invariant": "final_logits=context_logits+delta_logits",
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(config, indent=2, sort_keys=True), flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, object]] = []
    best_uar, best_loss, stale = -math.inf, math.inf, 0
    checkpoint_path = args.output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {name: 0.0 for name in (
            "total", "final", "context", "emotion", "contrastive", "null", "nuisance"
        )}
        seen = 0
        for batch_index, batch in enumerate(train_loader):
            if args.limit_train_batches is not None and batch_index >= args.limit_train_batches:
                break
            device_batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(device_batch)
                losses = compute_losses(
                    output, device_batch, args.variant, source_mapping, weights,
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
            for name in sums:
                sums[name] += float(losses[name].detach()) * current
            seen += current
        validation = evaluate(model, val_loader, device, args.limit_val_batches)
        scheduler.step(validation["final_loss"])
        row = {
            "epoch": epoch,
            **{f"train_{name}_loss": total / seen for name, total in sums.items()},
            "val_final_loss": validation["final_loss"],
            "val_final_uar": validation["final_uar"],
            "val_final_war": validation["final_war"],
            "val_context_loss": validation["context_loss"],
            "val_context_uar": validation["context_uar"],
            "val_delta_l2_mean": validation["delta_l2_mean"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        improved = validation["final_uar"] > best_uar or (
            math.isclose(validation["final_uar"], best_uar)
            and validation["final_loss"] < best_loss
        )
        if improved:
            best_uar = float(validation["final_uar"])
            best_loss = float(validation["final_loss"])
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_uar": best_uar,
                    "val_loss": best_loss,
                    "config": config,
                    "source_mapping": source_mapping,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping after epoch {epoch}", flush=True)
                break

    with (args.output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation = evaluate(model, val_loader, device)
    save_predictions(args.output_dir / "val_predictions.npz", validation)
    metrics = {
        "protocol": "contextual-affective-residual-validation-v1",
        "variant": args.variant,
        "best_epoch": checkpoint["epoch"],
        "validation": public_metrics(validation),
        "test": None,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
