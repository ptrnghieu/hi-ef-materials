#!/usr/bin/env python3
"""Train leakage-safe frozen-feature baselines for Hi-EF."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
EMOTION_TO_ID = {emotion: index for index, emotion in enumerate(EMOTIONS)}


def seed_everything(seed: int) -> None:
    # Required by deterministic CUDA matrix multiplication. This must be set
    # before the first CUDA operation in the process.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    if torch.cuda.is_available():
        # Flash and memory-efficient scaled-dot-product attention may select
        # nondeterministic backward kernels. The math backend is deterministic.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@lru_cache(maxsize=512)
def load_feature(path: str) -> dict[str, object]:
    return torch.load(path, map_location="cpu", weights_only=False)


class HiEFFrozenDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, features_dir: Path, num_clips: int) -> None:
        self.rows = rows.reset_index(drop=True)
        self.features_dir = features_dir
        self.num_clips = num_clips

    def __len__(self) -> int:
        return len(self.rows)

    def _load_clip(self, clip_id: str) -> dict[str, torch.Tensor]:
        path = self.features_dir / f"{clip_id.replace('/', '_')}.pt"
        item = load_feature(str(path))
        if item.get("clip_id") != clip_id:
            raise ValueError(f"Feature id mismatch for {clip_id}: {item.get('clip_id')}")
        return {
            "face": item["face_features"].float(),
            "ori": item["ori_features"].float(),
            "text": item["text_feature"].float(),
            "audio": item["audio_feature"].float(),
            "face_mask": torch.as_tensor(item["face_valid_mask"], dtype=torch.bool),
            "audio_found": torch.tensor(bool(item.get("audio_found", False))),
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows.iloc[index]
        result: dict[str, object] = {
            "sample_id": row["sample_id"],
            "source_folder": row["source_folder"],
            "target": EMOTION_TO_ID[row["clip4_emotion"]],
        }
        for position in range(1, self.num_clips + 1):
            result[f"clip{position}"] = self._load_clip(row[f"clip{position}"])
        return result


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {
        "sample_id": [item["sample_id"] for item in batch],
        "source_folder": [item["source_folder"] for item in batch],
        "target": torch.tensor([item["target"] for item in batch], dtype=torch.long),
    }
    clip_keys = sorted(key for key in batch[0] if key.startswith("clip"))
    for clip_key in clip_keys:
        clips = [item[clip_key] for item in batch]
        result[clip_key] = {
            key: torch.stack([clip[key] for clip in clips])
            for key in ("face", "ori", "text", "audio", "face_mask", "audio_found")
        }
    return result


class TemporalEncoder(nn.Module):
    def __init__(self, d_model: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(512, d_model) if d_model != 512 else nn.Identity()
        self.position = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )

    def forward(
        self, features: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.input_proj(features)
        if valid_mask is None:
            valid_mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        valid_mask = valid_mask.bool()
        all_missing = ~valid_mask.any(dim=1)
        safe_mask = valid_mask.clone()
        safe_mask[all_missing, 0] = True
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        encoded = self.encoder(
            x + self.position[:, : x.size(1)],
            src_key_padding_mask=~safe_mask,
        )
        weights = valid_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled[all_missing] = 0.0
        return pooled


class ClipEncoder(nn.Module):
    def __init__(
        self, d_model: int, temporal_layers: int, dropout: float, face_pooling: str
    ) -> None:
        super().__init__()
        self.face_pooling = face_pooling
        self.face_encoder = TemporalEncoder(d_model, temporal_layers, dropout)
        self.ori_encoder = TemporalEncoder(d_model, temporal_layers, dropout)
        self.text_proj = nn.Linear(512, d_model) if d_model != 512 else nn.Identity()
        self.audio_proj = nn.Linear(527, d_model)
        self.modality_position = nn.Parameter(torch.randn(1, 4, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, clip: dict[str, torch.Tensor]) -> torch.Tensor:
        face_mask = clip["face_mask"] if self.face_pooling == "masked" else None
        face = self.face_encoder(clip["face"], face_mask)
        ori = self.ori_encoder(clip["ori"])
        text = self.text_proj(clip["text"])
        audio = self.audio_proj(F.normalize(clip["audio"], dim=-1))
        tokens = torch.stack([face, ori, text, audio], dim=1)
        fused = self.fusion(tokens + self.modality_position)
        return self.norm(fused.mean(dim=1))


class FrozenFeatureBaseline(nn.Module):
    def __init__(
        self,
        num_clips: int,
        d_model: int,
        temporal_layers: int,
        inter_layers: int,
        dropout: float,
        face_pooling: str,
    ) -> None:
        super().__init__()
        self.num_clips = num_clips
        self.clip_encoder = ClipEncoder(d_model, temporal_layers, dropout, face_pooling)
        self.clip_position = nn.Parameter(torch.randn(1, num_clips, d_model) * 0.02)
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
        clip_features = [
            self.clip_encoder(batch[f"clip{position}"])
            for position in range(1, self.num_clips + 1)
        ]
        sequence = torch.stack(clip_features, dim=1) + self.clip_position
        return self.head(self.inter_encoder(sequence).mean(dim=1))


def to_device(value: object, device: torch.device) -> object:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    predictions = logits.argmax(axis=1)
    recalls: list[float] = []
    per_class = {}
    for class_id, emotion in enumerate(EMOTIONS):
        mask = labels == class_id
        recall = float((predictions[mask] == class_id).mean()) if mask.any() else None
        if recall is not None:
            recalls.append(recall)
        per_class[emotion] = recall
    return {
        "war": float((predictions == labels).mean()),
        "uar": float(np.mean(recalls)),
        "per_class_recall": per_class,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    limit_batches: int | None = None,
) -> dict[str, object]:
    model.eval()
    losses, logits, labels, sample_ids, folders = [], [], [], [], []
    for batch_index, batch in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        device_batch = to_device(batch, device)
        batch_logits = model(device_batch)
        target = device_batch["target"]
        losses.append(float(criterion(batch_logits, target)) * len(target))
        logits.append(batch_logits.cpu().numpy())
        labels.append(target.cpu().numpy())
        sample_ids.extend(batch["sample_id"])
        folders.extend(batch["source_folder"])
    merged_logits = np.concatenate(logits)
    merged_labels = np.concatenate(labels)
    result = metrics_from_logits(merged_logits, merged_labels)
    result.update(
        {
            "loss": sum(losses) / len(merged_labels),
            "logits": merged_logits,
            "labels": merged_labels,
            "sample_ids": np.asarray(sample_ids),
            "source_folders": np.asarray(folders),
        }
    )
    return result


def save_predictions(path: Path, result: dict[str, object]) -> None:
    np.savez_compressed(
        path,
        logits=result["logits"],
        labels=result["labels"],
        predictions=result["logits"].argmax(axis=1),
        sample_ids=result["sample_ids"],
        source_folders=result["source_folders"],
    )


def public_metrics(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"logits", "labels", "sample_ids", "source_folders"}
    }


def build_loader(
    rows: pd.DataFrame,
    features_dir: Path,
    num_clips: int,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        HiEFFrozenDataset(rows, features_dir, num_clips),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=collate_batch,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("context", "full"), required=True)
    parser.add_argument("--face-pooling", choices=("masked", "unmasked"), default="masked")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--inter-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--evaluate-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_clips = 2 if args.model == "context" else 3

    manifest = pd.read_csv(args.manifest, dtype={"source_folder": str})
    split_rows = {
        split: manifest[manifest["split"] == split].reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    train_loader = build_loader(
        split_rows["train"], args.features_dir, num_clips, args.batch_size,
        args.workers, True, args.seed,
    )
    val_loader = build_loader(
        split_rows["val"], args.features_dir, num_clips, args.batch_size,
        args.workers, False, args.seed + 1,
    )
    model = FrozenFeatureBaseline(
        num_clips=num_clips,
        d_model=args.d_model,
        temporal_layers=args.temporal_layers,
        inter_layers=args.inter_layers,
        dropout=args.dropout,
        face_pooling=args.face_pooling,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint_path = args.output_dir / "best.pt"
    history = []
    best_uar, best_loss, stale_epochs = -math.inf, math.inf, 0

    config = vars(args).copy()
    config.update(
        {
            "manifest": str(args.manifest),
            "features_dir": str(args.features_dir),
            "output_dir": str(args.output_dir),
            "device": str(device),
            "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "test_evaluation_requested": args.evaluate_test,
        }
    )
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, seen = 0.0, 0
        for batch_index, batch in enumerate(train_loader):
            if args.limit_train_batches is not None and batch_index >= args.limit_train_batches:
                break
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
            batch_size = len(device_batch["target"])
            train_loss += float(loss.detach()) * batch_size
            seen += batch_size

        validation = evaluate(
            model, val_loader, device, criterion, args.limit_val_batches
        )
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
        print(json.dumps(row))

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
                    "config": config,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after epoch {epoch}")
                break

    with (args.output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation = evaluate(model, val_loader, device, criterion)
    save_predictions(args.output_dir / "val_predictions.npz", validation)
    metrics = {
        "best_epoch": checkpoint["epoch"],
        "validation": public_metrics(validation),
        "test": None,
    }

    if args.evaluate_test:
        test_loader = build_loader(
            split_rows["test"], args.features_dir, num_clips, args.batch_size,
            args.workers, False, args.seed + 2,
        )
        test = evaluate(model, test_loader, device, criterion)
        save_predictions(args.output_dir / "test_predictions.npz", test)
        metrics["test"] = public_metrics(test)

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
