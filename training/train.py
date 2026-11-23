from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import uuid

import lightning as L
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training"))

from cross_boundary_coref.decoder import connected_component_rectangles  # noqa: E402
from cross_boundary_coref.model import AffinityModel, _checkpoint_state  # noqa: E402
from dataset import (  # noqa: E402
    collate_examples,
    encode_dataset,
    load_datasets,
)
from loss import affinity_loss  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_initial_checkpoint(path: Path, expected_hash: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Task 21 initialization checkpoint not found: {path}")
    actual_hash = _sha256(path)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"Task 21 checkpoint SHA-256 mismatch: expected {expected_hash}, "
            f"found {actual_hash}"
        )
    return actual_hash


def _load_initial_checkpoint(path: Path, model: AffinityModel) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(_checkpoint_state(checkpoint), strict=True)


def _span_f1(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    denominator = (left[1] - left[0]) + (right[1] - right[0])
    return 2 * intersection / denominator if denominator else 0.0


def _partial_credit(predicted, gold) -> float:
    scores = np.zeros((len(predicted), len(gold)), dtype=float)
    for i, (pred_context, pred_chunk) in enumerate(predicted):
        for j, (gold_context, gold_chunk) in enumerate(gold):
            scores[i, j] = _span_f1(pred_context, gold_context) * _span_f1(
                pred_chunk, gold_chunk
            )
    if not scores.size:
        return 0.0
    rows, columns = linear_sum_assignment(-scores)
    return float(scores[rows, columns].sum())


def _f1(tp: float, predicted: int, gold: int) -> float:
    precision = tp / predicted if predicted else (1.0 if gold == 0 else 0.0)
    recall = tp / gold if gold else (1.0 if predicted == 0 else 0.0)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def validation_metrics(records, thresholds, connectivities):
    if not records:
        raise ValueError("validation completed without affinity examples")
    candidates = []
    for connectivity in connectivities:
        for threshold in thresholds:
            exact_tp = predicted_count = gold_count = 0
            partial_credit = 0.0
            for scores, gold in records:
                boxes = connected_component_rectangles(
                    scores, threshold=threshold, connectivity=connectivity
                )
                predicted = [
                    ((box.context.start, box.context.end), (box.chunk.start, box.chunk.end))
                    for box in boxes
                ]
                predicted_counter = Counter(predicted)
                gold_counter = Counter(gold)
                exact_tp += sum(
                    min(count, gold_counter[pair])
                    for pair, count in predicted_counter.items()
                )
                predicted_count += len(predicted)
                gold_count += len(gold)
                partial_credit += _partial_credit(predicted, gold)
            exact_f1 = _f1(exact_tp, predicted_count, gold_count)
            precision = exact_tp / predicted_count if predicted_count else (
                1.0 if gold_count == 0 else 0.0
            )
            partial_f1 = _f1(partial_credit, predicted_count, gold_count)
            candidates.append(
                {
                    "f1": exact_f1,
                    "precision": precision,
                    "partial_f1": partial_f1,
                    "threshold": threshold,
                    "connectivity": connectivity,
                }
            )
    selected = max(
        candidates,
        key=lambda row: (
            row["f1"],
            row["precision"],
            row["partial_f1"],
            row["threshold"],
            -row["connectivity"],
        ),
    )
    return selected


class AffinityTrainingModule(L.LightningModule):
    def __init__(
        self,
        model: AffinityModel,
        *,
        learning_rate: float,
        negative_weight: float,
        thresholds: list[float],
        connectivities: list[int],
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.negative_weight = negative_weight
        self.thresholds = thresholds
        self.connectivities = connectivities
        self.validation_records = []

    def training_step(self, batch, batch_idx):
        logits = self.model(batch["input_ids"], batch["attention_mask"])
        loss = affinity_loss(
            logits,
            batch["affinity_targets"],
            batch["context_mask"],
            batch["chunk_mask"],
            negative_weight=self.negative_weight,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite affinity training loss")
        self.log("train_loss", loss, on_epoch=True, batch_size=len(batch["id"]))
        return loss

    def on_validation_epoch_start(self):
        self.validation_records = []

    def validation_step(self, batch, batch_idx):
        logits = self.model(batch["input_ids"], batch["attention_mask"])
        loss = affinity_loss(
            logits,
            batch["affinity_targets"],
            batch["context_mask"],
            batch["chunk_mask"],
            negative_weight=self.negative_weight,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite affinity validation loss")
        self.log("val_loss", loss, on_epoch=True, batch_size=len(batch["id"]))
        probabilities = logits.detach().sigmoid()
        for index in range(len(batch["id"])):
            context_positions = torch.nonzero(
                batch["context_mask"][index].bool(), as_tuple=False
            ).flatten().tolist()
            chunk_positions = torch.nonzero(
                batch["chunk_mask"][index].bool(), as_tuple=False
            ).flatten().tolist()
            context_lookup = {position: local for local, position in enumerate(context_positions)}
            chunk_lookup = {position: local for local, position in enumerate(chunk_positions)}
            gold = []
            for c0, c1, h0, h1 in batch["target_boxes"][index]:
                context_tokens = [
                    context_lookup[position]
                    for position in range(c0, c1)
                    if position in context_lookup
                ]
                chunk_tokens = [
                    chunk_lookup[position]
                    for position in range(h0, h1)
                    if position in chunk_lookup
                ]
                if context_tokens and chunk_tokens:
                    gold.append(
                        (
                            (min(context_tokens), max(context_tokens) + 1),
                            (min(chunk_tokens), max(chunk_tokens) + 1),
                        )
                    )
            scores = probabilities[index][context_positions][:, chunk_positions]
            self.validation_records.append((scores.cpu().numpy(), gold))
        return loss

    def on_validation_epoch_end(self):
        result = validation_metrics(
            self.validation_records, self.thresholds, self.connectivities
        )
        self.log("val_exact_relation_f1", result["f1"], prog_bar=True)
        self.log("val_exact_relation_precision", result["precision"])
        self.log("val_partial_relation_f1", result["partial_f1"])
        self.log("val_decoder_threshold", result["threshold"])
        self.log("val_decoder_connectivity", result["connectivity"])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


def _read_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a mapping: {path}")
    return config


def train(
    config_path: Path,
    *,
    initial_checkpoint: Path | None = None,
    max_train_examples: int | None = None,
    max_val_examples: int | None = None,
    max_steps: int | None = None,
    accelerator: str | None = None,
) -> Path:
    config = _read_config(config_path)
    checkpoint_path = initial_checkpoint or Path(config["model"]["initial_checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = (config_path.parent / checkpoint_path).resolve()
    initial_checkpoint_hash = _verify_initial_checkpoint(
        checkpoint_path,
        config["model"]["initial_checkpoint_sha256"],
    )

    output_root = Path(config["training"]["output_dir"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    train_rows, validation_rows, source_counts = load_datasets(config)
    if max_train_examples is not None:
        train_rows = train_rows.select(range(min(max_train_examples, len(train_rows))))
    if max_val_examples is not None:
        validation_rows = validation_rows.select(
            range(min(max_val_examples, len(validation_rows)))
        )
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["encoder_name"])
    train_rows = encode_dataset(train_rows, tokenizer)
    validation_rows = encode_dataset(validation_rows, tokenizer)
    collate = lambda rows: collate_examples(rows, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_rows,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        validation_rows,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate,
    )

    L.seed_everything(config["seed"], workers=True)
    model = AffinityModel(
        config["model"]["encoder_name"],
        load_pretrained=False,
    )
    _load_initial_checkpoint(checkpoint_path, model)
    module = AffinityTrainingModule(
        model,
        learning_rate=config["training"]["learning_rate"],
        negative_weight=config["training"]["negative_loss_weight"],
        thresholds=config["decoder"]["thresholds"],
        connectivities=config["decoder"]["connectivities"],
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best",
        auto_insert_metric_name=False,
        monitor=config["training"]["monitor"],
        mode=config["training"]["monitor_mode"],
        save_top_k=1,
        save_last=False,
    )
    early_stopping = EarlyStopping(
        monitor=config["training"]["monitor"],
        mode=config["training"]["monitor_mode"],
        patience=config["training"]["early_stopping_patience"],
    )
    trainer = L.Trainer(
        max_epochs=config["training"]["max_epochs"],
        max_steps=max_steps if max_steps is not None else -1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=CSVLogger(save_dir=str(output_dir), name="lightning", version=""),
        accelerator=accelerator or config["training"]["accelerator"],
        devices=config["training"]["devices"],
        precision=config["training"]["precision"],
        log_every_n_steps=config["training"]["log_every_n_steps"],
        gradient_clip_val=config["training"]["gradient_clip_val"],
        deterministic=config["training"]["deterministic"],
        enable_progress_bar=False,
    )
    trainer.fit(module, train_loader, validation_loader)
    checkpoint = Path(checkpoint_callback.best_model_path)
    if not checkpoint.is_file():
        raise RuntimeError("Training completed without saving a selected checkpoint")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "initial_checkpoint": str(checkpoint_path),
                "initial_checkpoint_sha256": initial_checkpoint_hash,
                "training_examples": len(train_rows),
                "validation_examples": len(validation_rows),
                "source_training_examples": source_counts,
                "best_checkpoint": str(checkpoint),
                "best_epoch": int(trainer.current_epoch),
                "best_validation_f1": float(checkpoint_callback.best_model_score),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the cross-boundary affinity model")
    parser.add_argument("--config", type=Path, default=ROOT / "training/config.yaml")
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-val-examples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--accelerator")
    args = parser.parse_args()
    print(
        train(
            args.config,
            initial_checkpoint=args.initial_checkpoint,
            max_train_examples=args.max_train_examples,
            max_val_examples=args.max_val_examples,
            max_steps=args.max_steps,
            accelerator=args.accelerator,
        )
    )


if __name__ == "__main__":
    main()
