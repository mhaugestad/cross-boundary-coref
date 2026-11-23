from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerBase


REQUIRED_COLUMNS = {
    "id",
    "context",
    "chunk",
    "chunk_spans",
    "context_spans",
}


def _relation_id(span: dict[str, Any]) -> str:
    return str(span["label"]).rsplit("_", 1)[-1]


def tokenize_example(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "truncation": True,
        "return_offsets_mapping": True,
    }
    if max_length is not None:
        options["max_length"] = max_length
    encoding = tokenizer(example["context"], example["chunk"], **options)
    sequence_ids = encoding.sequence_ids()
    offsets = encoding["offset_mapping"]

    def token_span(span: dict[str, Any], sequence_id: int):
        indices = [
            index
            for index, ((start, end), sid) in enumerate(zip(offsets, sequence_ids))
            if sid == sequence_id and start < span["end"] and end > span["start"]
        ]
        return (indices[0], indices[-1] + 1) if indices else None

    anaphors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for span in example["chunk_spans"]:
        anaphors[_relation_id(span)].append(span)

    target_boxes = []
    for antecedent in example["context_spans"]:
        antecedent_tokens = token_span(antecedent, 0)
        if antecedent_tokens is None:
            continue
        for anaphor in anaphors.get(_relation_id(antecedent), []):
            anaphor_tokens = token_span(anaphor, 1)
            if anaphor_tokens is not None:
                target_boxes.append(
                    [
                        antecedent_tokens[0],
                        antecedent_tokens[1],
                        anaphor_tokens[0],
                        anaphor_tokens[1],
                    ]
                )
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "context_mask": [int(sid == 0) for sid in sequence_ids],
        "chunk_mask": [int(sid == 1) for sid in sequence_ids],
        "offset_mapping": offsets,
        "sequence_ids": sequence_ids,
        "target_boxes": target_boxes,
    }


def collate_examples(
    batch: list[dict[str, Any]],
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    max_len = max(len(row["input_ids"]) for row in batch)
    batch_size = len(batch)
    output: dict[str, Any] = {
        "id": [row["id"] for row in batch],
        "target_boxes": [row["target_boxes"] for row in batch],
    }
    for key in ("input_ids", "attention_mask", "context_mask", "chunk_mask"):
        pad_value = pad_token_id if key == "input_ids" else 0
        output[key] = torch.full((batch_size, max_len), pad_value, dtype=torch.long)
    output["affinity_targets"] = torch.zeros(
        batch_size, max_len, max_len, dtype=torch.float32
    )
    for row_index, row in enumerate(batch):
        length = len(row["input_ids"])
        for key in ("input_ids", "attention_mask", "context_mask", "chunk_mask"):
            output[key][row_index, :length] = torch.as_tensor(row[key], dtype=torch.long)
        for c0, c1, h0, h1 in row["target_boxes"]:
            output["affinity_targets"][row_index, c0:c1, h0:h1] = 1.0
    return output


def load_datasets(config: dict[str, Any]):
    train_parts = []
    source_counts = {}
    for name, source in config["datasets"].items():
        dataset = load_dataset(source["repository"], split=source["train_split"])
        missing = REQUIRED_COLUMNS - set(dataset.column_names)
        if missing:
            raise ValueError(
                f"{source['repository']} is missing columns: {', '.join(sorted(missing))}"
            )
        train_parts.append(dataset)
        source_counts[name] = len(dataset)
    validation = load_dataset(
        config["validation"]["repository"],
        split=config["validation"]["split"],
    )
    missing = REQUIRED_COLUMNS - set(validation.column_names)
    if missing:
        raise ValueError(
            f"{config['validation']['repository']} is missing columns: "
            f"{', '.join(sorted(missing))}"
        )
    return concatenate_datasets(train_parts), validation, source_counts


def encode_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int | None = None,
) -> Dataset:
    return dataset.map(
        tokenize_example,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        desc="Tokenizing context/chunk pairs",
    )
