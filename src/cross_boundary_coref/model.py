from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


@dataclass(frozen=True)
class AffinityResult:
    logits: np.ndarray
    affinity: np.ndarray
    context_tokens: tuple[str, ...]
    chunk_tokens: tuple[str, ...]
    context_offsets: tuple[tuple[int, int], ...]
    chunk_offsets: tuple[tuple[int, int], ...]
    valid_mask: np.ndarray


def _checkpoint_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint does not contain a non-empty state_dict")
    if all(key.startswith("model.") for key in state):
        state = {key.removeprefix("model."): value for key, value in state.items()}
    return state


class AffinityModel(nn.Module):
    """DistilBERT with a bias-free bilinear affinity-scoring head."""

    def __init__(
        self,
        encoder_name: str = "distilbert-base-cased",
        *,
        load_pretrained: bool = True,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        if load_pretrained:
            self.encoder = AutoModel.from_pretrained(encoder_name)
        else:
            with torch.random.fork_rng(devices=[]):
                self.encoder = AutoModel.from_config(
                    AutoConfig.from_pretrained(encoder_name)
                )
        hidden_size = self.encoder.config.dim
        self.affinity = nn.Bilinear(hidden_size, hidden_size, 1, bias=False)
        self.tokenizer = None

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        return torch.matmul(
            torch.matmul(hidden, self.affinity.weight[0]),
            hidden.transpose(1, 2),
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        encoder_name: str | None = None,
    ) -> AffinityModel:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        hyperparameters = checkpoint.get("hyper_parameters", {})
        model_name = encoder_name or hyperparameters.get(
            "encoder_name", "distilbert-base-cased"
        )
        model = cls(model_name, load_pretrained=False)
        model.load_state_dict(_checkpoint_state(checkpoint), strict=True)
        model.to(device)
        model.eval()
        model.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        revision: str | None = None,
        device: str | torch.device = "cpu",
    ) -> AffinityModel:
        config_path = hf_hub_download(
            repo_id,
            "affinity_config.json",
            revision=revision,
        )
        with Path(config_path).open(encoding="utf-8") as file:
            config = json.load(file)
        if config.get("format") != "cross-boundary-coref-affinity":
            raise ValueError(f"{repo_id} is not a supported affinity model repository")

        model_name = config["encoder_name"]
        model = cls(model_name, load_pretrained=False)
        weights_path = hf_hub_download(
            repo_id,
            "model.safetensors",
            revision=revision,
        )
        model.load_state_dict(load_file(weights_path, device="cpu"), strict=True)
        model.to(device)
        model.eval()
        model.tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
        return model

    @torch.inference_mode()
    def predict_affinity(
        self,
        context: str,
        chunk: str,
        *,
        max_length: int | None = None,
    ) -> AffinityResult:
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        options: dict[str, Any] = {
            "truncation": True,
            "return_offsets_mapping": True,
            "return_tensors": "pt",
        }
        if max_length is not None:
            options["max_length"] = max_length
        encoded = self.tokenizer(context, chunk, **options)
        sequence_ids = encoded.sequence_ids(0)
        context_positions = [i for i, sid in enumerate(sequence_ids) if sid == 0]
        chunk_positions = [i for i, sid in enumerate(sequence_ids) if sid == 1]
        if not context_positions or not chunk_positions:
            raise ValueError("Both context and chunk must retain at least one token")

        device = next(self.parameters()).device
        logits = self(
            encoded["input_ids"].to(device),
            encoded["attention_mask"].to(device),
        )[0]
        logits = logits[context_positions][:, chunk_positions]
        probabilities = logits.sigmoid()
        input_tokens = self.tokenizer.convert_ids_to_tokens(
            encoded["input_ids"][0].tolist()
        )
        offsets = encoded["offset_mapping"][0].tolist()
        context_offsets = tuple(tuple(offsets[i]) for i in context_positions)
        chunk_offsets = tuple(tuple(offsets[i]) for i in chunk_positions)
        return AffinityResult(
            logits=logits.cpu().numpy(),
            affinity=probabilities.cpu().numpy(),
            context_tokens=tuple(input_tokens[i] for i in context_positions),
            chunk_tokens=tuple(input_tokens[i] for i in chunk_positions),
            context_offsets=context_offsets,
            chunk_offsets=chunk_offsets,
            valid_mask=np.ones((len(context_positions), len(chunk_positions)), dtype=bool),
        )
