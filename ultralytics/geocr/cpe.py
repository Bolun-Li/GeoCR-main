"""Cross-Modal Prior Extraction (CPE).

This module contains the active CPE runtime used by the experiment code:
precomputed hierarchical text/image embeddings are looked up by image name,
then text and visual priors are combined by a learned gate.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ultralytics.utils import LOGGER


class CrossModalPriorStore:
    """Load and retrieve precomputed CPE embeddings from dataset configuration."""

    _ARRAY_KEYS = ("p3_text", "p4_text", "p5_text", "local_visual", "global_visual")

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.embedding_dim = int(config.get("embedding_dim", 1024))
        self.mapping: dict[str, int] = {}
        self.arrays: dict[str, np.ndarray | None] = {key: None for key in self._ARRAY_KEYS}

        for key in self._ARRAY_KEYS:
            self.arrays[key] = self._load_array(config.get(key), key)
        self._load_mapping(config.get("index"))

    @property
    def enabled(self) -> bool:
        """Return whether an index and at least one embedding array were loaded."""
        return bool(self.mapping) and any(value is not None for value in self.arrays.values())

    @staticmethod
    def _as_path(value: str | Path | None) -> Path | None:
        if value in (None, ""):
            return None
        return Path(value).expanduser()

    def _load_array(self, value: str | Path | None, label: str) -> np.ndarray | None:
        path = self._as_path(value)
        if path is None:
            return None
        if not path.is_file():
            LOGGER.warning("GeoCR CPE: %s embedding file not found: %s", label, path)
            return None
        array = np.load(path).astype(np.float32)
        LOGGER.info("GeoCR CPE: loaded %s embeddings with shape %s", label, array.shape)
        return array

    def _load_mapping(self, value: str | Path | None) -> None:
        path = self._as_path(value)
        if path is None:
            return
        if not path.is_file():
            LOGGER.warning("GeoCR CPE: prompt index not found: %s", path)
            return
        with path.open("r", encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning("GeoCR CPE: skipped invalid JSONL record %d in %s", index + 1, path)
                    continue
                file_name = item.get("file_name")
                if file_name:
                    self.mapping[Path(file_name).name] = index

    def _mean_embedding(self, key: str, file_names: Iterable[str]) -> torch.Tensor:
        array = self.arrays[key]
        if array is None:
            return torch.zeros(self.embedding_dim, dtype=torch.float32)
        vectors = []
        for file_name in file_names:
            index = self.mapping.get(Path(file_name).name)
            if index is not None and 0 <= index < len(array):
                vectors.append(array[index])
        if not vectors:
            return torch.zeros(self.embedding_dim, dtype=torch.float32)
        return torch.from_numpy(np.mean(vectors, axis=0).astype(np.float32))

    def get(self, file_names: Iterable[str]) -> dict[str, torch.Tensor]:
        """Return the active P3/P4/P5 text and local/global visual priors."""
        names = list(file_names)
        return {f"cpe_{key}": self._mean_embedding(key, names) for key in self._ARRAY_KEYS}


class GatedPromptFusion(nn.Module):
    """Implement the channel-wise cross-modal gate in Eqs. (3)-(4)."""

    def __init__(self, embed_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        del hidden_dim  # Kept in the signature for checkpoint/config compatibility.
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, text_prior: torch.Tensor, visual_prior: torch.Tensor) -> torch.Tensor:
        reference = self.gate[0].weight
        text_prior = text_prior.to(device=reference.device, dtype=reference.dtype)
        visual_prior = visual_prior.to(device=reference.device, dtype=reference.dtype)
        weight = self.gate(torch.cat((text_prior, visual_prior), dim=1))
        return weight * text_prior + (1.0 - weight) * visual_prior
