"""Geometric Difficulty Rebalancing (GDR).

The implementation combines Class Similarity Graph connectivity, validation
AP, and class frequency, then updates per-class loss weights with EMA
smoothing. It is training-only and adds no inference-time operations.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def _distributed_is_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class GeometricDifficultyRebalancing(nn.Module):
    """Maintain dynamic per-class weights for classification and localization losses."""

    def __init__(
        self,
        num_classes: int,
        warmup_epochs: int = 50,
        update_freq_front: int = 10,
        update_freq_back: int = 5,
        switch_epoch: int = 40,
        mu: float = 0.9,
        f_min: float = 0.7,
        f_max: float = 2.0,
        eps: float = 1e-6,
        device: str | torch.device = "cpu",
        count_weight_start: float = 0.8,
        count_transition_epochs: int = 20,
        geometry_weight: float = 0.5,
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = int(num_classes)
        self.warmup_epochs = int(warmup_epochs)
        self.update_freq_front = int(update_freq_front)
        self.update_freq_back = int(update_freq_back)
        self.switch_epoch = int(switch_epoch)
        self.mu = float(mu)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.eps = float(eps)
        self.count_weight_start = float(count_weight_start)
        self.count_transition_epochs = int(count_transition_epochs)
        self.geometry_weight = float(min(max(geometry_weight, 0.0), 1.0))

        self.register_buffer("f_n", torch.ones(num_classes, dtype=torch.float32))
        self.register_buffer("smooth_ap", torch.ones(num_classes, dtype=torch.float32))
        self.register_buffer("last_update_epoch", torch.tensor(-1, dtype=torch.long))
        self.register_buffer("class_counts", torch.ones(num_classes, dtype=torch.float32))
        self.register_buffer("geometric_scores", torch.zeros(num_classes, dtype=torch.float32))
        self.to(device)

        # Fixed shaping constants retained from the experiment implementation.
        self._anchor_q = 0.80
        self._weight_anchor = 1.60
        self._tau = 0.70

    @torch.no_grad()
    def reset(self) -> None:
        self.f_n.fill_(1.0)
        self.smooth_ap.fill_(1.0)
        self.last_update_epoch.fill_(-1)
        self.class_counts.fill_(1.0)
        self.geometric_scores.zero_()

    @torch.no_grad()
    def set_class_counts(self, counts) -> None:
        if counts is None:
            return
        try:
            values = np.asarray(counts, dtype=np.float32)
        except (TypeError, ValueError):
            return
        if values.size != self.num_classes:
            return
        tensor = torch.from_numpy(values).to(self.f_n.device, dtype=torch.float32) + 1.0
        self.class_counts.copy_(tensor)

    @torch.no_grad()
    def set_geometric_scores(self, scores) -> None:
        """Set per-class difficulty aggregated from the Class Similarity Graph."""
        if scores is None:
            return
        try:
            values = torch.as_tensor(scores, device=self.f_n.device, dtype=torch.float32).flatten()
        except (TypeError, ValueError, RuntimeError):
            return
        if values.numel() == self.num_classes:
            self.geometric_scores.copy_(values.clamp_min(0.0))

    @torch.no_grad()
    def _count_blend(self, epoch: int) -> float:
        if self.count_transition_epochs <= 0:
            return 0.0
        if epoch < self.warmup_epochs:
            return self.count_weight_start

        progress = (epoch - self.warmup_epochs) / max(1, self.count_transition_epochs)
        progress = min(max(progress, 0.0), 1.0)
        slope = 10.0
        value = 1.0 / (1.0 + math.exp(-slope * (progress - 0.5)))
        start = 1.0 / (1.0 + math.exp(slope * 0.5))
        end = 1.0 / (1.0 + math.exp(-slope * 0.5))
        normalized = (value - start) / max(1e-6, end - start)
        return float(self.count_weight_start * (1.0 - normalized))

    @torch.no_grad()
    def weight_gather(self, class_ids: torch.Tensor) -> torch.Tensor:
        if class_ids is None or class_ids.numel() == 0:
            return torch.ones_like(class_ids, dtype=torch.float32, device=self.f_n.device)
        return self.f_n.index_select(0, class_ids.to(self.f_n.device)).to(dtype=torch.float32)

    @torch.no_grad()
    def maybe_update(self, per_class_ap, epoch: int, logger=None, class_counts=None) -> None:
        prefix = "[GDR]"
        if per_class_ap is None:
            if logger:
                logger.info(f"{prefix} epoch {epoch}: per-class AP unavailable; skipping update")
            return

        try:
            ap = torch.as_tensor(per_class_ap, device=self.f_n.device, dtype=torch.float32).flatten()
        except (TypeError, ValueError, RuntimeError):
            if logger:
                logger.info(f"{prefix} epoch {epoch}: invalid per-class AP; skipping update")
            return
        if ap.numel() != self.num_classes:
            if logger:
                logger.info(f"{prefix} epoch {epoch}: AP length {ap.numel()} != {self.num_classes}; skipping update")
            return
        if epoch < self.warmup_epochs:
            if logger:
                logger.info(f"{prefix} epoch {epoch}: warmup")
            return

        update_frequency = self.update_freq_front if epoch < self.switch_epoch else self.update_freq_back
        if update_frequency <= 0 or epoch - int(self.last_update_epoch.item()) < update_frequency:
            return

        self.smooth_ap.copy_(self.mu * self.smooth_ap + (1.0 - self.mu) * ap)
        inverse_ap = 1.0 / (self.smooth_ap + self.eps)
        inverse_ap = inverse_ap / (inverse_ap.mean() + self.eps)
        ap_weights = inverse_ap.clamp(min=self.f_min, max=self.f_max)

        if self.geometric_scores.max() > self.eps:
            normalized_geometry = self.geometric_scores / (self.geometric_scores.mean() + self.eps)
            geometry_weights = (1.0 + normalized_geometry).clamp(min=self.f_min, max=self.f_max)
            difficulty_weights = (
                self.geometry_weight * geometry_weights + (1.0 - self.geometry_weight) * ap_weights
            )
        else:
            difficulty_weights = ap_weights

        counts = None
        if class_counts is not None:
            try:
                values = np.asarray(class_counts, dtype=np.float32)
                if values.size == self.num_classes:
                    counts = torch.from_numpy(values + 1.0).to(self.f_n.device, dtype=torch.float32)
            except (TypeError, ValueError):
                counts = None
        if counts is None:
            counts = self.class_counts.to(self.f_n.device, dtype=torch.float32)

        log_counts = torch.log(counts + self.eps)
        maximum, minimum = torch.max(log_counts), torch.min(log_counts)
        scarcity = (maximum - log_counts) / (maximum - minimum).clamp_min(self.eps)
        anchor = torch.quantile(scarcity, q=self._anchor_q).clamp_min(0.05)
        magnitude = (self._weight_anchor - 1.0) / (anchor**self._tau)
        magnitude_cap = max(0.0, self.f_max - 1.0 - 1e-6)
        magnitude = float(max(0.0, min(magnitude, magnitude_cap)))
        count_weights = (1.0 + magnitude * scarcity**self._tau).clamp(min=1.0, max=self.f_max)

        count_ratio = self._count_blend(epoch)
        updated = count_ratio * count_weights + (1.0 - count_ratio) * difficulty_weights
        self.f_n.copy_(self.mu * self.f_n + (1.0 - self.mu) * updated)
        self.last_update_epoch.fill_(int(epoch))
        self._broadcast_buffers()

        if logger:
            logger.info(
                f"{prefix} epoch {epoch}: weights={self.f_n.detach().cpu().tolist()} "
                f"(count_ratio={count_ratio:.3f})"
            )

    @torch.no_grad()
    def _broadcast_buffers(self) -> None:
        if not _distributed_is_ready():
            return
        for name in ("f_n", "smooth_ap", "last_update_epoch", "class_counts", "geometric_scores"):
            torch.distributed.broadcast(getattr(self, name), src=0)

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, warmup={self.warmup_epochs}, "
            f"freq_front={self.update_freq_front}, freq_back={self.update_freq_back}, "
            f"switch_epoch={self.switch_epoch}, mu={self.mu}, "
            f"clip=[{self.f_min},{self.f_max}], count_start={self.count_weight_start}, "
            f"transition_epochs={self.count_transition_epochs}, geometry_weight={self.geometry_weight}"
        )
