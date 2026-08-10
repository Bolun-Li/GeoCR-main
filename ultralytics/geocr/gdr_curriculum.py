"""Relation-aware curriculum augmentation used inside GDR.

It maintains a class similarity graph together with foreground and
background-environment prototypes, then schedules easy-to-hard sample
construction across training stages.
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Iterable

import cv2
import numpy as np
import torch

from ultralytics.utils.metrics import box_iou


class GDRCurriculumConfig:
    """Validated configuration for the relation-aware GDR curriculum."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.enabled = bool(config.get("enabled", True))
        self.num_target_classes = int(config.get("num_target_classes", 1))
        self.stage_end_epochs = list(config.get("stage_end_epochs", [30, 240, 300]))
        self.branch_ratios = list(
            config.get(
                "branch_ratios",
                [
                    [0.0, 0.0, 0.0],
                    [0.6, 0.3, 0.1],
                    [0.2, 0.3, 0.5],
                ],
            )
        )
        self.max_paste_per_image = int(config.get("max_paste_per_image", 6))
        self.max_recall_paste_per_image = int(
            config.get("max_recall_paste_per_image", config.get("max_paste_per_image_leak", 3))
        )
        self.apply_prob = float(config.get("apply_prob", 1.0))
        self.min_patch_size = int(config.get("min_patch_size", 16))
        self.paste_scale_choices = list(config.get("paste_scale_choices", [0.35, 0.5, 0.7, 1.0]))
        self.paste_scale_probs = list(config.get("paste_scale_probs", [0.4, 0.3, 0.2, 0.1]))
        self.avoid_iou_thr = float(config.get("avoid_iou_thr", 0.2))
        self.near_gt_bias = bool(config.get("near_gt_bias", True))
        self.use_light_blur = bool(config.get("use_light_blur", True))
        self.use_low_contrast = bool(config.get("use_low_contrast", True))
        self.use_haze_like = bool(config.get("use_haze_like", True))
        self.use_alpha_blend = bool(config.get("use_alpha_blend", True))
        self.alpha_range = list(config.get("alpha_range", [0.7, 0.9]))
        self.bank_size_per_class = int(config.get("bank_size_per_class", 300))
        self.bank_target_short_side = int(config.get("bank_target_short_side", 96))
        self.conf_fp_bg_min = float(config.get("conf_fp_bg_min", 0.35))
        self.graph_conf_threshold = float(config.get("graph_conf_threshold", 0.25))
        self.graph_nms_iou_threshold = float(config.get("graph_nms_iou_threshold", 0.6))
        self.graph_match_iou_threshold = float(config.get("graph_match_iou_threshold", 0.5))
        self.ema_beta = float(config.get("ema_beta", 0.995))
        self.eps = float(config.get("eps", 1e-6))

        if len(self.stage_end_epochs) != 3:
            raise ValueError("stage_end_epochs must contain [warmup_end, middle_end, final_end]")
        if len(self.branch_ratios) != 3:
            raise ValueError("branch_ratios must contain one triple for each of the three stages")
        if len(self.paste_scale_choices) != len(self.paste_scale_probs):
            raise ValueError("paste_scale_choices and paste_scale_probs must have the same length")

        normalized = []
        for ratios in self.branch_ratios:
            if len(ratios) != 3:
                raise ValueError("each branch-ratio entry must be [recall, foreground, background]")
            ratios = [float(value) for value in ratios]
            total = sum(ratios)
            normalized.append([0.0, 0.0, 0.0] if total <= 1e-12 else [value / total for value in ratios])
        self.branch_ratios = normalized

        scale_prob_sum = float(sum(self.paste_scale_probs))
        if scale_prob_sum <= 0:
            raise ValueError("paste_scale_probs must have a positive sum")
        self.paste_scale_probs = [float(value) / scale_prob_sum for value in self.paste_scale_probs]

    def current_stage(self, epoch: int) -> int:
        """Return the zero-based curriculum stage for an epoch."""
        if epoch < self.stage_end_epochs[0]:
            return 0
        if epoch < self.stage_end_epochs[1]:
            return 1
        return 2

    def current_ratios(self, epoch: int) -> tuple[float, float, float]:
        """Return recall, foreground-confusion, and hard-background ratios."""
        return tuple(self.branch_ratios[self.current_stage(epoch)])


class ClassSimilarityGraph:
    """EMA class similarity graph with an additional background node."""

    def __init__(self, num_classes: int, ema_beta: float = 0.995, eps: float = 1e-6):
        self.num_classes = int(num_classes)
        self.background_index = self.num_classes
        self.num_nodes = self.num_classes + 1
        self.eps = float(eps)
        self.ema_beta = float(ema_beta)
        self.matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float64)
        self._initialized = False

    def update(self, pairs: Iterable[tuple[int, int]] = (), counts: Iterable[tuple[int, int, float]] | None = None):
        """Update row-normalized ground-truth-to-prediction relations."""
        local = np.zeros_like(self.matrix)
        if counts is not None:
            for ground_truth, prediction, count in counts:
                local[int(ground_truth), int(prediction)] += float(count)
        else:
            for ground_truth, prediction in pairs:
                local[int(ground_truth), int(prediction)] += 1.0

        local /= local.sum(axis=1, keepdims=True) + self.eps
        if not self._initialized:
            self.matrix = local
            self._initialized = True
        else:
            self.matrix = self.ema_beta * self.matrix + (1.0 - self.ema_beta) * local

    def false_negative_scores(self) -> np.ndarray:
        """Return per-class c-to-background relation scores."""
        diagonal = np.clip(np.diag(self.matrix)[: self.num_classes], self.eps, None)
        return self.matrix[: self.num_classes, self.background_index] / diagonal

    def false_positive_scores(self) -> np.ndarray:
        """Return per-class background-to-c relation scores."""
        background_row = self.matrix[self.background_index]
        denominator = max(background_row[self.background_index], self.eps)
        return background_row[: self.num_classes] / denominator

    def top_classes(self, scores: np.ndarray, count: int) -> list[int]:
        """Select the highest-scoring valid classes."""
        count = max(1, min(int(count), self.num_classes))
        return np.argsort(-scores)[:count].tolist()

    def difficulty_scores(self) -> np.ndarray:
        """Aggregate inter-class confusion and foreground-background ambiguity."""
        foreground_rows = self.matrix[: self.num_classes]
        correct = np.diag(self.matrix)[: self.num_classes]
        outgoing_confusion = foreground_rows.sum(axis=1) - correct
        background_confusion = self.matrix[self.background_index, : self.num_classes]
        return np.maximum(outgoing_confusion + background_confusion, 0.0).astype(np.float32)


class PrototypeBank:
    """FIFO bank of spatial prototypes."""

    def __init__(self, num_classes: int, capacity: int):
        self.num_classes = int(num_classes)
        self.buckets = [deque(maxlen=int(capacity)) for _ in range(self.num_classes)]

    def push(self, class_id: int, patch: np.ndarray, original_size: tuple[int, int]):
        """Store a patch for a valid class."""
        if 0 <= int(class_id) < self.num_classes:
            self.buckets[int(class_id)].append((patch, original_size))

    def get(self, class_id: int):
        """Sample one patch for a class, or return None when its bank is empty."""
        if 0 <= int(class_id) < self.num_classes and self.buckets[int(class_id)]:
            return random.choice(tuple(self.buckets[int(class_id)]))
        return None

    def size(self, class_id: int) -> int:
        """Return the number of stored patches for a class."""
        return len(self.buckets[int(class_id)]) if 0 <= int(class_id) < self.num_classes else 0


class ForegroundPrototypeBank(PrototypeBank):
    """Foreground prototypes indexed by object class."""


class BackgroundEnvironmentPrototypeBank(PrototypeBank):
    """Background-environment prototypes indexed by the class they resemble."""


def _choice_with_probabilities(choices, probabilities):
    return choices[np.random.choice(len(choices), p=np.asarray(probabilities, dtype=np.float64))]


def _alpha_blend(destination: np.ndarray, source: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    selected = (mask > 0)[:, :, None]
    blended = alpha * destination.astype(np.float32) + (1.0 - alpha) * source.astype(np.float32)
    return np.where(selected, blended, destination).astype(destination.dtype)


def _light_degrade(image: np.ndarray, config: GDRCurriculumConfig) -> np.ndarray:
    output = image.copy()
    if config.use_low_contrast:
        output = cv2.convertScaleAbs(output, alpha=np.random.uniform(0.8, 1.0), beta=np.random.uniform(-8, 8))
    if config.use_light_blur and random.random() < 0.5:
        kernel = int(np.random.choice([3, 5]))
        output = cv2.GaussianBlur(output, (kernel, kernel), 0)
    if config.use_haze_like and random.random() < 0.5:
        height, width = output.shape[:2]
        noise = cv2.resize(np.random.randn(height // 16 + 1, width // 16 + 1).astype(np.float32), (width, height))
        haze = ((noise - noise.min()) / (noise.max() - noise.min() + 1e-6) * 25).astype(np.uint8)
        output = cv2.add(output, cv2.merge([haze, haze, haze]))
    return output


def _place_patch(
    image: np.ndarray,
    base_xywh: tuple[int, int, int, int],
    patch: np.ndarray,
    scale: float,
    config: GDRCurriculumConfig,
    ground_truth_boxes: np.ndarray,
):
    height, width = image.shape[:2]
    patch_height, patch_width = patch.shape[:2]
    if height < 2 or width < 2 or patch_height < 1 or patch_width < 1:
        return None, None

    scale = min(float(scale), (width - 1) / max(float(patch_width), 1.0), (height - 1) / max(float(patch_height), 1.0))
    if scale <= 0:
        return None, None
    new_width = max(int(patch_width * scale), 1)
    new_height = max(int(patch_height * scale), 1)
    resized = cv2.resize(patch, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    center_x, center_y, base_width, base_height = base_xywh
    attempts = 20 if config.near_gt_bias else 50
    radius = max(int(0.5 * max(base_width, base_height)), 1)
    for _ in range(attempts):
        if config.near_gt_bias:
            x1 = np.clip(int(center_x - new_width // 2 + np.random.randint(-radius, radius)), 0, width - new_width)
            y1 = np.clip(int(center_y - new_height // 2 + np.random.randint(-radius, radius)), 0, height - new_height)
        else:
            x1 = np.random.randint(0, width - new_width + 1)
            y1 = np.random.randint(0, height - new_height + 1)
        candidate = np.array([x1, y1, x1 + new_width, y1 + new_height], dtype=np.float32)
        if ground_truth_boxes is None or len(ground_truth_boxes) == 0:
            return resized, (int(x1), int(y1), new_width, new_height)
        iou = box_iou(torch.as_tensor(candidate)[None], torch.as_tensor(ground_truth_boxes)).numpy().max()
        if iou < config.avoid_iou_thr:
            return resized, (int(x1), int(y1), new_width, new_height)
    return None, None


class EnvironmentGuidedCurriculum:
    """Three-branch relation-aware augmentation used by GDR."""

    def __init__(
        self,
        config: GDRCurriculumConfig,
        similarity_graph: ClassSimilarityGraph,
        foreground_prototypes: ForegroundPrototypeBank,
        background_environment_prototypes: BackgroundEnvironmentPrototypeBank,
        names,
    ):
        self.config = config
        self.similarity_graph = similarity_graph
        self.foreground_prototypes = foreground_prototypes
        self.background_environment_prototypes = background_environment_prototypes
        self.names = names
        self.num_classes = len(names)

    def __call__(self, image: np.ndarray, boxes_xyxy: np.ndarray, labels: np.ndarray, epoch: int):
        if not self.config.enabled or random.random() > self.config.apply_prob:
            return image, boxes_xyxy, labels

        recall_ratio, foreground_ratio, background_ratio = self.config.current_ratios(epoch)
        boxes = boxes_xyxy.astype(np.float32)
        classes = labels.astype(np.int64)

        selected_recall = self.similarity_graph.top_classes(
            self.similarity_graph.false_negative_scores(), self.config.num_target_classes
        )
        selected_background = self.similarity_graph.top_classes(
            self.similarity_graph.false_positive_scores(), self.config.num_target_classes
        )
        total_budget = self.config.max_paste_per_image
        recall_budget = min(self.config.max_recall_paste_per_image, round(total_budget * recall_ratio))
        foreground_budget = round(total_budget * foreground_ratio)
        background_budget = round(total_budget * background_ratio)
        used = 0

        for _ in range(recall_budget):
            if used >= total_budget:
                break
            candidates = [index for index, class_id in enumerate(classes) if class_id in selected_recall]
            if not candidates:
                break
            index = random.choice(candidates)
            class_id = int(classes[index])
            x1, y1, x2, y2 = boxes[index]
            if min(int(x2 - x1), int(y2 - y1)) < self.config.min_patch_size:
                continue
            item = self.foreground_prototypes.get(class_id)
            if item is None:
                continue
            patch, _ = item
            if self.config.use_low_contrast or self.config.use_light_blur or self.config.use_haze_like:
                patch = _light_degrade(patch, self.config)
            scale = _choice_with_probabilities(self.config.paste_scale_choices, self.config.paste_scale_probs)
            placed, position = _place_patch(
                image,
                (int((x1 + x2) / 2), int((y1 + y2) / 2), int(x2 - x1), int(y2 - y1)),
                patch,
                scale,
                self.config,
                boxes,
            )
            if placed is None:
                continue
            px, py, patch_width, patch_height = position
            region = image[py : py + patch_height, px : px + patch_width].copy()
            if self.config.use_alpha_blend:
                alpha = np.random.uniform(*self.config.alpha_range)
                mask = np.full((patch_height, patch_width), 255, dtype=np.uint8)
                placed = _alpha_blend(region, placed, mask, alpha)
            image[py : py + patch_height, px : px + patch_width] = placed
            boxes = np.vstack((boxes, np.array([px, py, px + patch_width, py + patch_height], dtype=np.float32)))
            classes = np.hstack((classes, class_id))
            used += 1

        for _ in range(foreground_budget):
            if used >= total_budget or len(classes) == 0:
                break
            index = np.random.randint(0, len(classes))
            base_class = int(classes[index])
            x1, y1, x2, y2 = boxes[index]
            if min(int(x2 - x1), int(y2 - y1)) < self.config.min_patch_size:
                continue
            probabilities = self.similarity_graph.matrix[:, base_class].copy()[: self.num_classes]
            probabilities[base_class] = 0.0
            if probabilities.sum() <= 1e-9:
                continue
            probabilities /= probabilities.sum()
            mixed_class = int(np.random.choice(np.arange(self.num_classes), p=probabilities))
            item = self.foreground_prototypes.get(mixed_class)
            if item is None:
                continue
            patch, _ = item
            scale = _choice_with_probabilities(self.config.paste_scale_choices, self.config.paste_scale_probs)
            placed, position = _place_patch(
                image,
                (int((x1 + x2) / 2), int((y1 + y2) / 2), int(x2 - x1), int(y2 - y1)),
                patch,
                scale,
                self.config,
                boxes,
            )
            if placed is None:
                continue
            px, py, patch_width, patch_height = position
            image[py : py + patch_height, px : px + patch_width] = placed
            boxes = np.vstack((boxes, np.array([px, py, px + patch_width, py + patch_height], dtype=np.float32)))
            classes = np.hstack((classes, mixed_class))
            used += 1

        for _ in range(background_budget):
            if used >= total_budget:
                break
            candidates = [class_id for class_id in set(classes.tolist()) if class_id in selected_background]
            if not candidates:
                break
            class_id = random.choice(candidates)
            item = self.background_environment_prototypes.get(class_id)
            if item is None:
                continue
            patch, _ = item
            matching = [index for index, current in enumerate(classes) if current == class_id]
            if not matching:
                continue
            index = random.choice(matching)
            x1, y1, x2, y2 = boxes[index]
            if min(int(x2 - x1), int(y2 - y1)) < self.config.min_patch_size:
                continue
            scale = _choice_with_probabilities(self.config.paste_scale_choices, self.config.paste_scale_probs)
            placed, position = _place_patch(
                image,
                (int((x1 + x2) / 2), int((y1 + y2) / 2), int(x2 - x1), int(y2 - y1)),
                patch,
                scale,
                self.config,
                boxes,
            )
            if placed is None:
                continue
            px, py, patch_width, patch_height = position
            image[py : py + patch_height, px : px + patch_width] = placed
            used += 1

        return image, boxes, classes
