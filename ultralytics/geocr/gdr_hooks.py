"""Training hooks for the relation-aware curriculum inside GDR."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

import yaml
from ultralytics.utils import LOGGER, colorstr

from .gdr_curriculum import (
    BackgroundEnvironmentPrototypeBank,
    ClassSimilarityGraph,
    EnvironmentGuidedCurriculum,
    ForegroundPrototypeBank,
    GDRCurriculumConfig,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "cfg" / "geocr" / "gdr_curriculum.yaml"


def _xywhn_to_xyxy_abs(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    converted = boxes.copy()
    if np.nanmax(converted) <= 2.0:
        converted[:, [0, 2]] *= float(width)
        converted[:, [1, 3]] *= float(height)
        output = np.empty_like(converted)
        output[:, 0] = converted[:, 0] - converted[:, 2] / 2.0
        output[:, 1] = converted[:, 1] - converted[:, 3] / 2.0
        output[:, 2] = converted[:, 0] + converted[:, 2] / 2.0
        output[:, 3] = converted[:, 1] + converted[:, 3] / 2.0
    else:
        output = converted
    output[:, [0, 2]] = np.clip(output[:, [0, 2]], 0, width)
    output[:, [1, 3]] = np.clip(output[:, [1, 3]], 0, height)
    return output.astype(np.float32)


def _xyxy_abs_to_xywhn(boxes: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return boxes.reshape(0, 4), np.zeros((0,), dtype=bool)
    clipped = boxes.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, width)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, height)
    box_width = clipped[:, 2] - clipped[:, 0]
    box_height = clipped[:, 3] - clipped[:, 1]
    valid = (box_width >= 1.0) & (box_height >= 1.0)
    output = np.zeros_like(clipped, dtype=np.float32)
    output[:, 0] = ((clipped[:, 0] + clipped[:, 2]) / 2.0) / max(float(width), 1.0)
    output[:, 1] = ((clipped[:, 1] + clipped[:, 3]) / 2.0) / max(float(height), 1.0)
    output[:, 2] = box_width / max(float(width), 1.0)
    output[:, 3] = box_height / max(float(height), 1.0)
    return np.clip(output, 0.0, 1.0).astype(np.float32), valid


def on_train_start(trainer) -> None:
    """Create the GDR similarity graph, prototypes, and environment-guided curriculum."""
    if getattr(trainer.args, "task", "detect") != "detect" or getattr(trainer, "gdr", None) is None:
        trainer.gdr_curriculum_enabled = False
        return

    config_path = Path(getattr(trainer.args, "gdr_curriculum_cfg", DEFAULT_CONFIG_PATH))
    if not config_path.is_file() and config_path.name == DEFAULT_CONFIG_PATH.name:
        config_path = DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        LOGGER.warning("[GDR] curriculum config not found at '%s'; curriculum disabled", config_path)
        trainer.gdr_curriculum_enabled = False
        return
    with config_path.open("r", encoding="utf-8") as stream:
        config = GDRCurriculumConfig(yaml.safe_load(stream) or {})

    names = getattr(trainer.model, "names", None)
    if names is None:
        LOGGER.warning("[GDR] model class names unavailable; curriculum disabled")
        trainer.gdr_curriculum_enabled = False
        return
    num_classes = len(names)
    similarity_graph = ClassSimilarityGraph(num_classes, config.ema_beta, config.eps)
    foreground_prototypes = ForegroundPrototypeBank(num_classes, config.bank_size_per_class)
    background_environment_prototypes = BackgroundEnvironmentPrototypeBank(
        num_classes, config.bank_size_per_class
    )
    curriculum = EnvironmentGuidedCurriculum(
        config,
        similarity_graph,
        foreground_prototypes,
        background_environment_prototypes,
        names,
    )

    trainer.gdr_curriculum_config = config
    trainer.gdr_similarity_graph = similarity_graph
    trainer.gdr_foreground_prototypes = foreground_prototypes
    trainer.gdr_background_environment_prototypes = background_environment_prototypes
    trainer.gdr_curriculum = curriculum
    trainer.gdr_curriculum_enabled = bool(config.enabled)
    base_model = getattr(trainer.model, "module", trainer.model)
    base_model.gdr_collect_similarity_events = trainer.gdr_curriculum_enabled
    base_model.gdr_event_confidence_threshold = config.graph_conf_threshold
    base_model.gdr_event_nms_iou_threshold = config.graph_nms_iou_threshold
    base_model.gdr_event_match_iou_threshold = config.graph_match_iou_threshold
    status = "enabled" if trainer.gdr_curriculum_enabled else "disabled"
    LOGGER.info(
        "%s%s | classes=%d | config='%s'",
        colorstr("GDR curriculum: "),
        status,
        num_classes,
        config_path,
    )


def on_preprocess_batch(trainer) -> None:
    """Apply the GDR curriculum to images and labels before normalization."""
    if not getattr(trainer, "gdr_curriculum_enabled", False):
        return
    batch = getattr(trainer, "batch", None)
    if batch is None or not all(key in batch for key in ("img", "bboxes", "cls", "batch_idx")):
        return

    images = batch["img"]
    boxes = batch["bboxes"]
    classes = batch["cls"]
    batch_indices = batch["batch_idx"]
    image_device, box_device = images.device, boxes.device
    class_device, index_device = classes.device, batch_indices.device
    class_dtype, index_dtype = classes.dtype, batch_indices.dtype
    classes_are_column = classes.ndim == 2

    uint8_images = images if images.dtype == torch.uint8 else (images * 255.0).clamp(0, 255).to(torch.uint8)
    images_numpy = uint8_images.permute(0, 2, 3, 1).cpu().numpy()
    boxes_numpy = boxes.detach().cpu().numpy().astype(np.float32).reshape(-1, 4)
    classes_numpy = classes.detach().cpu().numpy().astype(np.int64).reshape(-1)
    indices_numpy = batch_indices.detach().cpu().numpy().astype(np.int64).reshape(-1)
    batch_size, height, width = images_numpy.shape[:3]

    grouped = [[] for _ in range(batch_size)]
    for batch_index, box, class_id in zip(indices_numpy, boxes_numpy, classes_numpy):
        if 0 <= batch_index < batch_size:
            absolute = _xywhn_to_xyxy_abs(box[None], height, width)[0]
            if absolute[2] > absolute[0] and absolute[3] > absolute[1]:
                grouped[batch_index].append((absolute, class_id))

    output_images, output_boxes, output_classes, output_indices = [], [], [], []
    epoch = getattr(trainer, "epoch", 0)
    for batch_index in range(batch_size):
        if grouped[batch_index]:
            image_boxes = np.asarray([item[0] for item in grouped[batch_index]], dtype=np.float32)
            image_classes = np.asarray([item[1] for item in grouped[batch_index]], dtype=np.int64)
        else:
            image_boxes = np.zeros((0, 4), dtype=np.float32)
            image_classes = np.zeros((0,), dtype=np.int64)
        output_image, image_boxes, image_classes = trainer.gdr_curriculum(
            images_numpy[batch_index], image_boxes, image_classes, epoch
        )
        output_images.append(output_image[None])
        normalized, valid = _xyxy_abs_to_xywhn(image_boxes, height, width)
        if valid.any():
            output_boxes.append(normalized[valid])
            output_classes.append(image_classes.reshape(-1)[valid])
            output_indices.append(np.full(int(valid.sum()), batch_index, dtype=np.int64))

    batch["img"] = (
        torch.from_numpy(np.concatenate(output_images)).permute(0, 3, 1, 2).to(image_device).type_as(images)
    )
    if output_boxes:
        class_output = np.concatenate(output_classes).astype(np.int64)
        if classes_are_column:
            class_output = class_output.reshape(-1, 1)
        batch["bboxes"] = torch.from_numpy(np.concatenate(output_boxes)).to(box_device, dtype=boxes.dtype)
        batch["cls"] = torch.from_numpy(class_output).to(class_device, dtype=class_dtype)
        batch["batch_idx"] = torch.from_numpy(np.concatenate(output_indices)).to(index_device, dtype=index_dtype)
    else:
        batch["bboxes"] = torch.zeros((0, 4), device=box_device, dtype=boxes.dtype)
        batch["cls"] = torch.zeros((0, 1) if classes_are_column else (0,), device=class_device, dtype=class_dtype)
        batch["batch_idx"] = torch.zeros((0,), device=index_device, dtype=index_dtype)


def on_train_batch_end(trainer) -> None:
    """Update the similarity graph and prototypes from matched detections."""
    if not getattr(trainer, "gdr_curriculum_enabled", False):
        return
    batch = getattr(trainer, "batch", None)
    if batch is None or "img" not in batch:
        return

    images = batch["img"]
    uint8_images = images if images.dtype == torch.uint8 else (images * 255.0).clamp(0, 255).to(torch.uint8)
    images_numpy = uint8_images.permute(0, 2, 3, 1).cpu().numpy()
    height, width = images_numpy.shape[1:3]
    config = trainer.gdr_curriculum_config

    if all(key in batch for key in ("bboxes", "cls", "batch_idx")):
        boxes = batch["bboxes"].detach().cpu().numpy().astype(np.float32).reshape(-1, 4)
        classes = batch["cls"].detach().cpu().numpy().astype(np.int32).reshape(-1)
        batch_indices = batch["batch_idx"].detach().cpu().numpy().astype(np.int32).reshape(-1)
        for batch_index, box, class_id in zip(batch_indices, boxes, classes):
            if not 0 <= batch_index < len(images_numpy):
                continue
            x1, y1, x2, y2 = [int(value) for value in _xywhn_to_xyxy_abs(box[None], height, width)[0]]
            crop = images_numpy[batch_index][max(y1, 0) : min(y2, height), max(x1, 0) : min(x2, width)].copy()
            if crop.size == 0 or min(crop.shape[:2]) < config.min_patch_size:
                continue
            original_height, original_width = crop.shape[:2]
            if min(original_height, original_width) != config.bank_target_short_side:
                scale = config.bank_target_short_side / float(min(original_height, original_width))
                crop = cv2.resize(
                    crop,
                    (max(1, round(original_width * scale)), max(1, round(original_height * scale))),
                    interpolation=cv2.INTER_LINEAR,
                )
            trainer.gdr_foreground_prototypes.push(int(class_id), crop, (original_height, original_width))

    base_model = getattr(trainer.model, "module", trainer.model)
    events = getattr(base_model, "gdr_similarity_events", ()) or ()
    event_error = getattr(base_model, "gdr_similarity_event_error", None)
    if event_error and not getattr(trainer, "_gdr_event_error_reported", False):
        LOGGER.warning("[GDR] similarity-event collection failed: %s", event_error)
        trainer._gdr_event_error_reported = True

    pairs = []
    background_index = trainer.gdr_similarity_graph.background_index
    for image_index, event in enumerate(events):
        pairs.extend(
            (int(match["ground_truth_class"]), int(match["predicted_class"]))
            for match in event.get("matches", ())
        )
        pairs.extend(
            (int(missed["class_id"]), background_index) for missed in event.get("false_negatives", ())
        )
        for false_positive in event.get("false_positives", ()):
            class_id = int(false_positive["class_id"])
            pairs.append((background_index, class_id))
            if float(false_positive["confidence"]) < config.conf_fp_bg_min or image_index >= len(images_numpy):
                continue
            x1, y1, x2, y2 = [round(value) for value in false_positive["box"]]
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, width), min(y2, height)
            prototype = images_numpy[image_index][y1:y2, x1:x2].copy()
            if prototype.size == 0 or min(prototype.shape[:2]) < 2:
                continue
            original_height, original_width = prototype.shape[:2]
            scale = config.bank_target_short_side / float(min(original_height, original_width))
            if abs(scale - 1.0) > 1e-6:
                prototype = cv2.resize(
                    prototype,
                    (max(1, round(original_width * scale)), max(1, round(original_height * scale))),
                    interpolation=cv2.INTER_LINEAR,
                )
            trainer.gdr_background_environment_prototypes.push(
                class_id, prototype, (original_height, original_width)
            )
    if pairs:
        trainer.gdr_similarity_graph.update(pairs)
    base_model.gdr_similarity_events = []
