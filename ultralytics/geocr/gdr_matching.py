"""Detection-to-ground-truth matching for the GDR class similarity graph."""

from __future__ import annotations

import torch
import torchvision

from ultralytics.utils.metrics import box_iou


@torch.no_grad()
def collect_similarity_events(
    predicted_boxes: torch.Tensor,
    predicted_scores: torch.Tensor,
    ground_truth_labels: torch.Tensor,
    ground_truth_boxes: torch.Tensor,
    ground_truth_mask: torch.Tensor,
    image_shape: tuple[int, int],
    confidence_threshold: float = 0.25,
    nms_iou_threshold: float = 0.6,
    match_iou_threshold: float = 0.5,
    max_detections: int = 300,
) -> list[dict]:
    """Match decoded detections to GT and return graph/prototype update events."""
    events = []
    max_coordinate = float(max(image_shape) * 2)
    for image_index in range(predicted_boxes.shape[0]):
        valid_gt = ground_truth_mask[image_index].reshape(-1).bool()
        gt_boxes = ground_truth_boxes[image_index][valid_gt]
        gt_classes = ground_truth_labels[image_index][valid_gt].reshape(-1).long()

        scores, classes = predicted_scores[image_index].max(dim=-1)
        candidates = scores >= float(confidence_threshold)
        boxes = predicted_boxes[image_index][candidates]
        scores = scores[candidates]
        classes = classes[candidates]
        if boxes.numel():
            offsets = classes.to(boxes.dtype).unsqueeze(1) * max_coordinate
            selected = torchvision.ops.nms(boxes + offsets, scores, float(nms_iou_threshold))[:max_detections]
            boxes, scores, classes = boxes[selected], scores[selected], classes[selected]

        matched_gt, matched_predictions = set(), set()
        matches = []
        if gt_boxes.numel() and boxes.numel():
            overlaps = box_iou(gt_boxes, boxes)
            for flat_index in overlaps.flatten().argsort(descending=True).tolist():
                gt_index = flat_index // overlaps.shape[1]
                prediction_index = flat_index % overlaps.shape[1]
                overlap = float(overlaps[gt_index, prediction_index])
                if overlap < match_iou_threshold:
                    break
                if gt_index in matched_gt or prediction_index in matched_predictions:
                    continue
                matched_gt.add(gt_index)
                matched_predictions.add(prediction_index)
                matches.append(
                    {
                        "ground_truth_class": int(gt_classes[gt_index]),
                        "predicted_class": int(classes[prediction_index]),
                        "iou": overlap,
                    }
                )

        false_negatives = [
            {"class_id": int(gt_classes[index]), "box": gt_boxes[index].detach().cpu().tolist()}
            for index in range(len(gt_boxes))
            if index not in matched_gt
        ]
        false_positives = [
            {
                "class_id": int(classes[index]),
                "box": boxes[index].detach().cpu().tolist(),
                "confidence": float(scores[index]),
            }
            for index in range(len(boxes))
            if index not in matched_predictions
        ]
        events.append(
            {
                "matches": matches,
                "false_negatives": false_negatives,
                "false_positives": false_positives,
            }
        )
    return events
