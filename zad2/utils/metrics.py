"""
utils/metrics.py - Detection metrics: Precision, Recall, F1 for IoU > threshold.

A detection is considered True Positive if IoU with the ground-truth box > iou_threshold.
One ground-truth box per frame is assumed (as in YoutubeFaces dataset).

Box formats:
    predicted:    (x1, y1, x2, y2) or None  — None means no detection
    ground_truth: (4, 2) corner array        — dataset format, converted internally
"""

from __future__ import annotations
import numpy as np


def iou(pred: tuple[int, int, int, int], gt_corners: np.ndarray) -> float:
    """
    Compute IoU between a predicted box (x1,y1,x2,y2) and a ground-truth
    box in (4,2) corner format.
    """
    px1, py1, px2, py2 = pred
    gx1 = int(gt_corners[:, 0].min())
    gy1 = int(gt_corners[:, 1].min())
    gx2 = int(gt_corners[:, 0].max())
    gy2 = int(gt_corners[:, 1].max())

    inter_x1 = max(px1, gx1)
    inter_y1 = max(py1, gy1)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    pred_area = max(0, px2 - px1) * max(0, py2 - py1)
    gt_area   = max(0, gx2 - gx1) * max(0, gy2 - gy1)
    union_area = pred_area + gt_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def evaluate_video(
    predicted_boxes: list[tuple[int, int, int, int] | None],
    gt_boxes:        np.ndarray,
    iou_threshold:   float = 0.5,
) -> dict:
    """
    Evaluate detections for a single video.

    Args:
        predicted_boxes: list of (x1,y1,x2,y2) or None per frame (N,)
        gt_boxes:        ground-truth boxes (N, 4, 2) corner format
        iou_threshold:   IoU threshold for TP classification

    Returns:
        dict with keys: tp, fp, fn, precision, recall, f1
    """
    tp = fp = fn = 0

    for pred, gt in zip(predicted_boxes, gt_boxes):
        if pred is None:
            fn += 1
        else:
            score = iou(pred, gt)
            if score > iou_threshold:
                tp += 1
            else:
                fp += 1
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def aggregate_metrics(per_video: list[dict]) -> dict:
    """
    Aggregate per-video TP/FP/FN counts into overall Precision, Recall, F1.

    Args:
        per_video: list of dicts returned by evaluate_video()

    Returns:
        dict with overall precision, recall, f1 and summed tp, fp, fn
    """
    tp = sum(v["tp"] for v in per_video)
    fp = sum(v["fp"] for v in per_video)
    fn = sum(v["fn"] for v in per_video)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}