"""
face_detection/mtcnn_detector.py - Face detection using MTCNN (CNN-based).

Architecture:
    MTCNN (Multi-task Cascaded Convolutional Networks, Zhang et al. 2016)
    is a three-stage cascaded CNN detector:
        P-Net  — fully convolutional proposal network; runs as a sliding window
                 over an image pyramid to generate face candidates at multiple scales
        R-Net  — refines candidates from P-Net (24×24 crops), filters weak detections
        O-Net  — final high-precision stage (48×48 crops), outputs bounding boxes
                 + 5 facial landmarks + confidence score

Training database:
    WIDER FACE (face bounding boxes) + CelebA (facial landmark annotations)

Hyperparameter:
    min_confidence — minimum detection confidence score to accept a bounding box.
    Range is [0, 1]; higher = more strict (fewer false positives, may miss faces).
    Tuned visually on a sample of videos; default 0.90 was selected as a good
    balance between precision and recall on this dataset.
"""

from __future__ import annotations

import numpy as np
from mtcnn import MTCNN


# ----- hyperparameter -----
MIN_CONFIDENCE = 0.90


def detect(frame_bgr: np.ndarray, detector: MTCNN, min_confidence: float = MIN_CONFIDENCE) -> list[tuple[int, int, int, int]]:
    """
    Detect faces in a BGR frame using MTCNN.

    Args:
        frame_bgr:      (H, W, 3) uint8 BGR image
        detector:       MTCNN instance (initialized once in main)
        min_confidence: minimum confidence score to keep a detection

    Returns:
        List of (x1, y1, x2, y2) bounding boxes clipped to image bounds
    """
    import cv2
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    H, W = frame_rgb.shape[:2]
    results = detector.detect_faces(frame_rgb)

    boxes = []
    for r in results:
        if r["confidence"] < min_confidence:
            continue
        x, y, w, h = r["box"]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(W, x + w)
        y2 = min(H, y + h)
        boxes.append((x1, y1, x2, y2))

    return boxes