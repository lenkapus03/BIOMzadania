"""
face_detection/hog_detector.py - Face detection using Dlib HOG+SVM (classic, non-CNN).

Architecture:
    Dlib's frontal face detector (King, 2009) uses a sliding window approach:
        1. HOG (Histogram of Oriented Gradients) features are extracted from
           each window — the image is divided into cells, gradient orientations
           are binned into a histogram per cell, and neighbouring cells are
           grouped into blocks for normalisation
        2. A linear SVM classifier trained on these HOG features predicts
           whether a window contains a face
        3. An image pyramid (multiple resized copies) allows detection at
           different scales; NMS removes duplicate detections

Training database:
    Trained on a subset of the Labeled Faces in the Wild (LFW) dataset
    plus additional data curated by the dlib authors.

Hyperparameter:
    upsample — number of times the image is upsampled before detection.
    upsample=0: fastest, may miss small/distant faces.
    upsample=1: default — good balance of speed and recall.
    upsample=2: finds smaller faces but significantly slower.
    Tuned visually on a sample of videos; default 1 is adequate for
    the face sizes present in the YoutubeFaces dataset.
"""

from __future__ import annotations

import numpy as np
import dlib


# ----- hyperparameter -----
UPSAMPLE = 1    # number of upsampling passes; increase for small faces


def detect(frame_bgr: np.ndarray, detector: dlib.fhog_object_detector, upsample: int = UPSAMPLE) -> list[tuple[int, int, int, int]]:
    """
    Detect faces in a BGR frame using Dlib HOG+SVM.

    Args:
        frame_bgr: (H, W, 3) uint8 BGR image
        detector:  dlib frontal face detector instance (initialized once in main)
        upsample:  number of times to upsample the image before detection

    Returns:
        List of (x1, y1, x2, y2) bounding boxes clipped to image bounds
    """
    import cv2
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    H, W = frame_rgb.shape[:2]
    dets = detector(frame_rgb, upsample)

    boxes = []
    for d in dets:
        x1 = max(0, d.left())
        y1 = max(0, d.top())
        x2 = min(W, d.right())
        y2 = min(H, d.bottom())
        boxes.append((x1, y1, x2, y2))

    return boxes