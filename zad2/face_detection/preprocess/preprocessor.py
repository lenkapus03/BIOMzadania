"""
face_detection/preprocessor.py - Image preprocessing pipeline before face detection.

Pipeline (in order):
    1. Gaussian blur  — reduces high-frequency noise
    2. CLAHE          — adaptive contrast enhancement (local histogram equalization)
"""

import cv2
import numpy as np


USE_BLUR  = True
USE_CLAHE = True

BLUR_KERNEL      = (5, 5)   # must be odd; larger = more smoothing
CLAHE_CLIP_LIMIT = 3.0      # higher = more contrast boost
CLAHE_TILE_SIZE  = (8, 8)   # grid size for local histogram; smaller = more local


def gaussian_blur(frame: np.ndarray) -> np.ndarray:
    """Reduce noise with a Gaussian kernel."""
    return cv2.GaussianBlur(frame, BLUR_KERNEL, sigmaX=0)


def clahe(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Converts to LAB color space, applies CLAHE only to the L (lightness) channel,
    then converts back. This avoids shifting hue/saturation while boosting contrast.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe_obj = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    l_eq = clahe_obj.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Run the full preprocessing pipeline on a single BGR frame.
    """
    result = frame.copy()

    if USE_BLUR:
        result = gaussian_blur(result)
    if USE_CLAHE:
        result = clahe(result)

    return result