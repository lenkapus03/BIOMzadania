from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoData:
    """
    One .npz video clip

    Raw .npz layout:
        colorImages  (H, W, 3, N)   -> frames          (N, H, W, 3)
        boundingBox  (4, 2, N)      -> boxes            (N, 4, 2)
        landmarks2D  (68, 2, N)     -> landmarks        (N, 68, 2)
        landmarks3D  (68, 3, N)     -> landmarks3d      (N, 68, 3)
        detectedBox  (N, 4)         -> detected_boxes   (N, 4)          optional
    """
    name:           str
    frames:         np.ndarray                  # (N, H, W, 3)  uint8  BGR
    boxes:          np.ndarray                  # (N, 4, 2)     float64  — 4 corners, each (x, y)
    landmarks:      np.ndarray                  # (N, 68, 2)    float64  — 68 landmarks, each (x, y)
    landmarks3d:    np.ndarray                  # (N, 68, 3)    float64  — 68 landmarks, each (x, y, z)
    detected_boxes: np.ndarray | None = None    # (N, 4) int32 x1y1x2y2, -1 if no detection


def load_video(path: Path) -> VideoData:
    """Load one .npz file and return a VideoData with BGR frames."""
    npz = np.load(path)

    # colorImages: (H, W, 3, N) -> (N, H, W, 3), then RGB -> BGR
    frames_rgb = np.transpose(npz["colorImages"], (3, 0, 1, 2))
    frames = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames_rgb])

    # boundingBox: (4, 2, N) -> (N, 4, 2)
    boxes = np.transpose(npz["boundingBox"], (2, 0, 1))

    # landmarks2D: (68, 2, N) -> (N, 68, 2)
    landmarks = np.transpose(npz["landmarks2D"], (2, 0, 1))

    # landmarks3D: (68, 3, N) -> (N, 68, 3)
    landmarks3d = np.transpose(npz["landmarks3D"], (2, 0, 1))

    detected_boxes = npz["detectedBox"] if "detectedBox" in npz else None

    return VideoData(
        name=path.stem,
        frames=frames,
        boxes=boxes,
        landmarks=landmarks,
        landmarks3d=landmarks3d,
        detected_boxes=detected_boxes,
    )