"""
face_detection/model_hyperparam_tuning.py - Hyperparameter tuning for face detectors.

Picks one random video, runs detection on every frame for each hyperparameter
value, and saves annotated .npz files to data/videos-K-O-detection-tuning/.

Output filename format: <original_name>_<model>_<param><value>.npz
    e.g. KieferSutherland_0_hog_up0.npz
         KieferSutherland_0_mtcnn_conf0.9.npz

Switch between models by changing MODEL:
    MODEL = "hog"
    MODEL = "mtcnn"
"""

import random
import time
from pathlib import Path

import numpy as np
import cv2
import dlib
from mtcnn import MTCNN

from zad2.utils.data_loader import load_video
from zad2.face_detection.hog_detector import detect as hog_detect
from zad2.face_detection.mtcnn_detector import detect as mtcnn_detect

# ─── switch between models ────────────────────────────────────────────────────
MODEL = "mtcnn"   # "hog" or "mtcnn"
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data/videos-K-O"
OUT_DIR  = Path(__file__).parent.parent / "data/videos-K-O-detection-tuning"

HOG_UPSAMPLE_VALUES     = [0, 1, 2, 3]
MTCNN_CONFIDENCE_VALUES = [0.1, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]


def boxes_to_array(boxes: list[tuple[int, int, int, int]], N: int) -> np.ndarray:
    """
    Convert list of (x1, y1, x2, y2) to (N, 4) array.
    Frames with no detection get [-1, -1, -1, -1].
    """
    arr = np.full((N, 4), -1, dtype=np.int32)
    for i, box in enumerate(boxes):
        if box is not None:
            arr[i] = box
    return arr


def detect_video(video, detector, detect_fn, param_value) -> tuple[list, float]:
    """Run detection on every frame. Returns list of box-per-frame and avg ms."""
    frame_boxes = []
    total_ms    = 0.0
    for frame in video.frames:
        t_start = time.perf_counter()
        boxes   = detect_fn(frame, detector, param_value)
        total_ms += (time.perf_counter() - t_start) * 1000
        # keep only the first detected box (one face expected per frame)
        frame_boxes.append(boxes[0] if boxes else None)
    avg_ms = total_ms / len(video.frames)
    return frame_boxes, avg_ms


def save_result(video, npz_path: Path, frame_boxes: list, suffix: str):
    """Save .npz with detectedBox field added, colorImages preserved."""
    original = np.load(npz_path)

    N           = len(video.frames)
    detected    = boxes_to_array(frame_boxes, N)  # (N, 4)

    # colorImages back to original (H, W, 3, N) RGB layout
    frames_rgb  = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in video.frames])
    color_images = np.transpose(frames_rgb, (1, 2, 3, 0))

    out_path = OUT_DIR / f"{npz_path.stem}_{suffix}.npz"
    np.savez(
        out_path,
        colorImages=color_images,
        boundingBox=original["boundingBox"],
        landmarks2D=original["landmarks2D"],
        landmarks3D=original["landmarks3D"],
        detectedBox=detected,        # (N, 4) int32, x1y1x2y2, -1 if no detection
    )
    return out_path


def run(model: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        print(f"[ERROR] No .npz files found in '{DATA_DIR}'.")
        return

    npz_path = random.choice(npz_files)
    video    = load_video(npz_path)
    print(f"Video: {video.name}  ({len(video.frames)} frames)\n")

    if model == "hog":
        detector   = dlib.get_frontal_face_detector()
        values     = HOG_UPSAMPLE_VALUES
        detect_fn  = lambda frame, det, v: hog_detect(frame, det, upsample=v)
        param_name = "up"
        print(f"Model: HOG+SVM  |  upsample values: {values}\n")

    elif model == "mtcnn":
        detector   = MTCNN()
        values     = MTCNN_CONFIDENCE_VALUES
        detect_fn  = lambda frame, det, v: mtcnn_detect(frame, det, min_confidence=v)
        param_name = "conf"
        print(f"Model: MTCNN  |  confidence values: {values}\n")

    else:
        print(f"[ERROR] Unknown model '{model}'. Use 'hog' or 'mtcnn'.")
        return

    for value in values:
        suffix = f"{model}_{param_name}{value}"
        print(f"  [{suffix}] detecting...", end=" ", flush=True)

        frame_boxes, avg_ms = detect_video(video, detector, detect_fn, value)

        detected  = sum(1 for b in frame_boxes if b is not None)
        out_path  = save_result(video, npz_path, frame_boxes, suffix)

        print(f"{detected}/{len(video.frames)} frames detected  |  avg {avg_ms:.1f} ms/frame  ->  {out_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    run(MODEL)