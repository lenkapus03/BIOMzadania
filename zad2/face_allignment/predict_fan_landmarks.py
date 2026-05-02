"""
face_alignment/predict_fan_landmarks.py - Predict facial landmarks using FAN and save to .npz.

For each video in videos-K-O, runs FAN landmark prediction on every frame
using MTCNN bounding boxes as input. Results are saved to videos-K-O-landmarks/
in the same .npz format as the original files, with an additional field:
    fan_landmarks2D  (N, 68, 2)  float32  — FAN predicted landmarks, -1 if skipped

Frames where MTCNN does not detect a face are saved with landmarks filled as -1.
"""

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from pathlib import Path

import cv2
import numpy as np
import face_alignment
from mtcnn import MTCNN

from zad2.utils.data_loader import load_video
from zad2.face_detection.mtcnn_detector import detect as mtcnn_detect

DATA_DIR  = Path(__file__).parent.parent / "data/videos-K-O"
OUT_DIR   = Path(__file__).parent.parent / "data/videos-K-O-landmarks"

MTCNN_CONFIDENCE = 0.8

# flip_input=True: FAN internally flips the image horizontally and averages
# the predictions with the non-flipped version, improving accuracy at the
# cost of ~2x inference time
FLIP_INPUT = True


def predict_landmarks(frame_bgr: np.ndarray, box: tuple, fa) -> np.ndarray | None:
    """
    Run FAN on a single frame given a bounding box.

    Args:
        frame_bgr: (H, W, 3) uint8 BGR image
        box:       (x1, y1, x2, y2) from MTCNN
        fa:        face_alignment.FaceAlignment instance

    Returns:
        (68, 2) float32 array of predicted (x, y) positions, or None if FAN fails
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = box

    preds = fa.get_landmarks(frame_rgb, detected_faces=[[x1, y1, x2, y2]])
    if preds is None or len(preds) == 0:
        return None

    # preds[0] is (68, 2) or (68, 3) — keep only x, y
    return preds[0][:, :2].astype(np.float32)


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fa             = face_alignment.FaceAlignment(
                         face_alignment.LandmarksType.TWO_D,
                         flip_input=FLIP_INPUT,
                         device="cpu",
                     )
    mtcnn_detector = MTCNN()

    npz_files = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        print(f"[ERROR] No .npz files found in '{DATA_DIR}'.")
        return

    print(f"Predicting FAN landmarks for {len(npz_files)} videos -> '{OUT_DIR}'\n")

    for i, npz_path in enumerate(npz_files, 1):
        out_path = OUT_DIR / npz_path.name
        if out_path.exists():
            print(f"  [{i}/{len(npz_files)}] {npz_path.stem}  (skipped, already exists)")
            continue

        video     = load_video(npz_path)
        N         = len(video.frames)
        skipped   = 0

        # (N, 68, 2) — filled with -1 for frames where MTCNN finds no face
        fan_landmarks = np.full((N, 68, 2), -1, dtype=np.float32)

        for idx, frame in enumerate(video.frames):
            boxes = mtcnn_detect(frame, mtcnn_detector, min_confidence=MTCNN_CONFIDENCE)

            if not boxes:
                skipped += 1
                continue

            lm = predict_landmarks(frame, boxes[0], fa)
            if lm is None:
                skipped += 1
                continue

            fan_landmarks[idx] = lm

        # preserve original .npz fields, add fan_landmarks2D
        original = np.load(npz_path)
        np.savez(
            out_path,
            colorImages=original["colorImages"],
            boundingBox=original["boundingBox"],
            landmarks2D=original["landmarks2D"],
            landmarks3D=original["landmarks3D"],
            fan_landmarks2D=fan_landmarks,   # (N, 68, 2), -1 where skipped
        )

        print(f"  [{i}/{len(npz_files)}] {npz_path.stem}  "
              f"frames={N}  skipped={skipped}  -> {out_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    run()