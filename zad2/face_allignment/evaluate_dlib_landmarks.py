"""
face_alignment/evaluate_dlib_landmarks.py - Evaluate Dlib 68-point shape predictor.

Uses MTCNN (confidence=0.8) detections as bounding box input for the shape predictor.
Compares predicted landmarks with ground-truth landmarks2D using MSE.

Skips frames where MTCNN does not detect a face.
"""

from pathlib import Path

import cv2
import numpy as np
import dlib
from mtcnn import MTCNN

from zad2.utils.data_loader import load_video
from zad2.face_detection.mtcnn_detector import detect as mtcnn_detect

DATA_DIR         = Path(__file__).parent.parent / "data/videos-K-O"
DAT_PATH         = Path(__file__).parent / "shape_predictor_68_face_landmarks.dat"
MTCNN_CONFIDENCE = 0.8


def mse(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute MSE between predicted and ground-truth landmarks.

    Args:
        predicted:    (68, 2) array of predicted (x, y) landmark positions
        ground_truth: (68, 2) array of ground-truth (x, y) landmark positions

    Returns:
        Mean squared error over all 68 points and both coordinates
    """
    return float(np.mean((predicted - ground_truth) ** 2))


def predict_landmarks(frame_bgr: np.ndarray, box: tuple, predictor) -> np.ndarray | None:
    """
    Run dlib shape predictor on a frame given a bounding box.

    Args:
        frame_bgr: (H, W, 3) uint8 BGR image
        box:       (x1, y1, x2, y2) bounding box from MTCNN
        predictor: dlib shape predictor instance

    Returns:
        (68, 2) array of predicted landmark positions, or None on failure
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = box
    rect = dlib.rectangle(x1, y1, x2, y2)
    shape = predictor(frame_rgb, rect)
    points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
    return points


def run():
    if not DAT_PATH.exists():
        print(f"[ERROR] Model file not found: {DAT_PATH}")
        return

    predictor      = dlib.shape_predictor(str(DAT_PATH))
    mtcnn_detector = MTCNN()

    npz_files = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        print(f"[ERROR] No .npz files found in '{DATA_DIR}'.")
        return

    print(f"Model: Dlib 68-point shape predictor")
    print(f"Input boxes: MTCNN (confidence={MTCNN_CONFIDENCE})")
    print(f"Dataset: {len(npz_files)} videos\n")

    all_mse      = []
    skipped_total = 0
    processed_total = 0

    for npz_path in npz_files:
        video        = load_video(npz_path)
        video_mse    = []
        skipped      = 0

        for frame, gt_landmarks in zip(video.frames, video.landmarks):
            boxes = mtcnn_detect(frame, mtcnn_detector, min_confidence=MTCNN_CONFIDENCE)

            if not boxes:
                skipped += 1
                continue

            landmarks_pred = predict_landmarks(frame, boxes[0], predictor)
            if landmarks_pred is None:
                skipped += 1
                continue

            frame_mse = mse(landmarks_pred, gt_landmarks)
            video_mse.append(frame_mse)

        if video_mse:
            avg_video_mse = float(np.mean(video_mse))
            all_mse.extend(video_mse)
            processed_total += len(video_mse)
        else:
            avg_video_mse = float("nan")

        skipped_total += skipped
        print(f"  {npz_path.stem:<40}  "
              f"frames={len(video.frames)}  "
              f"processed={len(video_mse)}  "
              f"skipped={skipped}  "
              f"MSE={avg_video_mse:.2f}")

    print(f"\n{'='*60}")
    print(f"OVERALL")
    print(f"  Processed frames : {processed_total}")
    print(f"  Skipped frames   : {skipped_total} (no MTCNN detection)")
    print(f"  Overall MSE      : {np.mean(all_mse):.2f}")
    print(f"  Std MSE          : {np.std(all_mse):.2f}")


if __name__ == "__main__":
    run()