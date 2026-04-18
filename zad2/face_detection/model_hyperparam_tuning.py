"""
face_detection/model_hyperparam_tuning.py - Hyperparameter tuning for face detectors.

Switch between models by changing MODEL:
    MODEL = "hog"
    MODEL = "mtcnn"

Controls:
    Q / ESC — close all windows
"""

import random
import time
from pathlib import Path

import cv2
import dlib
from mtcnn import MTCNN

from zad2.utils.data_loader import load_video
from zad2.utils.annotator import draw_detections
from zad2.face_detection.hog_detector import detect as hog_detect
from zad2.face_detection.mtcnn_detector import detect as mtcnn_detect

# ─── switch between models ────────────────────────────────────────────────────
MODEL = "mtcnn"   # "hog" or "mtcnn"
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data/videos-K-O"

HOG_UPSAMPLE_VALUES     = [0, 1, 2, 3]
MTCNN_CONFIDENCE_VALUES = [0.1, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]


def run(model: str):
    npz_files = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        print(f"[ERROR] No .npz files found in '{DATA_DIR}'.")
        return

    video     = load_video(random.choice(npz_files))
    frame_bgr = video.frames[random.randint(0, len(video.frames) - 1)]
    print(f"Video: {video.name}\n")

    windows = []

    if model == "hog":
        detector = dlib.get_frontal_face_detector()
        print(f"Model: HOG+SVM  |  upsample values: {HOG_UPSAMPLE_VALUES}\n")
        for value in HOG_UPSAMPLE_VALUES:
            t_start = time.perf_counter()
            boxes   = hog_detect(frame_bgr, detector, upsample=value)
            t_ms    = (time.perf_counter() - t_start) * 1000
            label   = f"HOG  upsample={value}"
            print(f"  upsample={value}  ->  {len(boxes)} face(s)  |  {t_ms:.1f} ms")
            cv2.namedWindow(label, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(label, draw_detections(frame_bgr, boxes))
            windows.append(label)

    elif model == "mtcnn":
        detector = MTCNN()
        print(f"Model: MTCNN  |  confidence values: {MTCNN_CONFIDENCE_VALUES}\n")
        for value in MTCNN_CONFIDENCE_VALUES:
            t_start = time.perf_counter()
            boxes   = mtcnn_detect(frame_bgr, detector, min_confidence=value)
            t_ms    = (time.perf_counter() - t_start) * 1000
            label   = f"MTCNN  confidence={value}"
            print(f"  confidence={value}  ->  {len(boxes)} face(s)  |  {t_ms:.1f} ms")
            cv2.namedWindow(label, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(label, draw_detections(frame_bgr, boxes))
            windows.append(label)

    else:
        print(f"[ERROR] Unknown model '{model}'. Use 'hog' or 'mtcnn'.")
        return

    print("\nPress Q to quit.")
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key in (ord("q"), 27):
            break
        if all(cv2.getWindowProperty(w, cv2.WND_PROP_VISIBLE) < 1 for w in windows):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(MODEL)