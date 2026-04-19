"""
face_detection/model_hyperparam_tuning_batch.py - Batch hyperparameter tuning.

Runs face detection over the entire dataset for each hyperparameter value,
computes Precision / Recall / F1 (IoU > 0.5) and reports problematic videos
(those with F1 = 0 for a given parameter value).

Switch between models by changing MODEL:
    MODEL = "hog"
    MODEL = "mtcnn"
"""

import time
from pathlib import Path

import dlib
from mtcnn import MTCNN

from zad2.utils.data_loader import load_video
from zad2.utils.metrics import evaluate_video, aggregate_metrics
from zad2.face_detection.hog_detector import detect as hog_detect
from zad2.face_detection.mtcnn_detector import detect as mtcnn_detect

# ─── switch between models ────────────────────────────────────────────────────
MODEL = "mtcnn"   # "hog" or "mtcnn"
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR      = Path(__file__).parent.parent / "data/videos-K-O"
IOU_THRESHOLD = 0.5

# HOG_UPSAMPLE_VALUES     = [0, 1, 2, 3]
HOG_UPSAMPLE_VALUES = [1]
MTCNN_CONFIDENCE_VALUES = [0.1, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]


def boxes_from_detected(detected: list) -> list:
    """Convert list of (x1,y1,x2,y2) detections to per-frame boxes (first box or None)."""
    return [boxes[0] if boxes else None for boxes in detected]


def run_video(video, detector, detect_fn, param_value) -> tuple[list, float]:
    """Run detection on every frame. Returns per-frame box list and avg ms."""
    frame_boxes = []
    total_ms    = 0.0
    for frame in video.frames:
        t0     = time.perf_counter()
        boxes  = detect_fn(frame, detector, param_value)
        total_ms += (time.perf_counter() - t0) * 1000
        frame_boxes.append(boxes[0] if boxes else None)
    return frame_boxes, total_ms / len(video.frames)


def print_metrics(label: str, metrics: dict, avg_ms: float):
    print(
        f"  {label:<25}  "
        f"P={metrics['precision']:.3f}  "
        f"R={metrics['recall']:.3f}  "
        f"F1={metrics['f1']:.3f}  "
        f"TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  "
        f"avg {avg_ms:.1f} ms/frame"
    )


def run(model: str):
    npz_files = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        print(f"[ERROR] No .npz files found in '{DATA_DIR}'.")
        return

    print(f"Dataset: {len(npz_files)} videos  |  IoU threshold: {IOU_THRESHOLD}\n")

    if model == "hog":
        detector   = dlib.get_frontal_face_detector()
        values     = HOG_UPSAMPLE_VALUES
        detect_fn  = lambda frame, det, v: hog_detect(frame, det, upsample=v)
        param_name = "upsample"
        print(f"Model: HOG+SVM  |  upsample values: {values}\n")

    elif model == "mtcnn":
        detector   = MTCNN()
        values     = MTCNN_CONFIDENCE_VALUES
        detect_fn  = lambda frame, det, v: mtcnn_detect(frame, det, min_confidence=v)
        param_name = "min_confidence"
        print(f"Model: MTCNN  |  confidence values: {values}\n")

    else:
        print(f"[ERROR] Unknown model '{model}'. Use 'hog' or 'mtcnn'.")
        return

    for value in values:
        label        = f"{param_name}={value}"
        per_video    = []
        total_ms_sum = 0.0
        problematic  = []

        print(f"── {label} ──────────────────────────────")

        for idx, npz_path in enumerate(npz_files, 1):
            video = load_video(npz_path)

            frame_boxes, avg_ms = run_video(video, detector, detect_fn, value)
            total_ms_sum += avg_ms

            vm = evaluate_video(frame_boxes, video.boxes, IOU_THRESHOLD)
            per_video.append(vm)

            if vm["f1"] == 0.0:
                problematic.append((video.name, vm))

            print(f"  {idx}/{len(npz_files)}  {npz_path.stem}", flush=True)

        overall  = aggregate_metrics(per_video)
        avg_ms   = total_ms_sum / len(npz_files)
        print_metrics("OVERALL", overall, avg_ms)

        if problematic:
            print(f"\n  Problematic videos ({len(problematic)}):")
            for name, vm in problematic:
                print(f"    {name:<40}  TP={vm['tp']}  FP={vm['fp']}  FN={vm['fn']}")
        else:
            print("  No problematic videos.")

        print()

    print("Done.")


if __name__ == "__main__":
    run(MODEL)