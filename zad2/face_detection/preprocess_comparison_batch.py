"""
face_detection/preprocess_comparison_batch.py - Compare detection on original vs preprocessed videos.

Runs both detectors (HOG upsample=1, MTCNN confidence=0.9) on the full dataset
for both original and preprocessed videos, and reports Precision / Recall / F1
(IoU > 0.5) for each combination.

Switch between models by changing MODEL:
    MODEL = "hog"
    MODEL = "mtcnn"
    MODEL = "both"
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
MODEL = "both"   # "hog", "mtcnn", or "both"
# ──────────────────────────────────────────────────────────────────────────────

ORIGINAL_DIR     = Path(__file__).parent.parent / "data/videos-K-O"
PREPROCESSED_DIR = Path(__file__).parent.parent / "data/videos-K-O-preprocessed"
IOU_THRESHOLD    = 0.5

HOG_UPSAMPLE     = 1
MTCNN_CONFIDENCE = 0.8

TOP_N = 10

def run_video(video, detector, detect_fn, param_value) -> tuple[list, float]:
    frame_boxes = []
    total_ms    = 0.0
    for frame in video.frames:
        t0    = time.perf_counter()
        boxes = detect_fn(frame, detector, param_value)
        total_ms += (time.perf_counter() - t0) * 1000
        frame_boxes.append(boxes[0] if boxes else None)
    return frame_boxes, total_ms / len(video.frames)


def print_metrics(label: str, metrics: dict, avg_ms: float):
    print(
        f"  {label:<35}  "
        f"P={metrics['precision']:.3f}  "
        f"R={metrics['recall']:.3f}  "
        f"F1={metrics['f1']:.3f}  "
        f"TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  "
        f"avg {avg_ms:.1f} ms/frame"
    )

def run_dataset(npz_files: list, detector, detect_fn, param_value, label: str) -> dict:
    per_video    = []
    total_ms_sum = 0.0
    n            = len(npz_files)

    for idx, npz_path in enumerate(npz_files, 1):
        video = load_video(npz_path)

        frame_boxes, avg_ms = run_video(video, detector, detect_fn, param_value)
        total_ms_sum += avg_ms

        video_metrics = evaluate_video(frame_boxes, video.boxes, IOU_THRESHOLD)
        video_metrics["name"] = video.name
        per_video.append(video_metrics)

        print(f"  {idx}/{n}  {npz_path.stem}", flush=True)

    overall = aggregate_metrics(per_video)
    avg_ms  = total_ms_sum / n
    print_metrics(label, overall, avg_ms)

    print(f"\n  Hard to detect (highest FN, top {TOP_N}):")
    for vm in sorted(per_video, key=lambda x: x["fn"], reverse=True)[:TOP_N]:
        print(f"    {vm['name']:<40}  TP={vm['tp']}  FP={vm['fp']}  FN={vm['fn']}")

    print(f"\n  Most false detections (highest FP, top {TOP_N}):")
    for vm in sorted(per_video, key=lambda x: x["fp"], reverse=True)[:TOP_N]:
        print(f"    {vm['name']:<40}  TP={vm['tp']}  FP={vm['fp']}  FN={vm['fn']}")

    return overall


def run_model(model_name: str, detector, detect_fn, param_value, param_label: str):
    orig_files = sorted(ORIGINAL_DIR.glob("*.npz"))
    pre_files  = sorted(PREPROCESSED_DIR.glob("*.npz"))

    if not orig_files:
        print(f"[ERROR] No .npz files found in '{ORIGINAL_DIR}'.")
        return
    if not pre_files:
        print(f"[ERROR] No .npz files found in '{PREPROCESSED_DIR}'.")
        return

    print(f"\n{'='*60}")
    print(f"Model: {model_name}  |  {param_label}  |  IoU > {IOU_THRESHOLD}")
    print(f"{'='*60}")

    print(f"\n── Original ({len(orig_files)} videos) ──────────────────────────")
    orig_metrics = run_dataset(orig_files, detector, detect_fn, param_value, "OVERALL original")

    print(f"\n── Preprocessed ({len(pre_files)} videos) ───────────────────────")
    pre_metrics = run_dataset(pre_files, detector, detect_fn, param_value, "OVERALL preprocessed")

    print(f"\n── Summary ──────────────────────────────────────────────────")
    print(f"  {'':35}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print(f"  {'Original':<35}  {orig_metrics['precision']:>10.3f}  {orig_metrics['recall']:>8.3f}  {orig_metrics['f1']:>8.3f}")
    print(f"  {'Preprocessed':<35}  {pre_metrics['precision']:>10.3f}  {pre_metrics['recall']:>8.3f}  {pre_metrics['f1']:>8.3f}")
    print()


def run(model: str):
    if model in ("hog", "both"):
        detector  = dlib.get_frontal_face_detector()
        detect_fn = lambda frame, det, v: hog_detect(frame, det, upsample=v)
        run_model("HOG+SVM", detector, detect_fn, HOG_UPSAMPLE, f"upsample={HOG_UPSAMPLE}")

    if model in ("mtcnn", "both"):
        detector  = MTCNN()
        detect_fn = lambda frame, det, v: mtcnn_detect(frame, det, min_confidence=v)
        run_model("MTCNN", detector, detect_fn, MTCNN_CONFIDENCE, f"min_confidence={MTCNN_CONFIDENCE}")

    if model not in ("hog", "mtcnn", "both"):
        print(f"[ERROR] Unknown model '{model}'. Use 'hog', 'mtcnn', or 'both'.")

    print("Done.")


if __name__ == "__main__":
    run(MODEL)