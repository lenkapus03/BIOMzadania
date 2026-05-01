"""
face_detection/visualize_problematic.py - Visualize detections on problematic videos.

For each category, explicitly defines which detector(s) to use for frame selection
and what condition makes a frame "ideal" to show.

Controls:
    Q / ESC — close all windows
"""

from pathlib import Path

import cv2
import dlib
from mtcnn import MTCNN

from zad2.utils.data_loader import load_video
from zad2.utils.metrics import iou
from zad2.face_detection.hog_detector import detect as hog_detect
from zad2.face_detection.mtcnn_detector import detect as mtcnn_detect

# ─── videos by category ───────────────────────────────────────────────────────
HIGH_FN = ["Nuon_Chea_0", "Katja_Riemann_2", "Nicolas_Cage_1"]   # hard to detect
HIGH_FP = ["Kyle_Shewfelt_0", "Kate_Winslet_5", "Mario_Gallegos_3"]  # most false detections
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR         = Path(__file__).parent.parent / "data/videos-K-O"
HOG_UPSAMPLE     = 1
MTCNN_CONFIDENCE = 0.8
IOU_THRESHOLD    = 0.5


GT_COLOR    = (0, 255, 0)
HOG_COLOR   = (255, 0, 0)
MTCNN_COLOR = (0, 0, 255)
THICKNESS   = 2
FONT        = cv2.FONT_HERSHEY_SIMPLEX


def is_tp(box, gt_corners):
    return iou(box, gt_corners) > IOU_THRESHOLD


def is_fp(box, gt_corners):
    return iou(box, gt_corners) <= IOU_THRESHOLD


def select_frame_high_fn(frames, boxes, hog_det, mtcnn_det):
    """
    Find frame where HOG detects nothing at all (0 boxes).
    HOG was the main problem for these videos (FP=0 meaning it simply missed).
    """
    candidates = list(range(len(frames)))
    for idx in candidates:
        frame      = frames[idx]
        hog_boxes  = hog_detect(frame, hog_det, upsample=HOG_UPSAMPLE)
        mtcnn_boxes = mtcnn_detect(frame, mtcnn_det, min_confidence=MTCNN_CONFIDENCE)
        if len(hog_boxes) == 0:   # HOG finds nothing
            return idx, hog_boxes, mtcnn_boxes
    # fallback: frame with fewest HOG boxes
    best_idx, best_hog, best_mtcnn = candidates[0], [], []
    for idx in candidates:
        frame      = frames[idx]
        hog_boxes  = hog_detect(frame, hog_det, upsample=HOG_UPSAMPLE)
        mtcnn_boxes = mtcnn_detect(frame, mtcnn_det, min_confidence=MTCNN_CONFIDENCE)
        if len(hog_boxes) < len(best_hog) or best_hog == []:
            best_idx, best_hog, best_mtcnn = idx, hog_boxes, mtcnn_boxes
    return best_idx, best_hog, best_mtcnn


def select_frame_high_fp(frames, boxes, hog_det, mtcnn_det):
    """
    Find frame where HOG produces the most false positive boxes.
    A box is FP if it does not overlap with GT (IoU <= threshold).
    """
    candidates  = list(range(len(frames)))
    best_idx    = candidates[0]
    best_hog    = []
    best_mtcnn  = []
    best_fp     = -1

    for idx in candidates:
        frame       = frames[idx]
        gt_corners  = boxes[idx]
        hog_boxes   = hog_detect(frame, hog_det, upsample=HOG_UPSAMPLE)
        mtcnn_boxes = mtcnn_detect(frame, mtcnn_det, min_confidence=MTCNN_CONFIDENCE)
        fp_count    = sum(1 for b in hog_boxes if is_fp(b, gt_corners))
        if fp_count > best_fp:
            best_fp, best_idx, best_hog, best_mtcnn = fp_count, idx, hog_boxes, mtcnn_boxes

    return best_idx, best_hog, best_mtcnn


OUT_DIR = Path(__file__).parent / "problematic"


def draw_boxes(frame_bgr, gt_box, hog_boxes, mtcnn_boxes):
    img = frame_bgr.copy()
    x1  = int(gt_box[:, 0].min())
    y1  = int(gt_box[:, 1].min())
    x2  = int(gt_box[:, 0].max())
    y2  = int(gt_box[:, 1].max())
    cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, THICKNESS)
    for (bx1, by1, bx2, by2) in hog_boxes:
        cv2.rectangle(img, (bx1, by1), (bx2, by2), HOG_COLOR, THICKNESS)
    for (bx1, by1, bx2, by2) in mtcnn_boxes:
        cv2.rectangle(img, (bx1, by1), (bx2, by2), MTCNN_COLOR, THICKNESS)
    return img


def process_video(video_name, hog_det, mtcnn_det, category):
    npz_path = DATA_DIR / f"{video_name}.npz"
    if not npz_path.exists():
        print(f"[WARN] {npz_path} not found, skipping.")
        return None, None

    video = load_video(npz_path)

    if category == "HIGH_FN":
        idx, hog_boxes, mtcnn_boxes = select_frame_high_fn(video.frames, video.boxes, hog_det, mtcnn_det)
    elif category == "HIGH_FP":
        idx, hog_boxes, mtcnn_boxes = select_frame_high_fp(video.frames, video.boxes, hog_det, mtcnn_det)

    gt_corners = video.boxes[idx]

    def frame_metrics(boxes):
        if not boxes:
            return {"tp": 0, "fp": 0, "fn": 1, "best_iou": 0.0, "f1": 0.0}
        best_iou = max(iou(b, gt_corners) for b in boxes)
        tp = sum(1 for b in boxes if is_tp(b, gt_corners))
        fp = len(boxes) - tp
        fn = 1 if tp == 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "best_iou": best_iou, "f1": f1}

    hog_m   = frame_metrics(hog_boxes)
    mtcnn_m = frame_metrics(mtcnn_boxes)

    print(f"  [{category}] {video_name}  frame={idx}")
    print(f"    HOG:   TP={hog_m['tp']}  FP={hog_m['fp']}  FN={hog_m['fn']}  best_IoU={hog_m['best_iou']:.3f}  F1={hog_m['f1']:.3f}")
    print(f"    MTCNN: TP={mtcnn_m['tp']}  FP={mtcnn_m['fp']}  FN={mtcnn_m['fn']}  best_IoU={mtcnn_m['best_iou']:.3f}  F1={mtcnn_m['f1']:.3f}")

    img      = draw_boxes(video.frames[idx], video.boxes[idx], hog_boxes, mtcnn_boxes)
    win_name = f"[{category}] {video_name}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{category}_{video_name}.png"
    cv2.imwrite(str(out_path), img)
    print(f"    saved -> {out_path}")

    return img, win_name


def main():
    hog_det   = dlib.get_frontal_face_detector()
    mtcnn_det = MTCNN()

    categories = [
        ("HIGH_FN", HIGH_FN),
        ("HIGH_FP", HIGH_FP),
    ]

    windows = []
    for category, video_list in categories:
        print(f"\n── {category} ──────────────────────────")
        for video_name in video_list:
            img, win_name = process_video(video_name, hog_det, mtcnn_det, category)
            if img is not None:
                cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(win_name, img)
                windows.append(win_name)

    print("\nPress Q to quit.")
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key in (ord("q"), 27):
            break
        if all(cv2.getWindowProperty(w, cv2.WND_PROP_VISIBLE) < 1 for w in windows):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()