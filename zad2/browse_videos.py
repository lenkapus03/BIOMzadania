"""
browse_videos.py - Browse and play videos with annotations.

Change DATA_DIR to switch between datasets:
    Path(__file__).parent / "data/videos-K-O"
    Path(__file__).parent / "data/videos-K-O-preprocessed"
    Path(__file__).parent / "data/videos-K-O-detection-tuning"
"""

import sys
from pathlib import Path

import cv2

from zad2.utils.data_loader import load_video
from zad2.utils.annotator import play_video

# ─── change this to switch dataset ────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data/videos-K-O-detection-tuning"
# ──────────────────────────────────────────────────────────────────────────────

# ─── toggle what is drawn ─────────────────────────────────────────────────────
SHOW_GT_BOX       = True   # green ground-truth bounding box
SHOW_LANDMARKS    = False   # 68 facial landmarks
SHOW_DETECTED_BOX = True   # blue detected bounding box (if present in file)
# ──────────────────────────────────────────────────────────────────────────────


def main():
    if not DATA_DIR.exists():
        print(f"[ERROR] Data directory '{DATA_DIR}' not found.")
        sys.exit(1)

    npz_files = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        print(f"[ERROR] No .npz files found in '{DATA_DIR}'.")
        sys.exit(1)

    print(f"Found {len(npz_files)} file(s) in '{DATA_DIR}'.")

    video_index = 0
    v = load_video(npz_files[video_index])

    while True:
        action = play_video(
            v, video_index, len(npz_files),
            show_gt_box=SHOW_GT_BOX,
            show_landmarks=SHOW_LANDMARKS,
            show_detected_box=SHOW_DETECTED_BOX,
        )

        if action == "next":
            video_index = min(video_index + 1, len(npz_files) - 1)
            v = load_video(npz_files[video_index])
        elif action == "prev":
            video_index = max(video_index - 1, 0)
            v = load_video(npz_files[video_index])
        elif action == "quit":
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()