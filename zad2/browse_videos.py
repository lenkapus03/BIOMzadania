"""
browse_videos.py - Browse and play videos with ground-truth annotations.

Change DATA_DIR to switch between original and preprocessed videos:
    Path(__file__).parent / "videos-K-O"
    Path(__file__).parent / "videos-K-O-preprocessed"
"""

import sys
from pathlib import Path

import cv2

from data_loader import load_video
from face_detection.annotator import play_video

# ─── change this to switch dataset ────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "videos-K-O"
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
        action = play_video(v, video_index, len(npz_files))

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