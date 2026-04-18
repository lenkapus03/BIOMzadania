"""
batch_preprocess.py - One-time batch preprocessing of all videos.

Loads each .npz from videos-K-O, preprocesses every frame (BGR),
and saves the result as a new .npz in videos-K-O-preprocessed/
with the same filename and the same structure as the original,
except colorImages is replaced with the preprocessed version.

Run once:
    python batch_preprocess.py
"""

from pathlib import Path

import cv2
import numpy as np

from zad2.utils.data_loader import load_video
from zad2.preprocess.preprocessor import preprocess

SRC_DIR = Path(__file__).parent.parent.parent / "videos-K-O"
DST_DIR = Path(__file__).parent.parent.parent / "videos-K-O-preprocessed"


def main():
    npz_files = sorted(SRC_DIR.glob("*.npz"))
    if not npz_files:
        print(f"[ERROR] No .npz files found in '{SRC_DIR}'.")
        return

    DST_DIR.mkdir(exist_ok=True)
    print(f"Preprocessing {len(npz_files)} file(s) -> '{DST_DIR}'\n")

    for i, npz_path in enumerate(npz_files, 1):
        dst_path = DST_DIR / npz_path.name

        if dst_path.exists():
            print(f"  [{i}/{len(npz_files)}] {npz_path.name}  (skipped, already exists)")
            continue

        video = load_video(npz_path)
        N = len(video.frames)

        # preprocess each BGR frame, then convert back to RGB for saving
        # (original colorImages are RGB — preserve that convention in the .npz)
        preprocessed_rgb = np.stack([
            cv2.cvtColor(preprocess(frame), cv2.COLOR_BGR2RGB)
            for frame in video.frames
        ])  # (N, H, W, 3) RGB

        # restore original .npz layout: (H, W, 3, N)
        color_images = np.transpose(preprocessed_rgb, (1, 2, 3, 0))

        # load original npz to copy all other arrays unchanged
        original = np.load(npz_path)
        np.savez(
            dst_path,
            colorImages=color_images,
            boundingBox=original["boundingBox"],
            landmarks2D=original["landmarks2D"],
            landmarks3D=original["landmarks3D"],
        )

        print(f"  [{i}/{len(npz_files)}] {npz_path.name}  ({N} frames)")

    print("\nDone.")


if __name__ == "__main__":
    main()