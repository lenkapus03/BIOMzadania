"""
face_alignment/average_face.py - Compute and display the average face from the normalized dataset.

Loads all normalized face crops from videos-K-O-normalized/ and computes
the pixel-wise mean across all frames. If normalization is correct (eyes
always at the same position), the average face will be sharp and recognizable.
If normalization failed, the result will be blurry and unrecognizable.

Saves the result to face_alignment/average_face.png.
"""

from pathlib import Path

import cv2
import numpy as np

NORMALIZED_DIR = Path(__file__).parent.parent / "data/videos-K-O-normalized"
OUT_PATH       = Path(__file__).parent / "average_face.png"


def run():
    npz_files = sorted(NORMALIZED_DIR.glob("*.npz"))
    if not npz_files:
        print(f"[ERROR] No .npz files found in '{NORMALIZED_DIR}'.")
        return

    print(f"Computing average face from {len(npz_files)} videos...")

    # accumulate sum in float64 to avoid overflow
    acc   = None
    total = 0

    for i, npz_path in enumerate(npz_files, 1):
        frames = np.load(npz_path)["normalized_faces"]  # (N, 112, 112, 3) uint8 BGR
        if acc is None:
            acc = np.zeros(frames.shape[1:], dtype=np.float64)  # (112, 112, 3)
        acc   += frames.sum(axis=0).astype(np.float64)
        total += len(frames)
        print(f"  [{i}/{len(npz_files)}]  {npz_path.stem}  frames={len(frames)}  total={total}",
              flush=True)

    avg = (acc / total).astype(np.uint8)

    cv2.imwrite(str(OUT_PATH), avg)
    print(f"\nAverage face saved -> {OUT_PATH}")
    print(f"Total frames used  : {total}")

    cv2.namedWindow("Average face", cv2.WINDOW_NORMAL)
    cv2.imshow("Average face", avg)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()