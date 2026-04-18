"""
Biometry Project - Face Detection, Normalization, Recognition
Dataset: YoutubeFaces subset (videos-K-O)
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2

from face_detection.annotator import play_video
from face_detection.preprocessor import preprocess


DATA_DIR = Path("videos-K-O")

SHOW_SAMPLES   = False
PLAY_ANNOTATED = True


@dataclass
class VideoData:
    """
    One .npz video clip with all arrays in (N, ...) convention.

    Raw .npz layout:
        colorImages  (H, W, 3, N)   -> frames     (N, H, W, 3)
        boundingBox  (4, 2, N)      -> boxes      (N, 4, 2)
        landmarks2D  (68, 2, N)     -> landmarks  (N, 68, 2)
        landmarks3D  (68, 3, N)     -> landmarks3d(N, 68, 3)
    """
    name:        str
    frames:      np.ndarray   # (N, H, W, 3)  uint8  — original RGB frames
    frames_pre:  np.ndarray   # (N, H, W, 3)  uint8  — preprocessed BGR frames
    boxes:       np.ndarray   # (N, 4, 2)     float64  — 4 corners, each (x, y)
    landmarks:   np.ndarray   # (N, 68, 2)    float64  — 68 landmarks, each (x, y)
    landmarks3d: np.ndarray   # (N, 68, 3)    float64  — 68 landmarks, each (x, y, z)


def load_video(path: Path) -> VideoData:
    """Load one .npz file and return a VideoData with (N, ...) arrays."""
    npz = np.load(path)

    # colorImages: (H, W, 3, N) -> (N, H, W, 3)
    frames = np.transpose(npz["colorImages"], (3, 0, 1, 2))

    # preprocess each frame: RGB -> BGR -> preprocess -> store as BGR
    frames_pre = np.stack([
        preprocess(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        for f in frames
    ])

    # boundingBox: (4, 2, N) -> (N, 4, 2)
    boxes = np.transpose(npz["boundingBox"], (2, 0, 1))

    # landmarks2D: (68, 2, N) -> (N, 68, 2)
    landmarks = np.transpose(npz["landmarks2D"], (2, 0, 1))

    # landmarks3D: (68, 3, N) -> (N, 68, 3)
    landmarks3d = np.transpose(npz["landmarks3D"], (2, 0, 1))

    return VideoData(
        name=path.stem,
        frames=frames,
        frames_pre=frames_pre,
        boxes=boxes,
        landmarks=landmarks,
        landmarks3d=landmarks3d,
    )


def main():
    if not DATA_DIR.exists():
        print(f"[ERROR] Data directory '{DATA_DIR}' not found.")
        sys.exit(1)

    npz_files = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        print(f"[ERROR] No .npz files found in '{DATA_DIR}'.")
        sys.exit(1)

    print(f"Found {len(npz_files)} file(s) in '{DATA_DIR}'.")

    if SHOW_SAMPLES:
        for i in range(min(len(npz_files), 20)):
            v = load_video(npz_files[i])
            print(f"\n  {v.name}")
            print(f"    frames      {v.frames.shape}     {v.frames.dtype}")
            print(f"    frames_pre  {v.frames_pre.shape}  {v.frames_pre.dtype}")
            print(f"    boxes       {v.boxes.shape}       {v.boxes.dtype}")
            print(f"    landmarks   {v.landmarks.shape}   {v.landmarks.dtype}")
            print(f"    landmarks3d {v.landmarks3d.shape} {v.landmarks3d.dtype}")

    if PLAY_ANNOTATED:
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