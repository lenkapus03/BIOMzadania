import sys
from pathlib import Path

import cv2

from zad2.data_loader import load_video
from zad2.detection.annotator import play_video

DATA_DIR = Path("videos-K-O")

SHOW_SAMPLES = False
PLAY_ANNOTATED  = True


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
            print(f"    frames      {v.frames.shape}   {v.frames.dtype}")
            print(f"    boxes       {v.boxes.shape}     {v.boxes.dtype}")
            print(f"    landmarks   {v.landmarks.shape}  {v.landmarks.dtype}")
            print(f"    landmarks3d {v.landmarks3d.shape}  {v.landmarks3d.dtype}")

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