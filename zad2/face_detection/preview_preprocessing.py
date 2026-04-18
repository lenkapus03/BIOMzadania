"""
preview_preprocessing.py - Visual tuning helper for preprocessing parameters.

Opens 3 randomly selected frames (one per video) side-by-side:
    LEFT  — original frame
    RIGHT — preprocessed frame

Adjust parameters in face_detection/preprocessor.py and re-run this script.
Close all windows or press Q to quit.
"""

import random
from pathlib import Path

import cv2

from zad2.data_loader import load_video
from zad2.face_detection.preprocessor import preprocess, BLUR_KERNEL, CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE

DATA_DIR   = Path(__file__).parent.parent / "videos-K-O"

N_PREVIEWS = 3


def main():
    npz_files = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        print(f"[ERROR] No .npz files found in '{DATA_DIR}'.")
        return

    sample = random.sample(npz_files, min(N_PREVIEWS, len(npz_files)))

    print(f"Parameters:  BLUR={BLUR_KERNEL}  CLAHE clip={CLAHE_CLIP_LIMIT}  tile={CLAHE_TILE_SIZE}")
    print("Press Q in any window to close all.\n")

    windows = []
    for npz_path in sample:
        video   = load_video(npz_path)
        frame   = video.frames[random.randint(0, len(video.frames) - 1)]  # BGR
        pre     = preprocess(frame)                                         # BGR

        label = f"blur={BLUR_KERNEL} clip={CLAHE_CLIP_LIMIT} tile={CLAHE_TILE_SIZE}"
        cv2.putText(pre, label, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        side_by_side = cv2.hconcat([frame, pre])

        win_name = f"{video.name}  |  original  vs  preprocessed"
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(win_name, side_by_side)
        windows.append(win_name)
        print(f"  {video.name}  ({len(video.frames)} frames)")

    print("\nClose windows or press Q to quit.")
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord("q"):
            break
        if all(cv2.getWindowProperty(w, cv2.WND_PROP_VISIBLE) < 1 for w in windows):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()