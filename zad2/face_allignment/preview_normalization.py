"""
face_alignment/preview_normalization.py - Preview original vs normalized face.

Picks a random video and random frame, shows the original frame
and the normalized face side by side.

Controls:
    Q / ESC — close window
"""

import random
from pathlib import Path

import cv2

from zad2.utils.data_loader import load_video
from zad2.face_allignment.normalize_faces import align_face
from zad2.face_allignment.predict_fan_landmarks import predict_landmarks
import face_alignment as fa_module
from mtcnn import MTCNN
from zad2.face_detection.mtcnn_detector import detect as mtcnn_detect

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

DATA_DIR         = Path(__file__).parent.parent / "data/videos-K-O"
MTCNN_CONFIDENCE = 0.8


def main():
    npz_files = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        print(f"[ERROR] No .npz files found in '{DATA_DIR}'.")
        return

    fa             = fa_module.FaceAlignment(
                         fa_module.LandmarksType.TWO_D,
                         flip_input=True,
                         device="cpu",
                     )
    mtcnn_detector = MTCNN()

    # try random videos until we find a frame with a valid detection
    random.shuffle(npz_files)
    for npz_path in npz_files:
        video   = load_video(npz_path)
        indices = list(range(len(video.frames)))
        random.shuffle(indices)

        for idx in indices:
            frame = video.frames[idx]
            boxes = mtcnn_detect(frame, mtcnn_detector, min_confidence=MTCNN_CONFIDENCE)
            if not boxes:
                continue

            landmarks = predict_landmarks(frame, boxes[0], fa)
            if landmarks is None:
                continue

            aligned = align_face(frame, landmarks)
            if aligned is None:
                continue

            print(f"Video: {npz_path.stem}  frame={idx}")

            cv2.namedWindow("original", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("normalized", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("original", frame)
            cv2.imshow("normalized", aligned)

            print("Press Q to quit.")
            while True:
                key = cv2.waitKey(100) & 0xFF
                if key in (ord("q"), 27):
                    break
                if (cv2.getWindowProperty("original", cv2.WND_PROP_VISIBLE) < 1 and
                        cv2.getWindowProperty("normalized", cv2.WND_PROP_VISIBLE) < 1):
                    break

            cv2.destroyAllWindows()
            return

    print("[ERROR] No valid frame found.")


if __name__ == "__main__":
    main()