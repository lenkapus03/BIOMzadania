"""
face_alignment/normalize_faces.py - Normalize faces using FAN predicted landmarks.

Loads .npz files from videos-K-O-landmarks/ (which contain fan_landmarks2D),
aligns each face so that the eyes are at fixed positions in the output image,
and saves normalized face crops to videos-K-O-normalized/.

Normalization approach:
    - Eye centers are computed as the mean of the 6 landmark points per eye:
        Source: https://www.researchgate.net/figure/A-68-facial-landmark-detector-pretrained-from-the-iBUG-300-W-dataset-Each-number_fig1_353685970
        left eye:  points 36-41  (standard iBUG 68-point convention)
        right eye: points 42-47
    - The face is rotated so the line between the eyes is horizontal,
      then scaled and cropped so the eyes land at fixed target positions.

Output image size: 112x120 px
    Source: https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf
    - Chosen to match ArcFace input size, which will
      be used in the verification part of this project.

Target eye positions (as fraction of image size):
    Source: https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
    - Left eye:  (0.35 * W, 0.35 * H)
    - Right eye: (0.65 * W, 0.35 * H)

Frames with skipped landmarks (fan_landmarks2D == -1) are not saved.
"""

from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).parent.parent / "data/videos-K-O-landmarks"
OUT_DIR = Path(__file__).parent.parent / "data/videos-K-O-normalized"

# Output image size — matches ArcFace input
OUTPUT_SIZE = 112

LEFT_EYE_TARGET  = (0.35, 0.35)
RIGHT_EYE_TARGET = (0.65, 0.35)

# 68-point landmark indices for eyes:
LEFT_EYE_POINTS  = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))


def eye_center(landmarks: np.ndarray, indices: list) -> np.ndarray:
    """
    Compute eye center as the mean of the given landmark points.

    Using the mean of all 6 eye points (rather than e.g. just the outer
    corners) gives a more stable estimate robust to individual point errors.

    Args:
        landmarks: (68, 2) array of (x, y) landmark positions
        indices:   list of point indices belonging to one eye

    Returns:
        (2,) array — (x, y) center of the eye
    """
    return landmarks[indices].mean(axis=0)


def align_face(frame_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray | None:
    """
    Align and crop a face so that both eyes land at fixed target positions.

    Source: https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/

    Steps:
        1. Compute left and right eye centers from landmarks
        2. Compute the rotation angle to make the eye line horizontal
        3. Compute scale so the inter-eye distance matches the target
        4. Apply affine transform (rotation + scale + translation) to the image

    Args:
        frame_bgr: (H, W, 3) uint8 BGR image
        landmarks: (68, 2) float32 array of FAN predicted landmark positions

    Returns:
        (OUTPUT_SIZE, OUTPUT_SIZE, 3) uint8 normalized face image, or None on error
    """
    left_eye = eye_center(landmarks, LEFT_EYE_POINTS)
    right_eye = eye_center(landmarks, RIGHT_EYE_POINTS)

    # compute angle between eye centroids
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # compute desired right eye x-coordinate based on desired left eye
    desiredRightEyeX = 1.0 - LEFT_EYE_TARGET[0]

    # determine scale based on desired distance between eyes
    dist = np.sqrt((dX ** 2) + (dY ** 2))
    desiredDist = (desiredRightEyeX - LEFT_EYE_TARGET[0]) * OUTPUT_SIZE
    if dist < 1e-6:
        return None
    scale = desiredDist / dist

    # compute center (x, y)-coordinates between the two eyes
    eyesCenter = (
        int((left_eye[0] + right_eye[0]) // 2),
        int((left_eye[1] + right_eye[1]) // 2),
    )

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

    # update the translation component of the matrix
    tX = OUTPUT_SIZE * 0.5
    tY = OUTPUT_SIZE * LEFT_EYE_TARGET[1]
    M[0, 2] += (tX - eyesCenter[0])
    M[1, 2] += (tY - eyesCenter[1])

    # apply the affine transformation
    aligned = cv2.warpAffine(frame_bgr, M, (OUTPUT_SIZE, OUTPUT_SIZE),
                             flags=cv2.INTER_LINEAR)
    return aligned


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(SRC_DIR.glob("*.npz"))
    if not npz_files:
        print(f"[ERROR] No .npz files found in '{SRC_DIR}'.")
        return

    print(f"Normalizing faces from {len(npz_files)} videos -> '{OUT_DIR}'\n")
    print(f"Output size : {OUTPUT_SIZE}x{OUTPUT_SIZE} px")
    print(f"Left eye    : {LEFT_EYE_TARGET}")
    print(f"Right eye   : {RIGHT_EYE_TARGET}\n")

    total_saved   = 0
    total_skipped = 0

    for i, npz_path in enumerate(npz_files, 1):
        out_path = OUT_DIR / npz_path.name
        if out_path.exists():
            print(f"  [{i}/{len(npz_files)}] {npz_path.stem}  (skipped, already exists)")
            continue

        data          = np.load(npz_path)
        frames_rgb    = np.transpose(data["colorImages"], (3, 0, 1, 2))  # (N, H, W, 3)
        fan_landmarks = data["fan_landmarks2D"]                           # (N, 68, 2)
        N             = len(frames_rgb)

        normalized = []
        skipped    = 0

        for idx in range(N):
            landmarks = fan_landmarks[idx]

            # skip frames where prediction was not available
            if np.all(landmarks == -1):
                skipped += 1
                continue

            frame_bgr = cv2.cvtColor(frames_rgb[idx], cv2.COLOR_RGB2BGR)
            aligned   = align_face(frame_bgr, landmarks)

            if aligned is None:
                skipped += 1
                continue

            normalized.append(aligned)

        if not normalized:
            print(f"  [{i}/{len(npz_files)}] {npz_path.stem}  no valid frames, skipping")
            continue

        # save as (M, OUTPUT_SIZE, OUTPUT_SIZE, 3) uint8 array
        normalized_arr = np.stack(normalized)
        np.savez(out_path, normalized_faces=normalized_arr)

        total_saved   += len(normalized)
        total_skipped += skipped
        print(f"  [{i}/{len(npz_files)}] {npz_path.stem}  "
              f"saved={len(normalized)}  skipped={skipped}")

    print(f"\nDone.  Total saved={total_saved}  total skipped={total_skipped}")


if __name__ == "__main__":
    run()