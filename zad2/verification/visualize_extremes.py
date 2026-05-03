"""
verification/visualize_extremes.py - Visualize frame pairs with highest and lowest similarity.

For True and False pairs separately, finds the top N frame combinations
with the highest and lowest cosine similarity, and saves them as images
for visual analysis of common properties (blur, expression, pose, noise).

Output structure:
    data/extremes/true_max_1.png  ... true_max_N.png
    data/extremes/true_min_1.png  ... true_min_N.png
    data/extremes/false_max_1.png ... false_max_N.png
    data/extremes/false_min_1.png ... false_min_N.png
"""

import csv
from pathlib import Path

import cv2
import numpy as np

from zad2.verification.extract_embeddings import get_embeddings

DATA_DIR       = Path(__file__).parent.parent / "data"
NORMALIZED_DIR = DATA_DIR / "videos-K-O-normalized"
TRUE_CSV       = DATA_DIR / "sampled_true_pairs.csv"
FALSE_CSV      = DATA_DIR / "sampled_false_pairs.csv"
OUT_DIR        = DATA_DIR / "extremes"

# how many top/bottom frame pairs to save per category
TOP_N = 5


def load_frames(video_name: str) -> np.ndarray | None:
    """Load normalized frames for a video. Returns (N, 112, 112, 3) uint8 BGR."""
    npz_path = NORMALIZED_DIR / f"{video_name}.npz"
    if not npz_path.exists():
        print(f"[WARN] Not found: {npz_path}")
        return None
    return np.load(npz_path)["normalized_faces"]


def save_pair(frame_a: np.ndarray, frame_b: np.ndarray,
              sim: float, label: str, rank: int):
    """Save two frames side by side with similarity score in filename."""
    # add small white border between images
    border = np.ones((112, 4, 3), dtype=np.uint8) * 200
    combined = np.hstack([frame_a, border, frame_b])

    out_path = OUT_DIR / f"{label}_{rank:02d}_sim{sim:.3f}.png"
    cv2.imwrite(str(out_path), combined)
    return out_path


def process_pairs(csv_path: Path, label: str):
    """Find top/bottom N frame combinations across all pairs in a CSV."""
    with open(csv_path, newline="") as f:
        pairs = list(csv.DictReader(f))

    # collect all frame combinations with their similarity scores
    # each entry: (sim, frame_a, frame_b, pair_name)
    all_combos = []

    for row in pairs:
        name_a = Path(row["videoA"]).stem
        name_b = Path(row["videoB"]).stem

        emb_a  = get_embeddings(name_a)
        emb_b  = get_embeddings(name_b)
        frames_a = load_frames(name_a)
        frames_b = load_frames(name_b)

        if any(x is None for x in [emb_a, emb_b, frames_a, frames_b]):
            print(f"  [SKIP] {name_a} vs {name_b}")
            continue

        sim_mat = emb_a @ emb_b.T   # (N_A, N_B)

        # keep only the best (max) and worst (min) frame combination per pair
        # to prevent one long video from dominating the entire top N list
        max_idx = np.unravel_index(np.argmax(sim_mat), sim_mat.shape)
        min_idx = np.unravel_index(np.argmin(sim_mat), sim_mat.shape)

        pair_name = f"{name_a}_vs_{name_b}"
        all_combos.append((
            float(sim_mat[max_idx]),
            frames_a[max_idx[0]],
            frames_b[max_idx[1]],
            pair_name,
            "max"
        ))
        all_combos.append((
            float(sim_mat[min_idx]),
            frames_a[min_idx[0]],
            frames_b[min_idx[1]],
            pair_name,
            "min"
        ))

    # sort by similarity — one entry per pair
    all_combos.sort(key=lambda x: x[0], reverse=True)

    print(f"\n── {label} ──────────────────────────")

    # save top N (highest similarity) — one per pair
    print(f"  Top {TOP_N} highest similarity:")
    for rank, (sim, frame_a, frame_b, pair_name, _) in enumerate(all_combos[:TOP_N], 1):
        out_path = save_pair(frame_a, frame_b, sim, f"{label}_max", rank)
        print(f"    {rank}. sim={sim:.4f}  {pair_name}  -> {out_path.name}")

    # save top N (lowest similarity) — one per pair
    print(f"  Top {TOP_N} lowest similarity:")
    for rank, (sim, frame_a, frame_b, pair_name, _) in enumerate(all_combos[-TOP_N:][::-1], 1):
        out_path = save_pair(frame_a, frame_b, sim, f"{label}_min", rank)
        print(f"    {rank}. sim={sim:.4f}  {pair_name}  -> {out_path.name}")


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    process_pairs(TRUE_CSV,  "true")
    process_pairs(FALSE_CSV, "false")
    print(f"\nDone. Images saved to '{OUT_DIR}'")


if __name__ == "__main__":
    run()