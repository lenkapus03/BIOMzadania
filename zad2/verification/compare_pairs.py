"""
verification/compare_pairs.py - Compare frame pairs for sampled True/False video pairs.

For each pair (videoA, videoB):
  1. Extract ArcFace embeddings for all frames of both videos
  2. Compute cosine similarity for every combination of frames (N_A x N_B matrix)
  3. Report: random, mean, max, min similarity

Reads sampled_true_pairs.csv and sampled_false_pairs.csv.
"""

import csv
import random
from pathlib import Path

import numpy as np

from zad2.verification.extract_embeddings import get_embeddings, cosine_similarity

RANDOM_SEED = 42

DATA_DIR       = Path(__file__).parent.parent / "data"
TRUE_CSV       = DATA_DIR / "sampled_true_pairs.csv"
FALSE_CSV      = DATA_DIR / "sampled_false_pairs.csv"


def similarity_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity for all frame combinations of two videos.

    Args:
        emb_a: (N_A, 512) embeddings for video A
        emb_b: (N_B, 512) embeddings for video B

    Returns:
        (N_A, N_B) matrix of cosine similarities — entry [i, j] is the
        similarity between frame i of video A and frame j of video B.
        Uses matrix multiplication since embeddings are L2-normalized:
        sim(a, b) = a . b  =>  all similarities = emb_a @ emb_b.T
    """
    return emb_a @ emb_b.T


def pair_stats(sim_mat: np.ndarray) -> dict:
    """
    Compute summary statistics from a similarity matrix.

    Returns random, mean, max, min similarity across all frame combinations.
    """
    flat = sim_mat.flatten()
    return {
        "random": float(flat[random.randint(0, len(flat) - 1)]),
        "mean":   float(np.mean(flat)),
        "max":    float(np.max(flat)),
        "min":    float(np.min(flat)),
    }


def process_csv(csv_path: Path, label: str):
    print(f"\n{'='*60}")
    print(f"{label}  ({csv_path.name})")
    print(f"{'='*60}")
    print(f"  {'Pair':<55}  {'random':>8}  {'mean':>8}  {'max':>8}  {'min':>8}")
    print(f"  {'-'*55}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    with open(csv_path, newline="") as f:
        pairs = list(csv.DictReader(f))

    for row in pairs:
        name_a = Path(row["videoA"]).stem
        name_b = Path(row["videoB"]).stem

        emb_a = get_embeddings(name_a)
        emb_b = get_embeddings(name_b)

        if emb_a is None or emb_b is None:
            print(f"  {name_a} vs {name_b}  [SKIPPED — embeddings not found]")
            continue

        sim_mat = similarity_matrix(emb_a, emb_b)
        stats   = pair_stats(sim_mat)

        pair_label = f"{name_a}  vs  {name_b}"
        print(
            f"  {pair_label:<55}  "
            f"{stats['random']:>8.4f}  "
            f"{stats['mean']:>8.4f}  "
            f"{stats['max']:>8.4f}  "
            f"{stats['min']:>8.4f}"
        )


def run():
    random.seed(RANDOM_SEED)
    process_csv(TRUE_CSV,  "TRUE PAIRS  (same person)")
    process_csv(FALSE_CSV, "FALSE PAIRS (different person)")
    print("\nDone.")


if __name__ == "__main__":
    run()