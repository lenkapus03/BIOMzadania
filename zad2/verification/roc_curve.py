"""
verification/roc_curve.py - ROC curve for face verification on sampled pairs.

For each pair, computes a similarity matrix (N_A x N_B) and derives one
similarity score per aggregation strategy (random, mean, max, min).

For each strategy, plots a ROC curve and reports AUC + optimal threshold.
"""

import csv
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from zad2.verification.extract_embeddings import get_embeddings

RANDOM_SEED = 42

DATA_DIR   = Path(__file__).parent.parent / "data"
TRUE_CSV   = DATA_DIR / "sampled_true_pairs.csv"
FALSE_CSV  = DATA_DIR / "sampled_false_pairs.csv"
OUT_DIR    = Path(__file__).parent.parent / "data"

STRATEGIES = ["random", "mean", "max", "min"]


def similarity_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """(N_A, N_B) cosine similarity matrix for all frame combinations."""
    return emb_a @ emb_b.T


def aggregate(sim_mat: np.ndarray, strategy: str) -> float:
    """Reduce similarity matrix to one score using the given strategy."""
    flat = sim_mat.flatten()
    if strategy == "random":
        return float(flat[random.randint(0, len(flat) - 1)])
    elif strategy == "mean":
        return float(np.mean(flat))
    elif strategy == "max":
        return float(np.max(flat))
    elif strategy == "min":
        return float(np.min(flat))
    raise ValueError(f"Unknown strategy: {strategy}")


def load_pairs(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def compute_scores(pairs: list[dict]) -> list[np.ndarray]:
    """For each pair compute and return its similarity matrix."""
    results = []
    for row in pairs:
        name_a = Path(row["videoA"]).stem
        name_b = Path(row["videoB"]).stem
        emb_a  = get_embeddings(name_a)
        emb_b  = get_embeddings(name_b)
        if emb_a is None or emb_b is None:
            print(f"  [SKIP] {name_a} vs {name_b}")
            continue
        results.append(similarity_matrix(emb_a, emb_b))
    return results


def optimal_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> float:
    """
    Find threshold closest to top-left corner of ROC curve (0, 1).
    This maximizes TPR while minimizing FPR.
    """
    distances = np.sqrt(fpr**2 + (1 - tpr)**2)
    return float(thresholds[np.argmin(distances)])


def run():
    random.seed(RANDOM_SEED)

    true_pairs  = load_pairs(TRUE_CSV)
    false_pairs = load_pairs(FALSE_CSV)

    print("Extracting embeddings for True pairs...")
    true_mats  = compute_scores(true_pairs)

    print("Extracting embeddings for False pairs...")
    false_mats = compute_scores(false_pairs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random classifier")

    print(f"\n{'Strategy':<10}  {'AUC':>6}  {'Optimal threshold':>18}")
    print(f"{'-'*10}  {'-'*6}  {'-'*18}")

    for strategy in STRATEGIES:
        scores = ([aggregate(m, strategy) for m in true_mats] +
                  [aggregate(m, strategy) for m in false_mats])
        labels = [1] * len(true_mats) + [0] * len(false_mats)

        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc  = auc(fpr, tpr)
        opt_thr  = optimal_threshold(fpr, tpr, thresholds)

        ax.plot(fpr, tpr, label=f"{strategy}  (AUC={roc_auc:.3f}, thr={opt_thr:.3f})")
        print(f"{strategy:<10}  {roc_auc:>6.3f}  {opt_thr:>18.4f}")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC krivka — porovnanie stratégií výberu framov")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = OUT_DIR / "roc_curve_sampled.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nROC curve saved -> {out_path}")
    plt.show()


if __name__ == "__main__":
    run()