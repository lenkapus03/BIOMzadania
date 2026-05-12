"""
verification/full_database_eval.py - Full-database face verification evaluation.

Runs the chosen method (ArcFace + min aggregation) on all 234 True and 234 False
pairs, plots the ROC curve, draws the confusion matrix, and reports accuracy at
the fixed threshold of 0.004.
"""

import csv
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

from zad2.verification.extract_embeddings import get_embeddings

RANDOM_SEED = 42
STRATEGY    = "min"
THRESHOLD   = 0.004

DATA_DIR = Path(__file__).parent.parent / "data"
TRUE_CSV = DATA_DIR / "TruePairs.csv"
FALSE_CSV = DATA_DIR / "FalsePairs.csv"
OUT_DIR  = DATA_DIR


def similarity_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """(N_A, N_B) cosine-similarity matrix (L2-normalised embeddings)."""
    return emb_a @ emb_b.T


def aggregate_min(sim_mat: np.ndarray) -> float:
    return float(np.min(sim_mat))


def load_pairs(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def compute_scores(pairs: list[dict], label_name: str) -> list[float]:
    scores = []
    total = len(pairs)
    for i, row in enumerate(pairs, 1):
        name_a = Path(row["videoA"]).stem
        name_b = Path(row["videoB"]).stem
        print(f"  [{i:3d}/{total}] {name_a} vs {name_b}", end="  ")
        emb_a = get_embeddings(name_a)
        emb_b = get_embeddings(name_b)
        if emb_a is None or emb_b is None:
            print("[SKIP]")
            scores.append(None)
            continue
        sim = aggregate_min(similarity_matrix(emb_a, emb_b))
        print(f"score={sim:.4f}")
        scores.append(sim)
    return scores


def optimal_threshold(fpr, tpr, thresholds):
    distances = np.sqrt(fpr**2 + (1 - tpr)**2)
    return float(thresholds[np.argmin(distances)])


def run():
    random.seed(RANDOM_SEED)

    true_pairs  = load_pairs(TRUE_CSV)
    false_pairs = load_pairs(FALSE_CSV)

    print(f"Loaded {len(true_pairs)} True pairs and {len(false_pairs)} False pairs.\n")

    print("=== Computing scores for True pairs ===")
    true_scores = compute_scores(true_pairs, "True")

    print("\n=== Computing scores for False pairs ===")
    false_scores = compute_scores(false_pairs, "False")

    # Drop pairs where embedding extraction failed
    valid_true  = [s for s in true_scores  if s is not None]
    valid_false = [s for s in false_scores if s is not None]

    scores = valid_true + valid_false
    labels = [1] * len(valid_true) + [0] * len(valid_false)

    print(f"\nValid pairs: {len(valid_true)} True, {len(valid_false)} False "
          f"(dropped {len(true_scores) - len(valid_true) + len(false_scores) - len(valid_false)})\n")

    # ── ROC curve ──────────────────────────────────────────────────────────────
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc  = auc(fpr, tpr)
    opt_thr  = optimal_threshold(fpr, tpr, thresholds)

    print(f"Strategy : {STRATEGY}")
    print(f"AUC      : {roc_auc:.4f}")
    print(f"Optimal threshold (top-left ROC): {opt_thr:.4f}")
    print(f"Fixed threshold used            : {THRESHOLD:.4f}")

    # ── Confusion matrix at fixed threshold ────────────────────────────────────
    preds = [1 if s >= THRESHOLD else 0 for s in scores]
    cm    = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy  = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    print(f"\nConfusion matrix at threshold={THRESHOLD}:")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1       : {f1:.4f}")

    # ── Figure: ROC + confusion matrix side by side ───────────────────────────
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.4, 1])

    # ROC curve panel
    ax_roc = fig.add_subplot(gs[0])
    ax_roc.plot(fpr, tpr,
                color="steelblue", linewidth=2,
                label=f"ArcFace – min  (AUC={roc_auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random classifier")

    # Mark fixed threshold
    idx_fixed = np.argmin(np.abs(thresholds - THRESHOLD))
    ax_roc.scatter(fpr[idx_fixed], tpr[idx_fixed],
                   color="red", zorder=5,
                   label=f"Fixed thr={THRESHOLD} "
                         f"(FPR={fpr[idx_fixed]:.3f}, TPR={tpr[idx_fixed]:.3f})")

    # Mark optimal threshold
    idx_opt = np.argmin(np.sqrt(fpr**2 + (1 - tpr)**2))
    ax_roc.scatter(fpr[idx_opt], tpr[idx_opt],
                   color="green", marker="^", zorder=5,
                   label=f"Optimal thr={opt_thr:.4f} "
                         f"(FPR={fpr[idx_opt]:.3f}, TPR={tpr[idx_opt]:.3f})")

    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC krivka – celá databáza\n(stratégia: {STRATEGY})")
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.grid(True, alpha=0.3)

    # Confusion matrix panel
    ax_cm = fig.add_subplot(gs[1])
    disp  = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["False pair", "True pair"])
    disp.plot(ax=ax_cm, colorbar=False, cmap="Blues")
    ax_cm.set_title(
        f"Konfúzna matica\n"
        f"threshold={THRESHOLD}  accuracy={accuracy:.3f}"
    )

    plt.tight_layout()
    out_path = OUT_DIR / "roc_and_cm_full_database.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nFigure saved -> {out_path}")
    plt.show()


if __name__ == "__main__":
    run()