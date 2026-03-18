import numpy as np

# Source: https://www.geeksforgeeks.org/area-of-intersection-of-two-circles/
def circle_iou(cx1, cy1, r1, cx2, cy2, r2):
    # Euklidovská vzdialenosť stredov
    d = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

    # Prípad 1: kruhy sa nepretínajú
    if d > r1 + r2:
        return 0.0

    # Prípad 2a: kruh 2 je celý vnútri kruhu 1
    if d <= (r1 - r2) and r1 >= r2:
        intersection = np.pi * r2 ** 2
        union = np.pi * r1 ** 2
        return intersection / union

    # Prípad 2b: kruh 1 je celý vnútri kruhu 2
    if d <= (r2 - r1) and r2 >= r1:
        intersection = np.pi * r1 ** 2
        union = np.pi * r2 ** 2
        return intersection / union

    # Prípad 3: čiastočný prienik
    alpha = np.arccos((r1**2 + d**2 - r2**2) / (2 * r1 * d)) * 2
    beta  = np.arccos((r2**2 + d**2 - r1**2) / (2 * r2 * d)) * 2

    a1 = 0.5 * beta  * r2**2 - 0.5 * r2**2 * np.sin(beta)
    a2 = 0.5 * alpha * r1**2 - 0.5 * r1**2 * np.sin(alpha)

    intersection = a1 + a2
    union = np.pi * (r1**2 + r2**2) - intersection
    return intersection / union


def evaluate_single(detected, ground_truth, iou_threshold=0.75):
    """
    Vyhodnotí jeden detekovaný kruh oproti ground truth.

    detected:      (x, y, r) alebo None ak nebol nič detekovaný
    ground_truth:  (x, y, r) z anotačného CSV
    iou_threshold: minimálne IoU pre True Positive (default 0.75)

    Vracia (tp, fp, fn, iou):
      tp=1 ak IoU >= threshold
      fp=1 ak detekcia existuje ale IoU < threshold
      fn=1 ak nebola detekcia alebo IoU < threshold
    """
    gt_x, gt_y, gt_r = ground_truth

    # Preskočiť neplatné anotácie (záporný alebo nulový polomer)
    if gt_r <= 0:
        return 0, 0, 0, 0.0

    # Nič nedetekované, ale ground truth existuje → False Negative
    if detected is None:
        return 0, 0, 1, 0.0

    iou = circle_iou(detected[0], detected[1], detected[2], gt_x, gt_y, gt_r)

    if iou >= iou_threshold:
        return 1, 0, 0, iou  # True Positive
    else:
        return 0, 1, 1, iou  # False Positive + False Negative


def compute_metrics(results):
    metrics = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for name, r in results.items():
        tp, fp, fn = r["tp"], r["fp"], r["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": r["iou"],
            "tp": tp, "fp": fp, "fn": fn
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Celkové metriky cez všetky 4 kružnice
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    total_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    total_f1        = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0.0
    metrics["total"] = {
        "precision": total_precision,
        "recall": total_recall,
        "f1": total_f1,
        "iou": None,
        "tp": total_tp, "fp": total_fp, "fn": total_fn
    }

    return metrics

def print_metrics(metrics, image_name=None, iou_threshold=0.75):
    print(f"\n{'='*65}")
    if image_name:
        print(f"Vyhodnotenie: {image_name}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"{'='*65}")
    print(f"{'Kružnica':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'IoU':>8} {'TP':>4} {'FP':>4} {'FN':>4}")
    print(f"{'-'*65}")
    for name, m in metrics.items():
        iou_str = f"{m['iou']:>8.3f}" if m["iou"] is not None else f"{'N/A':>8}"
        print(f"{name:<12} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} {iou_str} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}")
    print(f"{'='*65}\n")

def compute_batch_metrics(totals):
    """Vypočíta precision, recall, F1 z agregovaných tp/fp/fn cez 100 záznamov."""
    metrics = {}
    for name, t in totals.items():
        tp, fp, fn = t["tp"], t["fp"], t["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[name] = {
            "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn
        }
    return metrics


def print_batch_metrics(metrics, iou_threshold=0.75):
    print(f"\n{'='*65}")
    print(f"Batch evaluácia (100 záznamov), IoU threshold: {iou_threshold}")
    print(f"{'='*65}")
    print(f"{'Kružnica':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'TP':>4} {'FP':>4} {'FN':>4}")
    print(f"{'-'*65}")
    for name, m in metrics.items():
        print(f"{name:<12} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} "
              f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4}")
    print(f"{'='*65}\n")