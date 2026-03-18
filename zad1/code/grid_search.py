import os
import json
import cv2
import copy
import itertools

from zad1.code.core.constants import CIRCLE_NAMES, CIRCLE_CSV_INDEX
from zad1.code.data.load_data import load_valid_records
from zad1.code.core.image_processor import ImageProcessor
from zad1.code.core.renderer import Renderer
from zad1.code.core.detector import detect_circle
from zad1.code.core.evaluator import evaluate_single
from zad1.code.state.app_state import AppState
from zad1.code.ui.ui_parameters import PARAMETER_CONFIG

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.normpath(os.path.join(CURRENT_DIR, "..", "data"))
CONFIG_FOLDER = os.path.normpath(os.path.join(CURRENT_DIR, "..", "config"))
CSV_PATH = os.path.join(DATA_FOLDER, "iris_annotation.csv")

IOU_THRESHOLD = 0.75


def make_grid(center, step, n=1, min_val=None, max_val=None):
    """
    Vygeneruje 2*n+1 hodnôt okolo centra s daným krokom.
    Hodnoty sa prispôsobujú podľa [min_val, max_val] v PARAMETER_CONFIG.
    """
    values = [round(center + i * step, 4) for i in range(-n, n + 1)]
    if min_val is not None:
        values = [max(min_val, v) for v in values]
    if max_val is not None:
        values = [min(max_val, v) for v in values]
    return values


def load_config(circle_name):
    """Načíta uloženú konfiguráciu pre danú kružnicu z JSON súboru."""
    path = os.path.join(CONFIG_FOLDER, f"{circle_name}_config.json")
    with open(path) as f:
        return json.load(f)


def build_param_grid(cfg):
    """
    Grid search cez 6 najdôležitejších Hough parametrov.
    3^6 = 729 kombinácií — pod hranicou 1000.
    Predspracovanie zostáva fixné podľa uloženej konfigurácie.
    """
    grid = {}

    # ── Hough
    # grid["hough_param1"] = make_grid(float(cfg["hough_param1"]), step=_grid_step("hough_param1"), n=1,
    #                                  min_val=PARAMETER_CONFIG["hough_param1"]["min"],
    #                                  max_val=PARAMETER_CONFIG["hough_param1"]["max"])
    # grid["hough_param2"] = make_grid(float(cfg["hough_param2"]), step=_grid_step("hough_param2"), n=1,
    #                                  min_val=PARAMETER_CONFIG["hough_param2"]["min"],
    #                                  max_val=PARAMETER_CONFIG["hough_param2"]["max"])
    # grid["hough_mindist"] = make_grid(float(cfg["hough_mindist"]), step=_grid_step("hough_mindist"), n=1,
    #                                   min_val=PARAMETER_CONFIG["hough_mindist"]["min"],
    #                                   max_val=PARAMETER_CONFIG["hough_mindist"]["max"])
    # grid["hough_minr"] = make_grid(float(cfg["hough_minr"]), step=_grid_step("hough_minr"), n=1,
    #                                min_val=PARAMETER_CONFIG["hough_minr"]["min"],
    #                                max_val=PARAMETER_CONFIG["hough_minr"]["max"])
    # grid["hough_maxr"] = make_grid(float(cfg["hough_maxr"]), step=_grid_step("hough_maxr"), n=1,
    #                                min_val=PARAMETER_CONFIG["hough_maxr"]["min"],
    #                                max_val=PARAMETER_CONFIG["hough_maxr"]["max"])
    grid["hough_dp"] = make_grid(float(cfg["hough_dp"]), step=_grid_step("hough_dp"), n=1,
                                 min_val=PARAMETER_CONFIG["hough_dp"]["min"],
                                 max_val=PARAMETER_CONFIG["hough_dp"]["max"])

    # ── CLAHE
    grid["clahe_clip"] = make_grid(float(cfg["clahe_clip"]), step=_grid_step("clahe_clip"), n=1, min_val=PARAMETER_CONFIG["clahe_clip"]["min"], max_val=PARAMETER_CONFIG["clahe_clip"]["max"])
    grid["clahe_tile"] = make_grid(float(cfg["clahe_tile"]), step=_grid_step("clahe_tile"), n=1, min_val=PARAMETER_CONFIG["clahe_tile"]["min"], max_val=PARAMETER_CONFIG["clahe_tile"]["max"])

    # ── Canny
    grid["threshold_1"] = make_grid(float(cfg["threshold_1"]), step=_grid_step("threshold_1"), n=1, min_val=PARAMETER_CONFIG["threshold_1"]["min"], max_val=PARAMETER_CONFIG["threshold_1"]["max"])
    grid["threshold_2"] = make_grid(float(cfg["threshold_2"]), step=_grid_step("threshold_2"), n=1, min_val=PARAMETER_CONFIG["threshold_2"]["min"], max_val=PARAMETER_CONFIG["threshold_2"]["max"])

    return grid


def _grid_step(tag):
    """
    Definuje krok gridu pre každý parameter.
    Menší krok = jemnejšie prehľadávanie, väčší krok = rýchlejšie ale hrubšie.
    """
    steps = {
        "clahe_clip":    0.5,
        "clahe_tile":    2,
        "gauss_kernel":  2,
        "gauss_sigma":   0.5,
        "threshold_1":   10,
        "threshold_2":   20,
        "hough_dp":      0.1,
        "hough_mindist": 10,
        "hough_param1":  10,
        "hough_param2":  3,
        "hough_minr":    10,
        "hough_maxr":    10,
    }
    return steps[tag]


def evaluate_on_all_records(processor, renderer, circle_name, cfg, records):
    """
    Vyhodnotí danú konfiguráciu na všetkých záznamoch.
    Pre každý záznam načíta obrazok, spustí detekciu a porovná s ground truth.
    Vracia agregované tp, fp, fn cez všetky záznamy.
    """
    total_tp, total_fp, total_fn = 0, 0, 0
    idx = CIRCLE_CSV_INDEX[circle_name]

    for record in records:
        img_path = os.path.normpath(os.path.join(DATA_FOLDER, record.image.replace("\\", "/")))
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Nastav nový obrazok do processora — ostatné nastavenia zostávajú
        processor.original_image = img
        processor.processed_image = img.copy()

        try:
            detected = detect_circle(processor, renderer, cfg, circle_name, headless=True)
        except Exception as e:
            print(f"  Chyba pri detekcii: {e}")
            detected = None

        gt = (
            getattr(record, f"center_x_{idx}"),
            getattr(record, f"center_y_{idx}"),
            getattr(record, f"polomer_{idx}")
        )

        tp, fp, fn, _ = evaluate_single(detected, gt, IOU_THRESHOLD)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    return total_tp, total_fp, total_fn


def grid_search():
    """
    Hlavná funkcia grid searchu.
    Pre každú kružnicu:
      1. Načíta uloženú konfiguráciu ako základ
      2. Zostrojí grid hodnôt okolo každého parametra
      3. Vyskúša všetky kombinácie na 100 záznamoch
      4. Uloží najlepšiu konfiguráciu podľa F1-skóre
    """
    records = load_valid_records(CSV_PATH, DATA_FOLDER, n=100)
    records = [
        r for r in records
        if os.path.exists(os.path.normpath(os.path.join(DATA_FOLDER, r.image.replace("\\", "/"))))
    ]

    # Vytvor processor a renderer — obrázok sa bude meniť v slučke pre každý záznam
    dummy_img = None
    for r in records:
        path = os.path.normpath(os.path.join(DATA_FOLDER, r.image.replace("\\", "/")))
        dummy_img = cv2.imread(path)
        if dummy_img is not None:
            break

    if dummy_img is None:
        print("Žiadny platný obrazok nenájdený")
        return
    processor = ImageProcessor(original_image=dummy_img)
    state = AppState()
    renderer = Renderer(processor=processor, state=state)

    best_results = {}

    for circle_name in CIRCLE_NAMES:
        print(f"\nGrid search pre: {circle_name}")
        base_cfg = load_config(circle_name)
        param_grid = build_param_grid(base_cfg)

        # Vygeneruj kartézsky súčin všetkých hodnôt parametrov
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        total = 1
        for v in values:
            total *= len(v)
        print(f"  Počet kombinácií: {total}")

        best_score = -1
        best_cfg = None
        best_metrics = None

        for i, combo in enumerate(itertools.product(*values)):
            cfg = base_cfg.copy()
            for k, v in zip(keys, combo):
                # Integer parametre zaokrúhli — float hodnoty nechaj ako sú
                if k in ("clahe_tile", "gauss_kernel", "threshold_1", "threshold_2",
                         "hough_mindist", "hough_param1", "hough_param2",
                         "hough_minr", "hough_maxr"):
                    cfg[k] = max(1, int(v))
                else:
                    cfg[k] = v

            if cfg["hough_maxr"] > 0 and cfg["hough_minr"] >= cfg["hough_maxr"]:
                continue

            tp, fp, fn = evaluate_on_all_records(processor, renderer, circle_name, cfg, records)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            # Ulož konfiguráciu ak je lepšia ako doterajšie maximum
            score = (precision + recall + f1) / 3
            if score > best_score:
                best_score = score
                best_cfg = copy.deepcopy(cfg)
                best_metrics = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "score": score,
                    "tp": tp, "fp": fp, "fn": fn
                }
                print(f"  [{i}/{total}] Nové najlepšie "
                      f"score={score:.3f} F1={f1:.3f} P={precision:.3f} R={recall:.3f} | " +
                      " ".join(f"{k}={cfg[k]}" for k in keys))

            if i % 1 == 0 and i > 0:
                print(f"  [{i}/{total}] priebežne najlepšie score={best_score:.3f} "
                      f"F1={best_metrics['f1']:.3f} P={best_metrics['precision']:.3f} R={best_metrics['recall']:.3f}")

        best_results[circle_name] = {
            "best_config": best_cfg,
            "metrics": best_metrics
        }

    # Výpis súhrnných výsledkov
    print(f"\n{'='*65}")
    print(f"{'Kružnica':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'TP':>4} {'FP':>4} {'FN':>4}")
    print(f"{'-'*65}")
    for name, r in best_results.items():
        m = r["metrics"]
        print(f"{name:<12} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} "
              f"{m['tp']:>4} {m['fp']:>4} {m['fn']:>4}")

    # Ulož najlepšie konfigurácie pre každú kružnicu do JSON
    output_path = os.path.join(CONFIG_FOLDER, "grid_search_results.json")
    with open(output_path, "w") as f:
        json.dump(best_results, f, indent=2)
    print(f"\nVýsledky uložené do: {output_path}")


if __name__ == "__main__":
    grid_search()