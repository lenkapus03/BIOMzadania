from abc import ABC, abstractmethod

class Command(ABC):
    @abstractmethod
    def execute(self, sender=None, app_data=None, user_data=None):
        pass

from zad1.code.commands.image_command import ImageCommand
import dearpygui.dearpygui as dpg


class CannyCommand(ImageCommand):
    def execute(self, sender=None, app_data=None, user_data=None):
        self.processor.canny_threshold_1 = int(dpg.get_value("threshold_1"))
        self.processor.canny_threshold_2 = float(dpg.get_value("threshold_2"))

        self.refresh()

from zad1.code.commands.image_command import ImageCommand
import dearpygui.dearpygui as dpg

class CLAHECommand(ImageCommand):
    def execute(self, sender=None, app_data=None, user_data=None):
        self.processor.clip_limit = float(dpg.get_value("clahe_clip"))
        self.processor.grid_size = int(dpg.get_value("clahe_tile"))
        self.refresh()

class CommandDispatcher:
    def __init__(self):
        self._commands = {}

    def register(self, name, command):
        self._commands[name] = command

    def execute(self, name, sender=None, app_data=None, user_data=None):
        if name in self._commands:
            self._commands[name].execute(sender, app_data, user_data)

import os
import json
from datetime import datetime

from zad1.code.commands.image_command import ImageCommand
from zad1.code.core.helpers import apply_params
from zad1.code.core.detector import evaluate_record, evaluate_batch
from zad1.code.core.evaluator import print_metrics, compute_metrics, compute_batch_metrics, print_batch_metrics
from zad1.code.ui.ui_parameters import PARAMETER_CONFIG, TOGGLE_TAGS

CONFIG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))
IOU_THRESHOLD = 0.75

class EvaluateCommand(ImageCommand):
    def __init__(self, processor, renderer, state):
        super().__init__(processor, renderer)
        self.state = state

    def execute(self, sender=None, app_data=None, user_data=None):
        record = self.state.current_record
        if record is None or self.processor.original_image is None:
            print("Žiadny obrazok načítaný")
            return

        current_params = self._backup_processor()

        # 1. Vyhodnoť aktuálny obrazok
        print("\n--- Vyhodnotenie aktuálneho obrazka ---")
        single_result = evaluate_record(
            self.processor, self.renderer, record, self.state.circle_params, IOU_THRESHOLD
        )
        single_metrics = compute_metrics(single_result)
        print_metrics(single_metrics, image_name=record.image, iou_threshold=IOU_THRESHOLD)

        # 2. Batch evaluácia na 100 záznamoch
        print("\n--- Batch evaluácia na 100 záznamoch ---")
        batch_totals = evaluate_batch(self.processor, self.renderer, self.state.circle_params, IOU_THRESHOLD)
        batch_metrics = compute_batch_metrics(batch_totals)
        print_batch_metrics(batch_metrics, IOU_THRESHOLD)

        # Obnov pôvodné parametre a prerendruj
        apply_params(current_params, self.processor, self.renderer)
        self.refresh(apply_processor=True)

        self._save_results(single_result, single_metrics, batch_metrics, record.image)

    def _backup_processor(self):
        backup = {}
        # back-up processor parametre
        for tag, cfg in PARAMETER_CONFIG.items():
            backup[tag] = getattr(self.processor, tag, cfg["default"])
        for tag in TOGGLE_TAGS:
            backup[tag] = getattr(self.processor, tag, False)
        # back-up renderer parametre
        backup["canvas_width"] = self.renderer.canvas_width
        backup["canvas_height"] = self.renderer.canvas_height
        return backup

    def _save_results(self, single_result, single_metrics, batch_metrics, image_name):
        output = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "iou_threshold": IOU_THRESHOLD,
            "single_image": {
                "image": image_name,
                "per_circle": single_result,
                "metrics": single_metrics
            },
            "batch_100": {
                "metrics": batch_metrics
            }
        }
        path = os.path.join(CONFIG_DIR, "eval_results.json")
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Výsledky uložené do: {path}")

from zad1.code.commands.image_command import ImageCommand
import dearpygui.dearpygui as dpg

class GaussianBlurCommand(ImageCommand):
    def execute(self, sender=None, app_data=None, user_data=None):
        self.processor.gauss_kernel = int(dpg.get_value("gauss_kernel"))
        self.processor.gauss_sigma = float(dpg.get_value("gauss_sigma"))

        self.refresh()

from zad1.code.commands.image_command import ImageCommand
import dearpygui.dearpygui as dpg

class HoughCommand(ImageCommand):
    def execute(self, sender=None, app_data=None, user_data=None):
        self.processor.hough_dp = float(dpg.get_value("hough_dp"))
        self.processor.hough_mindist = int(dpg.get_value("hough_mindist"))
        self.processor.hough_param1 = int(dpg.get_value("hough_param1"))
        self.processor.hough_param2 = int(dpg.get_value("hough_param2"))
        self.processor.hough_minr = int(dpg.get_value("hough_minr"))
        self.processor.hough_maxr = int(dpg.get_value("hough_maxr"))
        self.refresh()

from abc import ABC
from zad1.code.commands.base_command import Command

class ImageCommand(Command, ABC):
    def __init__(self, processor, renderer):
        self.processor = processor
        self.renderer = renderer

    def refresh(self, apply_processor=True):
        self.renderer.refresh_texture(apply_processor)

from zad1.code.commands.image_command import ImageCommand

class PreviewOriginalCommand(ImageCommand):
    def __init__(self, processor, renderer, show_original: bool):
        super().__init__(processor, renderer)
        self.show_original = show_original

    def execute(self, sender=None, app_data=None, user_data=None):
        self.processor.preview_original = bool(app_data)
        self.refresh(apply_processor=False)

import json
import os
import dearpygui.dearpygui as dpg
from zad1.code.commands.image_command import ImageCommand
from zad1.code.ui.ui_parameters import PARAMETER_CONFIG, TOGGLE_TAGS, CANVAS_TAGS

CONFIG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))

class SaveSettingsCommand(ImageCommand):
    def __init__(self, processor, renderer, state):
        super().__init__(processor, renderer)
        self.state = state

    def execute(self, sender=None, app_data=None, user_data=None):
        current = self.state.active_circle

        self.state.circle_params[current] = {
            tag: dpg.get_value(tag)
            for tag in PARAMETER_CONFIG
        }
        for tag in TOGGLE_TAGS:
            self.state.circle_params[current][tag] = dpg.get_value(tag)
        for tag in CANVAS_TAGS:
            val = dpg.get_value(tag)
            print(f"Saving canvas tag {tag} = {val}")
            self.state.circle_params[current][tag] = val

        os.makedirs(CONFIG_DIR, exist_ok=True)
        for circle, params in self.state.circle_params.items():
            print(f"Saving {circle}: {params}")
            path = os.path.join(CONFIG_DIR, f"{circle}_config.json")
            with open(path, "w") as f:
                json.dump(params, f, indent=2)

import json
import os
from zad1.code.commands.image_command import ImageCommand
import dearpygui.dearpygui as dpg

from zad1.code.core.helpers import apply_params
from zad1.code.ui.ui_parameters import PARAMETER_CONFIG, TOGGLE_TAGS, CANVAS_TAGS

CONFIG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))

class SelectCircleCommand(ImageCommand):
    def __init__(self, processor, renderer, state):
        super().__init__(processor, renderer)
        self.state = state

    def execute(self, sender=None, app_data=None, user_data=None):
        old = self.state.active_circle
        self.state.circle_params[old] = {
            tag: dpg.get_value(tag)
            for tag in PARAMETER_CONFIG
        }
        for tag in TOGGLE_TAGS:
            self.state.circle_params[old][tag] = dpg.get_value(tag)
        for tag in CANVAS_TAGS:
            self.state.circle_params[old][tag] = dpg.get_value(tag)

        self.state.active_circle = app_data

        saved = self.state.circle_params[app_data]
        apply_params(saved, self.processor, self.renderer)
        self.refresh(apply_processor=True)

from zad1.code.commands.image_command import ImageCommand


class ToggleCommand(ImageCommand):
    def __init__(self, processor, renderer, attr_name):
        super().__init__(processor, renderer)
        self.attr_name = attr_name

    def execute(self, sender=None, app_data=None, user_data=None):
        setattr(self.processor, self.attr_name, bool(app_data))
        self.refresh()

from zad1.code.commands.image_command import ImageCommand

class ToggleRendererCommand(ImageCommand):
    def __init__(self, processor, renderer, attr_name):
        super().__init__(processor, renderer)
        self.attr_name = attr_name

    def execute(self, sender=None, app_data=None, user_data=None):
        setattr(self.renderer, self.attr_name, bool(app_data))
        self.refresh(apply_processor=False)

from zad1.code.commands.image_command import ImageCommand
import dearpygui.dearpygui as dpg

class UpdateCanvasCommand(ImageCommand):
    def execute(self, sender=None, app_data=None, user_data=None):
        self.renderer.canvas_width = int(dpg.get_value("canvas_width"))
        self.renderer.canvas_height = int(dpg.get_value("canvas_height"))

        self.refresh(apply_processor=True)

CIRCLE_NAMES = ["iris", "pupil", "lower_lid", "upper_lid"]
CIRCLE_CSV_INDEX = {
    "pupil":     1,
    "iris":      2,
    "lower_lid": 3,
    "upper_lid": 4,
}

import os
import cv2

from zad1.code.core.constants import CIRCLE_CSV_INDEX, CIRCLE_NAMES
from zad1.code.core.helpers import apply_params_headless, apply_params
from zad1.code.data.load_data import load_valid_records
from zad1.code.core.evaluator import evaluate_single

def detect_circle(processor, renderer, cfg, circle_name, headless=False):
    """
    Detekuje jednu kružnicu na obrazku pomocou uložených parametrov.

    Postup:
      1. Nastav parametre processora a renderera z konfigurácie
      2. Spracuj obrazok (histogram eq, CLAHE, blur, canny)
      3. Postav canvas s paddingom
      4. Spusti Houghovu transformáciu na canvase
      5. Filtruj kandidátov a vyber najlepší kruh
      6. Prepočítaj súradnice z canvas priestoru späť do obrazkového priestoru

    Vracia (x, y, r) v súradniciach pôvodného obrazka, alebo None ak nič nenašlo.
    """
    # Nastav parametre processora a renderera z konfigurácie
    if headless:
        apply_params_headless(cfg, processor, renderer)
    else:
        apply_params(cfg, processor, renderer)

    # Spracuj obrazok (histogram eq, CLAHE, blur, canny)
    processor.apply()

    # Postav canvas s bielym paddingom okolo obrazka
    canvas = renderer._build_canvas(processor.processed_image)

    # Konvertuj na odtiene šedej — HoughCircles vyžaduje grayscale vstup
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    # Detekuj kružnice pomocou Houghovej transformácie
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=processor.hough_dp,
        minDist=processor.hough_mindist,
        param1=processor.hough_param1,
        param2=processor.hough_param2,
        minRadius=processor.hough_minr,
        maxRadius=processor.hough_maxr
    )

    # Filtruj kandidátov podľa pravidiel pre daný typ kružnice a vyber najlepší
    best = processor.filter_circles(
        circles,
        processor.original_image.shape,  # rozmery pôvodného obrazka pre filtrovacie pravidlá
        circle_name,
        canvas_shape=canvas.shape         # rozmery canvasu pre prepočet súradníc
    )

    if best is None:
        return None

    # Prepočet súradníc z canvas priestoru do obrazkového priestoru
    # Canvas má padding — stred obrazka je posunutý o (canv_x1, canv_y1)
    img_h, img_w = processor.original_image.shape[:2]
    canv_x1 = max(0, (renderer.canvas_width - img_w) // 2)
    canv_y1 = max(0, (renderer.canvas_height - img_h) // 2)
    x, y, r = int(best[0][0][0]), int(best[0][0][1]), int(best[0][0][2])
    return (x - canv_x1, y - canv_y1, r)


def evaluate_record(processor, renderer, record, circle_params, iou_threshold=0.75):
    """
    Vyhodnotí detekciu všetkých kružníc pre jeden záznam.

    Pre každú kružnicu:
      - detekuje kruh pomocou uložených parametrov
      - porovná ho s ground truth anotáciou pomocou IoU
      - vypočíta TP, FP, FN

    Vracia dict {circle_name: {detected, ground_truth, iou, tp, fp, fn}}
    """
    result = {}
    for circle_name in CIRCLE_NAMES:
        cfg = circle_params.get(circle_name, {})
        if not cfg:
            # Konfigurácia pre túto kružnicu nebola uložená — preskočíme
            continue

        # Detekuj kruh s parametrami pre túto kružnicu
        detected = detect_circle(processor, renderer, cfg, circle_name)

        # Načítaj ground truth súradnice z CSV záznamu
        idx = CIRCLE_CSV_INDEX[circle_name]
        gt = (
            getattr(record, f"center_x_{idx}"),
            getattr(record, f"center_y_{idx}"),
            getattr(record, f"polomer_{idx}")
        )

        # Vyhodnoť detekciu oproti ground truth
        tp, fp, fn, iou = evaluate_single(detected, gt, iou_threshold)
        result[circle_name] = {
            "detected": list(detected) if detected else None,
            "ground_truth": list(gt),
            "iou": iou,
            "tp": tp, "fp": fp, "fn": fn
        }
    return result

def evaluate_batch(processor, renderer, circle_params, iou_threshold=0.75):
    """
    Vyhodnotí detekciu na 100 náhodných záznamoch.
    Vracia agregované tp/fp/fn pre každú kružnicu.
    """
    DATA_FOLDER = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    CSV_PATH = os.path.join(DATA_FOLDER, "iris_annotation.csv")

    records = load_valid_records(CSV_PATH, DATA_FOLDER, n=100)
    totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CIRCLE_NAMES}

    for record in records:
        img_path = os.path.normpath(os.path.join(DATA_FOLDER, record.image.replace("\\", "/")))
        img = cv2.imread(img_path)
        if img is None:
            continue

        for circle_name in CIRCLE_NAMES:
            cfg = circle_params.get(circle_name, {})
            if not cfg:
                continue

            processor.original_image = img
            processor.processed_image = img.copy()

            try:
                detected = detect_circle(processor, renderer, cfg, circle_name, headless=True)
            except Exception:
                detected = None

            idx = CIRCLE_CSV_INDEX[circle_name]
            gt = (
                getattr(record, f"center_x_{idx}"),
                getattr(record, f"center_y_{idx}"),
                getattr(record, f"polomer_{idx}")
            )

            tp, fp, fn, _ = evaluate_single(detected, gt, iou_threshold)
            totals[circle_name]["tp"] += tp
            totals[circle_name]["fp"] += fp
            totals[circle_name]["fn"] += fn

    return totals

import numpy as np

# Zdroj: https://www.geeksforgeeks.org/area-of-intersection-of-two-circles/
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

import dearpygui.dearpygui as dpg
from zad1.code.ui.ui_parameters import PARAMETER_CONFIG, TOGGLE_TAGS, CANVAS_TAGS

def update_texture(state, image_data, width, height):
    """
    Aktualizuje DPG textúru novými dátami obrazka.
    Odstráni starú textúru ak existuje a vytvorí novú s danými rozmermi.
    Ak existuje widget 'displayed_image_widget', aktualizuje ho na novú textúru.
    """
    if image_data is None:
        return None

    # Odstran predchadzajucu texturu ak existuje
    if getattr(state, 'texture_id', None) is not None and dpg.does_item_exist(state.texture_id):
        dpg.delete_item(state.texture_id)

    # Vytvor novu texturu
    with dpg.texture_registry():
        tex_id = dpg.add_dynamic_texture(width=width, height=height, default_value=image_data)

    state.texture_id = tex_id

    # Update-ni widget obrazok
    if dpg.does_item_exist("displayed_image_widget"):
        dpg.configure_item(
            "displayed_image_widget",
            texture_tag=tex_id,
            width=width,
            height=height
        )

    return tex_id

def apply_params(saved, processor, renderer):
    """
    Aplikuje uložené parametre do UI (DPG slidery), processora a renderera.
    Používa sa v rámci aplikácie kde je dostupný DPG kontext.
    Na konci zavolá processor.apply() aby sa obraz prerendroval s novými parametrami.
    """
    for tag, cfg in PARAMETER_CONFIG.items():
        value = saved.get(tag, cfg["default"])
        dpg.set_value(tag, value)

    for tag in TOGGLE_TAGS:
        value = saved.get(tag, False)
        dpg.set_value(tag, value)
        setattr(processor, tag, value)

    for tag in CANVAS_TAGS:
        default = 320 if tag == "canvas_width" else 280
        dpg.set_value(tag, saved.get(tag, default))

    renderer.canvas_width = int(saved.get("canvas_width", 320))
    renderer.canvas_height = int(saved.get("canvas_height", 280))

    processor.clip_limit = saved.get("clahe_clip", PARAMETER_CONFIG["clahe_clip"]["default"])
    processor.grid_size = int(saved.get("clahe_tile", PARAMETER_CONFIG["clahe_tile"]["default"]))
    processor.gauss_kernel = int(saved.get("gauss_kernel", PARAMETER_CONFIG["gauss_kernel"]["default"]))
    processor.gauss_sigma = float(saved.get("gauss_sigma", PARAMETER_CONFIG["gauss_sigma"]["default"]))
    processor.canny_threshold_1 = int(saved.get("threshold_1", PARAMETER_CONFIG["threshold_1"]["default"]))
    processor.canny_threshold_2 = int(saved.get("threshold_2", PARAMETER_CONFIG["threshold_2"]["default"]))
    processor.hough_dp = float(saved.get("hough_dp", PARAMETER_CONFIG["hough_dp"]["default"]))
    processor.hough_mindist = int(saved.get("hough_mindist", PARAMETER_CONFIG["hough_mindist"]["default"]))
    processor.hough_param1 = int(saved.get("hough_param1", PARAMETER_CONFIG["hough_param1"]["default"]))
    processor.hough_param2 = int(saved.get("hough_param2", PARAMETER_CONFIG["hough_param2"]["default"]))
    processor.hough_minr = int(saved.get("hough_minr", PARAMETER_CONFIG["hough_minr"]["default"]))
    processor.hough_maxr = int(saved.get("hough_maxr", PARAMETER_CONFIG["hough_maxr"]["default"]))

    processor.apply()

def apply_params_headless(cfg, processor, renderer):
    """Verzia apply_params bez DPG — pre použitie mimo aplikácie (grid search)."""
    for tag in TOGGLE_TAGS:
        if tag in cfg:
            setattr(processor, tag, cfg[tag])

    processor.clip_limit = cfg.get("clahe_clip", PARAMETER_CONFIG["clahe_clip"]["default"])
    processor.grid_size = int(cfg.get("clahe_tile", PARAMETER_CONFIG["clahe_tile"]["default"]))
    processor.gauss_kernel = int(cfg.get("gauss_kernel", PARAMETER_CONFIG["gauss_kernel"]["default"]))
    processor.gauss_sigma = float(cfg.get("gauss_sigma", PARAMETER_CONFIG["gauss_sigma"]["default"]))
    processor.canny_threshold_1 = int(cfg.get("threshold_1", PARAMETER_CONFIG["threshold_1"]["default"]))
    processor.canny_threshold_2 = int(cfg.get("threshold_2", PARAMETER_CONFIG["threshold_2"]["default"]))
    processor.hough_dp = float(cfg.get("hough_dp", PARAMETER_CONFIG["hough_dp"]["default"]))
    processor.hough_mindist = int(cfg.get("hough_mindist", PARAMETER_CONFIG["hough_mindist"]["default"]))
    processor.hough_param1 = int(cfg.get("hough_param1", PARAMETER_CONFIG["hough_param1"]["default"]))
    processor.hough_param2 = int(cfg.get("hough_param2", PARAMETER_CONFIG["hough_param2"]["default"]))
    processor.hough_minr = int(cfg.get("hough_minr", PARAMETER_CONFIG["hough_minr"]["default"]))
    processor.hough_maxr = int(cfg.get("hough_maxr", PARAMETER_CONFIG["hough_maxr"]["default"]))

    renderer.canvas_width = int(cfg.get("canvas_width", 320))
    renderer.canvas_height = int(cfg.get("canvas_height", 280))

    processor.apply()

import cv2

class ImageManager:
    def __init__(self):
        self.original_image = None
        self.processed_image = None

    def load_image(self, path: str):
        """Načíta obraz zo súboru. Vyhodí ValueError ak súbor neexistuje alebo nie je čitateľný."""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot load image from {path}")
        self.original_image = img
        self.processed_image = img.copy()

    def reset(self):
        """Obnoví spracovaný obraz na pôvodný."""
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()

import cv2
import numpy as np

CIRCLE_COLORS = {
    "iris":      (255, 0, 0),    # modrá
    "pupil":     (0, 255, 0),    # zelená
    "upper_lid": (0, 165, 255),  # oranžová
    "lower_lid": (0, 0, 255),    # červená
}

class ImageProcessor:
    def __init__(self, original_image=None):
        self.original_image = original_image
        self.processed_image = original_image.copy() if original_image is not None else None

        # Nastavenia
        self.use_histogram_eq = False

        self.use_clahe = False
        self.clip_limit = 2
        self.grid_size = 8

        self.use_blur = False
        self.gauss_kernel = 5
        self.gauss_sigma = 1.0

        self.use_canny = False
        self.canny_threshold_1 = 50
        self.canny_threshold_2 = 150

        self.show_hough = False
        self.show_rejected_circles = False
        self.hough_dp = 1.2
        self.hough_mindist = 50
        self.hough_param1 = 100
        self.hough_param2 = 30
        self.hough_minr = 0
        self.hough_maxr = 0

        self.preview_original = False

    # Zdroj: https://www.freedomvc.com/index.php/2021/09/11/color-image-histograms/
    def histogram_equalization(self, image):
        """Aplikuje histogramovú ekvalizáciu samostatne na každý BGR kanál."""
        b, g, r = cv2.split(image)
        return cv2.merge([
            cv2.equalizeHist(b),
            cv2.equalizeHist(g),
            cv2.equalizeHist(r)
        ])

    # Zdroj: https://medium.com/@lin.yong.hui.jason/histogram-equalization-for-color-images-using-opencv-655ae13b9dd0
    def clahe(self, image):
        """Aplikuje CLAHE (Contrast Limited Adaptive Histogram Equalization) na každý BGR kanál."""
        clahe_obj = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.grid_size, self.grid_size)
        )
        return cv2.merge([
            clahe_obj.apply(image[:, :, 0]),
            clahe_obj.apply(image[:, :, 1]),
            clahe_obj.apply(image[:, :, 2])
        ])

    # Zdroj: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
    def gaussian_blur(self, image):
        """
        Aplikuje Gaussovské rozmazanie. Ak sú kernel aj sigma nulové, vráti obraz bez zmeny.
        Párny kernel sa automaticky zvýši o 1 aby bol nepárny.
        """
        if self.gauss_kernel == 0 and self.gauss_sigma == 0:
            return image
        kernel = self.gauss_kernel
        if kernel > 0 and kernel % 2 == 0:
            kernel += 1
        return cv2.GaussianBlur(image, (kernel, kernel), self.gauss_sigma)

    # Zdroj: https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
    def canny(self, image):
        """Aplikuje Cannyho detektor hrán a konvertuje výsledok späť do BGR formátu."""
        edges = cv2.Canny(image, self.canny_threshold_1, self.canny_threshold_2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def hough(self, image, active_circle="iris", original_shape=None, show_rejected=True, detect_on=None,
              filter_canvas_shape=None, draw_offset=(0, 0)):
        """
        Detekuje kružnice pomocou Houghovej transformácie a nakreslí ich na obraz.
        Detekcia prebieha na detect_on ak je zadaný, inak priamo na image.
        draw_offset posúva nakreslené kruhy do display canvas priestoru pri show_all_circles.
        """

        # Detekuj kruhy na detect_on ak je zadaný, inak na image
        source = detect_on if detect_on is not None else image

        # Konverzia na odtiene šedej
        gray_image = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

        # Aplikácia Houghovej transformácie
        circles = cv2.HoughCircles(
            gray_image,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_dp,  # Rozlíšenie akumulátora
            minDist=self.hough_mindist,  # Min. vzdialenosť stredov kružníc
            param1=self.hough_param1,  # Vyššia hranica v Cannyho detektore
            param2=self.hough_param2,  # Hranica akumulátora pre stredy kružníc
            minRadius=self.hough_minr,  # Min. veľkosť kružníc
            maxRadius=self.hough_maxr  # Max. veľkosť kružníc
        )

        result = image.copy()
        color = CIRCLE_COLORS.get(active_circle, (255, 0, 255))

        if circles is not None:
            filter_shape = original_shape if original_shape is not None else image.shape
            # Použi filter_canvas_shape ak je zadaný, inak použi tvar source
            fcs = filter_canvas_shape if filter_canvas_shape is not None else source.shape
            accepted = self.filter_circles(circles, filter_shape, active_circle, canvas_shape=fcs)
            accepted_set = set()
            if accepted is not None:
                for c in accepted[0, :]:
                    accepted_set.add((int(c[0]), int(c[1]), int(c[2])))

            ox, oy = draw_offset
            for c in circles[0, :]:
                # Porovnávaj bez offsetu (súradnice v detect canvas priestore)
                is_accepted = (int(c[0]), int(c[1]), int(c[2])) in accepted_set

                if not is_accepted and not show_rejected:
                    continue

                # Kresli s offsetom (presunuté do display canvas priestoru)
                center = (int(c[0]) + ox, int(c[1]) + oy)
                radius = int(c[2])
                draw_color = color if is_accepted else (150, 150, 150)  # sivá = odmietnutý
                cv2.circle(result, center, 1, draw_color, 1)  # stred
                cv2.circle(result, center, radius, draw_color, 1)  # kružnica

        return result

    def filter_circles(self, circles, image_shape, active_circle, canvas_shape=None):
        """
        Filtruje detekované kružnice podľa geometrických pravidiel pre daný typ kružnice.
        Vracia najlepší kandidát — kruh s najmenšou vzdialenosťou od stredu obrazka.
        """
        if circles is None:
            return None

        h, w = image_shape[:2]

        if canvas_shape is not None:
            ch, cw = canvas_shape[:2]
            # Stred canvasu = stred obrazka v canvas súradniciach (obrazok je vycentrovaný)
            cx, cy = cw // 2, ch // 2
            # Hranice pôvodného obrazka v canvas súradniciach
            img_x1 = cx - w // 2
            img_y1 = cy - h // 2
            img_x2 = img_x1 + w
            img_y2 = img_y1 + h
        else:
            cx, cy = w // 2, h // 2
            img_x1, img_y1, img_x2, img_y2 = 0, 0, w, h

        best = None
        best_score = float('inf')

        for c in circles[0, :]:
            x, y, r = int(c[0]), int(c[1]), int(c[2])
            # Vzdialenosť stredu kruhu od stredu obrazka — čím menšia, tým lepšie
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5

            if active_circle == "iris":
                # Dúhovka — stredne veľký kruh (20%-60% šírky obrazka)
                if not (w * 0.2 < r < w * 0.6):
                    continue
                # Celý kruh sa musí zmestiť do pôvodného obrazka
                if x - r < img_x1 or x + r > img_x2 or y - r < img_y1 or y + r > img_y2:
                    continue
                # Preferujeme kruh najbližší k stredu obrazka
                score = dist

            elif active_circle == "pupil":
                # Zrenička — malý kruh (menej ako 25% šírky obrazka)
                if not (r < w * 0.25):
                    continue
                # Celý kruh sa musí zmestiť do pôvodného obrazka
                if x - r < img_x1 or x + r > img_x2 or y - r < img_y1 or y + r > img_y2:
                    continue
                # Preferujeme kruh najbližší k stredu obrazka
                score = dist

            elif active_circle == "upper_lid":
                # Horné viečko — veľký polomer (viac ako 40% šírky obrazka)
                if not (r > w * 0.4):
                    continue
                # Stred kruhu musí byť v DOLNEJ polovici canvasu
                # (kruh s veľkým polomerom ktorého stred je dole pokrýva hornú časť obrazka)
                if y < cy:
                    continue
                # Horizontálne musí byť stred blízko stredu obrazka
                if abs(x - cx) > w * 0.4:
                    continue
                # Horný okraj kruhu (y - r) musí byť v HORNEJ polovici obrazka
                if (y - r) > cy:
                    continue
                # Preferujeme kruh najbližší k stredu obrazka
                score = dist

            elif active_circle == "lower_lid":
                # Dolné viečko — veľký polomer (viac ako 40% šírky obrazka)
                if not (r > w * 0.4):
                    continue
                # Stred kruhu musí byť v HORNEJ polovici canvasu
                # (kruh s veľkým polomerom ktorého stred je hore pokrýva dolnú časť obrazka)
                if y > cy:
                    continue
                # Horizontálne musí byť stred blízko stredu obrazka
                if abs(x - cx) > w * 0.4:
                    continue
                # Dolný okraj kruhu (y + r) musí byť v DOLNEJ polovici obrazka
                if (y + r) < cy:
                    continue
                # Preferujeme kruh najbližší k stredu obrazka
                score = dist

            else:
                continue

            if score < best_score:
                best_score = score
                best = c

        if best is None:
            return None
        return np.array([[best]], dtype=np.float32)

    def apply(self):
        """
        Aplikuje všetky zapnuté metódy predspracovania v poradí:
        histogram eq → CLAHE → Gaussian blur → Canny.
        Výsledok uloží do processed_image.
        """
        if self.original_image is None:
            self.processed_image = None
            return None

        img = self.original_image.copy()
        if self.use_histogram_eq:
            img = self.histogram_equalization(img)
        if self.use_clahe:
            img = self.clahe(img)
        if self.use_blur and not self.use_canny:
            img = self.gaussian_blur(img)
        if self.use_canny:
            img = self.canny(img)

        self.processed_image = img
        return self.processed_image

import cv2
import numpy as np
from zad1.code.core.helpers import apply_params
from zad1.code.core.helpers import update_texture

class Renderer:
    def __init__(self, processor, state=None, canvas_width=320, canvas_height=280):
        self.texture_id = None
        self.processor = processor
        self.state = state
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.preview_original = False
        self.show_ground_truth = False
        self.show_all_circles = False

    def _build_canvas(self, img):
        """Umiestni obraz vycentrovaný na biely canvas s aktuálnymi rozmermi canvas_width x canvas_height."""
        img_h, img_w = img.shape[:2]
        canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255

        x_offset = (self.canvas_width - img_w) // 2
        y_offset = (self.canvas_height - img_h) // 2

        img_x1 = max(0, -x_offset)
        img_y1 = max(0, -y_offset)
        canv_x1 = max(0, x_offset)
        canv_y1 = max(0, y_offset)
        draw_w = min(img_w - img_x1, self.canvas_width - canv_x1)
        draw_h = min(img_h - img_y1, self.canvas_height - canv_y1)

        if draw_w > 0 and draw_h > 0:
            canvas[canv_y1:canv_y1 + draw_h, canv_x1:canv_x1 + draw_w] = \
                img[img_y1:img_y1 + draw_h, img_x1:img_x1 + draw_w]
        return canvas

    def get_scaled_texture(self):
        """
        Zostaví výslednú textúru pre zobrazenie v DPG.
        Podľa aktívnych prepínačov aplikuje Houghovu detekciu alebo show_all_circles,
        prípadne nakreslí ground truth kružnice.
        Vracia tuple (rgba_data, width, height).
        """
        if self.processor.processed_image is None:
            return None

        processed_canvas = self._build_canvas(self.processor.processed_image)

        if self.processor.preview_original:
            display_canvas = self._build_canvas(self.processor.original_image)
        else:
            display_canvas = processed_canvas.copy()

        original_shape = self.processor.original_image.shape if self.processor.original_image is not None else None

        # Rozmery pre výslednu textúru — môžu sa zmeniť pri show_all_circles
        display_w = self.canvas_width
        display_h = self.canvas_height

        # Detekuj a nakresli každú kružnicu s jej vlastnými parametrami
        if self.show_all_circles:
            # Nájdi maximálne rozlíšenie cez všetky kružnice pre zobrazenie
            max_w = int(max(
                (cfg.get("canvas_width", self.canvas_width) for cfg in self.state.circle_params.values() if cfg),
                default=self.canvas_width
            ))
            max_h = int(max(
                (cfg.get("canvas_height", self.canvas_height) for cfg in self.state.circle_params.values() if cfg),
                default=self.canvas_height
            ))

            backup_w = self.canvas_width
            backup_h = self.canvas_height
            backup_processed = self.processor.processed_image.copy()
            active_cfg = self.state.circle_params.get(self.state.active_circle, {})

            # Nastav maximálne rozlíšenie pre display canvas
            self.canvas_width = max_w
            self.canvas_height = max_h
            display_w = max_w
            display_h = max_h

            # Display canvas vždy z pôvodného obrazka — nechceme zobrazovať spracovaný obraz
            display_canvas = self._build_canvas(self.processor.original_image)

            for circle_name in ["iris", "pupil", "upper_lid", "lower_lid"]:
                cfg = self.state.circle_params.get(circle_name, {})
                if not cfg:
                    continue

                # Nastav parametre tejto kružnice vrátane jej canvas rozlíšenia
                apply_params(cfg, self.processor, self)

                # Postav detect canvas s rozlíšením tejto kružnice
                circle_processed = self._build_canvas(self.processor.processed_image)
                circle_canvas_shape = circle_processed.shape

                # Vypočítaj offset medzi detect canvas a display canvas
                # Keďže obrazok je vycentrovaný, menší canvas má iný offset ako max canvas
                circle_w = int(cfg.get("canvas_width", backup_w))
                circle_h = int(cfg.get("canvas_height", backup_h))
                detect_offset_x = (max_w - circle_w) // 2
                detect_offset_y = (max_h - circle_h) // 2

                # Obnov max rozmery pre kreslenie na display canvas
                self.canvas_width = max_w
                self.canvas_height = max_h

                display_canvas = self.processor.hough(
                    display_canvas,
                    circle_name,
                    original_shape,
                    show_rejected=False,
                    detect_on=circle_processed,
                    filter_canvas_shape=circle_canvas_shape,
                    draw_offset=(detect_offset_x, detect_offset_y)
                )

            # Obnov pôvodné parametre a rozmery
            self.canvas_width = backup_w
            self.canvas_height = backup_h
            self.processor.processed_image = backup_processed
            if active_cfg:
                apply_params(active_cfg, self.processor, self)
                self.canvas_width = backup_w
                self.canvas_height = backup_h

        elif self.processor.show_hough:
            display_canvas = self.processor.hough(
                display_canvas,
                self.state.active_circle,
                original_shape,
                show_rejected=self.processor.show_rejected_circles,
                detect_on=processed_canvas
            )

        # Nakresli ground truth kružnice
        if self.show_ground_truth and self.state.ground_truth_circles is not None:
            circles_to_draw = (
                self.state.ground_truth_circles  # všetky pri show_all_circles
                if self.show_all_circles
                else {self.state.active_circle: self.state.ground_truth_circles.get(self.state.active_circle)}
            )
            display_canvas = self._draw_ground_truth(display_canvas, circles_to_draw)

        canvas = cv2.cvtColor(display_canvas, cv2.COLOR_BGR2RGBA)
        return canvas.astype(np.float32).flatten() / 255.0, display_w, display_h

    def refresh_texture(self, apply_processor=True):
        """
        Prerendruje textúru — voliteľne najprv aplikuje predspracovanie obrazka.
        Aktualizuje DPG textúru pomocou update_texture.
        """
        if apply_processor:
            self.processor.apply()

        result = self.get_scaled_texture()
        if result is None:
            return None

        # Rozbaľ textúru a rozmery — rozmery sa môžu líšiť pri show_all_circles
        tex_data, display_w, display_h = result

        self.texture_id = update_texture(self, tex_data, display_w, display_h)
        return self.texture_id

    def _draw_ground_truth(self, canvas, circles):
        """Nakreslí ground truth kružnice na canvas fialovou farbou s prepočtom do canvas súradníc."""
        result = canvas.copy()
        img_h, img_w = self.processor.original_image.shape[:2]
        x_offset = (self.canvas_width - img_w) // 2
        y_offset = (self.canvas_height - img_h) // 2

        for circle_name, gt in circles.items():
            if gt is None:
                continue
            gx, gy, gr = gt
            print(f"[GT] {circle_name}: gx={gx}, gy={gy}, gr={gr}, canvas_cy={gy + y_offset}")
            if gr <= 0:
                continue
            cx = gx + x_offset
            cy = gy + y_offset
            cv2.circle(result, (int(cx), int(cy)), int(gr), (255, 0, 255), 1)  # fialová
            cv2.circle(result, (int(cx), int(cy)), 2, (255, 0, 255), 2)
        return result

import cv2
import numpy as np
import os
from zad1.code.data.load_data import load_valid_records
from zad1.code.core.image_processor import ImageProcessor
from zad1.code.core.renderer import Renderer
from zad1.code.core.detector import detect_circle
from zad1.code.state.app_state import AppState
import json
from zad1.code.core.image_processor import CIRCLE_COLORS

def create_circle_mask(shape, cx, cy, r):
    """Vytvorí binárnu masku - vnútro kruhu sú 1tky"""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 1, -1)
    return mask


def segment_iris(image, circles):
    """
    Segmentuje dúhovku na základe detekovaných kružníc.

    circles: dict {
        "iris":      (cx, cy, r),
        "pupil":     (cx, cy, r),
        "upper_lid": (cx, cy, r),
        "lower_lid": (cx, cy, r),
    }

    Vracia binárnu masku (0/1) rovnakých rozmerov ako vstupný obraz.
    """
    h, w = image.shape[:2]

    iris = circles.get("iris")
    if iris is None:
        return np.zeros((h, w), dtype=np.uint8)

    # 1. Vytvor masku oka ako prienik viečok
    upper_lid = circles.get("upper_lid")
    lower_lid = circles.get("lower_lid")

    if upper_lid is not None and lower_lid is not None:
        upper_mask = create_circle_mask((h, w), upper_lid[0], upper_lid[1], upper_lid[2])
        lower_mask = create_circle_mask((h, w), lower_lid[0], lower_lid[1], lower_lid[2])
        # Oko = prienik oboch viečok
        eye_mask = cv2.bitwise_and(upper_mask, lower_mask)
    else:
        # Ak viečka nie sú nájdené, použij celý obrazok
        eye_mask = np.ones((h, w), dtype=np.uint8)

    # 2. Iris + pupil = prienik oka a iris kružnice
    iris_mask = create_circle_mask((h, w), iris[0], iris[1], iris[2])
    mask = cv2.bitwise_and(eye_mask, iris_mask)

    # 3. Vyreš zrenicu
    pupil = circles.get("pupil")
    if pupil is not None:
        pupil_mask = create_circle_mask((h, w), pupil[0], pupil[1], pupil[2])
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(pupil_mask))

    return mask


def apply_segmentation_mask(image, mask):
    """
    Aplikuje binárnu masku na obraz.
    Pixely mimo masky = 0, ostatné = pôvodná hodnota.
    """
    result = image.copy()
    result[mask == 0] = 0
    return result

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_FOLDER = os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", "data"))
    CONFIG_FOLDER = os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", "config"))
    CSV_PATH = os.path.join(DATA_FOLDER, "iris_annotation.csv")

    CIRCLE_NAMES = ["iris", "pupil", "upper_lid", "lower_lid"]

    # Načítaj jeden náhodný platný záznam
    records = load_valid_records(CSV_PATH, DATA_FOLDER, n=1)
    record = records[0]
    img_path = os.path.normpath(os.path.join(DATA_FOLDER, record.image.replace("\\", "/")))
    image = cv2.imread(img_path)
    print(f"Obrazok: {record.image}")

    # Vytvor processor a renderer
    processor = ImageProcessor(original_image=image)
    state = AppState()
    renderer = Renderer(processor=processor, state=state)

    # Detekuj kruhy pre každú kružnicu
    circles = {}
    for circle_name in CIRCLE_NAMES:
        cfg_path = os.path.join(CONFIG_FOLDER, f"{circle_name}_config.json")
        if not os.path.exists(cfg_path):
            print(f"Konfigurácia pre {circle_name} nenájdená, preskakujem")
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)

        processor.original_image = image
        processor.processed_image = image.copy()
        detected = detect_circle(processor, renderer, cfg, circle_name, headless=True)
        if detected is not None:
            circles[circle_name] = detected
            print(f"  {circle_name}: {detected}")
        else:
            print(f"  {circle_name}: nenájdený")

    # Segmentuj dúhovku a zobraz
    mask = segment_iris(image, circles)
    segmented = apply_segmentation_mask(image, mask)

    annotated = image.copy()
    for circle_name, (cx, cy, r) in circles.items():
        color = CIRCLE_COLORS.get(circle_name, (255, 255, 255))
        cv2.circle(annotated, (int(cx), int(cy)), int(r), color, 1)
        cv2.circle(annotated, (int(cx), int(cy)), 2, color, 2)

    cv2.imshow("Najdene kruznice", annotated)
    cv2.imshow("Segmentovana duhovka", segmented)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import cv2

# Zdroj: https://www.kaggle.com/code/farahalarbeed/convert-binary-masks-to-yolo-format
def mask_to_yolo_polygon(mask, img_w, img_h, class_id=0):
    """Konvertuje binárnu masku na YOLO polygon formát."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    lines = []
    for contour in contours:
        if len(contour) < 3:
            continue  # nie je polygon

        normalized_points = []
        for point in contour.squeeze():
            x = point[0] / img_w
            y = point[1] / img_h
            # zabezpeč rozsah 0-1
            if 0 <= x <= 1 and 0 <= y <= 1:
                normalized_points.append(f"{x:.6f} {y:.6f}")

        if normalized_points:
            lines.append(f"{class_id} " + " ".join(normalized_points))

    return "\n".join(lines) if lines else None

import os

import pandas as pd
from dataclasses import dataclass

ALL_RECORDS = []
RANDOM_RECORDS = []

@dataclass
class ImageRecord:
    image: str
    center_x_1: int
    center_y_1: int
    polomer_1: int
    center_x_2: int
    center_y_2: int
    polomer_2: int
    center_x_3: int
    center_y_3: int
    polomer_3: int
    center_x_4: int
    center_y_4: int
    polomer_4: int


def load_random_records(csv_path, n=100):
    """
    Načíta n náhodných záznamov
    """
    global RANDOM_RECORDS

    df = pd.read_csv(csv_path)
    n = min(n, len(df))
    sampled_df = df.sample(n=n)

    records = []

    for _, row in sampled_df.iterrows():
        row_dict = row.to_dict()
        records.append(ImageRecord(**row_dict))

    RANDOM_RECORDS = records
    return records

def load_valid_records(csv_path, data_folder, n=100):
    """
    Načíta záznamy kým nemá n platných (súbor existuje na disku).
    Záznamy sa vyberajú náhodne z celého datasetu.
    """
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1)  # náhodné poradie
    valid = []
    for _, row in df.iterrows():
        record = ImageRecord(**row.to_dict())
        path = os.path.normpath(os.path.join(data_folder, record.image.replace("\\", "/")))
        if os.path.exists(path):
            valid.append(record)
        if len(valid) >= n:
            break
    print(f"Načítaných {len(valid)} platných záznamov z {len(df)} celkovo")
    return valid

import json
import os

CONFIG_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))


class AppState:
    """Uchováva globálny stav aplikácie — aktívnu kružnicu, parametre, aktuálny záznam a textúru."""

    def __init__(self):
        self.texture_id = None           # ID aktuálnej DPG textúry
        self.current_image_path = None   # Cesta k aktuálne načítanému obrazku
        self.active_circle = "iris"      # Aktuálne vybraná kružnica (iris/pupil/upper_lid/lower_lid)
        self.current_record = None       # Aktuálny ImageRecord z CSV
        self.ground_truth_circles = None # Ground truth kružnice pre aktuálny záznam

        # Uložené parametre predspracovania a detekcie pre každú kružnicu
        self.circle_params = {
            "iris": {},
            "pupil": {},
            "upper_lid": {},
            "lower_lid": {}
        }
        # Načítaj uložené konfigurácie pri štarte aplikácie
        self._load_all_configs()

    def _load_all_configs(self):
        """Načíta uložené JSON konfigurácie pre každú kružnicu z config priečinka."""
        for circle in self.circle_params:
            path = os.path.join(CONFIG_DIR, f"{circle}_config.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    self.circle_params[circle] = json.load(f)

import dearpygui.dearpygui as dpg
from zad1.code.ui.ui_helpers import add_slider, add_checkbox, add_section_separator
from zad1.code.ui.ui_parameters import PARAMETER_CONFIG

class ControlsWindow:
    def __init__(self, dispatcher, fonts, width, height):
        self.dispatcher = dispatcher
        self.large_font, self.small_font = fonts
        self.width = width
        self.height = height

    def build(self):
        """Zostaví celé okno so všetkými sekciami."""
        with dpg.window(label="Controls", width=self.width, height=self.height, pos=(0, 0)):
            self._build_circle_selector()
            self._add_section_separator()

            self._build_view_section()
            self._add_section_separator()

            self._build_on_off_section()
            self._add_section_separator()

            self._build_parameter_sections()
            self._add_section_separator()

            self._build_bottom_buttons()

    def _build_circle_selector(self):
        """Vytvorí combo box pre výber aktívnej kružnice."""
        title = dpg.add_text("Kruznica")
        dpg.bind_item_font(title, self.large_font)
        dpg.add_combo(
            items=["iris", "pupil", "upper_lid", "lower_lid"],
            default_value="iris",
            tag="active_circle",
            callback=lambda s, a, u: self.dispatcher.execute("select_circle", s, a, u),
            width=-1
        )

    # Sections
    def _build_view_section(self):
        """Vytvorí sekciu so slidermi pre nastavenie rozlíšenia canvasu."""
        title = dpg.add_text("Zobrazenie")
        dpg.bind_item_font(title, self.large_font)

        self._add_slider("Canvas width", "canvas_width", "update_canvas", 320, 320, 1600)
        self._add_slider("Canvas height", "canvas_height", "update_canvas", 280, 280, 1200)

    def _build_on_off_section(self):
        """Vytvorí sekciu s checkboxmi pre zapínanie/vypínanie metód predspracovania a zobrazenia."""
        title = dpg.add_text("ON / OFF")
        dpg.bind_item_font(title, self.large_font)

        # Namapuj labels na atribúty processora
        toggles = {
            "Histogram equation": "use_histogram_eq",
            "CLAHE": "use_clahe",
            "Gaussian blur": "use_blur",
            "Show canny edges": "use_canny",
            "Hough circles": "show_hough",
            "Show rejected circles": "show_rejected_circles",
            "Show ground truth": "show_ground_truth",
            "Show all circles": "show_all_circles",
        }

        with dpg.group(horizontal=True):
            with dpg.group():
                for label, attr in list(toggles.items())[:3]:
                    self._add_checkbox(label, attr, command_name=attr)
            dpg.add_spacer(width=20)
            with dpg.group():
                for label, attr in list(toggles.items())[3:]:
                    self._add_checkbox(label, attr, command_name=attr)

    def _build_parameter_sections(self):
        """Vytvorí sekcie so slidermi pre parametre CLAHE, Gaussian blur, Canny a HoughCircles."""
        sections = [
            ("CLAHE", ["clahe_clip", "clahe_tile"]),
            ("Gaussian Blur", ["gauss_kernel", "gauss_sigma"]),
            ("Canny", ["threshold_1", "threshold_2"]),
            (
                "HoughCircles",
                [
                    "hough_dp", "hough_mindist", "hough_param1", "hough_param2", "hough_minr", "hough_maxr"
                ],
            ),
        ]

        for section_title, param_tags in sections:
            title = dpg.add_text(section_title)
            dpg.bind_item_font(title, self.large_font)

            for tag in param_tags:
                cfg = PARAMETER_CONFIG[tag]
                self._add_slider(
                    label=tag.replace("_", " ").title(),
                    tag=tag,
                    command_name=cfg["command"],
                    default=cfg["default"],
                    min_v=cfg["min"],
                    max_v=cfg["max"],
                    pre_callback = self._snap_odd if tag == "gauss_kernel" else None
                )

            self._add_section_separator()

    def _build_bottom_buttons(self):
        """Vytvorí tlačidlá Reset defaults, Save settings, Evaluate a checkbox Preview original."""
        with dpg.group(horizontal=True):
            dpg.add_button(label="Reset defaults", width=120, callback=self._global_callback,
                           user_data="reset_defaults")
            dpg.add_button(label="Save settings", width=120, callback=self._global_callback,
                           user_data="save_settings")
            dpg.add_button(label="Evaluate", width=120, callback=self._global_callback,
                           user_data="evaluate")
        dpg.add_checkbox(
            label="Preview original",
            tag="preview_toggle",
            callback=lambda s, a, u: self.dispatcher.execute("preview_original", s, a, u)
        )

    # Pomocné funkcie
    def _add_slider(self, label, tag, command_name=None, default=0, min_v=0, max_v=2000, pre_callback=None):
        """
        Pridá slider do UI a nastaví naň callback.
        pre_callback sa zavolá pred hlavným callbackom — používa sa napríklad na snap na nepárne číslo.
        """
        def callback(sender, app_data, user_data):
            if pre_callback:
                pre_callback(sender, app_data, user_data)
                app_data = dpg.get_value(sender)
            self._global_callback(sender, app_data, user_data)

        add_slider(
            label=label,
            tag=tag,
            small_font=self.small_font,
            callback=callback if command_name else None,
            default=default,
            min_v=min_v,
            max_v=max_v
        )
        if command_name:
            dpg.set_item_user_data(tag, command_name)

    def _add_checkbox(self, label, processor_attr, command_name="use_histogram_eq"):
        """Pridá checkbox ktorý pri zmene odošle command cez dispatcher."""
        add_checkbox(
            label=label,
            tag=processor_attr,
            small_font=self.small_font,
            callback=lambda s, a, u: self.dispatcher.execute(command_name, s, a, u),
            user_data=processor_attr
        )

    def _add_section_separator(self):
        """Zaokrúhli hodnotu slidera na najbližšie nepárne číslo — používa sa pre gauss_kernel."""
        add_section_separator()

    def _snap_odd(self, sender, app_data, user_data):
        """Zaokrúhli hodnotu slidera na najbližšie nepárne číslo — používa sa pre gauss_kernel."""
        if app_data != 0 and app_data % 2 == 0:
            dpg.set_value(sender, app_data + 1)

    def _global_callback(self, sender, app_data, user_data):
        """Univerzálny callback — načíta command name z user_data položky a odošle ho cez dispatcher."""
        command_name = dpg.get_item_user_data(sender)  # ← always the command name
        self.dispatcher.execute(command_name, sender, app_data, user_data)

import dearpygui.dearpygui as dpg

class ImageWindow:
    def __init__(self, state, width, height, pos_x):
        self.state = state
        self.width = width
        self.height = height
        self.pos_x = pos_x

    def build(self):
        """Zostaví okno a vytvorí image widget."""
        with dpg.window(
            label="View image",
            width=self.width,
            height=self.height,
            pos=(self.pos_x, 0),
        ):
            self._create_image_widget()

    def _create_image_widget(self):
        """Pridá DPG image widget s aktuálnou textúrou. Ak textúra neexistuje, nič nevykoná."""
        if self.state.texture_id is None:
            return

        dpg.add_image(
            self.state.texture_id,
            tag="displayed_image_widget",
        )

import dearpygui.dearpygui as dpg
from zad1.code.ui.ui_parameters import SECTION_SPACING, FIELD_SPACING

def add_spacing(height):
    dpg.add_spacer(height=height)

def add_slider(label, tag, small_font, callback=None, default=280, min_v=1, max_v=900):
    is_float = any(isinstance(v, float) for v in (default, min_v, max_v))
    slider_func = dpg.add_slider_float if is_float else dpg.add_slider_int

    slider = slider_func(
        label=label,
        tag=tag,
        default_value=default,
        min_value=min_v,
        max_value=max_v,
        callback=callback
    )
    dpg.bind_item_font(slider, small_font)
    add_spacing(FIELD_SPACING)
    return slider

def add_checkbox(label, tag, small_font, callback=None, command=None, user_data=None):
    checkbox = dpg.add_checkbox(
        label=label,
        tag=tag,
        callback=callback,
        user_data=user_data
    )
    dpg.bind_item_font(checkbox, small_font)
    return checkbox

def add_section_separator():
    dpg.add_separator()
    add_spacing(SECTION_SPACING)

TOGGLE_TAGS = [
    "use_histogram_eq",
    "use_clahe",
    "use_blur",
    "use_canny",
    "show_hough"
]

CANVAS_TAGS = ["canvas_width", "canvas_height"]

FIELD_SPACING = 1
SECTION_SPACING = 4

PARAMETER_CONFIG = {
    # CLAHE
    "clahe_clip": {
        "command": "clahe",
        "default": 2.0,
        "min": 1.0,
        "max": 10.0,
    },
    "clahe_tile": {
        "command": "clahe",
        "default": 8,
        "min": 2,
        "max": 32,
    },

    # Gaussian
    "gauss_kernel": {
        "command": "gaussian_blur",
        "default": 5,
        "min": 0,
        "max": 31,
    },
    "gauss_sigma": {
        "command": "gaussian_blur",
        "default": 1.0,
        "min": 0.0,
        "max": 10.0,
    },

    # Canny
    "threshold_1": {
        "command": "canny",
        "default": 50,
        "min": 0,
        "max": 500,
    },
    "threshold_2": {
        "command": "canny",
        "default": 150,
        "min": 0,
        "max": 500,
    },

    # Hough
    "hough_dp": {
        "command": "hough",
        "default": 1.2,
        "min": 1.0,
        "max": 3.0,
    },
    "hough_mindist": {
        "command": "hough",
        "default": 50,
        "min": 1,
        "max": 500,
    },
    "hough_param1": {
        "command": "hough",
        "default": 100,
        "min": 1,
        "max": 500,
    },
    "hough_param2": {
        "command": "hough",
        "default": 30,
        "min": 1,
        "max": 200,
    },
    "hough_minr": {
        "command": "hough",
        "default": 0,
        "min": 0,
        "max": 500,
    },
    "hough_maxr": {
        "command": "hough",
        "default": 0,
        "min": 0,
        "max": 500,
    },
}

from ultralytics import YOLO
import os
import cv2
import random

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_YAML = os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", "yolo_dataset", "data.yaml"))
VAL_IMAGES = os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", "yolo_dataset", "images", "val"))

def visualize_predictions(val_images_dir, model):
    """Zobrazí predikcie modelu na 3 náhodných obrazkoch z validačnej množiny."""
    val_images = [f for f in os.listdir(val_images_dir) if f.endswith(".jpg")]
    sample = random.sample(val_images, min(3, len(val_images)))

    for img_name in sample:
        img_path = os.path.join(val_images_dir, img_name)
        results = model(img_path)
        annotated = results[0].plot()
        cv2.imshow(f"Predikcia: {img_name}", annotated)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def evaluate():
    model = YOLO("../../runs/segment/yolo_runs/iris_seg/weights/best.pt")
    metrics = model.val(data=DATASET_YAML)
    print(f"Precision:  {metrics.seg.p.mean():.3f}")
    print(f"Recall:     {metrics.seg.r.mean():.3f}")
    print(f"mAP50:      {metrics.seg.map50:.3f}")
    print(f"mAP50-95:   {metrics.seg.map:.3f}")
    visualize_predictions(VAL_IMAGES, model)

if __name__ == "__main__":
    evaluate()

import os
import cv2
import shutil
import random
import numpy as np
from zad1.code.data.load_data import load_valid_records
from zad1.code.core.segmentation import segment_iris, create_circle_mask
from zad1.code.core.yolo_mask import mask_to_yolo_polygon

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", "data"))
OUTPUT_FOLDER = os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", "yolo_dataset"))
CSV_PATH = os.path.join(DATA_FOLDER, "iris_annotation.csv")

TRAIN_RATIO = 0.9


def record_to_gt_circles(record):
    """Načíta ground truth kruhy z CSV záznamu."""
    return {
        "pupil":     (record.center_x_1, record.center_y_1, record.polomer_1),
        "iris":      (record.center_x_2, record.center_y_2, record.polomer_2),
        "lower_lid": (record.center_x_3, record.center_y_3, record.polomer_3),
        "upper_lid": (record.center_x_4, record.center_y_4, record.polomer_4),
    }


def create_dataset_structure(output_folder):
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_folder, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "labels", split), exist_ok=True)
    print(f"Štruktúra datasetu vytvorená v: {output_folder}")


def create_data_yaml():
    """Vytvorí data.yaml konfiguračný súbor pre YOLO."""
    yaml_content = f"""path: {OUTPUT_FOLDER}
train: images/train
val: images/val

nc: 1
names: ['iris']
"""
    yaml_path = os.path.join(OUTPUT_FOLDER, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"data.yaml vytvorený: {yaml_path}")

def prepare_dataset_generic(output_folder, data_folder, csv_path, label_generator_fn):
    """
    Generická funkcia pre prípravu YOLO datasetu.
    label_generator_fn(image, circles, w, h) -> list of label strings
    """
    records = load_valid_records(csv_path, data_folder, n=100)
    print(f"Načítaných {len(records)} platných záznamov")

    create_dataset_structure(output_folder)

    random.shuffle(records)
    split_idx = int(len(records) * TRAIN_RATIO)
    train_records = records[:split_idx]
    val_records = records[split_idx:]
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")

    for split, split_records in [("train", train_records), ("val", val_records)]:
        ok = 0
        skip = 0
        for record in split_records:
            img_path = os.path.normpath(os.path.join(data_folder, record.image.replace("\\", "/")))
            image = cv2.imread(img_path)
            if image is None:
                skip += 1
                continue

            h, w = image.shape[:2]
            circles = record_to_gt_circles(record)

            if circles["iris"][2] <= 0:
                skip += 1
                continue

            labels = label_generator_fn(image, circles, w, h)
            if not labels:
                skip += 1
                continue

            filename = record.image.replace("\\", "/").replace("/", "_").replace(".jpg", "")
            img_out = os.path.join(output_folder, "images", split, f"{filename}.jpg")
            shutil.copy(img_path, img_out)

            label_out = os.path.join(output_folder, "labels", split, f"{filename}.txt")
            with open(label_out, "w") as f:
                for label in labels:
                    f.write(label + "\n")

            ok += 1

        print(f"  {split}: {ok} spracovaných, {skip} preskočených")

def prepare_dataset():
    def label_fn(image, circles, w, h):
        mask = segment_iris(image, circles)
        label = mask_to_yolo_polygon(mask, w, h, class_id=0)
        return [label] if label else []

    prepare_dataset_generic(OUTPUT_FOLDER, DATA_FOLDER, CSV_PATH, label_fn)
    create_data_yaml()
    print("\nDataset pripravený.")


if __name__ == "__main__":
    prepare_dataset()

from ultralytics import YOLO
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_YAML = os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", "yolo_dataset", "data.yaml"))

def train():
    model = YOLO("yolo11n-seg.pt")

    model.train(
        data=DATASET_YAML,
        epochs=300,
        imgsz=(280, 320),
        batch=8,
        pretrained=True,
        project="yolo_runs",
        name="iris_seg"
    )

if __name__ == "__main__":
    train()

import os
from ultralytics import YOLO
from zad1.code.yolo_iris.evaluate_yolo import visualize_predictions

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_YAML = os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", "yolo_multiclass_dataset", "data.yaml"))
VAL_IMAGES = os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", "yolo_multiclass_dataset", "images", "val"))
MODEL_PATH = "../../../runs/segment/yolo_runs/iris_pupil_seg/weights/best.pt"

def evaluate():
    model = YOLO(MODEL_PATH)

    metrics = model.val(data=DATASET_YAML)
    print(f"Precision:  {metrics.seg.p.mean():.3f}")
    print(f"Recall:     {metrics.seg.r.mean():.3f}")
    print(f"mAP50:      {metrics.seg.map50:.3f}")
    print(f"mAP50-95:   {metrics.seg.map:.3f}")

    visualize_predictions(VAL_IMAGES, model)

if __name__ == "__main__":
    evaluate()

import os
import numpy as np
from zad1.code.core.segmentation import segment_iris, create_circle_mask
from zad1.code.core.yolo_mask import mask_to_yolo_polygon
from zad1.code.yolo_iris.prepare_yolo_dataset import prepare_dataset_generic

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", "data"))
OUTPUT_FOLDER = os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", "yolo_multiclass_dataset"))
CSV_PATH = os.path.join(DATA_FOLDER, "iris_annotation.csv")

TRAIN_RATIO = 0.9


def segment_pupil(image, circles):
    """Vytvorí masku len pre zrenicu."""
    h, w = image.shape[:2]
    pupil = circles.get("pupil")
    if pupil is None or pupil[2] <= 0:
        return np.zeros((h, w), dtype=np.uint8)
    return create_circle_mask((h, w), pupil[0], pupil[1], pupil[2])


def create_multiclass_data_yaml():
    yaml_content = f"""path: {OUTPUT_FOLDER}
train: images/train
val: images/val

nc: 2
names: ['iris', 'pupil']
"""
    with open(os.path.join(OUTPUT_FOLDER, "data.yaml"), "w") as f:
        f.write(yaml_content)


def prepare_dataset():
    def label_fn(image, circles, w, h):
        iris_mask = segment_iris(image, circles)
        pupil_mask = segment_pupil(image, circles)
        iris_label = mask_to_yolo_polygon(iris_mask, w, h, class_id=0)
        pupil_label = mask_to_yolo_polygon(pupil_mask, w, h, class_id=1)
        return [l for l in [iris_label, pupil_label] if l]

    prepare_dataset_generic(OUTPUT_FOLDER, DATA_FOLDER, CSV_PATH, label_fn)
    create_multiclass_data_yaml()
    print("\nDataset pripravený.")


if __name__ == "__main__":
    prepare_dataset()

from ultralytics import YOLO
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_YAML = os.path.normpath(os.path.join(CURRENT_DIR, "..", "..", "yolo_multiclass_dataset", "data.yaml"))

def train():
    model = YOLO("yolo11n-seg.pt")
    model.train(
        data=DATASET_YAML,
        epochs=300,
        imgsz=(280, 320),
        batch=8,
        pretrained=True,
        project="yolo_runs",
        name="iris_pupil_seg"
    )

if __name__ == "__main__":
    train()

import os
import dearpygui.dearpygui as dpg

from zad1.code.commands.canny_command import CannyCommand
from zad1.code.commands.clahe_command import CLAHECommand
from zad1.code.commands.dispatcher import CommandDispatcher
from zad1.code.commands.evaluate_command import EvaluateCommand
from zad1.code.commands.gaussian_blur_command import GaussianBlurCommand
from zad1.code.commands.hough_command import HoughCommand
from zad1.code.commands.save_settings import SaveSettingsCommand
from zad1.code.commands.select_circle_command import SelectCircleCommand
from zad1.code.commands.toggle_command import ToggleCommand
from zad1.code.commands.toggle_renderer_command import ToggleRendererCommand
from zad1.code.commands.update_canvas_command import UpdateCanvasCommand
from zad1.code.core.helpers import apply_params
from zad1.code.core.image_manager import ImageManager
from zad1.code.core.image_processor import ImageProcessor
from zad1.code.core.renderer import Renderer
from zad1.code.state.app_state import AppState
from zad1.code.ui.controls_window import ControlsWindow
from zad1.code.ui.image_window import ImageWindow
from zad1.code.data import load_data


class Application:
    def __init__(self):
        # Hlavné objekty
        self.state = AppState()
        self.image_manager = ImageManager()
        self.processor = None
        self.renderer = None
        self.dispatcher = CommandDispatcher()

        # Fonts
        self.fonts = (None, None)

        # UI windows
        self.controls_window = None
        self.image_window = None

        # Inicializuj appku
        self._load_first_image()
        self._init_processor()
        self._init_renderer()
        self._register_commands()

    def _load_first_image(self):
        if not load_data.RANDOM_RECORDS:
            return

        first_record = load_data.RANDOM_RECORDS[0]
        self.state.current_record = first_record
        self.state.ground_truth_circles = self._record_to_gt(first_record)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.normpath(os.path.join(current_dir, "..", "data"))
        full_path = os.path.normpath(os.path.join(data_folder, first_record.image.replace("\\", "/")))

        self.image_manager.load_image(full_path)

    def _record_to_gt(self, record):
        return {
            "pupil": (record.center_x_1, record.center_y_1, record.polomer_1),
            "iris": (record.center_x_2, record.center_y_2, record.polomer_2),
            "lower_lid": (record.center_x_3, record.center_y_3, record.polomer_3),
            "upper_lid": (record.center_x_4, record.center_y_4, record.polomer_4),
        }

    def _init_processor(self):
        self.processor = ImageProcessor(
            original_image=self.image_manager.original_image
        )

    def _init_renderer(self):
        self.renderer = Renderer(
            processor=self.processor,
            state=self.state
        )

    def _create_initial_texture(self):
        if self.processor is None or self.processor.processed_image is None:
            return

        width = self.renderer.canvas_width
        height = self.renderer.canvas_height
        if width == 0 or height == 0:
            return

        result = self.renderer.get_scaled_texture()
        if result is None:
            canvas_data = [0.0] * (width * height * 4)
        else:
            canvas_data, width, height = result  # rozbaľ tuple

        if not hasattr(self, "_texture_registry"):
            self._texture_registry = dpg.add_texture_registry(show=False)

        if hasattr(self.state, "texture_id") and dpg.does_item_exist(self.state.texture_id):
            dpg.delete_item(self.state.texture_id)

        self.state.texture_id = dpg.add_dynamic_texture(
            width=width,
            height=height,
            default_value=canvas_data,
            parent=self._texture_registry
        )

    def _register_commands(self):
        self.dispatcher.register("update_canvas", UpdateCanvasCommand(self.processor, self.renderer))

        self.dispatcher.register("use_histogram_eq", ToggleCommand(self.processor, self.renderer, "use_histogram_eq"))

        self.dispatcher.register("use_clahe", ToggleCommand(self.processor, self.renderer, "use_clahe"))
        self.dispatcher.register("clahe", CLAHECommand(self.processor, self.renderer))

        self.dispatcher.register("use_blur", ToggleCommand(self.processor, self.renderer, "use_blur"))
        self.dispatcher.register("gaussian_blur", GaussianBlurCommand(self.processor, self.renderer))

        self.dispatcher.register("use_canny", ToggleCommand(self.processor, self.renderer, "use_canny"))
        self.dispatcher.register("canny", CannyCommand(self.processor, self.renderer))

        self.dispatcher.register("show_hough", ToggleCommand(self.processor, self.renderer, "show_hough"))
        self.dispatcher.register("hough", HoughCommand(self.processor, self.renderer))

        self.dispatcher.register("select_circle", SelectCircleCommand(self.processor, self.renderer, self.state))
        self.dispatcher.register("show_rejected_circles",ToggleCommand(self.processor, self.renderer, "show_rejected_circles"))
        self.dispatcher.register("save_settings", SaveSettingsCommand(self.processor, self.renderer, self.state))
        self.dispatcher.register("preview_original", ToggleCommand(self.processor, self.renderer, "preview_original"))

        self.dispatcher.register("evaluate", EvaluateCommand(self.processor, self.renderer, self.state))

        self.dispatcher.register("show_ground_truth",ToggleRendererCommand(self.processor, self.renderer, "show_ground_truth"))
        self.dispatcher.register("show_all_circles",ToggleRendererCommand(self.processor, self.renderer, "show_all_circles"))    # UI helpers

    def _setup_fonts(self):
        with dpg.font_registry():
            large_font = dpg.add_font("C:/Windows/Fonts/arial.ttf", 12)
            small_font = dpg.add_font("C:/Windows/Fonts/arial.ttf", 10)
        self.fonts = (large_font, small_font)

    def _create_windows(self):
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()

        controls_width = int(viewport_width * 0.3)
        image_width = int(viewport_width * 0.7)

        self.controls_window = ControlsWindow(
            dispatcher=self.dispatcher,
            fonts=self.fonts,
            width=controls_width,
            height=viewport_height
        )

        self.image_window = ImageWindow(
            state=self.state,
            width=image_width,
            height=viewport_height,
            pos_x=controls_width
        )

    def _apply_initial_params(self):
        saved = self.state.circle_params[self.state.active_circle]
        apply_params(saved, self.processor, self.renderer)

    def _build_ui(self):
        self._setup_fonts()
        self._create_windows()
        self.controls_window.build()
        self.image_window.build()
        self._apply_initial_params()

    def run(self):
        dpg.create_context()
        dpg.create_viewport(title="Zad 1")
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.maximize_viewport()

        self._create_initial_texture()
        self._build_ui()
        self.renderer.refresh_texture(apply_processor=True)

        dpg.start_dearpygui()
        dpg.destroy_context()

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

import os

from zad1.code.data import load_data
from app import Application

if __name__ == "__main__":
    # Načítaj CSV dáta
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.normpath(os.path.join(current_dir, "..", "data"))
    csv_path = os.path.join(data_folder, "iris_annotation.csv")
    load_data.load_random_records(csv_path, n=100)

    app = Application()
    app.run()