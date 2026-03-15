import os
import json
import cv2
from datetime import datetime

from zad1.code.commands.image_command import ImageCommand
from zad1.code.core.evaluator import evaluate_single, print_metrics, compute_metrics
from zad1.code.core.helpers import apply_params

# Poradie kružníc podľa CSV stĺpcov: 1=iris, 2=pupil, 3=lower_lid, 4=upper_lid
CIRCLE_NAMES = ["iris", "pupil", "lower_lid", "upper_lid"]
CIRCLE_CSV_INDEX = {
    "pupil":     1,
    "iris":      2,
    "lower_lid": 3,
    "upper_lid": 4,
}
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

        # Ulož aktuálne parametre — obnovíme ich po vyhodnotení
        current_params = self._backup_processor()

        # Vyhodnoť aktuálny obrázok
        single_result = self._evaluate_record(record)

        metrics = compute_metrics(single_result)

        # Obnov pôvodné parametre a prerendruj
        apply_params(current_params, self.processor, self.renderer)
        self.refresh(apply_processor=True)

        # Vypíš a ulož výsledky
        print_metrics(metrics, image_name=record.image, iou_threshold=IOU_THRESHOLD)
        self._save_results(single_result, metrics, record.image)

    def _evaluate_record(self, record):
        """
        Vyhodnotí jeden záznam pre všetky kružnice.
        Táto metóda je reusable — neskôr ju zavoláme v slučke pre batch vyhodnotenie.

        Vracia dict {circle_name: {detected, ground_truth, iou, tp, fp, fn}}
        """
        result = {}
        for circle_name in CIRCLE_NAMES:
            cfg = self.state.circle_params.get(circle_name, {})
            if not cfg:
                print(f"Žiadna konfigurácia pre {circle_name}, preskakujem")
                continue

            # Detekuj kruh pomocou uložených parametrov
            detected = self._detect_circle(cfg, circle_name)

            # Načítaj ground truth z záznamu
            idx = CIRCLE_CSV_INDEX[circle_name]
            gt = (
                getattr(record, f"center_x_{idx}"),
                getattr(record, f"center_y_{idx}"),
                getattr(record, f"polomer_{idx}")
            )

            tp, fp, fn, iou = evaluate_single(detected, gt, IOU_THRESHOLD)
            print(f"[{circle_name}] detected={detected}, gt={gt}, iou={iou:.3f}")
            result[circle_name] = {
                "detected": list(detected) if detected else None,
                "ground_truth": list(gt),
                "iou": iou,
                "tp": tp, "fp": fp, "fn": fn
            }
        return result

    def _detect_circle(self, cfg, circle_name):
        """
        Detekuje kruh pre danú kružnicu pomocou uložených parametrov.
        Znovupoužíva:
          - processor.apply() na spracovanie obrazka
          - renderer._build_canvas() na vytvorenie canvasu s paddingom
          - processor.filter_circles() na výber najlepšieho kruhu
        """
        # Nastav parametre processora a renderera z konfigurácie
        apply_params(cfg, self.processor, self.renderer)

        # Spracuj obrazok (histogram eq, CLAHE, blur, canny)
        self.processor.apply()

        # Postav canvas s paddingom — padding je dôležitý pre detekciu viečok
        canvas = self.renderer._build_canvas(self.processor.processed_image)

        # Detekuj kruhy na canvase pomocou Houghovej transformácie
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.processor.hough_dp,
            minDist=self.processor.hough_mindist,
            param1=self.processor.hough_param1,
            param2=self.processor.hough_param2,
            minRadius=self.processor.hough_minr,
            maxRadius=self.processor.hough_maxr
        )

        # Filtruj a vyber najlepší kruh podľa pravidiel pre danú kružnicu
        best = self.processor.filter_circles(
            circles,
            self.processor.original_image.shape,
            circle_name,
            canvas_shape=canvas.shape
        )

        if best is None:
            return None

        # Prepočet súradníc z canvas priestoru späť do obrazkového priestoru
        # Canvas má padding — treba odčítať offset
        img_h, img_w = self.processor.original_image.shape[:2]
        canv_x1 = max(0, (self.renderer.canvas_width - img_w) // 2)
        canv_y1 = max(0, (self.renderer.canvas_height - img_h) // 2)
        x, y, r = int(best[0][0][0]), int(best[0][0][1]), int(best[0][0][2])
        return (x - canv_x1, y - canv_y1, r)

    def _backup_processor(self):
        """Uloží aktuálne parametre processora a renderera pre neskoršie obnovenie."""
        from zad1.code.ui.ui_parameters import PARAMETER_CONFIG, TOGGLE_TAGS
        backup = {}
        for tag, cfg in PARAMETER_CONFIG.items():
            backup[tag] = getattr(self.processor, tag, cfg["default"])
        for tag in TOGGLE_TAGS:
            backup[tag] = getattr(self.processor, tag, False)
        backup["canvas_width"] = self.renderer.canvas_width
        backup["canvas_height"] = self.renderer.canvas_height
        return backup

    def _save_results(self, single_result, metrics, image_name):
        """Uloží výsledky do JSON súboru v config priečinku."""
        output = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image": image_name,
            "iou_threshold": IOU_THRESHOLD,
            # Detailné výsledky pre každú kružnicu
            "per_circle": single_result,
            # Agregované metriky (precision, recall, F1)
            "metrics": metrics
        }
        path = os.path.join(CONFIG_DIR, "eval_results.json")
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Výsledky uložené do: {path}")