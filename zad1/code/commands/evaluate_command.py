import os
import json
from datetime import datetime

from zad1.code.commands.image_command import ImageCommand
from zad1.code.core.helpers import apply_params
from zad1.code.core.detector import evaluate_record, evaluate_batch
from zad1.code.core.evaluator import print_metrics, compute_metrics, compute_batch_metrics, print_batch_metrics


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

    def _save_results(self, single_result, single_metrics, batch_metrics, image_name):
        """Uloží výsledky do JSON súboru v config priečinku."""
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