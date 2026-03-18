import os

import cv2

from zad1.code.core.constants import CIRCLE_CSV_INDEX, CIRCLE_NAMES
from zad1.code.core.evaluator import evaluate_single
from zad1.code.core.helpers import apply_params_headless, apply_params


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
    import cv2
    from zad1.code.data.load_data import load_valid_records
    from zad1.code.core.evaluator import evaluate_single

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