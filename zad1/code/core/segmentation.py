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