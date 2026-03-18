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


def create_dataset_structure():
    """Vytvorí priečinkovú štruktúru datasetu pre YOLO."""
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_FOLDER, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_FOLDER, "labels", split), exist_ok=True)
    print(f"Štruktúra datasetu vytvorená v: {OUTPUT_FOLDER}")


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


def prepare_dataset():
    records = load_valid_records(CSV_PATH, DATA_FOLDER, n=100)

    create_dataset_structure()

    # Náhodne rozdeľ na train/val
    random.shuffle(records)
    split_idx = int(len(records) * TRAIN_RATIO)
    train_records = records[:split_idx]
    val_records = records[split_idx:]
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")

    for split, split_records in [("train", train_records), ("val", val_records)]:
        ok = 0
        skip = 0
        for record in split_records:
            img_path = os.path.normpath(os.path.join(DATA_FOLDER, record.image.replace("\\", "/")))
            image = cv2.imread(img_path)
            if image is None:
                skip += 1
                continue

            h, w = image.shape[:2]
            circles = record_to_gt_circles(record)

            # Preskočiť ak dúhovka nemá platný polomer
            if circles["iris"][2] <= 0:
                skip += 1
                continue

            # Vytvor masku z ground truth kruhov
            mask = segment_iris(image, circles)

            # Konvertuj masku na YOLO polygon
            yolo_label = mask_to_yolo_polygon(mask, w, h)
            if yolo_label is None:
                skip += 1
                continue

            # Názov súboru bez lomítok
            filename = record.image.replace("\\", "/").replace("/", "_").replace(".jpg", "")

            # Ulož obrazok
            img_out = os.path.join(OUTPUT_FOLDER, "images", split, f"{filename}.jpg")
            shutil.copy(img_path, img_out)

            # Ulož label
            label_out = os.path.join(OUTPUT_FOLDER, "labels", split, f"{filename}.txt")
            with open(label_out, "w") as f:
                f.write(yolo_label + "\n")

            ok += 1

        print(f"  {split}: {ok} spracovaných, {skip} preskočených")

    create_data_yaml()
    print("\nDataset pripravený.")


if __name__ == "__main__":
    prepare_dataset()