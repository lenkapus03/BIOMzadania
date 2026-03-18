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