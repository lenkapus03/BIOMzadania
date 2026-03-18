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