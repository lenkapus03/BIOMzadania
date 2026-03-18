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