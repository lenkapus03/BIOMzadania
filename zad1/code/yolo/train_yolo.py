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