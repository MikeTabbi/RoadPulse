"""
RoadPulse - YOLOv8 Training Script
Trains YOLOv8s on the prepared RDD2022 dataset.

Usage:
    python models/train.py
"""

from ultralytics import YOLO

if __name__ == '__main__':
    # Load pretrained YOLOv8s model
    model = YOLO("yolov8s.pt")

    # Train on RDD2022
    results = model.train(
        data="data/rdd2022_yolo/rdd2022.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project="runs",
        name="rdd2022_yolov8s",
        patience=20,
        save=True,
        plots=True,
    )

    print("Training complete!")
    print(f"Best model saved to: runs/rdd2022_yolov8s/weights/best.pt")