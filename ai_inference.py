from ultralytics import YOLO
from config import MODEL_PATH

model = YOLO(MODEL_PATH)

def analyze_image(image_path):
    results = model(image_path)[0]

    summary = {
        "healthy_grass": 0,
        "dead_grass": 0,
        "water": 0
    }

    for box in results.boxes:
        cls = results.names[int(box.cls)]
        summary[cls] = summary.get(cls, 0) + 1

    total = sum(summary.values()) or 1
    for k in summary:
        summary[k] /= total

    return summary