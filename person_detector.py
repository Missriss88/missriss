# person_detector.py
import os
from ultralytics import YOLO

def load_person_model(model_path='yolov8n.pt'):
    abspath = os.path.abspath(model_path)
    if not os.path.exists(abspath):
        raise FileNotFoundError(f"person model file not found: {abspath}")
    return YOLO(abspath)

def detect_person(frame, model):
    results = model.predict(source=frame, verbose=False)
    boxes = []
    for r in results:
        names = r.names
        for b in r.boxes:
            if names.get(int(b.cls), '') == 'person':
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                boxes.append([x1, y1, x2, y2])
    return boxes
