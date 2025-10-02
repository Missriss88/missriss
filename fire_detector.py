# fire_detector.py
import os
from ultralytics import YOLO

def load_fire_model(model_path='fire2.pt'):
    abspath = os.path.abspath(model_path)
    if not os.path.exists(abspath):
        raise FileNotFoundError(f"fire model file not found: {abspath}")
    return YOLO(abspath)  # 로컬 pt 직접 로드

def detect_fire(frame, model):
    results = model.predict(source=frame, verbose=False)  # numpy BGR 가능
    boxes = []
    for r in results:
        names = r.names  # 클래스 이름 dict
        for b in r.boxes:
            cls_id = int(b.cls)
            name = names.get(cls_id, str(cls_id))
            if name.lower() == 'fire':   # 학습 시 클래스명이 다르면 여길 맞춰주세요
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                boxes.append([x1, y1, x2, y2])
    return boxes
