import cv2
from ultralytics import YOLO
import numpy as np


class ObjectDetector:
    def __init__(self, model_size="yolo11n.pt", confidence=0.5):
        self.model = YOLO(model_size)
        self.confidence = confidence

    def detect(self, frame):
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1

                detections.append({
                    "label": label,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "track_id": track_id
                })

        return detections

    def draw(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            confidence = det["confidence"]
            track_id = det["track_id"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            text = f"{label} {confidence:.2f}"
            if track_id != -1:
                text += f" ID:{track_id}"

            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return frame