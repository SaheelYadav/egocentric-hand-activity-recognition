import numpy as np


class InteractionDetector:
    def __init__(self, distance_threshold=100):
        self.distance_threshold = distance_threshold

    def get_hand_center(self, keypoints):
        x_coords = [kp["x"] for kp in keypoints]
        y_coords = [kp["y"] for kp in keypoints]
        return np.mean(x_coords), np.mean(y_coords)

    def get_bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def get_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def is_hand_inside_bbox(self, hand_center, bbox, frame_w, frame_h):
        x1, y1, x2, y2 = bbox
        hx = hand_center[0] * frame_w
        hy = hand_center[1] * frame_h
        return x1 <= hx <= x2 and y1 <= hy <= y2

    def detect(self, tracked_hands, detections, frame_w, frame_h):
        interactions = []

        for hand_label, hand_data in tracked_hands.items():
            if hand_data is None:
                continue

            keypoints = hand_data["keypoints"]
            hand_center_norm = self.get_hand_center(keypoints)

            hand_center_px = (
                hand_center_norm[0] * frame_w,
                hand_center_norm[1] * frame_h
            )

            for det in detections:
                bbox = det["bbox"]
                obj_label = det["label"]
                confidence = det["confidence"]

                inside = self.is_hand_inside_bbox(hand_center_norm, bbox, frame_w, frame_h)
                obj_center = self.get_bbox_center(bbox)
                distance = self.get_distance(hand_center_px, obj_center)

                if inside or distance < self.distance_threshold:
                    interaction_type = "holding" if inside else "near"

                    interactions.append({
                        "hand": hand_label,
                        "object": obj_label,
                        "interaction_type": interaction_type,
                        "distance": round(distance, 2),
                        "confidence": round(confidence, 2),
                        "bbox": bbox
                    })

        return interactions

    def draw(self, frame, interactions):
        import cv2
        for interaction in interactions:
            x1, y1, x2, y2 = interaction["bbox"]
            hand = interaction["hand"].upper()
            obj = interaction["object"]
            itype = interaction["interaction_type"]

            color = (0, 255, 0) if itype == "holding" else (0, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{hand} {itype} {obj}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame