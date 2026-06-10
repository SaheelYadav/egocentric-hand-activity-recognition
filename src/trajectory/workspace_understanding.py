import numpy as np


class WorkspaceUnderstanding:
    def __init__(self, proximity_threshold=150):
        self.proximity_threshold = proximity_threshold

    def get_bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def get_hand_center_px(self, keypoints, frame_w, frame_h):
        x = np.mean([kp["x"] for kp in keypoints]) * frame_w
        y = np.mean([kp["y"] for kp in keypoints]) * frame_h
        return (int(x), int(y))

    def get_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def is_on_surface(self, bbox, frame_h, threshold=0.75):
        _, _, _, y2 = bbox
        return y2 > frame_h * threshold

    def is_inside_region(self, obj_center, region_bbox):
        x1, y1, x2, y2 = region_bbox
        return x1 <= obj_center[0] <= x2 and y1 <= obj_center[1] <= y2

    def analyze(self, tracked_hands, detections, frame_w, frame_h):
        workspace_events = []

        hand_centers = {}
        for label, data in tracked_hands.items():
            if data is not None:
                hand_centers[label] = self.get_hand_center_px(
                    data["keypoints"], frame_w, frame_h
                )

        for det in detections:
            bbox = det["bbox"]
            obj_label = det["label"]
            obj_center = self.get_bbox_center(bbox)

            # Check if object is on surface
            on_surface = self.is_on_surface(bbox, frame_h)

            # Check proximity to each hand
            for hand_label, hand_center in hand_centers.items():
                distance = self.get_distance(obj_center, hand_center)

                if distance < self.proximity_threshold:
                    relation = "near"
                    if distance < self.proximity_threshold * 0.5:
                        relation = "holding"

                    workspace_events.append({
                        "object": obj_label,
                        "hand": hand_label,
                        "relation": relation,
                        "distance": round(float(distance), 2),
                        "on_surface": on_surface,
                        "obj_center": obj_center,
                        "bbox": bbox
                    })
                else:
                    workspace_events.append({
                        "object": obj_label,
                        "hand": hand_label,
                        "relation": "far",
                        "distance": round(float(distance), 2),
                        "on_surface": on_surface,
                        "obj_center": obj_center,
                        "bbox": bbox
                    })

        return workspace_events

    def draw(self, frame, workspace_events):
        import cv2
        seen = set()

        for event in workspace_events:
            bbox = event["bbox"]
            key = tuple(bbox)
            if key in seen:
                continue
            seen.add(key)

            x1, y1, x2, y2 = bbox
            relation = event["relation"]
            obj = event["object"]
            hand = event["hand"].upper()
            on_surface = event["on_surface"]

            if relation == "holding":
                color = (0, 255, 0)
            elif relation == "near":
                color = (0, 255, 255)
            else:
                color = (200, 200, 200)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{hand}:{relation} {obj}"
            if on_surface:
                label += " [surface]"

            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame