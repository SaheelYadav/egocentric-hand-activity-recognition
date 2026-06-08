import numpy as np
from collections import deque


class ActivityRecognizer:
    def __init__(self, history_size=20):
        self.history_size = history_size
        self.velocity_history = {
            "left": deque(maxlen=history_size),
            "right": deque(maxlen=history_size)
        }
        self.position_history = {
            "left": deque(maxlen=history_size),
            "right": deque(maxlen=history_size)
        }
        self.interaction_history = deque(maxlen=history_size)

    def update(self, tracked_hands, analytics, workspace_events, frame_w, frame_h):
        for label, data in tracked_hands.items():
            if data is not None:
                wrist = data["keypoints"][0]
                pos = (wrist["x"] * frame_w, wrist["y"] * frame_h)
                self.position_history[label].append(pos)

        for label in ["left", "right"]:
            if label in analytics:
                self.velocity_history[label].append(
                    analytics[label]["velocity"]
                )

        self.interaction_history.append(workspace_events)

    def get_dominant_hand(self):
        left_vel = np.mean(list(self.velocity_history["left"])) \
            if self.velocity_history["left"] else 0
        right_vel = np.mean(list(self.velocity_history["right"])) \
            if self.velocity_history["right"] else 0
        return "left" if left_vel > right_vel else "right"

    def detect_vertical_motion(self, label):
        pts = list(self.position_history[label])
        if len(pts) < 5:
            return None
        y_values = [p[1] for p in pts]
        dy = y_values[-1] - y_values[0]
        if dy < -30:
            return "up"
        elif dy > 30:
            return "down"
        return None

    def detect_horizontal_motion(self, label):
        pts = list(self.position_history[label])
        if len(pts) < 5:
            return None
        x_values = [p[0] for p in pts]
        dx = x_values[-1] - x_values[0]
        if dx < -30:
            return "left"
        elif dx > 30:
            return "right"
        return None

    def get_avg_velocity(self, label):
        if not self.velocity_history[label]:
            return 0
        return np.mean(list(self.velocity_history[label]))

    def recognize(self, tracked_hands, workspace_events, frame_w, frame_h):
        activities = []

        for label, data in tracked_hands.items():
            if data is None:
                continue

            avg_vel = self.get_avg_velocity(label)
            vertical = self.detect_vertical_motion(label)
            horizontal = self.detect_horizontal_motion(label)

            # Get interactions for this hand
            hand_events = [
                e for e in workspace_events if e["hand"] == label
            ]
            holding = any(e["relation"] == "holding" for e in hand_events)
            near = any(e["relation"] == "near" for e in hand_events)
            on_surface = any(e["on_surface"] for e in hand_events)

            activity = "idle"

            # Rule based activity recognition
            if holding and vertical == "up":
                activity = "pick"
            elif holding and vertical == "down":
                activity = "place"
            elif holding and avg_vel > 15:
                activity = "move_object"
            elif holding and avg_vel < 3:
                activity = "hold"
            elif near and vertical == "up":
                activity = "reach"
            elif near and avg_vel > 20:
                activity = "wave"
            elif avg_vel > 25:
                activity = "fast_motion"
            elif avg_vel > 10:
                activity = "moving"
            elif avg_vel < 2:
                activity = "idle"

            # Detect typing/writing if both hands near surface
            if on_surface and avg_vel > 3 and avg_vel < 12:
                activity = "type_or_write"

            activities.append({
                "hand": label,
                "activity": activity,
                "velocity": round(float(avg_vel), 2),
                "vertical_motion": vertical,
                "horizontal_motion": horizontal,
                "holding": holding
            })

        return activities

    def draw(self, frame, activities):
        import cv2
        y_offset = 30
        for act in activities:
            text = (f"{act['hand'].upper()}: {act['activity']} "
                    f"(v={act['velocity']})")
            color = (0, 255, 0) if act["holding"] else (255, 255, 0)
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 35
        return frame
