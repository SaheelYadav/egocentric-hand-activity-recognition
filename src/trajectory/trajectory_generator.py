import numpy as np
from collections import deque


class TrajectoryGenerator:
    def __init__(self, max_length=30):
        self.max_length = max_length
        self.trajectories = {
            "left": deque(maxlen=max_length),
            "right": deque(maxlen=max_length),
            "objects": {}
        }

    def get_hand_wrist(self, keypoints, frame_w, frame_h):
        wrist = keypoints[0]
        return (
            int(wrist["x"] * frame_w),
            int(wrist["y"] * frame_h)
        )

    def update_hand(self, tracked_hands, frame_w, frame_h):
        for label, data in tracked_hands.items():
            if data is not None:
                point = self.get_hand_wrist(
                    data["keypoints"], frame_w, frame_h
                )
                self.trajectories[label].append(point)
            else:
                self.trajectories[label].append(None)

    def update_objects(self, detections):
        for det in detections:
            track_id = det["track_id"]
            if track_id == -1:
                continue

            key = f"{det['label']}_{track_id}"
            if key not in self.trajectories["objects"]:
                self.trajectories["objects"][key] = deque(maxlen=self.max_length)

            x1, y1, x2, y2 = det["bbox"]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            self.trajectories["objects"][key].append((cx, cy))

    def update(self, tracked_hands, detections, frame_w, frame_h):
        self.update_hand(tracked_hands, frame_w, frame_h)
        self.update_objects(detections)

    def get_trajectories(self):
        return {
            "left": list(self.trajectories["left"]),
            "right": list(self.trajectories["right"]),
            "objects": {
                k: list(v) for k, v in self.trajectories["objects"].items()
            }
        }

    def draw(self, frame, color_left=(0, 0, 255),
             color_right=(255, 0, 0), color_obj=(0, 255, 255)):
        import cv2

        # Draw left hand trajectory
        pts = [p for p in self.trajectories["left"] if p is not None]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color_left, 2)

        # Draw right hand trajectory
        pts = [p for p in self.trajectories["right"] if p is not None]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color_right, 2)

        # Draw object trajectories
        for key, traj in self.trajectories["objects"].items():
            pts = list(traj)
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], color_obj, 2)

        return frame
