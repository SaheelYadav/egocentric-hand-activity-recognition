import json
import os
from datetime import datetime


class DatasetExporter:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_dir, f"session_{self.session_id}")
        os.makedirs(self.session_dir, exist_ok=True)

        # Data buffers
        self.hand_keypoints = []
        self.object_annotations = []
        self.interactions = []
        self.trajectories = []
        self.motion_statistics = []
        self.pseudo_imu = []
        self.workspace_relationships = []
        self.actions = []
        self.frame_count = 0

        print(f"Session started: {self.session_dir}")

    def record_frame(self, frame_idx, tracked_hands, detections,
                     interactions, workspace_events, trajs,
                     analytics, activities):

        timestamp = frame_idx / 30.0  # Assume 30fps

        # Hand keypoints
        hand_data = {"frame": frame_idx, "timestamp": timestamp, "hands": {}}
        for label, data in tracked_hands.items():
            if data is not None:
                hand_data["hands"][label] = {
                    "keypoints": data["keypoints"]
                }
        self.hand_keypoints.append(hand_data)

        # Object annotations
        obj_data = {"frame": frame_idx, "timestamp": timestamp, "objects": []}
        for det in detections:
            obj_data["objects"].append({
                "label": det["label"],
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "track_id": det["track_id"]
            })
        self.object_annotations.append(obj_data)

        # Interactions
        self.interactions.append({
            "frame": frame_idx,
            "timestamp": timestamp,
            "interactions": interactions
        })

        # Trajectories
        self.trajectories.append({
            "frame": frame_idx,
            "timestamp": timestamp,
            "left": trajs["left"][-1] if trajs["left"] else None,
            "right": trajs["right"][-1] if trajs["right"] else None
        })

        # Motion statistics
        motion_data = {"frame": frame_idx, "timestamp": timestamp}
        for label in ["left", "right"]:
            if label in analytics:
                motion_data[label] = {
                    "velocity": analytics[label]["velocity"],
                    "acceleration": analytics[label]["acceleration"]
                }
        self.motion_statistics.append(motion_data)

        # Pseudo IMU
        imu_data = {"frame": frame_idx, "timestamp": timestamp}
        for label in ["left", "right"]:
            if label in analytics:
                imu_data[label] = analytics[label]["pseudo_imu"]
        self.pseudo_imu.append(imu_data)

        # Workspace relationships
        self.workspace_relationships.append({
            "frame": frame_idx,
            "timestamp": timestamp,
            "events": workspace_events
        })

        # Actions
        self.actions.append({
            "frame": frame_idx,
            "timestamp": timestamp,
            "activities": activities
        })

        self.frame_count += 1

    def save(self):
        files = {
            "hand_keypoints_3d.json": self.hand_keypoints,
            "object_annotations.json": self.object_annotations,
            "hand_object_interactions.json": self.interactions,
            "trajectories.json": self.trajectories,
            "motion_statistics.json": self.motion_statistics,
            "pseudo_imu.json": self.pseudo_imu,
            "workspace_relationships.json": self.workspace_relationships,
            "actions.json": self.actions
        }

        for filename, data in files.items():
            filepath = os.path.join(self.session_dir, filename)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"Saved: {filepath}")

        # Save metadata
        metadata = {
            "session_id": self.session_id,
            "total_frames": self.frame_count,
            "output_dir": self.session_dir,
            "created_at": datetime.now().isoformat()
        }
        with open(os.path.join(self.session_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nDataset saved! Total frames: {self.frame_count}")
        print(f"Location: {self.session_dir}")
