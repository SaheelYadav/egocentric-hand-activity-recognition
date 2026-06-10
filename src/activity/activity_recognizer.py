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

    @staticmethod
    def _normalize_object_label(label: str | None) -> str | None:
        if not label:
            return None
        return label.lower().strip()

    def get_natural_label(self, activity, hand_label, interactions=None):
        """
        Generates clear descriptive annotations by combining the activity + the object.
        Supports both (activity_str, hand_label, interactions) and (activity_dict, interactions)
        for backward compatibility.
        """
        if isinstance(activity, dict):
            activity_dict = activity
            activity_str = activity_dict.get("activity", "idle")
            hand = activity_dict.get("hand")
            interactions_list = hand_label or []
        else:
            activity_str = activity
            hand = hand_label
            interactions_list = interactions or []

        # Find object holding or near
        obj = None
        for e in interactions_list:
            if e.get("hand") == hand:
                rel = e.get("relation") or e.get("interaction_type")
                if rel == "holding":
                    obj = e.get("object")
                    break

        if not obj:
            for e in interactions_list:
                if e.get("hand") == hand:
                    rel = e.get("relation") or e.get("interaction_type")
                    if rel == "near":
                        obj = e.get("object")
                        break

        norm_obj = self._normalize_object_label(obj) if obj else None
        if norm_obj in ["person", "nail", "scissors"]:
            norm_obj = None

        if activity_str == "pick":
            return f"lifting {norm_obj}" if norm_obj else "picking up object"
        if activity_str == "place":
            return f"placing {norm_obj} down" if norm_obj else "placing object down"
        if activity_str == "hold":
            return f"holding {norm_obj}" if norm_obj else "holding object"
        if activity_str == "move_object":
            return f"moving {norm_obj}" if norm_obj else "moving object"
        if activity_str == "reach":
            return f"reaching for {norm_obj}" if norm_obj else "reaching"
        if activity_str == "type_or_write":
            return "typing or writing"
        if activity_str == "wave":
            return "waving hand"
        if activity_str == "idle":
            return "hand at rest"
        if activity_str == "moving":
            return "moving hand"
        if activity_str == "fast_motion":
            return "fast hand motion"

        if norm_obj:
            return f"{activity_str} {norm_obj}"
        return activity_str

    def get_scene_summary(self, activities, interactions):
        """
        Generates ONE single summary sentence describing what is happening in the scene overall.
        """
        held_objects = {}
        near_objects = {}
        
        # Parse interactions
        for inter in (interactions or []):
            hand = inter.get("hand")
            if not hand:
                continue
            rel = inter.get("relation") or inter.get("interaction_type")
            obj = inter.get("object")
            if obj:
                obj = self._normalize_object_label(obj)
                if obj in ["person", "nail", "scissors"]:
                    obj = None
            if obj:
                if rel == "holding":
                    held_objects[hand] = obj
                elif rel == "near":
                    near_objects[hand] = obj

        active_hands = [act for act in activities if act is not None]
        
        # 1. Knife detection rules
        if "knife" in held_objects.values():
            return "Person is cutting or chopping"
        if "knife" in near_objects.values():
            return "Person is using a knife"

        # 2. Both hands moving at high velocity (>15)
        if len(active_hands) == 2:
            if all(act["velocity"] > 15 for act in active_hands):
                return "Person is performing active task"

        # 3. Both hands detected and holding objects
        if len(active_hands) == 2 and len(held_objects) == 2:
            objs = list(set(held_objects.values()))
            if len(objs) == 1:
                return f"Person is manipulating {objs[0]} with both hands"
            else:
                return f"Person is manipulating {held_objects.get('left')} and {held_objects.get('right')} with both hands"

        # 4. Any hand moving at velocity > 20
        if any(act["velocity"] > 20 for act in active_hands):
            return "Person is performing fast motion"

        # 5. One hand holding object + picking
        for act in active_hands:
            hand = act["hand"]
            if hand in held_objects and act["activity"] == "pick":
                return f"Person is picking up {held_objects[hand]}"

        # 6. One hand holding object + placing
        for act in active_hands:
            hand = act["hand"]
            if hand in held_objects and act["activity"] == "place":
                return f"Person is placing {held_objects[hand]} down"

        # 7. One hand holding object + velocity between 5-15
        for act in active_hands:
            hand = act["hand"]
            if hand in held_objects and (5 <= act["velocity"] <= 15):
                return f"Person is handling {held_objects[hand]}"

        # 8. One hand holding object + moving
        for act in active_hands:
            hand = act["hand"]
            if hand in held_objects:
                if act["activity"] in ["move_object", "moving"] or act["velocity"] > 15:
                    return f"Person is moving the {held_objects[hand]}"

        # 9. One hand holding object default fallback
        for act in active_hands:
            hand = act["hand"]
            if hand in held_objects:
                return f"Person is handling {held_objects[hand]}"

        # 10. Hand near object + slow movement
        for act in active_hands:
            hand = act["hand"]
            if hand in near_objects and hand not in held_objects:
                if act["velocity"] < 10 or act["activity"] in ["idle", "reach"]:
                    return f"Person is working with {near_objects[hand]}"

        # 11. Default when hands detected but no clear activity
        if active_hands:
            return "Person is working with hands"

        # 12. Idle
        return "Person is at rest"

    def draw(self, frame, activities, interactions=None):
        import cv2
        y_offset = 30
        for act in activities:
            label_text = self.get_natural_label(act["activity"], act["hand"], interactions)
            text = (f"{act['hand'].upper()}: {label_text} "
                    f"(v={act['velocity']})")
            color = (0, 255, 0) if act["holding"] else (255, 255, 0)
            
            font_scale = 0.55 if len(text) > 35 else 0.7
            
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            y_offset += 35
        return frame
