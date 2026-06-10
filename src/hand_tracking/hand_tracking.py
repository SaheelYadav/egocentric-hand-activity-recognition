import cv2
import mediapipe as mp
import numpy as np


# ── Landmark index names for all 21 points ──────────────────────────────────
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

# Per-finger colours (BGR) used for both dots and connections
FINGER_COLORS = {
    "THUMB":  (  0, 204, 255),   # yellow
    "INDEX":  (  0, 255,   0),   # green
    "MIDDLE": (255, 128,   0),   # blue
    "RING":   (255,   0, 255),   # magenta
    "PINKY":  (  0,  80, 255),   # red-orange
    "WRIST":  (255, 255, 255),   # white
}

# Finger → landmark index ranges
FINGER_RANGES = {
    "THUMB":  (1,  5),
    "INDEX":  (5,  9),
    "MIDDLE": (9,  13),
    "RING":   (13, 17),
    "PINKY":  (17, 21),
}


def _preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Light preprocessing that helps MediaPipe in dim / workshop environments:
      1. CLAHE on the L channel  → local contrast boost
      2. Mild sharpening kernel  → crisp edges for landmark detection
    Does NOT alter the frame passed in; returns a new array.
    """
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    l     = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    kernel    = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened


def _finger_for_idx(idx: int) -> str:
    if idx == 0:
        return "WRIST"
    for name, (lo, hi) in FINGER_RANGES.items():
        if lo <= idx < hi:
            return name
    return "WRIST"


class HandTracker:
    """
    Robust 21-point hand tracker built on MediaPipe Hands.

    Improvements over the original:
      • CLAHE + sharpening pre-processing for workshop / low-light scenes
      • Lower (but still reliable) detection threshold with model_complexity=1
      • Correct Left/Right label — MediaPipe returns mirrored labels for webcam
        input; we flip them when `mirror_labels=True` (default ON for webcam)
      • Rich per-finger coloured landmark drawing with index labels
      • Per-landmark pixel coordinates exposed in `keypoints`
      • Finger-state (extended / bent) helper
      • draw_landmarks_detailed() replaces the default MediaPipe style
    """

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.4,   # lowered from 0.7 for workshop
        tracking_confidence: float  = 0.4,
        model_complexity: int       = 1,      # 0=fast, 1=balanced, 2=accurate
        mirror_labels: bool         = True,   # flip L/R for front-facing webcam
        preprocess: bool            = True,   # CLAHE + sharpen
    ):
        self.mp_hands  = mp.solutions.hands
        self.mp_draw   = mp.solutions.drawing_utils

        self.mirror_labels = mirror_labels
        self.preprocess    = preprocess

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    # ────────────────────────────────────────────────────────────────────
    #  Core processing
    # ────────────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray, mirror_labels: bool | None = None) -> dict:
        """
        Detect hands in *frame* (BGR).

        Returns:
            {
              "left":  None | { keypoints, landmarks, finger_states, bbox },
              "right": None | { keypoints, landmarks, finger_states, bbox },
            }

        keypoints: list of 21 dicts
            { name, idx, x, y, z, px, py }   ← px/py are pixel coords
        finger_states: dict  e.g. {"THUMB": True, "INDEX": False, ...}
            True = finger extended
        bbox: (x1, y1, x2, y2) pixel bounding box of the hand
        """
        h, w = frame.shape[:2]
        src  = _preprocess(frame) if self.preprocess else frame
        rgb  = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        output = {"left": None, "right": None}

        if not (results.multi_hand_landmarks and results.multi_handedness):
            return output

        mirror = self.mirror_labels if mirror_labels is None else mirror_labels

        for hand_lm, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            raw_label = handedness.classification[0].label.lower()  # mediapipe label

            # MediaPipe mirrors left↔right for selfie/webcam streams
            if mirror:
                label = "right" if raw_label == "left" else "left"
                if output[label] is not None:
                    label = raw_label
            else:
                label = raw_label

            # Build rich keypoints list
            keypoints = []
            xs, ys = [], []
            for idx, lm in enumerate(hand_lm.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                xs.append(px); ys.append(py)
                keypoints.append({
                    "name": LANDMARK_NAMES[idx],
                    "idx":  idx,
                    "x":    lm.x,
                    "y":    lm.y,
                    "z":    lm.z,
                    "px":   px,
                    "py":   py,
                })

            bbox          = (min(xs), min(ys), max(xs), max(ys))
            finger_states = self._finger_states(keypoints)

            output[label] = {
                "keypoints":     keypoints,
                "landmarks":     hand_lm,
                "finger_states": finger_states,
                "bbox":          bbox,
            }

        return output

    # ────────────────────────────────────────────────────────────────────
    #  Finger state helper
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _finger_states(keypoints: list[dict]) -> dict[str, bool]:
        """
        Simple heuristic: finger is 'extended' when its TIP y-coord is
        above (smaller py) its MCP y-coord.  Thumb uses x-axis.
        Returns dict {finger_name: bool}.
        """
        kp = {p["name"]: p for p in keypoints}
        states = {}

        # Thumb: tip to the left of IP (for right hand front-facing)
        states["THUMB"] = kp["THUMB_TIP"]["px"] < kp["THUMB_IP"]["px"]

        for finger in ("INDEX", "MIDDLE", "RING", "PINKY"):
            tip = kp[f"{finger}_TIP"]["py"]
            mcp = kp[f"{finger}_MCP"]["py"]
            states[finger] = tip < mcp   # extended when tip above MCP

        return states

    # ────────────────────────────────────────────────────────────────────
    #  Drawing
    # ────────────────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, tracked_hands: dict) -> np.ndarray:
        """
        Full-detail draw: coloured per-finger connections, numbered dots,
        hand label + bounding box + finger-state bar.
        """
        for label, data in tracked_hands.items():
            if data is None:
                continue

            kps = data["keypoints"]
            h_img, w_img = frame.shape[:2]

            # ── Draw connections first (so dots sit on top) ──────────────
            connections = list(self.mp_hands.HAND_CONNECTIONS)
            for (a_idx, b_idx) in connections:
                finger = _finger_for_idx(a_idx)
                color  = FINGER_COLORS[finger]
                pa = (kps[a_idx]["px"], kps[a_idx]["py"])
                pb = (kps[b_idx]["px"], kps[b_idx]["py"])
                cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)

            # ── Draw all 21 landmark dots with index labels ──────────────
            for kp in kps:
                finger = _finger_for_idx(kp["idx"])
                color  = FINGER_COLORS[finger]
                cx, cy = kp["px"], kp["py"]

                # Outer white ring + coloured fill
                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                cv2.circle(frame, (cx, cy), 5, color, -1)

                # Landmark index number (tiny, above the dot)
                cv2.putText(
                    frame, str(kp["idx"]),
                    (cx - 5, cy - 9),
                    cv2.FONT_HERSHEY_PLAIN, 0.8,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

            # ── Bounding box ─────────────────────────────────────────────
            x1, y1, x2, y2 = data["bbox"]
            box_color = (0, 255, 120) if label == "right" else (255, 180, 0)
            cv2.rectangle(frame, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10),
                          box_color, 1)

            # ── Hand label ───────────────────────────────────────────────
            cv2.putText(
                frame, label.upper(),
                (x1 - 10, y1 - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2, cv2.LINE_AA,
            )

            # ── Finger-state bar (bottom of bbox) ────────────────────────
            self._draw_finger_bar(frame, data["finger_states"],
                                  x1 - 10, y2 + 14)

        return frame

    @staticmethod
    def _draw_finger_bar(
        frame: np.ndarray,
        states: dict[str, bool],
        x: int,
        y: int,
    ) -> None:
        """Small row of coloured pills showing which fingers are extended."""
        order = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
        short = ["T", "I", "M", "R", "P"]
        pill_w, pill_h, gap = 18, 14, 3

        for i, (finger, short_name) in enumerate(zip(order, short)):
            px  = x + i * (pill_w + gap)
            col = FINGER_COLORS[finger] if states.get(finger) else (60, 60, 60)
            cv2.rectangle(frame, (px, y), (px + pill_w, y + pill_h), col, -1)
            cv2.putText(frame, short_name,
                        (px + 4, y + pill_h - 2),
                        cv2.FONT_HERSHEY_PLAIN, 0.85,
                        (255, 255, 255), 1, cv2.LINE_AA)

    def close(self):
        self.hands.close()


# ─────────────────────────────────────────────
#  Standalone test  (python hand_tracking.py [source])
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    source  = sys.argv[1] if len(sys.argv) > 1 else 0
    tracker = HandTracker(
        max_hands=2,
        detection_confidence=0.5,
        tracking_confidence=0.5,
        model_complexity=1,
        mirror_labels=True,   # set False for recorded video files
        preprocess=True,
    )

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    print("[HandTracker] Running… press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracked = tracker.process(frame)
        frame   = tracker.draw(frame, tracked)

        # Print keypoints to console (optional)
        for side, data in tracked.items():
            if data:
                extended = [f for f, s in data["finger_states"].items() if s]
                print(f"  {side.upper():5s} | fingers up: {extended}")

        cv2.imshow("Hand Tracker – 21 pts", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()