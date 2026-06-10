import cv2
import numpy as np
from ultralytics import YOLO, YOLOWorld


# ─────────────────────────────────────────────
#  Industrial / workshop tool classes for YOLO-World
# ─────────────────────────────────────────────
INDUSTRIAL_CLASSES = [
    # Hand tools
    "pliers", "needle-nose pliers", "cutting pliers", "wire stripper",
    "wrench", "adjustable wrench", "spanner", "screwdriver",
    "phillips screwdriver", "flathead screwdriver", "hammer", "mallet",
    "utility knife", "cutter", "box cutter", "scissors",
    "tape measure", "ruler", "caliper", "level",
    # Electrical / electronics
    "soldering iron", "multimeter", "wire", "cable", "connector",
    "circuit board", "pcb", "breadboard", "resistor", "capacitor",
    "battery", "power supply",
    # Fasteners & hardware
    "bolt", "nut", "screw", "nail", "washer", "rivet",
    # Power tools & equipment
    "drill", "drill bit", "angle grinder", "clamp", "vise",
    # Safety
    "safety gloves", "safety glasses", "helmet",
]

# Colour palette: COCO detections → blue, industrial → orange
COCO_COLOR      = (255, 140,   0)   # orange-blue  BGR
INDUSTRY_COLOR  = (  0, 200, 255)   # yellow-cyan  BGR


class ObjectDetector:
    """
    Dual-model detector that combines:
      • YOLO11  – all 80 standard COCO classes
      • YOLO-World – custom industrial / workshop tool classes

    Results are merged and deduplicated via class-aware NMS so that
    overlapping boxes from both models don't stack up.
    """

    def __init__(
        self,
        coco_model: str = "yolo11n.pt",
        world_model: str = "yolov8s-worldv2.pt",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        custom_industrial_classes: list[str] | None = None,
    ):
        self.confidence    = confidence
        self.iou_threshold = iou_threshold

        # ── COCO model (standard YOLO11 / YOLOv8) ──────────────────────
        print(f"[Detector] Loading COCO model: {coco_model}")
        self.coco_model = YOLO(coco_model)

        # ── YOLO-World model for industrial tools ───────────────────────
        print(f"[Detector] Loading YOLO-World model: {world_model}")
        self.world_model = YOLOWorld(world_model)
        classes = custom_industrial_classes or INDUSTRIAL_CLASSES
        self.world_model.set_classes(classes)
        print(f"[Detector] Industrial classes set ({len(classes)} total)")

    # ────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ────────────────────────────────────────────────────────────────────

    def _run_model(self, model, frame: np.ndarray, source_tag: str) -> list[dict]:
        """Run a single model and return normalised detection dicts."""
        results = model(frame, conf=self.confidence, verbose=False)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label      = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                track_id   = int(box.id[0]) if box.id is not None else -1
                detections.append({
                    "label":      label,
                    "bbox":       [x1, y1, x2, y2],
                    "confidence": confidence,
                    "track_id":   track_id,
                    "source":     source_tag,   # "coco" | "industrial"
                })
        return detections

    @staticmethod
    def _iou(a: list[int], b: list[int]) -> float:
        """Compute IoU between two [x1,y1,x2,y2] boxes."""
        ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (area_a + area_b - inter)

    def _merge_and_nms(self, coco_dets: list[dict], industrial_dets: list[dict]) -> list[dict]:
        """
        Merge detections from both models and suppress duplicates.
        Strategy:
          1. Industrial detections are kept as-is (high priority).
          2. COCO detections are suppressed when they overlap heavily
             (IoU > iou_threshold) with an already-accepted detection.
          3. Within each pool, standard confidence-based NMS is applied.
        """
        all_dets = industrial_dets + coco_dets   # industrial first = higher priority
        kept: list[dict] = []

        # Sort by descending confidence
        all_dets.sort(key=lambda d: d["confidence"], reverse=True)

        for det in all_dets:
            suppressed = False
            for kept_det in kept:
                if self._iou(det["bbox"], kept_det["bbox"]) > self.iou_threshold:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(det)

        return kept

    # ────────────────────────────────────────────────────────────────────
    #  Public API  (same interface as your original class)
    # ────────────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list[dict]:
        coco_dets = self._run_model(self.coco_model, frame, "coco")
        
        # Run YOLO-World every 10 frames only to save CPU
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
            self._last_industrial = []
        
        self._frame_count += 1
        
        if self._frame_count % 10 == 0:
            try:
                self._last_industrial = self._run_model(
                    self.world_model, frame, "industrial"
                )
            except Exception:
                self._last_industrial = []
        
        return self._merge_and_nms(coco_dets, self._last_industrial)

    def draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """
        Draw detections on frame.
          • Orange boxes  → COCO classes
          • Cyan boxes    → Industrial / YOLO-World classes
        """
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label      = det["label"]
            confidence = det["confidence"]
            track_id   = det["track_id"]
            source     = det.get("source", "coco")

            color = INDUSTRY_COLOR if source == "industrial" else COCO_COLOR

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label text
            text = f"{label} {confidence:.2f}"
            if track_id != -1:
                text += f" #{track_id}"

            # Background pill for readability
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - th - baseline - 6),
                (x1 + tw + 4, y1),
                color, -1
            )
            cv2.putText(
                frame, text,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 2, cv2.LINE_AA
            )

        return frame

    def draw_legend(self, frame: np.ndarray) -> np.ndarray:
        """Overlay a small legend in the top-right corner."""
        h, w = frame.shape[:2]
        items = [
            ("COCO (80 classes)",       COCO_COLOR),
            ("Industrial tools",        INDUSTRY_COLOR),
        ]
        pad, line_h = 8, 22
        box_w = 200
        x0 = w - box_w - 10
        y0 = 10

        cv2.rectangle(frame, (x0-4, y0-4),
                      (x0+box_w, y0 + len(items)*line_h + pad),
                      (30, 30, 30), -1)

        for i, (name, color) in enumerate(items):
            cy = y0 + i*line_h + line_h//2
            cv2.rectangle(frame, (x0, cy-6), (x0+14, cy+6), color, -1)
            cv2.putText(frame, name, (x0+20, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1, cv2.LINE_AA)

        return frame


# ─────────────────────────────────────────────
#  Quick standalone test  (python object_detector.py)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else 0   # path or webcam

    detector = ObjectDetector(
        coco_model  = "yolo11n.pt",       # swap to yolo11s/m/l for accuracy
        world_model = "yolov8s-worldv2.pt", # swap to yolov8m-worldv2.pt for accuracy
        confidence  = 0.35,
    )

    cap = cv2.VideoCapture(source if source != "0" else 0)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    print("[Detector] Running… press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        frame      = detector.draw(frame, detections)
        frame      = detector.draw_legend(frame)

        # HUD: detection count
        coco_n = sum(1 for d in detections if d["source"] == "coco")
        ind_n  = sum(1 for d in detections if d["source"] == "industrial")
        cv2.putText(frame, f"COCO: {coco_n}  Industrial: {ind_n}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Dual Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
