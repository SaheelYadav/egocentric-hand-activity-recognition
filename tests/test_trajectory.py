import cv2
from src.hand_tracking.hand_tracking import HandTracker
from src.object_detection.object_detector import ObjectDetector
from src.trajectory.trajectory_generator import TrajectoryGenerator

tracker = HandTracker(max_hands=2)
detector = ObjectDetector(model_size="yolo11n.pt", confidence=0.5)
trajectory = TrajectoryGenerator(max_length=30)

cap = cv2.VideoCapture(0)

print("Starting trajectory generator... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found!")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    tracked_hands = tracker.process(frame)
    detections = detector.detect(frame)

    trajectory.update(tracked_hands, detections, w, h)

    frame = tracker.draw(frame, tracked_hands)
    frame = detector.draw(frame, detections)
    frame = trajectory.draw(frame)

    trajs = trajectory.get_trajectories()

    for label in ["left", "right"]:
        pts = [p for p in trajs[label] if p is not None]
        if pts:
            print(f"{label.upper()} hand trajectory points: {len(pts)}")

    cv2.imshow("Trajectory", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
tracker.close()
print("Done!")
