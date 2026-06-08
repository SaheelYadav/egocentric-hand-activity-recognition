import cv2
from src.hand_tracking.hand_tracking import HandTracker
from src.object_detection.object_detector import ObjectDetector
from src.trajectory.trajectory_generator import TrajectoryGenerator
from src.trajectory.motion_analytics import MotionAnalytics

tracker = HandTracker(max_hands=2)
detector = ObjectDetector(model_size="yolo11n.pt", confidence=0.5)
trajectory = TrajectoryGenerator(max_length=30)
motion = MotionAnalytics(window_size=10)

cap = cv2.VideoCapture(0)

print("Starting motion analytics... Press Q to quit")

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
    trajs = trajectory.get_trajectories()

    motion.update(trajs)
    analytics = motion.get_analytics()

    frame = tracker.draw(frame, tracked_hands)
    frame = trajectory.draw(frame)

    for label, data in analytics.items():
        if data["velocity"] > 0:
            print(f"{label.upper()} | Velocity: {data['velocity']} | "
                  f"Acceleration: {data['acceleration']} | "
                  f"IMU: {data['pseudo_imu']}")

    cv2.imshow("Motion Analytics", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
tracker.close()
print("Done!")
