import cv2
from src.hand_tracking.hand_tracking import HandTracker
from src.object_detection.object_detector import ObjectDetector
from src.trajectory.trajectory_generator import TrajectoryGenerator
from src.trajectory.motion_analytics import MotionAnalytics
from src.interaction.workspace_understanding import WorkspaceUnderstanding
from src.activity.activity_recognizer import ActivityRecognizer

tracker = HandTracker(max_hands=2)
detector = ObjectDetector(model_size="yolo11n.pt", confidence=0.5)
trajectory = TrajectoryGenerator(max_length=30)
motion = MotionAnalytics(window_size=10)
workspace = WorkspaceUnderstanding(proximity_threshold=150)
activity = ActivityRecognizer(history_size=20)

cap = cv2.VideoCapture(0)

print("Starting activity recognizer... Press Q to quit")

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

    workspace_events = workspace.analyze(tracked_hands, detections, w, h)

    activity.update(tracked_hands, analytics, workspace_events, w, h)
    activities = activity.recognize(tracked_hands, workspace_events, w, h)

    frame = tracker.draw(frame, tracked_hands)
    frame = detector.draw(frame, detections)
    frame = activity.draw(frame, activities)

    for act in activities:
        print(f"{act['hand'].upper()} | Activity: {act['activity']} | "
              f"Velocity: {act['velocity']} | "
              f"Holding: {act['holding']}")

    cv2.imshow("Activity Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
tracker.close()
print("Done!")
