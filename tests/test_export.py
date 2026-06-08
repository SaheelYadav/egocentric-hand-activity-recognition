import cv2
from src.hand_tracking.hand_tracking import HandTracker
from src.object_detection.object_detector import ObjectDetector
from src.trajectory.trajectory_generator import TrajectoryGenerator
from src.trajectory.motion_analytics import MotionAnalytics
from src.interaction.interaction_detector import InteractionDetector
from src.interaction.workspace_understanding import WorkspaceUnderstanding
from src.activity.activity_recognizer import ActivityRecognizer
from src.utils.dataset_exporter import DatasetExporter

tracker = HandTracker(max_hands=2)
detector = ObjectDetector(model_size="yolo11n.pt", confidence=0.5)
trajectory = TrajectoryGenerator(max_length=30)
motion = MotionAnalytics(window_size=10)
interaction = InteractionDetector(distance_threshold=100)
workspace = WorkspaceUnderstanding(proximity_threshold=150)
activity = ActivityRecognizer(history_size=20)
exporter = DatasetExporter(output_dir="output")

cap = cv2.VideoCapture(0)
frame_idx = 0
max_frames = 100  # Record 100 frames then save

print("Recording dataset... Please move your hands!")

while frame_idx < max_frames:
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

    interactions = interaction.detect(tracked_hands, detections, w, h)
    workspace_events = workspace.analyze(tracked_hands, detections, w, h)

    activity.update(tracked_hands, analytics, workspace_events, w, h)
    activities = activity.recognize(tracked_hands, workspace_events, w, h)

    exporter.record_frame(
        frame_idx, tracked_hands, detections,
        interactions, workspace_events, trajs,
        analytics, activities
    )

    print(f"Frame {frame_idx + 1}/{max_frames} recorded")
    frame_idx += 1

    cv2.imshow("Recording Dataset", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

exporter.save()
cap.release()
cv2.destroyAllWindows()
tracker.close()
print("Done!")
