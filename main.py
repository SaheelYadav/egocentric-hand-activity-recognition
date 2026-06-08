import cv2
import argparse
from src.hand_tracking.hand_tracking import HandTracker
from src.object_detection.object_detector import ObjectDetector
from src.trajectory.trajectory_generator import TrajectoryGenerator
from src.trajectory.motion_analytics import MotionAnalytics
from src.interaction.interaction_detector import InteractionDetector
from src.interaction.workspace_understanding import WorkspaceUnderstanding
from src.activity.activity_recognizer import ActivityRecognizer
from src.utils.dataset_exporter import DatasetExporter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hand Egocentric Activity Intelligence Platform"
    )
    parser.add_argument("--source", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--save", action="store_true",
                        help="Save dataset to output folder")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames to record (0 = unlimited)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Detection confidence threshold")
    parser.add_argument("--depth", action="store_true",
                        help="Enable depth estimation (slow on CPU)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*50)
    print(" Hand Egocentric Activity Intelligence Platform")
    print("="*50)

    # Initialize all modules
    print("\nInitializing modules...")
    tracker = HandTracker(max_hands=2)
    detector = ObjectDetector(model_size="yolo11n.pt",
                              confidence=args.confidence)
    trajectory = TrajectoryGenerator(max_length=30)
    motion = MotionAnalytics(window_size=10)
    interaction = InteractionDetector(distance_threshold=100)
    workspace = WorkspaceUnderstanding(proximity_threshold=150)
    activity = ActivityRecognizer(history_size=20)

    # Depth estimator only if flag is passed
    depth_estimator = None
    if args.depth:
        from src.depth.depth_estimator import DepthEstimator
        depth_estimator = DepthEstimator()
        print("Depth estimator loaded!")

    exporter = None
    if args.save:
        exporter = DatasetExporter(output_dir="output")

    print("All modules loaded!")
    print("\nControls:")
    print("  Q  — Quit")
    print("  S  — Toggle dataset saving")
    print("  R  — Reset trajectories")
    print("  D  — Toggle depth window (if depth enabled)")
    print("="*50)

    cap = cv2.VideoCapture(args.source)
    frame_idx = 0
    saving = args.save
    show_depth = args.depth
    depth_output = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not found!")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Run all modules
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

        # Run depth every 5 frames to reduce lag
        if depth_estimator is not None and frame_idx % 5 == 0:
            depth_output = depth_estimator.estimate(frame)

        # Draw all visualizations
        frame = tracker.draw(frame, tracked_hands)
        frame = detector.draw(frame, detections)
        frame = trajectory.draw(frame)
        frame = interaction.draw(frame, interactions)
        frame = activity.draw(frame, activities)

        # Show frame info
        cv2.putText(frame, f"Frame: {frame_idx}", (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if saving:
            cv2.putText(frame, "RECORDING", (10, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if depth_estimator is not None:
            cv2.putText(frame, "DEPTH ON", (w - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show main window
        cv2.imshow("Hand Egocentric Platform", frame)

        # Show depth window separately
        if show_depth and depth_output is not None:
            cv2.imshow("Depth Map", depth_output["colored"])

        # Save dataset
        if saving and exporter:
            exporter.record_frame(
                frame_idx, tracked_hands, detections,
                interactions, workspace_events, trajs,
                analytics, activities
            )

        # Print activity to terminal
        for act in activities:
            if act["velocity"] > 1:
                print(f"Frame {frame_idx} | "
                      f"{act['hand'].upper()} | "
                      f"Activity: {act['activity']} | "
                      f"Velocity: {act['velocity']}")

        frame_idx += 1

        # Check max frames
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            print(f"Reached max frames: {args.max_frames}")
            break

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            saving = not saving
            if saving and exporter is None:
                exporter = DatasetExporter(output_dir="output")
            print(f"Saving: {saving}")
        elif key == ord('r'):
            trajectory = TrajectoryGenerator(max_length=30)
            print("Trajectories reset!")
        elif key == ord('d'):
            show_depth = not show_depth
            if not show_depth:
                cv2.destroyWindow("Depth Map")
            print(f"Depth window: {show_depth}")

    # Save on exit
    if saving and exporter:
        exporter.save()

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("\nPlatform stopped!")


if __name__ == "__main__":
    main()