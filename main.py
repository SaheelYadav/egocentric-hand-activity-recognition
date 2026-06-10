import cv2
import argparse
import os
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
    parser.add_argument("--source", default="0",
                        help="Camera index (0) or video file path")
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

    # Determine source — webcam or video file
    if args.source.isdigit():
        source = int(args.source)
        is_video_file = False
        print(f"\nSource: Webcam (index {source})")
    else:
        source = args.source
        is_video_file = True
        if not os.path.exists(source):
            print(f"Error: Video file not found: {source}")
            return
        print(f"\nSource: Video file — {source}")

    # Initialize all modules
    print("Initializing modules...")
    tracker = HandTracker(max_hands=2)
    detector = ObjectDetector(
        coco_model="yolo11n.pt",
        world_model="yolov8s-worldv2.pt",
        confidence=0.25
    )
    trajectory = TrajectoryGenerator(max_length=30)
    motion = MotionAnalytics(window_size=10)
    interaction = InteractionDetector(distance_threshold=100)
    workspace = WorkspaceUnderstanding(proximity_threshold=150)
    activity = ActivityRecognizer(history_size=20)

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
    print("  SPACE — Pause / Resume (video only)")
    print("="*50)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source!")
        return

    # Get video info
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if is_video_file:
        duration = total_frames / fps if fps > 0 else 0
        print(f"\nVideo info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.1f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.1f} seconds")
        print()

    frame_idx = 0
    saving = args.save
    paused = False
    depth_output = None
    show_depth = args.depth

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    print("\nVideo ended!")
                else:
                    print("Camera not found!")
                break

            # Only flip for webcam not video files
            if not is_video_file:
                frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            # Run all modules
            tracked_hands = tracker.process(frame, mirror_labels=False)
            detections = detector.detect(frame)

            trajectory.update(tracked_hands, detections, w, h)
            trajs = trajectory.get_trajectories()

            motion.update(trajs)
            analytics = motion.get_analytics()

            interactions = interaction.detect(tracked_hands, detections, w, h)
            workspace_events = workspace.analyze(
                tracked_hands, detections, w, h)

            activity.update(tracked_hands, analytics, workspace_events, w, h)
            activities = activity.recognize(
                tracked_hands, workspace_events, w, h)

            # Depth every 5 frames
            if depth_estimator is not None and frame_idx % 5 == 0:
                depth_output = depth_estimator.estimate(frame)

            # Draw visualizations
            frame = tracker.draw(frame, tracked_hands)
            interacting_bboxes = {tuple(inter['bbox']) for inter in interactions}
            interacting_detections = [
                det for det in detections
                if tuple(det['bbox']) in interacting_bboxes
            ]
            frame = detector.draw(frame, interacting_detections)
            frame = trajectory.draw(frame)
            frame = interaction.draw(frame, interactions)
            frame = activity.draw(frame, activities, interactions=workspace_events)

            # Frame info overlay
            cv2.putText(frame, f"Frame: {frame_idx}", (10, h - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Progress bar for video files
            if is_video_file and total_frames > 0:
                progress = int((frame_idx / total_frames) * w)
                cv2.rectangle(frame, (0, h - 8), (w, h),
                              (50, 50, 50), -1)
                cv2.rectangle(frame, (0, h - 8), (progress, h),
                              (0, 255, 100), -1)
                pct = int((frame_idx / total_frames) * 100)
                cv2.putText(frame, f"{pct}%", (10, h - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1)

            if saving:
                cv2.putText(frame, "RECORDING", (10, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)

            if depth_estimator:
                cv2.putText(frame, "DEPTH ON", (w - 120, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255), 2)

            cv2.imshow("Hand Egocentric Platform", frame)

            if show_depth and depth_output is not None:
                cv2.imshow("Depth Map", depth_output["colored"])

            # Save dataset
            if saving and exporter:
                exporter.record_frame(
                    frame_idx, tracked_hands, detections,
                    interactions, workspace_events, trajs,
                    analytics, activities
                )

            # Print to terminal
            for act in activities:
                if act["velocity"] > 1:
                    nat_label = activity.get_natural_label(act["activity"], act["hand"], workspace_events)
                    print(f"Frame {frame_idx} | "
                          f"{act['hand'].upper()} | "
                          f"Activity: {nat_label} | "
                          f"Velocity: {act['velocity']}")

            frame_idx += 1

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
        elif key == ord(' '):
            if is_video_file:
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")

    if saving and exporter:
        exporter.save()

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("\nPlatform stopped!")


if __name__ == "__main__":
    main()