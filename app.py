from flask import Flask, render_template, Response, request, json, jsonify, send_from_directory
import cv2
import threading
import time
import os
import base64
import numpy as np
from src.hand_tracking.hand_tracking import HandTracker
from src.object_detection.object_detector import ObjectDetector
from src.trajectory.trajectory_generator import TrajectoryGenerator
from src.trajectory.motion_analytics import MotionAnalytics
from src.interaction.interaction_detector import InteractionDetector
from src.interaction.workspace_understanding import WorkspaceUnderstanding
from src.activity.activity_recognizer import ActivityRecognizer

class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            ret, frame = self.stream.read()
            if ret:
                with self.read_lock:
                    self.ret = ret
                    self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.read_lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join()
        self.stream.release()


app = Flask(__name__)

# Global state
tracker = HandTracker(max_hands=2)
detector = ObjectDetector(
    coco_model="yolo11x.pt",
    world_model="yolov8x-worldv2.pt",
    confidence=0.25,
    iou_threshold=0.45
)
trajectory = TrajectoryGenerator(max_length=30)
motion = MotionAnalytics(window_size=10)
interaction = InteractionDetector(distance_threshold=100)
workspace = WorkspaceUnderstanding(proximity_threshold=150)
activity = ActivityRecognizer(history_size=20)

cap = None
webcam_stream = None
is_running = False
current_frame = None
current_stats = {}
lock = threading.Lock()


def reset_modules():
    global tracker, trajectory, motion, interaction, workspace, activity
    tracker = HandTracker(max_hands=2)
    trajectory = TrajectoryGenerator(max_length=30)
    motion = MotionAnalytics(window_size=10)
    interaction = InteractionDetector(distance_threshold=100)
    workspace = WorkspaceUnderstanding(proximity_threshold=150)
    activity = ActivityRecognizer(history_size=20)


def process_frame(frame, is_webcam=True):
    global current_stats

    if is_webcam:
        frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    tracked_hands = tracker.process(frame, mirror_labels=False)
    detections = detector.detect(frame)

    trajectory.update(tracked_hands, detections, w, h)
    trajs = trajectory.get_trajectories()

    motion.update(trajs)
    analytics = motion.get_analytics()

    interactions = interaction.detect(tracked_hands, detections, w, h)
    workspace_events = workspace.analyze(tracked_hands, detections, w, h)

    activity.update(tracked_hands, analytics, workspace_events, w, h)
    activities = activity.recognize(tracked_hands, workspace_events, w, h)

    frame = tracker.draw(frame, tracked_hands)
    frame = activity.draw(frame, activities, interactions=workspace_events)

    # Build stats
    stats = {
        "hands": [],
        "objects": [],
        "interactions": [],
        "activities": [],
        "summary": activity.get_scene_summary(activities, workspace_events)
    }

    for label, data in tracked_hands.items():
        if data is not None:
            stats["hands"].append(label.upper())

    IGNORE_LABELS = ['person', 'nail', 'scissors', 'bed', 'couch', 'chair', 'dining table', 'floor']
    for det in detections:
        if det['label'] not in IGNORE_LABELS:
            stats["objects"].append(
                f"{det['label']} ({det['confidence']:.0%})"
            )

    for inter in interactions:
        stats["interactions"].append(
            f"{inter['hand'].upper()} {inter['interaction_type']} {inter['object']}"
        )

    for act in activities:
        if act["velocity"] > 0.5:
            nat_label = activity.get_natural_label(act["activity"], act["hand"], workspace_events)
            stats["activities"].append(
                f"{act['hand'].upper()}: {nat_label} "
                f"(v={act['velocity']})"
            )

    for label in ["left", "right"]:
        if label in analytics:
            stats[f"{label}_velocity"] = analytics[label]["velocity"]
            stats[f"{label}_imu"] = analytics[label]["pseudo_imu"]

    current_stats = stats
    return frame


def generate_webcam():
    global is_running, current_frame, webcam_stream
    webcam_stream = WebcamVideoStream(src=0).start()
    is_running = True

    while is_running:
        ret, frame = webcam_stream.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame = process_frame(frame, is_webcam=True)

        ret2, buffer = cv2.imencode('.jpg', frame, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        with lock:
            current_frame = frame_bytes

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               frame_bytes + b'\r\n')

    if webcam_stream:
        webcam_stream.stop()
        webcam_stream = None


def generate_video(video_path):
    global cap, is_running, current_frame
    cap = cv2.VideoCapture(video_path)
    is_running = True
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while is_running:
        ret, frame = cap.read()
        if not ret:
            is_running = False
            break

        frame = process_frame(frame, is_webcam=False)

        progress = int((frame_idx / total_frames) * 100) \
            if total_frames > 0 else 0
        h, w, _ = frame.shape
        bar_w = int((frame_idx / total_frames) * w) if total_frames > 0 else 0
        cv2.rectangle(frame, (0, h-8), (w, h), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, h-8), (bar_w, h), (0, 220, 100), -1)
        cv2.putText(frame, f"Frame {frame_idx}/{total_frames} ({progress}%)",
                    (10, h-15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)

        ret2, buffer = cv2.imencode('.jpg', frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        with lock:
            current_frame = frame_bytes

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + 
               frame_bytes + b'\r\n')

        # Yield execution to respect video FPS
        time.sleep(1.0 / fps if fps > 0 else 0.03)

        frame_idx += 1

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/webcam_feed')
def webcam_feed():
    reset_modules()
    return Response(generate_webcam(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed', methods=['POST'])
def video_feed():
    reset_modules()
    video_path = request.form.get('video_path', '')
    if not os.path.exists(video_path):
        return "Video file not found", 404
    return Response(generate_video(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    os.makedirs('data/uploads', exist_ok=True)
    save_path = os.path.join('data/uploads', file.filename)
    file.save(save_path)

    return jsonify({
        "success": True,
        "path": save_path,
        "filename": file.filename
    })


@app.route('/video_stream')
def video_stream():
    video_path = request.args.get('path', '')
    reset_modules()
    return Response(generate_video(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory('data/uploads', filename)


@app.route('/process_single_frame', methods=['POST'])
def process_single_frame():
    # Receive image file
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image sent"}), 400
        
    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Process the frame
    processed_frame = process_frame(frame, is_webcam=False)
    
    # Encode back to JPEG
    _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    encoded_img = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        "image": f"data:image/jpeg;base64,{encoded_img}",
        "stats": current_stats
    })


@app.route('/stats')
def stats():
    return jsonify(current_stats)


@app.route('/stop')
def stop():
    global is_running, cap, webcam_stream
    is_running = False
    if cap:
        cap.release()
        cap = None
    if webcam_stream:
        webcam_stream.stop()
        webcam_stream = None
    return jsonify({"status": "stopped"})


if __name__ == '__main__':
    os.makedirs('data/uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
