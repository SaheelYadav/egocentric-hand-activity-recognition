import cv2
from src.hand_tracking.hand_tracking import HandTracker
from src.object_detection.object_detector import ObjectDetector
from src.interaction.workspace_understanding import WorkspaceUnderstanding

tracker = HandTracker(max_hands=2)
detector = ObjectDetector(model_size="yolo11n.pt", confidence=0.5)
workspace = WorkspaceUnderstanding(proximity_threshold=150)

cap = cv2.VideoCapture(0)

print("Starting workspace understanding... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found!")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    tracked_hands = tracker.process(frame)
    detections = detector.detect(frame)
    events = workspace.analyze(tracked_hands, detections, w, h)

    frame = tracker.draw(frame, tracked_hands)
    frame = workspace.draw(frame, events)

    for event in events:
        if event["relation"] != "far":
            print(f"{event['hand'].upper()} hand is {event['relation']} "
                  f"{event['object']} | Distance: {event['distance']}px | "
                  f"On Surface: {event['on_surface']}")

    cv2.imshow("Workspace Understanding", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
tracker.close()
print("Done!")
