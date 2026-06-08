import cv2
from src.hand_tracking.hand_tracking import HandTracker
from src.object_detection.object_detector import ObjectDetector
from src.interaction.interaction_detector import InteractionDetector

# Initialize the three modules
hand_tracker = HandTracker(max_hands=2)
object_detector = ObjectDetector(model_size="yolo11n.pt", confidence=0.5)
interaction_detector = InteractionDetector(distance_threshold=100)

cap = cv2.VideoCapture(0)

print("Starting egocentric interaction platform... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found!")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # 1. Hand tracking
    tracked_hands = hand_tracker.process(frame)

    # 2. Object detection
    detections = object_detector.detect(frame)

    # 3. Interaction detection
    interactions = interaction_detector.detect(tracked_hands, detections, w, h)

    # 4. Drawing and visualization
    frame = hand_tracker.draw(frame, tracked_hands)
    frame = object_detector.draw(frame, detections)
    frame = interaction_detector.draw(frame, interactions)

    # Print active interactions
    for inter in interactions:
        print(f"Interaction: {inter['hand'].upper()} hand is {inter['interaction_type']} {inter['object']} (Dist: {inter['distance']}px)")

    cv2.imshow("Egocentric Interaction Platform", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hand_tracker.close()
print("Done!")
