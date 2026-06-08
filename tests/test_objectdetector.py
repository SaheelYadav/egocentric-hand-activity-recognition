import cv2
from src.object_detection.object_detector import ObjectDetector

detector = ObjectDetector(model_size="yolo11n.pt", confidence=0.5)

cap = cv2.VideoCapture(0)

print("Starting object detector... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found!")
        break

    frame = cv2.flip(frame, 1)

    detections = detector.detect(frame)

    frame = detector.draw(frame, detections)

    for det in detections:
        print(f"Detected: {det['label']} | Confidence: {det['confidence']:.2f} | BBox: {det['bbox']}")

    cv2.imshow("Object Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")
