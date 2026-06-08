import cv2
from src.hand_tracking.hand_tracking import HandTracker

tracker = HandTracker(max_hands=2)

cap = cv2.VideoCapture(0)

print("Starting hand tracker... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found!")
        break

    frame = cv2.flip(frame, 1)

    tracked_hands = tracker.process(frame)

    frame = tracker.draw(frame, tracked_hands)

    # Show keypoint count
    for label, data in tracked_hands.items():
        if data is not None:
            print(f"{label.upper()} hand detected - {len(data['keypoints'])} keypoints")

    cv2.imshow("Hand Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
tracker.close()
print("Done!")
