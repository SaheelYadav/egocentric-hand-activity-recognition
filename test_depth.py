import cv2
from src.depth.depth_estimator import DepthEstimator

estimator = DepthEstimator()

cap = cv2.VideoCapture(0)

print("Starting depth estimator... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found!")
        break

    frame = cv2.flip(frame, 1)

    depth_output = estimator.estimate(frame)

    depth_colored = depth_output["colored"]
    depth_raw = depth_output["raw"]

    h, w, _ = frame.shape
    center_depth = estimator.get_point_depth(depth_raw, w//2, h//2)

    print(f"Center depth value: {center_depth:.4f}")

    combined = cv2.hconcat([frame, depth_colored])
    cv2.imshow("RGB | Depth", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")