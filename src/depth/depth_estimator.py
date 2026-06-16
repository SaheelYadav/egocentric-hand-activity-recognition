import cv2
import numpy as np
import torch
from transformers import pipeline


class DepthEstimator:
    def __init__(self):
        print("Loading Depth Anything V2 model...")
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device
        )
        print(f"Depth model loaded on device: {device}!")

    def estimate(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_image = Image.fromarray(rgb)

        result = self.pipe(pil_image)
        depth = np.array(result["depth"])

        # Normalize to 0-255
        depth_normalized = cv2.normalize(
            depth, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)

        return {
            "raw": depth,
            "normalized": depth_normalized,
            "colored": depth_colored
        }

    def get_point_depth(self, depth_raw, x, y):
        h, w = depth_raw.shape
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        return float(depth_raw[y, x])
