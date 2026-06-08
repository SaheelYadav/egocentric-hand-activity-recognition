import numpy as np
from collections import deque


class MotionAnalytics:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = {
            "left": deque(maxlen=window_size),
            "right": deque(maxlen=window_size)
        }

    def update(self, trajectories):
        for label in ["left", "right"]:
            pts = [p for p in trajectories[label] if p is not None]
            if pts:
                self.history[label].append(pts[-1])

    def compute_velocity(self, points):
        if len(points) < 2:
            return 0.0
        velocities = []
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            v = np.sqrt(dx**2 + dy**2)
            velocities.append(v)
        return float(np.mean(velocities))

    def compute_acceleration(self, points):
        if len(points) < 3:
            return 0.0
        velocities = []
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            velocities.append(np.sqrt(dx**2 + dy**2))
        accels = []
        for i in range(1, len(velocities)):
            accels.append(abs(velocities[i] - velocities[i-1]))
        return float(np.mean(accels))

    def compute_pseudo_imu(self, points):
        if len(points) < 3:
            return {
                "accel_x": 0.0,
                "accel_y": 0.0,
                "gyro_z": 0.0
            }

        # Acceleration in x and y
        dx1 = points[-1][0] - points[-2][0]
        dy1 = points[-1][1] - points[-2][1]
        dx2 = points[-2][0] - points[-3][0]
        dy2 = points[-2][1] - points[-3][1]

        accel_x = dx1 - dx2
        accel_y = dy1 - dy2

        # Gyro z — angular change
        angle1 = np.arctan2(dy1, dx1)
        angle2 = np.arctan2(dy2, dx2)
        gyro_z = float(angle1 - angle2)

        return {
            "accel_x": round(float(accel_x), 4),
            "accel_y": round(float(accel_y), 4),
            "gyro_z": round(gyro_z, 4)
        }

    def get_analytics(self):
        analytics = {}

        for label in ["left", "right"]:
            points = list(self.history[label])

            velocity = self.compute_velocity(points)
            acceleration = self.compute_acceleration(points)
            pseudo_imu = self.compute_pseudo_imu(points)

            analytics[label] = {
                "velocity": round(velocity, 4),
                "acceleration": round(acceleration, 4),
                "pseudo_imu": pseudo_imu
            }

        return analytics
