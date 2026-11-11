"""휠 속도 + 자이로로 하는 추측 항법(오도메트리). 바이어스·스케일 오차로 점점 틀어진다."""
import math
import numpy as np


class Odometry:
    def __init__(self, pose0, speed_scale_err=0.01, speed_sigma=0.05,
                 gyro_bias=0.003, gyro_sigma=0.01, rng=None):
        self.rng = rng or np.random.default_rng()
        self.scale = 1.0 + speed_scale_err   # 타이어 반경 오차
        self.speed_sigma = speed_sigma        # [m/s]
        self.gyro_bias = gyro_bias            # [rad/s]
        self.gyro_sigma = gyro_sigma          # [rad/s]
        self.reset(pose0)

    def reset(self, pose0):
        self.x, self.y, self.yaw = pose0

    @property
    def pose(self):
        return (self.x, self.y, self.yaw)

    def measure(self, vehicle):
        """실제 차량에서 센서값을 만든다. 휠 속도는 횡 미끄러짐(vy)을 모른다."""
        v = vehicle.vx * self.scale + self.rng.normal(0, self.speed_sigma)
        w = vehicle.r + self.gyro_bias + self.rng.normal(0, self.gyro_sigma)
        if abs(vehicle.vx) < 1e-3:
            v = 0.0  # 정지 시 휠 속도는 정확히 0
        return v, w

    def update(self, v, w, dt):
        th = self.yaw + 0.5 * w * dt
        self.x += v * math.cos(th) * dt
        self.y += v * math.sin(th) * dt
        self.yaw = (self.yaw + w * dt + math.pi) % (2 * math.pi) - math.pi
