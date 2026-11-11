"""차량 동역학: 저속은 기구학 자전거 모델, 그 이상은 선형 타이어 + 마찰 한계의 동적 자전거 모델."""
import math
from dataclasses import dataclass


@dataclass
class VehicleParams:
    m: float = 1500.0          # 질량 [kg]
    Iz: float = 2250.0         # 요 관성 모멘트 [kg m^2]
    lf: float = 1.2            # 무게중심 → 앞축 [m]
    lr: float = 1.4            # 무게중심 → 뒤축 [m]
    Cf: float = 80000.0        # 앞축 코너링 강성 [N/rad]
    Cr: float = 90000.0        # 뒤축 코너링 강성 [N/rad]
    mu: float = 0.9            # 노면 마찰 계수
    max_steer: float = math.radians(35)
    steer_rate: float = math.radians(70)   # 조향 속도 한계 [rad/s]
    max_accel: float = 4.0     # [m/s^2]
    max_brake: float = 8.0
    max_reverse: float = 3.0   # 후진 최고 속도 [m/s]
    c_roll: float = 0.15       # 구름 저항 [m/s^2]
    c_aero: float = 0.0006     # 공기 저항 계수 [1/m]
    kin_speed: float = 1.5     # 이 속도 미만은 기구학 모델
    length: float = 4.5
    width: float = 1.85

    @property
    def L(self):
        return self.lf + self.lr


class Vehicle:
    """상태: x, y, yaw(전역), vx, vy(차체 좌표), r(요레이트), steer(실제 조향각)."""

    G = 9.81

    def __init__(self, x=0.0, y=0.0, yaw=0.0, params=None):
        self.p = params or VehicleParams()
        self.reset(x, y, yaw)

    def reset(self, x, y, yaw):
        self.x, self.y, self.yaw = x, y, yaw
        self.vx = self.vy = self.r = 0.0
        self.steer = 0.0
        self.ax = 0.0

    @property
    def pose(self):
        return (self.x, self.y, self.yaw)

    @property
    def slip_angle(self):
        """차체 슬립각 beta [rad]."""
        return math.atan2(self.vy, abs(self.vx)) if abs(self.vx) > 0.1 else 0.0

    def _long_accel(self, cmd):
        p = self.p
        if cmd >= 0:
            a = cmd * p.max_accel
        elif self.vx > 0.1:
            a = cmd * p.max_brake           # 브레이크
        else:
            a = cmd * 2.0                    # 후진
            if self.vx <= -p.max_reverse:
                a = 0.0
        # 저항
        v = self.vx
        drag = p.c_aero * v * abs(v) + (p.c_roll * math.copysign(1, v) if abs(v) > 0.05 else 0.0)
        a -= drag
        # 정지 근처에서 브레이크가 후진으로 넘어가지 않게
        if cmd < 0 and self.vx > 0 and self.vx + a * 0.002 < 0:
            a = -self.vx / 0.002
        return a

    def step(self, steer_cmd, accel_cmd, dt):
        """steer_cmd: -1..1 (왼쪽 +), accel_cmd: -1..1. dt는 2ms 정도로 작게."""
        p = self.p
        target = max(-1.0, min(1.0, steer_cmd)) * p.max_steer
        ds = max(-p.steer_rate * dt, min(p.steer_rate * dt, target - self.steer))
        self.steer += ds
        d = self.steer
        a = self._long_accel(max(-1.0, min(1.0, accel_cmd)))
        self.ax = a

        if abs(self.vx) < p.kin_speed:
            # 기구학 모델 (저속, 후진 포함)
            self.vx += a * dt
            self.r = self.vx * math.tan(d) / p.L
            self.vy = self.r * p.lr
        else:
            vx, vy, r = self.vx, self.vy, self.r
            af = d - math.atan2(vy + p.lf * r, vx)
            ar = -math.atan2(vy - p.lr * r, vx)
            Fzf = p.m * self.G * p.lr / p.L
            Fzr = p.m * self.G * p.lf / p.L
            Fyf = max(-p.mu * Fzf, min(p.mu * Fzf, p.Cf * af))
            Fyr = max(-p.mu * Fzr, min(p.mu * Fzr, p.Cr * ar))
            dvx = a - Fyf * math.sin(d) / p.m + vy * r
            dvy = (Fyf * math.cos(d) + Fyr) / p.m - vx * r
            dr = (p.lf * Fyf * math.cos(d) - p.lr * Fyr) / p.Iz
            self.vx += dvx * dt
            self.vy += dvy * dt
            self.r += dr * dt

        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (self.vx * c - self.vy * s) * dt
        self.y += (self.vx * s + self.vy * c) * dt
        self.yaw = (self.yaw + self.r * dt + math.pi) % (2 * math.pi) - math.pi

    def outline(self, pose=None):
        """차체 사각형 꼭짓점 (5개, 닫힌 다각형)."""
        x, y, th = pose if pose is not None else self.pose
        p = self.p
        rear, front, hw = -p.lr - 0.9, p.lf + 0.9, p.width / 2
        pts = [(rear, -hw), (front, -hw), (front, hw), (rear, hw), (rear, -hw)]
        c, s = math.cos(th), math.sin(th)
        return [(x + c * u - s * v, y + s * u + c * v) for u, v in pts]
