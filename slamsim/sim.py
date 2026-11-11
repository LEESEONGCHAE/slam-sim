"""시뮬레이션 루프: 차량 + 오도메트리 + 라이다 + 지도 두 장(실제 위치 / 오도메트리 위치)."""
import math
import numpy as np

from .vehicle import Vehicle
from .world import World, Lidar
from .sensors import Odometry
from .mapping import OccupancyGrid


class PurePursuit:
    """경로 추종 자동 주행. 조향이 클수록 속도를 줄인다."""

    def __init__(self, waypoints, lookahead=5.0, v_max=9.0, closed=True, step=0.5):
        pts = list(waypoints) + ([waypoints[0]] if closed else [])
        path = []
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
            for k in range(n):
                path.append((x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n))
        self.path = np.array(path)
        self.lookahead, self.v_max = lookahead, v_max
        self.i = 0

    def control(self, veh):
        n = len(self.path)
        win = (self.i + np.arange(40)) % n
        d = np.hypot(self.path[win, 0] - veh.x, self.path[win, 1] - veh.y)
        self.i = int(win[np.argmin(d)])
        j = (self.i + int(self.lookahead / 0.5)) % n
        tx, ty = self.path[j]
        alpha = math.atan2(ty - veh.y, tx - veh.x) - veh.yaw
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        delta = math.atan2(2 * veh.p.L * math.sin(alpha), self.lookahead)
        steer = delta / veh.p.max_steer
        v_ref = self.v_max / (1.0 + 4.0 * abs(delta))
        accel = max(-1.0, min(1.0, 0.5 * (v_ref - veh.vx)))
        return max(-1.0, min(1.0, steer)), accel


class Sim:
    DT = 0.002          # 동역학 적분 간격 [s]
    SCAN_PERIOD = 0.1   # 라이다 10 Hz

    def __init__(self, world=None, start=(7.0, 7.0, 0.0), seed=0):
        self.rng = np.random.default_rng(seed)
        self.world = world or World.default()
        self.start = start
        self.vehicle = Vehicle(*start)
        self.lidar = Lidar(rng=self.rng)
        self.odom = Odometry(start, rng=self.rng)
        self.autopilot = PurePursuit(World.LOOP)
        xmin, ymin, xmax, ymax = self.world.bounds
        m = 2.0
        self.grid_true = OccupancyGrid(xmin - m, ymin - m, xmax + m, ymax + m)
        self.grid_odom = OccupancyGrid(xmin - m, ymin - m, xmax + m, ymax + m)
        self.reset()

    def reset(self):
        self.vehicle.reset(*self.start)
        self.odom.reset(self.start)
        self.autopilot.i = 0
        self.grid_true.L[:] = 0
        self.grid_odom.L[:] = 0
        self.t = 0.0
        self._next_scan = 0.0
        self.path_true = [self.start[:2]]
        self.path_odom = [self.start[:2]]
        self.scan = None   # (ranges, hit)

    def step(self, steer_cmd, accel_cmd, duration, auto=False):
        n = max(1, int(round(duration / self.DT)))
        for _ in range(n):
            if auto:
                steer_cmd, accel_cmd = self.autopilot.control(self.vehicle)
            self.vehicle.step(steer_cmd, accel_cmd, self.DT)
            v, w = self.odom.measure(self.vehicle)
            self.odom.update(v, w, self.DT)
            self.t += self.DT
            if self.t >= self._next_scan:
                self._next_scan += self.SCAN_PERIOD
                self._do_scan()

    def _do_scan(self):
        r, hit = self.lidar.scan(self.world, self.vehicle.pose)
        self.scan = (r, hit)
        a, R = self.lidar.angles, self.lidar.max_range
        self.grid_true.update(self.vehicle.pose, a, r, hit, R)
        self.grid_odom.update(self.odom.pose, a, r, hit, R)
        self.path_true.append((self.vehicle.x, self.vehicle.y))
        self.path_odom.append((self.odom.x, self.odom.y))

    @property
    def odom_error(self):
        dx = self.odom.x - self.vehicle.x
        dy = self.odom.y - self.vehicle.y
        dth = (self.odom.yaw - self.vehicle.yaw + math.pi) % (2 * math.pi) - math.pi
        return math.hypot(dx, dy), math.degrees(dth)
