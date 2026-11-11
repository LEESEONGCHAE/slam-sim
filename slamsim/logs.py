"""센서 로그: 시뮬레이터와 실제 데이터(Intel Research Lab, CARMEN 형식)를 같은 형식으로 내보낸다.

SLAM 알고리즘이 받는 것은 두 가지 메시지뿐이다.
  OdomMsg : 로봇이 바퀴(+자이로)로 적분한 위치. 실제 로봇의 /odom 과 같다. 점점 틀어진다.
  ScanMsg : 2D 라이다 한 바퀴. 각도별 거리 + 유효 여부 + 그 순간의 오도메트리 위치.
정답 위치(truth)는 시뮬레이터에만 있고, 알고리즘에는 주지 않는다(평가용).
"""
import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class OdomMsg:
    t: float
    x: float
    y: float
    theta: float
    v: float = float("nan")    # 전진 속도 [m/s] (있으면)
    w: float = float("nan")    # 요레이트 [rad/s] (있으면)

    @property
    def pose(self):
        return (self.x, self.y, self.theta)


@dataclass
class ScanMsg:
    t: float
    ranges: np.ndarray          # (N,) [m]
    valid: np.ndarray           # (N,) bool. False = 반사 없음/측정 실패
    odom: tuple                 # 이 스캔 시점의 오도메트리 위치 (x, y, theta)


@dataclass
class SensorLog:
    name: str
    angles: np.ndarray          # 라이다 빔 각도 (로봇 기준, 왼쪽 +) [rad]
    max_range: float
    messages: list              # OdomMsg / ScanMsg 시간순
    truth: np.ndarray = None    # (K, 4): t, x, y, theta  (시뮬레이터만)
    bounds: tuple = None        # 지도 범위 제안 (xmin, ymin, xmax, ymax)
    notes: dict = field(default_factory=dict)

    @property
    def scans(self):
        return [m for m in self.messages if isinstance(m, ScanMsg)]

    @property
    def duration(self):
        return self.messages[-1].t - self.messages[0].t if self.messages else 0.0

    def info(self):
        """알고리즘에 넘기는 정보 (정답은 빠져 있다)."""
        return {"name": self.name, "angles": self.angles, "max_range": self.max_range,
                "bounds": self.bounds}

    def truth_at(self, t):
        """시간 t의 정답 위치 (선형 보간). 없으면 None."""
        if self.truth is None:
            return None
        T = self.truth
        i = int(np.clip(np.searchsorted(T[:, 0], t), 1, len(T) - 1))
        t0, t1 = T[i - 1, 0], T[i, 0]
        a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        x = T[i - 1, 1] + a * (T[i, 1] - T[i - 1, 1])
        y = T[i - 1, 2] + a * (T[i, 2] - T[i - 1, 2])
        dth = (T[i, 3] - T[i - 1, 3] + math.pi) % (2 * math.pi) - math.pi
        return (x, y, T[i - 1, 3] + a * dth)

    # ---- 저장/불러오기 (npz) ----
    def save(self, path):
        od = np.array([(m.t, m.x, m.y, m.theta, m.v, m.w) for m in self.messages if isinstance(m, OdomMsg)])
        sc = [m for m in self.messages if isinstance(m, ScanMsg)]
        np.savez_compressed(
            path, name=self.name, angles=self.angles, max_range=self.max_range,
            odom=od, scan_t=np.array([m.t for m in sc]), scan_r=np.array([m.ranges for m in sc]),
            scan_v=np.array([m.valid for m in sc]), scan_odom=np.array([m.odom for m in sc]),
            truth=self.truth if self.truth is not None else np.zeros((0, 4)),
            bounds=np.array(self.bounds if self.bounds else [np.nan] * 4))

    @staticmethod
    def load(path):
        z = np.load(path)
        msgs = [OdomMsg(*row) for row in z["odom"]]
        msgs += [ScanMsg(t, r, v, tuple(o)) for t, r, v, o in
                 zip(z["scan_t"], z["scan_r"], z["scan_v"], z["scan_odom"])]
        msgs.sort(key=lambda m: m.t)
        b = z["bounds"]
        return SensorLog(str(z["name"]), z["angles"], float(z["max_range"]), msgs,
                         z["truth"] if len(z["truth"]) else None,
                         None if np.isnan(b).any() else tuple(b))


def _bounds_from(poses, margin):
    p = np.asarray(poses)
    return (p[:, 0].min() - margin, p[:, 1].min() - margin, p[:, 0].max() + margin, p[:, 1].max() + margin)


# ---------------------------------------------------------------- 실제 데이터
def load_carmen(path, name=None, invalid_from=80.0):
    """CARMEN 로그(.clf). Intel Research Lab 데이터셋이 이 형식이다.

    FLASER n r1..rn x y theta odom_x odom_y odom_theta ipc_t host logger_t
    ODOM x y theta tv rv accel ipc_t host logger_t
    라이다: 정면 기준 -90°..+89°, 1° 간격 (180빔). invalid_from 이상은 반사 없음으로 본다.
    """
    msgs, n_beams = [], None
    t0 = None
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == "ODOM":
                t = float(p[7])
                t0 = t if t0 is None else t0
                msgs.append(OdomMsg(t - t0, float(p[1]), float(p[2]), float(p[3]), float(p[4]), float(p[5])))
            elif p[0] == "FLASER":
                n = int(p[1])
                n_beams = n
                r = np.array(p[2:2 + n], dtype=float)
                ox, oy, oth = (float(v) for v in p[2 + n + 3:2 + n + 6])
                t = float(p[2 + n + 6])
                t0 = t if t0 is None else t0
                msgs.append(ScanMsg(t - t0, r, r < invalid_from, (ox, oy, oth)))
    msgs.sort(key=lambda m: m.t)
    angles = np.deg2rad(-90.0 + np.arange(n_beams))
    poses = [m.odom for m in msgs if isinstance(m, ScanMsg)]
    return SensorLog(name or path, angles, invalid_from, msgs, None, _bounds_from(poses, 15.0),
                     {"source": "CARMEN", "beam_angles": "assumed -90..+89 deg, 1 deg step"})


# ---------------------------------------------------------------- 시뮬레이터
class SimRig:
    """시뮬레이터 차량에 센서를 단 것. 운전 입력을 넣으면 센서 메시지가 나온다.

    realistic=True 이면 실제 라이다에서 흔한 현상을 넣는다:
      - 빔 2%: 반사 없음(valid=False)
      - 빔 0.5%: 엉뚱한 짧은 값(먼지·반사 노이즈)
      - 모션 왜곡: 한 바퀴(0.1 s) 도는 동안 차가 움직여서 빔마다 쏜 위치가 다르다
    """

    def __init__(self, seed=0, odom_hz=50.0, realistic=True):
        from .sim import Sim
        self.sim = Sim(seed=seed)
        self.seed, self.odom_hz, self.realistic = seed, odom_hz, realistic
        self.odom_every = max(1, int(round(1.0 / odom_hz / self.sim.DT)))
        self.scan_every = int(round(self.sim.SCAN_PERIOD / self.sim.DT))
        self.reset()

    def reset(self):
        self.sim.reset()
        self.k = 0
        self.messages, self.truth = [], []
        self.scan_truth = {}      # 스캔 시각 → 정답 위치 (평가용)
        self._scan_start = self.sim.vehicle.pose

    @property
    def vehicle(self):
        return self.sim.vehicle

    def step(self, steer, accel, duration, auto=False):
        """duration 초만큼 진행하고, 그동안 나온 새 메시지 목록을 돌려준다."""
        sim, lid, rng = self.sim, self.sim.lidar, self.sim.rng
        new = []
        for _ in range(max(1, int(round(duration / sim.DT)))):
            self.k += 1
            if auto and self.k % 10 == 1:
                steer, accel = sim.autopilot.control(sim.vehicle)
            sim.vehicle.step(steer, accel, sim.DT)
            v, w = sim.odom.measure(sim.vehicle)
            sim.odom.update(v, w, sim.DT)
            sim.t += sim.DT
            if self.k % self.odom_every == 0:
                new.append(OdomMsg(sim.t, *sim.odom.pose, v, w))
                self.truth.append((sim.t, *sim.vehicle.pose))
            if self.k % self.scan_every == 0:
                end = sim.vehicle.pose
                start = self._scan_start if self.realistic else end
                r, hit = lid.scan(sim.world, end, noisy=True, pose_start=start)
                valid = hit.copy()
                if self.realistic:
                    valid &= ~(rng.random(r.shape) < 0.02)
                    spur = rng.random(r.shape) < 0.005
                    r = np.where(spur, rng.uniform(0.2, r), r)
                    valid |= spur
                r = np.where(valid, r, lid.max_range)
                new.append(ScanMsg(sim.t, r, valid, sim.odom.pose))
                self.truth.append((sim.t, *end))
                self.scan_truth[sim.t] = end
                self._scan_start = end
        self.messages += new
        return new

    def truth_pose(self):
        return self.sim.vehicle.pose

    def info(self):
        b = self.sim.world.bounds
        return {"name": f"sim(seed={self.seed})", "angles": self.sim.lidar.angles.copy(),
                "max_range": self.sim.lidar.max_range, "bounds": (b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2)}

    def to_log(self, name=None):
        b = self.sim.world.bounds
        return SensorLog(name or f"sim(seed={self.seed})", self.sim.lidar.angles.copy(),
                         self.sim.lidar.max_range, list(self.messages),
                         np.array(self.truth) if self.truth else None,
                         (b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2),
                         {"source": "simulator", "realistic": self.realistic})


def record_sim(seconds=180.0, seed=0, odom_hz=50.0, realistic=True):
    """시뮬레이터를 자동 주행으로 돌려 센서 로그를 만든다."""
    rig = SimRig(seed, odom_hz, realistic)
    rig.step(0, 0, seconds, auto=True)
    return rig.to_log()
