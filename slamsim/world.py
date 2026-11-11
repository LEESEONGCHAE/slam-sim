"""2D 세계(벽 선분)와 레이캐스트 라이다."""
import numpy as np


def box(x0, y0, x1, y1):
    return [(x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)]


class World:
    def __init__(self, segments):
        self.seg = np.asarray(segments, dtype=float)  # (M, 4): x0 y0 x1 y1

    @property
    def bounds(self):
        xs = np.r_[self.seg[:, 0], self.seg[:, 2]]
        ys = np.r_[self.seg[:, 1], self.seg[:, 3]]
        return xs.min(), ys.min(), xs.max(), ys.max()

    @staticmethod
    def default():
        s = []
        s += box(0, 0, 80, 60)          # 바깥 벽
        s += box(15, 15, 30, 25)
        s += box(45, 12, 55, 30)
        s += box(20, 38, 40, 45)
        s += box(60, 38, 70, 47)
        for cx, cy in [(10, 30), (35, 31), (65, 22), (50, 48)]:
            s += box(cx - 1, cy - 1, cx + 1, cy + 1)   # 기둥
        return World(s)

    # 자동 주행용 기본 순환 경로
    LOOP = [(7, 7), (73, 7), (73, 53), (7, 53)]


class Lidar:
    def __init__(self, n_beams=181, fov_deg=270.0, max_range=30.0, sigma=0.03, rng=None):
        fov = np.deg2rad(fov_deg)
        self.angles = np.linspace(-fov / 2, fov / 2, n_beams)
        self.max_range = max_range
        self.sigma = sigma
        self.rng = rng or np.random.default_rng()

    def scan(self, world, pose, noisy=True):
        """반환: ranges (N,), hit (N,) bool. 맞은 것이 없으면 max_range."""
        x, y, th = pose
        ang = self.angles + th
        d = np.stack([np.cos(ang), np.sin(ang)], axis=1)        # (N, 2)
        p = world.seg[:, :2]
        e = world.seg[:, 2:] - p                                  # (M, 2)
        w = p - np.array([x, y])                                  # (M, 2)
        denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[None, :, 0] * e[None, :, 1] - w[None, :, 1] * e[None, :, 0]) / denom
            u = (w[None, :, 0] * d[:, None, 1] - w[None, :, 1] * d[:, None, 0]) / denom
        ok = (np.abs(denom) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
        t = np.where(ok, t, np.inf).min(axis=1)
        hit = t < self.max_range
        r = np.where(hit, t, self.max_range)
        if noisy:
            r = np.where(hit, r + self.rng.normal(0, self.sigma, r.shape), r)
        return r, hit

    def points(self, pose, ranges, hit):
        """맞은 점의 전역 좌표 (K, 2)."""
        x, y, th = pose
        a = self.angles[hit] + th
        r = ranges[hit]
        return np.stack([x + r * np.cos(a), y + r * np.sin(a)], axis=1)
