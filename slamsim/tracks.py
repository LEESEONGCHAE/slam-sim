"""닫힌 트랙(중심선) 만들기와 경로 오차 계산.

트랙은 극좌표 r(θ) = R0 (1 + Σ a_k sin(kθ + φ_k)) 로 만든다. seed 마다 모양이 다르다.
학습용 seed 와 시험용 seed 를 나눠서, RL 이 '처음 보는 트랙'에서도 되는지 본다.
"""
import math
import numpy as np


class Track:
    def __init__(self, xy, width=4.0, name="track"):
        xy = np.asarray(xy, float)
        # 0.5 m 간격으로 다시 찍기
        seg = np.diff(np.vstack([xy, xy[:1]]), axis=0)
        s = np.r_[0, np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))]
        self.length = s[-1]
        n = int(self.length / 0.5)
        ss = np.linspace(0, self.length, n, endpoint=False)
        closed = np.vstack([xy, xy[:1]])
        self.xy = np.c_[np.interp(ss, s, closed[:, 0]), np.interp(ss, s, closed[:, 1])]
        self.s = ss
        self.n = n
        self.ds = self.length / n
        d = np.roll(self.xy, -1, 0) - np.roll(self.xy, 1, 0)
        self.heading = np.arctan2(d[:, 1], d[:, 0])
        dh = (np.roll(self.heading, -1) - np.roll(self.heading, 1) + np.pi) % (2 * np.pi) - np.pi
        k = dh / (2 * self.ds)                                  # 곡률 [1/m], 왼쪽 회전 +
        self.kappa = np.convolve(np.r_[k[-3:], k, k[:3]], np.ones(7) / 7, "valid")
        self.width = width        # 이 폭(반폭 = width/2)을 벗어나면 이탈
        self.name = name

    @staticmethod
    def random(seed, R0=60.0, width=4.0):
        rng = np.random.default_rng(seed)
        th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        r = np.ones_like(th)
        for k in (2, 3, 4):
            r += rng.uniform(0.0, 0.22 / k * 2) * np.sin(k * th + rng.uniform(0, 2 * np.pi))
        xy = np.c_[R0 * r * np.cos(th), R0 * r * np.sin(th)]
        return Track(xy, width, name=f"random{seed}")

    @staticmethod
    def oval(a=80.0, b=40.0, width=4.0):
        th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
        return Track(np.c_[a * np.cos(th), b * np.sin(th)], width, name="oval")

    def start_pose(self):
        return (self.xy[0, 0], self.xy[0, 1], self.heading[0])

    def locate(self, x, y, hint=None, window=60):
        """가장 가까운 중심선 점 index. hint 주변만 찾는다(빠름)."""
        if hint is None:
            idx = np.arange(self.n)
        else:
            idx = (hint + np.arange(-window // 4, window)) % self.n
        d = (self.xy[idx, 0] - x) ** 2 + (self.xy[idx, 1] - y) ** 2
        return int(idx[np.argmin(d)])

    def errors(self, i, x, y, yaw):
        """(횡 오차 e: 왼쪽 +, 방향 오차 epsi) at index i."""
        h = self.heading[i]
        dx, dy = x - self.xy[i, 0], y - self.xy[i, 1]
        e = -math.sin(h) * dx + math.cos(h) * dy
        epsi = (yaw - h + math.pi) % (2 * math.pi) - math.pi
        return e, epsi

    def preview(self, i, distances):
        """앞쪽 거리별 곡률."""
        return np.array([self.kappa[(i + int(d / self.ds)) % self.n] for d in distances])
