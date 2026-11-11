"""로그 오즈 점유 격자 지도."""
import numpy as np


class OccupancyGrid:
    def __init__(self, xmin, ymin, xmax, ymax, res=0.2,
                 l_occ=0.85, l_free=-0.4, l_min=-4.0, l_max=4.0):
        self.xmin, self.ymin, self.res = xmin, ymin, res
        self.nx = int(np.ceil((xmax - xmin) / res))
        self.ny = int(np.ceil((ymax - ymin) / res))
        self.extent = (xmin, xmin + self.nx * res, ymin, ymin + self.ny * res)
        self.l_occ, self.l_free, self.l_min, self.l_max = l_occ, l_free, l_min, l_max
        self.L = np.zeros((self.ny, self.nx))

    def _index(self, px, py):
        ix = np.floor((px - self.xmin) / self.res).astype(int)
        iy = np.floor((py - self.ymin) / self.res).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return np.unique(iy[ok] * self.nx + ix[ok])

    def update(self, pose, angles, ranges, hit, max_range):
        x, y, th = pose
        a = angles + th
        c, s = np.cos(a), np.sin(a)
        steps = np.arange(0.0, max_range, self.res * 0.5)
        free_mask = steps[None, :] < (ranges[:, None] - self.res)
        px = (x + c[:, None] * steps[None, :])[free_mask]
        py = (y + s[:, None] * steps[None, :])[free_mask]
        free = self._index(px, py)
        occ = self._index(x + c[hit] * ranges[hit], y + s[hit] * ranges[hit])
        free = np.setdiff1d(free, occ, assume_unique=True)
        flat = self.L.reshape(-1)
        flat[free] += self.l_free
        flat[occ] += self.l_occ
        np.clip(self.L, self.l_min, self.l_max, out=self.L)

    def prob(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.L))
