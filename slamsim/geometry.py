"""2D 자세(x, y, theta) 계산 도우미. SLAM 구현할 때 자주 쓴다."""
import math
import numpy as np


def wrap(a):
    """각도를 -pi..pi 로."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def compose(a, b):
    """a ⊕ b : a 자세에서 (로봇 기준) b 만큼 움직인 자세."""
    x, y, th = a
    c, s = math.cos(th), math.sin(th)
    return (x + c * b[0] - s * b[1], y + s * b[0] + c * b[1], wrap(th + b[2]))


def relative(a, b):
    """a ⊖ b 의 반대: a 자세에서 본 b 자세 (= a 기준 이동량). compose(a, relative(a, b)) == b"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    c, s = math.cos(a[2]), math.sin(a[2])
    return (c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2]))


def scan_to_points(angles, ranges, valid=None):
    """라이다 스캔 → 로봇 기준 점들 (K, 2)."""
    if valid is not None:
        angles, ranges = angles[valid], ranges[valid]
    return np.stack([ranges * np.cos(angles), ranges * np.sin(angles)], axis=1)


def transform_points(pose, pts):
    """로봇 기준 점들 → pose 기준(전역) 점들."""
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([x, y])
