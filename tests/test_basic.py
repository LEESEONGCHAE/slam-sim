import math
import numpy as np

from slamsim import Sim, Vehicle, World, Lidar


def test_straight_line_accel():
    v = Vehicle()
    for _ in range(int(3 / 0.002)):
        v.step(0, 1, 0.002)
    assert 9 < v.vx < 12.5           # 4 m/s^2 에서 저항을 뺀 값
    assert abs(v.y) < 1e-6 and abs(v.yaw) < 1e-9


def test_steady_turn_has_slip_at_speed():
    v = Vehicle()
    v.vx = 15.0
    for _ in range(int(4 / 0.002)):
        v.step(0.3, 0.15, 0.002)
    assert v.r > 0                    # 왼쪽 조향 → 반시계 회전
    assert abs(v.vy) > 0.05           # 동적 모델에서는 횡속도가 생긴다


def test_lidar_hits_wall_distance():
    w = World([(10, -50, 10, 50)])
    lid = Lidar(n_beams=3, fov_deg=90, sigma=0.0)
    r, hit = lid.scan(w, (0, 0, 0), noisy=False)
    assert hit[1] and math.isclose(r[1], 10.0, abs_tol=1e-9)
    assert math.isclose(r[0], 10 / math.cos(math.radians(45)), rel_tol=1e-9)


def test_autopilot_completes_loop_without_collision():
    sim = Sim(seed=1)
    sim.step(0, 0, 60, auto=True)
    p = np.array(sim.path_true)
    assert p[:, 0].max() > 70 and p[:, 1].max() > 50   # 코너를 다 돌았다
    xmin, ymin, xmax, ymax = sim.world.bounds
    assert (p[:, 0] > xmin + 1).all() and (p[:, 1] > ymin + 1).all()
    assert (p[:, 0] < xmax - 1).all() and (p[:, 1] < ymax - 1).all()
