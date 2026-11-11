"""직접 구현하는 SLAM. play.py 가 센서 메시지를 시간순으로 넣어 준다.

지금 들어 있는 것은 '오도메트리를 그대로 믿는' 기준선이다.
on_scan 의 TODO 자리에 스캔 매칭(ICP 등)을 넣어서 휘어진 지도를 펴 보자.

받는 메시지 (slamsim/logs.py):
  OdomMsg(t, x, y, theta, v, w)          로봇이 적분한 위치. 점점 틀어진다.
  ScanMsg(t, ranges, valid, odom)        라이다 한 바퀴. valid=False 는 반사 없음.
info: angles(빔 각도, 로봇 기준), max_range, bounds(지도 범위 제안), name
"""
import numpy as np

from slamsim import OccupancyGrid
from slamsim.geometry import compose, relative, scan_to_points, transform_points  # noqa: F401


class MySlam:
    def __init__(self, info):
        self.angles = info["angles"]
        self.use_range = min(info["max_range"], 20.0)   # 너무 먼 값은 지도에 안 씀
        xmin, ymin, xmax, ymax = info["bounds"]
        self.grid = OccupancyGrid(xmin, ymin, xmax, ymax, res=0.1)
        self.pose = None          # 추정 위치 (x, y, theta)
        self.prev_odom = None
        self.prev_points = None   # 이전 스캔 점 (로봇 기준) - 스캔 매칭에 쓰면 된다

    def on_odom(self, msg):
        pass  # 스캔 사이에 더 촘촘한 예측이 필요하면 여기서

    def on_scan(self, msg):
        points = scan_to_points(self.angles, msg.ranges, msg.valid & (msg.ranges < self.use_range))

        if self.pose is None:
            self.pose = msg.odom                      # 시작점은 오도메트리와 같게
        else:
            # 1) 예측: 지난 스캔 이후 오도메트리가 말하는 이동량 (로봇 기준)
            delta = relative(self.prev_odom, msg.odom)

            # 2) TODO 보정: prev_points 와 points 를 맞춰서 delta 를 고친다 (ICP 등)
            # delta = my_icp(self.prev_points, points, initial=delta)

            self.pose = compose(self.pose, delta)

        self.prev_odom = msg.odom
        self.prev_points = points

        # 3) 지도 갱신 (유효한 빔만)
        v = msg.valid
        r = np.minimum(msg.ranges[v], self.use_range)
        self.grid.update(self.pose, self.angles[v], r, msg.ranges[v] < self.use_range, self.use_range)

    # play.py 가 화면·평가에 쓰는 것
    def estimate(self):
        return self.pose

    def map(self):
        return self.grid.prob(), self.grid.extent
