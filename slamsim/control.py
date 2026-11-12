"""고전 제어: 조향 PID(+ 곡률 피드포워드)와 공통 속도 규칙.

PID 조향 (부호: e 는 경로 왼쪽이 +, 조향은 왼쪽이 +)
  δ = −(Kp·e + Ki·∫e + Kd·de/dt) − Kψ·eψ + δ_ff
  δ_ff = atan(L·κ)       경로 곡률만큼 미리 꺾어 두는 값 (기구학 자전거 모델, 2장)
피드포워드를 끄면 순수 PID 가 된다.
"""
import math

import numpy as np


def speed_command(track, i, vx, v_max=15.0, a_lat=5.0, horizon=(0, 5, 10, 15, 20, 30)):
    """앞 30 m 안의 가장 급한 커브에서 횡가속도가 a_lat 을 넘지 않게 목표 속도를 정한다.
    PID 와 RL 이 똑같이 쓴다 (조향만 비교하려고)."""
    k = np.abs(track.preview(i, horizon)).max()
    v_ref = min(v_max, math.sqrt(a_lat / max(k, 1e-4)))
    return max(-1.0, min(1.0, 0.6 * (v_ref - vx))), v_ref


class PIDSteer:
    def __init__(self, kp=0.25, ki=0.01, kd=0.08, k_psi=1.0, feedforward=True, dt=0.05):
        self.kp, self.ki, self.kd, self.k_psi = kp, ki, kd, k_psi
        self.feedforward, self.dt = feedforward, dt
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_e = None

    def __call__(self, e, epsi, kappa, L, max_steer):
        self.integral = max(-5.0, min(5.0, self.integral + e * self.dt))   # 적분 와인드업 제한
        de = 0.0 if self.prev_e is None else (e - self.prev_e) / self.dt
        self.prev_e = e
        delta = -(self.kp * e + self.ki * self.integral + self.kd * de) - self.k_psi * epsi
        if self.feedforward:
            delta += math.atan(L * kappa)
        return max(-1.0, min(1.0, delta / max_steer))     # 차량 입력 −1..1 로
