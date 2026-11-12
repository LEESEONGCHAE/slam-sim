"""직접 짜 보는 조향 제어기. rl/compare.py 가 이 파일을 찾으면 '내 제어기'로 비교표에 넣는다.

policy(obs, env) 가 매 0.05 초마다 불린다. 조향 명령 −1..1 을 돌려주면 된다 (1 = 왼쪽 35°).

쓸 수 있는 값
  e, epsi = env.meas            # 횡 오차 [m] (경로 왼쪽 +), 방향 오차 [rad]. 노이즈 조건이면 노이즈 포함
  kappa = env.track.kappa[env.i]                  # 지금 위치 경로 곡률 [1/m] (왼쪽 회전 +)
  ahead = env.track.preview(env.i, (5, 10, 20))   # 앞쪽 거리별 곡률
  v = env.veh                   # 차량: v.vx, v.vy, v.r(요레이트), v.steer, v.p.L(축거), v.p.max_steer
  env.t                         # 에피소드 시간 [s] (0 이면 새 에피소드 시작)
  obs                           # RL 이 보는 12개 값 (envs.py 설명 참고)

지금 들어 있는 것: P 제어 + 방향 오차 항 + 곡률 피드포워드. 게인을 바꾸거나 새로 짜 보자.
  python rl/compare.py         → rl/results.md 에 '내 제어기' 줄이 생긴다
"""
import math

KP = 0.3        # 횡 오차 게인 [rad/m]
K_PSI = 1.0     # 방향 오차 게인


def policy(obs, env):
    e, epsi = env.meas
    v = env.veh
    kappa = env.track.kappa[env.i]
    delta = -KP * e - K_PSI * epsi + math.atan(v.p.L * kappa)
    return max(-1.0, min(1.0, delta / v.p.max_steer))
