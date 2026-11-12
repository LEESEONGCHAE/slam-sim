"""강화학습 환경: 경로 따라가기 (Gymnasium).

과제: 닫힌 트랙의 중심선을 따라 달린다. 속도는 공통 규칙(control.speed_command)이 정하고,
      에이전트는 조향만 정한다. 중심선에서 반폭(2 m) 넘게 벗어나면 이탈로 끝난다.

관측 (12개, 대략 −1..1 로 맞춤)
  e, eψ, vx, vy, r, 현재 조향, 앞쪽 0/5/10/15/20/30 m 곡률
행동: 조향 명령 −1..1 (차량 조향 속도 한계 70°/s 를 거친다)
보상 (reward="v1"): 1 − (e/반폭)² − 0.5·eψ² − 0.1·(조향 변화)²,  이탈 시 −20 하고 끝
     v1 은 23 cm 벗어나도 벌점이 0.013 뿐이라, RL 이 '이탈 안 하기'만 배우고 정확도는 안 올렸다.
보상 (reward="v2"): exp(−(e/0.25 m)²) − 0.5·eψ² − 0.5·(조향 변화)²,  이탈 시 −20
     작은 오차에도 보상이 크게 변하고(23 cm 에서 0.43), 조향 흔들기 벌점을 5배로.

시험용 변화(학습 때 안 본 조건): 마찰 μ, 질량, 코너링 강성, 관측 노이즈, 조향 지연(step 단위)
"""
import math
from collections import deque

import gymnasium as gym
import numpy as np

from .control import speed_command
from .tracks import Track
from .vehicle import Vehicle, VehicleParams

PREVIEW = (0, 5, 10, 15, 20, 30)
TRAIN_SEEDS = list(range(0, 50))
TEST_SEEDS = list(range(100, 110))


class PathFollowEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seeds=TRAIN_SEEDS, randomize=True, episode_time=60.0, dt=0.05,
                 mu=None, mass_scale=1.0, cr_scale=1.0, obs_noise=0.0, delay_steps=0, reward="v1"):
        super().__init__()
        self.reward_kind = reward
        self.seeds, self.randomize = list(seeds), randomize
        self.episode_time, self.dt = episode_time, dt
        self.fixed = dict(mu=mu, mass_scale=mass_scale, cr_scale=cr_scale)
        self.obs_noise, self.delay_steps = obs_noise, delay_steps
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (6 + len(PREVIEW),), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        self._tracks = {}

    def _track(self, s):
        if s not in self._tracks:
            self._tracks[s] = Track.random(s)
        return self._tracks[s]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        opt = options or {}
        self.track = opt.get("track") or self._track(int(rng.choice(self.seeds)))
        p = VehicleParams()
        if self.randomize:      # 도메인 랜덤화: 학습 때 차 특성을 매번 조금씩 바꾼다
            p.mu = float(rng.uniform(0.75, 1.0))
            p.m *= float(rng.uniform(0.9, 1.1))
            p.Cf *= float(rng.uniform(0.85, 1.15))
            p.Cr *= float(rng.uniform(0.85, 1.15))
        if self.fixed["mu"] is not None:
            p.mu = self.fixed["mu"]
        p.m *= self.fixed["mass_scale"]
        p.Iz *= self.fixed["mass_scale"]
        p.Cr *= self.fixed["cr_scale"]
        self.veh = Vehicle(*self.track.start_pose(), params=p)
        self.veh.vx = 5.0
        self.i = 0
        self.t = 0.0
        self.dist = 0.0
        self.prev_action = 0.0
        self.queue = deque([0.0] * self.delay_steps)
        self.log = []
        return self._obs(), {}

    def errors(self):
        v = self.veh
        self.i = self.track.locate(v.x, v.y, self.i)
        e, epsi = self.track.errors(self.i, v.x, v.y, v.yaw)
        return e, epsi

    def _obs(self):
        v = self.veh
        e, epsi = self.errors()
        if self.obs_noise:
            e += self.np_random.normal(0, 0.1 * self.obs_noise)
            epsi += self.np_random.normal(0, 0.02 * self.obs_noise)
        self.meas = (e, epsi)       # 제어기가 보는 값 (노이즈 포함)
        k = self.track.preview(self.i, PREVIEW)
        o = [e / 2.0, epsi / 0.5, v.vx / 15.0, v.vy / 2.0, v.r / 1.0, v.steer / v.p.max_steer] + list(k * 20.0)
        return np.array(o, dtype=np.float32)

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1, 1))
        self.queue.append(a)
        steer = self.queue.popleft()      # 지연이 있으면 몇 step 전 명령이 들어간다
        v = self.veh
        accel, v_ref = speed_command(self.track, self.i, v.vx)
        i0 = self.i
        for _ in range(int(round(self.dt / 0.002))):
            v.step(steer, accel, 0.002)
        self.t += self.dt
        obs = self._obs()
        e, epsi = self.errors()
        di = (self.i - i0) % self.track.n
        self.dist += di * self.track.ds if di < self.track.n // 2 else 0.0
        half = self.track.width / 2
        off = abs(e) > half
        if self.reward_kind == "v2":
            reward = math.exp(-(e / 0.25) ** 2) - 0.5 * epsi ** 2 - 0.5 * (a - self.prev_action) ** 2
        else:
            reward = 1.0 - (e / half) ** 2 - 0.5 * epsi ** 2 - 0.1 * (a - self.prev_action) ** 2
        if off:
            reward -= 20.0
        self.prev_action = a
        self.log.append((self.t, v.x, v.y, e, epsi, v.vx, steer, v.slip_angle))
        trunc = self.t >= self.episode_time
        return obs, float(reward), off, trunc, {"e": e, "off": off, "v_ref": v_ref}


def run_episode(env, policy, track=None, seed=0):
    """policy(obs, env) -> 조향(−1..1). 결과 지표 dict 와 기록을 돌려준다."""
    obs, _ = env.reset(seed=seed, options={"track": track} if track else None)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step([policy(obs, env)])
        done = term or trunc
    L = np.array(env.log)
    e, steer = L[:, 3], L[:, 6]
    return {
        "track": env.track.name,
        "off": bool(term),
        "time": env.t,
        "dist": float(env.dist),
        "rms_e": float(np.sqrt(np.mean(e ** 2))),
        "max_e": float(np.abs(e).max()),
        "steer_rate": float(np.sqrt(np.mean(np.diff(steer) ** 2)) / env.dt),   # 조향 떨림 [1/s]
        "max_slip_deg": float(np.degrees(np.abs(L[:, 7]).max())),
    }, L


def pid_policy(pid):
    """PIDSteer 를 run_episode 용 policy 로."""
    def f(obs, env):
        if env.t == 0.0:
            pid.reset()
        e, epsi = env.meas
        k = env.track.kappa[env.i]
        return pid(e, epsi, k, env.veh.p.L, env.veh.p.max_steer)
    return f
