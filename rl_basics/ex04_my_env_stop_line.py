"""예제 4. 환경을 직접 만들기 - 정지선에 차 세우기 (RL vs PD 제어)

상황
  직선 도로. 차가 0 m 에서 멈춰 있다. 10 m 앞 정지선에 정확히 멈춰야 한다.
  - 상태 2개: 정지선까지 남은 거리, 속도
  - 행동: 가속 페달 −1 ~ +1  (×3 m/s². 음수는 브레이크)
  - 보상: 매 스텝 −(남은 거리) × 0.1  − 0.01 × 페달²
          정지선에 가까울수록 감점이 작다. 페달을 세게 밟으면 조금 감점(부드럽게 운전하라는 뜻)
  - 0.1 초마다 한 번, 10 초(100 스텝) 하면 끝

배울 것
  1. gymnasium.Env 를 상속해서 reset() 과 step() 두 함수만 쓰면 환경이 된다
  2. 같은 문제를 사람이 짠 PD 제어와 RL 로 풀어서 비교한다
  3. 보상 식을 바꿔 보면 RL 의 운전 습관이 바뀐다 (아래 '해 볼 것')

실행: .venv\\Scripts\\python rl_basics/ex04_my_env_stop_line.py   (1분 정도)

해 볼 것
  - 보상에서 페달 감점 0.01 을 0 이나 0.1 로 바꿔 보기 → 운전이 거칠어지거나 느려진다
  - PD 게인 KP, KD 를 바꿔 보기 → KD 를 0 으로 하면 정지선을 지나쳤다 돌아온다(오버슈트)
  - 마찰(FRICTION)을 0.5 로 바꿔 RL 은 다시 학습하지 말고 시험만 해 보기 → 누가 더 잘 버티나
"""
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC

torch.set_num_threads(4)

TARGET = 10.0      # 정지선 위치 [m]
DT = 0.1           # 한 스텝 시간 [s]
MAX_ACCEL = 3.0    # 페달 1 일 때 가속도 [m/s²]


# ---------------------------------------------------------------- 1. 환경 만들기
class StopLineEnv(gym.Env):
    def __init__(self, friction=0.0):
        super().__init__()
        self.friction = friction   # 속도에 비례해 줄어드는 저항 (0 이면 없음)
        # 관측: [남은 거리, 속도]. 범위는 넉넉히
        self.observation_space = gym.spaces.Box(-50.0, 50.0, (2,), np.float32)
        # 행동: 페달 하나, −1 ~ 1
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x, self.v, self.t = 0.0, 0.0, 0
        return self._obs(), {}

    def _obs(self):
        return np.array([TARGET - self.x, self.v], dtype=np.float32)

    def step(self, action):
        pedal = float(np.clip(action[0], -1, 1))
        a = pedal * MAX_ACCEL - self.friction * self.v      # 운동 방정식: 가속도 = 페달 − 저항
        self.v += a * DT
        self.x += self.v * DT
        self.t += 1
        dist = abs(TARGET - self.x)
        reward = -0.1 * dist - 0.01 * pedal ** 2
        truncated = self.t >= 100           # 10 초 지나면 끝
        return self._obs(), reward, False, truncated, {}


# ---------------------------------------------------------------- 2. 사람이 짠 PD 제어
KP, KD = 0.5, 1.0     # 남은 거리 게인, 속도 게인


def pd_controller(obs):
    remain, v = obs
    return np.clip([KP * remain - KD * v], -1, 1)    # 많이 남았으면 밟고, 빠르면 브레이크


# ---------------------------------------------------------------- 3. 평가 함수
def drive(policy, friction=0.0):
    env = StopLineEnv(friction)
    obs, _ = env.reset()
    total, xs = 0.0, []
    for _ in range(100):
        obs, r, term, trunc, _ = env.step(policy(obs))
        total += r
        xs.append(env.x)
    xs = np.array(xs)
    overshoot = max(0.0, xs.max() - TARGET)
    reach = next((i * DT + DT for i, x in enumerate(xs) if abs(x - TARGET) < 0.1), None)
    return total, xs[-1], overshoot, reach, env.v


def report(name, res):
    total, final_x, over, reach, v = res
    reach_s = f"{reach:.1f}초" if reach else "도착 못 함"
    print(f"  {name:6s} 점수 {total:7.2f} | 최종 위치 {final_x:6.2f} m (속도 {v:+.2f}) | "
          f"정지선 지나침 {over:.2f} m | 10 cm 안 도착 {reach_s}")


# ---------------------------------------------------------------- 4. RL 학습
env = StopLineEnv()
model = SAC("MlpPolicy", env, learning_rate=1e-3, learning_starts=500, seed=0, verbose=0,
            policy_kwargs=dict(net_arch=[64, 64]))


def rl_policy(obs):
    return model.predict(obs, deterministic=True)[0]


print("학습 전")
report("RL", drive(rl_policy))
for k in range(1, 5):
    model.learn(2500, reset_num_timesteps=False)
    print(f"{k * 2500} 스텝 학습 후")
    report("RL", drive(rl_policy))

print("\n=== 비교 (마찰 없음, 학습한 조건) ===")
report("PD", drive(pd_controller))
report("RL", drive(rl_policy))
print("\n=== 비교 (마찰 0.5, 학습 때 못 본 조건) ===")
report("PD", drive(pd_controller, friction=0.5))
report("RL", drive(rl_policy, friction=0.5))
