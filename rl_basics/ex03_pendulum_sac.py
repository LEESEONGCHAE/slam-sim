"""예제 3. 진자 세우기 (Pendulum) - SAC

상황
  한쪽 끝이 고정된 막대(진자)를 모터로 돌려서 거꾸로 세워 유지한다.
  - 상태 3개: cos(각도), sin(각도), 각속도
  - 행동: 모터 토크 −2 ~ +2  ← 예제 2와 달리 '연속값'
  - 보상: −(각도² + 0.1·각속도² + 0.001·토크²). 똑바로 서 있으면 0 에 가깝고, 매달려 있으면 약 −10
  - 200 스텝이면 끝. 점수(합)는 대략 −1500(못함) ~ −150(잘함)

예제 2와 다른 점
  행동이 연속값이라 DQN 처럼 "가장 좋은 행동 고르기(max)"를 할 수 없다.
  SAC 는 행동을 직접 내는 신경망(액터)과 그 행동의 점수를 평가하는 신경망(크리틱)을 같이 학습한다.
  모터 힘이 약해서 한 번에 못 올린다 → 앞뒤로 흔들어 반동을 써야 한다는 걸 스스로 찾아야 한다.
  slam-sim 의 경로 따라가기 RL 이 쓴 것도 이 SAC 다.

실행
  .venv\\Scripts\\python rl_basics/ex03_pendulum_sac.py           학습(1~2분) 후 결과 출력
  .venv\\Scripts\\python rl_basics/ex03_pendulum_sac.py --show    학습 후 창으로 보기
"""
import sys

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC

torch.set_num_threads(4)


def evaluate(model, episodes=5):
    env = gym.make("Pendulum-v1")
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=100 + ep)
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            total += reward
            done = term or trunc
        scores.append(total)
    return np.mean(scores)


env = gym.make("Pendulum-v1")
model = SAC("MlpPolicy", env, learning_rate=1e-3, learning_starts=1000, seed=0, verbose=0,
            policy_kwargs=dict(net_arch=[64, 64]))

print(f"학습 전 평균 점수: {evaluate(model):7.1f}   (−1500 근처면 못 하는 것)")
for k in range(1, 6):
    model.learn(3000, reset_num_timesteps=False)
    print(f"{k * 3000:>6} 스텝 학습 후 평균 점수: {evaluate(model):7.1f}   (−200 위면 거의 세운 것)")

if "--show" in sys.argv:
    env = gym.make("Pendulum-v1", render_mode="human")
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    env.close()
