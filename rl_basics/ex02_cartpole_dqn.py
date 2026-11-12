"""예제 2. 막대 세우기 (CartPole) - DQN

상황
  수레 위에 막대가 서 있다. 수레를 왼쪽/오른쪽으로 밀어서 막대가 쓰러지지 않게 한다.
  - 상태 4개: 수레 위치, 수레 속도, 막대 각도, 막대 각속도
  - 행동 2개: 0 = 왼쪽으로 밀기, 1 = 오른쪽으로 밀기
  - 보상: 버틴 스텝마다 +1. 막대가 12° 넘게 기울거나 수레가 밖으로 나가면 끝. 최대 500

예제 1과 다른 점
  예제 1은 칸이 25개라 Q 를 표에 담을 수 있었다.
  여기는 상태가 연속값(각도 0.013°, 0.014°, ...)이라 표로 못 만든다.
  그래서 Q 표 대신 '신경망'이 Q 값을 계산한다 → DQN (Deep Q-Network)

실행
  .venv\\Scripts\\python rl_basics/ex02_cartpole_dqn.py           학습(1~2분) 후 결과 출력
  .venv\\Scripts\\python rl_basics/ex02_cartpole_dqn.py --show    학습 후 창으로 보기
"""
import sys

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN

torch.set_num_threads(4)   # 이 PC 에서는 스레드를 줄여야 빠르다


def evaluate(model, episodes=10):
    """학습된 모델을 탐험 없이 돌려 평균 점수를 낸다"""
    env = gym.make("CartPole-v1")
    scores = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        total, done = 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            total += reward
            done = term or trunc
        scores.append(total)
    return np.mean(scores)


env = gym.make("CartPole-v1")

# DQN 설정. 숫자는 RL Baselines3 Zoo 에 공개된 CartPole 값을 따랐다.
model = DQN(
    "MlpPolicy", env,
    learning_rate=2.3e-3,          # 학습률
    buffer_size=100_000,           # 리플레이 버퍼: 경험을 쌓아 두는 곳
    learning_starts=1000,          # 처음 1000 스텝은 무작위로 경험만 모은다
    batch_size=64,                 # 한 번 배울 때 버퍼에서 뽑는 경험 수
    gamma=0.99,                    # 할인율
    target_update_interval=10,     # 타깃 네트워크(천천히 따라오는 사본) 갱신 간격
    train_freq=256, gradient_steps=128,
    exploration_fraction=0.16,     # 학습의 처음 16% 동안 탐험 비율 ε 을 1 → 0.04 로 줄인다
    exploration_final_eps=0.04,
    policy_kwargs=dict(net_arch=[256, 256]),   # 신경망: 은닉층 256개 × 2
    seed=0, verbose=0,
)

print("학습 전 평균 점수:", evaluate(model), "/ 500   (무작위에 가깝다)")
for k in range(1, 6):
    model.learn(10_000, reset_num_timesteps=False)
    print(f"{k * 10_000:>6} 스텝 학습 후 평균 점수: {evaluate(model):6.1f} / 500")

if "--show" in sys.argv:
    env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    env.close()
