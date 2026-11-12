"""RL 학습: 경로 따라가기 조향 정책 (SAC, Stable-Baselines3).
  python rl/train.py --steps 150000               (보상 v1)
  python rl/train.py --steps 150000 --reward v2   (보상 v2)
결과: rl/models/sac_pathfollow.zip, 학습 곡선 rl/models/train_log.csv"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from slamsim.envs import PathFollowEnv  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class Progress(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.f = open(path, "w")
        self.f.write("step,episode_return,episode_len\n")
        self.t0 = time.time()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self.f.write(f"{self.num_timesteps},{ep['r']:.2f},{ep['l']}\n")
                self.f.flush()
        if self.num_timesteps % 10000 == 0:
            print(f"{self.num_timesteps} steps, {time.time() - self.t0:.0f} s", flush=True)
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward", default="v1", choices=["v1", "v2"])
    args = ap.parse_args()
    torch.set_num_threads(4)   # 16 스레드 기본값은 경합 때문에 오히려 느렸다
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    env = Monitor(PathFollowEnv(episode_time=30.0, reward=args.reward))
    model = SAC("MlpPolicy", env, seed=args.seed, verbose=0, learning_starts=5000, train_freq=2, gradient_steps=1,
                policy_kwargs=dict(net_arch=[128, 128]))
    model.learn(args.steps, callback=Progress(os.path.join(HERE, "models", f"train_log{'' if args.reward == 'v1' else '_v2'}.csv")))
    name = "sac_pathfollow" if args.reward == "v1" else "sac_pathfollow_v2"
    model.save(os.path.join(HERE, "models", name))
    print(f"saved rl/models/{name}.zip")
