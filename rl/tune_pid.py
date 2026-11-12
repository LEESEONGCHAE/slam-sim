"""PID 게인 탐색 (python rl/tune_pid.py, 강건 튜닝은 --robust): 학습용 트랙에서 격자 탐색 (RL 과 같은 트랙·같은 랜덤화 조건).
점수 = 평균 RMS 오차 + 이탈 1회당 1.0 + 0.01·조향 떨림.  결과는 rl/pid_gains.json"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from slamsim.control import PIDSteer  # noqa: E402
from slamsim.envs import PathFollowEnv, run_episode, pid_policy  # noqa: E402


ROBUST_CONDITIONS = [{}, {"obs_noise": 1.0}, {"delay_steps": 3}]


def score(gains, ff, episodes=6, conditions=({},)):
    s = []
    for kw in conditions:
        env = PathFollowEnv(episode_time=30.0, **kw)
        for ep in range(episodes):
            m, _ = run_episode(env, pid_policy(PIDSteer(**gains, feedforward=ff)), seed=1000 + ep)
            s.append(m["rms_e"] + (1.0 if m["off"] else 0.0) + 0.01 * m["steer_rate"])
    return float(np.mean(s))


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pid_gains.json")
    if "--robust" in sys.argv:
        # 노이즈·지연 조건도 함께 넣어 튜닝 (RL 과 공정하게: RL 도 이 조건을 학습 때 보지 않았지만,
        # PID 는 구조가 단순해 튜닝 조건이 결과를 크게 좌우하므로 따로 비교한다)
        best = None
        for kp, kd, kpsi, ki in itertools.product([0.1, 0.25, 0.5, 1.0], [0.0, 0.05, 0.1],
                                                  [0.5, 1.0, 1.5, 2.0], [0.0, 0.01, 0.05]):
            g = dict(kp=kp, ki=ki, kd=kd, k_psi=kpsi)
            sc = score(g, True, episodes=3, conditions=ROBUST_CONDITIONS)
            if best is None or sc < best[0]:
                best = (sc, g)
        out = json.load(open(path))
        out["pid_ff_robust"] = best[1]
        json.dump(out, open(path, "w"), indent=2)
        print("PID+FF robust best", best[1], "score", round(best[0], 4))
        sys.exit()
    out = {}
    for ff in (True, False):
        best = None
        for kp, kd, kpsi, ki in itertools.product([0.1, 0.25, 0.5, 1.0], [0.0, 0.05, 0.1, 0.2],
                                                  [0.5, 1.0, 1.5, 2.0], [0.0, 0.01, 0.05]):
            g = dict(kp=kp, ki=ki, kd=kd, k_psi=kpsi)
            sc = score(g, ff)
            if best is None or sc < best[0]:
                best = (sc, g)
        out["pid_ff" if ff else "pid"] = best[1]
        print(("PID+feedforward" if ff else "PID only"), "best", best[1], "score", round(best[0], 4))
    json.dump(out, open(path, "w"), indent=2)
