"""PID vs RL 비교. 학습 때 안 본 트랙 10개(seed 100~109) × 여러 조건.
  python rl/compare.py
결과: rl/results.md (표), out/compare.png (그림)

제어기
  PID       : 조향 PID (곡률 피드포워드 없음), 게인은 rl/tune_pid.py 탐색값
  PID+FF    : PID + 곡률 피드포워드 atan(L·κ)
  PID+FF 강건: 같은 구조, 게인을 노이즈·지연 조건까지 넣어 탐색 (rl/tune_pid.py --robust)
  RL 보상v1 : rl/train.py 로 학습한 SAC 정책 (보상 v1)
  RL 보상v2 : 같은 설정, 보상만 v2 (rl/train.py --reward v2)
조건
  기본, 젖은 노면(μ 0.6), 짐 40% 추가, 뒤 타이어 마모(Cr 0.7배), 센서 노이즈, 조향 지연 0.15 s
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import warnings  # noqa: E402

warnings.filterwarnings("ignore")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from slamsim.control import PIDSteer  # noqa: E402
from slamsim.envs import TEST_SEEDS, PathFollowEnv, pid_policy, run_episode  # noqa: E402
from slamsim.tracks import Track  # noqa: E402

SCENARIOS = [
    ("기본 (μ 0.9)", {}),
    ("젖은 노면 μ 0.6", {"mu": 0.6}),
    ("짐 40% 추가", {"mass_scale": 1.4}),
    ("뒤 타이어 마모 Cr×0.7", {"cr_scale": 0.7}),
    ("센서 노이즈 (e 10 cm, eψ 1.1°)", {"obs_noise": 1.0}),
    ("조향 지연 0.15 s", {"delay_steps": 3}),
]


def controllers():
    gains = json.load(open(os.path.join(HERE, "pid_gains.json")))
    out = [("PID", lambda: pid_policy(PIDSteer(**gains["pid"], feedforward=False))),
           ("PID+FF", lambda: pid_policy(PIDSteer(**gains["pid_ff"], feedforward=True)))]
    if "pid_ff_robust" in gains:
        out.append(("PID+FF 강건", lambda: pid_policy(PIDSteer(**gains["pid_ff_robust"], feedforward=True))))
    mine = os.path.join(HERE, "my_controller.py")
    if os.path.exists(mine):
        import importlib.util
        spec = importlib.util.spec_from_file_location("my_controller", mine)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out.append(("내 제어기", lambda: mod.policy))
    from stable_baselines3 import SAC
    for label, fname in (("RL 보상v1", "sac_pathfollow.zip"), ("RL 보상v2", "sac_pathfollow_v2.zip")):
        path = os.path.join(HERE, "models", fname)
        if not os.path.exists(path):
            print(f"{fname} 없음 (python rl/train.py) - {label} 은 빼고 비교")
            continue
        model = SAC.load(path, device="cpu")
        out.append((label, (lambda m: (lambda: (lambda obs, env: float(m.predict(obs, deterministic=True)[0][0]))))(model)))
    return out


def main():
    ctrls = controllers()
    tracks = [Track.random(s) for s in TEST_SEEDS]
    rows, traces = [], {}
    for sname, kw in SCENARIOS:
        for cname, make in ctrls:
            ms = []
            for k, tr in enumerate(tracks):
                env = PathFollowEnv(seeds=TEST_SEEDS, randomize=False, episode_time=60.0, **kw)
                m, L = run_episode(env, make(), track=tr, seed=k)
                ms.append(m)
                if k == 0:
                    traces[(sname, cname)] = (L, tr)
            off = sum(m["off"] for m in ms)
            ok = [m for m in ms if not m["off"]]
            rows.append({
                "scenario": sname, "ctrl": cname, "off": off,
                "rms_cm": 100 * np.mean([m["rms_e"] for m in ok]) if ok else float("nan"),
                "max_cm": 100 * np.max([m["max_e"] for m in ok]) if ok else float("nan"),
                "steer_rate": np.mean([m["steer_rate"] for m in ms]),
                "slip": np.max([m["max_slip_deg"] for m in ms]),
            })
            r = rows[-1]
            print(f"{sname:28s} {cname:9s} 이탈 {off}/10  RMS {r['rms_cm']:6.1f} cm  최대 {r['max_cm']:6.1f} cm"
                  f"  조향떨림 {r['steer_rate']:.2f}/s", flush=True)

    # 표
    lines = ["| 조건 | 제어기 | 이탈 (10개 중) | 평균 RMS 오차 [cm] | 최대 오차 [cm] | 조향 떨림 [1/s] |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['ctrl']} | {r['off']} | {r['rms_cm']:.1f} | {r['max_cm']:.1f} | {r['steer_rate']:.2f} |")
    note = ("\n오차는 이탈하지 않은 트랙만으로 계산했다. 조향 떨림은 조향 명령 변화율의 RMS(값이 클수록 핸들을 자주 흔든다).\n"
            "트랙 seed 100~109 는 학습·게인 탐색에 쓰지 않았다. 속도는 모든 제어기가 같은 규칙(앞 30 m 곡률로 횡가속도 5 m/s² 이하)이다.\n"
            "RL 과 PID, PID+FF 는 학습·튜닝 때 노이즈·지연을 보지 않았다. 'PID+FF 강건'만 노이즈·지연 조건까지 넣어 게인을 골랐으므로 그 두 조건에서는 유리하다.\n"
            "RL 은 학습 때 μ 0.75~1.0, 질량 ±10%, 코너링 강성 ±15% 범위로 랜덤화했다(젖은 노면 0.6, 짐 40%, 마모 0.7배는 범위 밖).\n")
    open(os.path.join(HERE, "results.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n" + note)

    # 그림: 조건별 횡 오차 시간 그래프 (트랙 seed 100)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axs = plt.subplots(len(SCENARIOS), 1, figsize=(11, 2.1 * len(SCENARIOS)), sharex=True)
    colors = {"PID": "tab:red", "PID+FF": "tab:orange", "PID+FF 강건": "tab:green", "RL 보상v1": "tab:purple", "RL 보상v2": "tab:blue", "내 제어기": "tab:gray"}
    for ax, (sname, _) in zip(axs, SCENARIOS):
        for cname, _ in ctrls:
            L, _ = traces[(sname, cname)]
            ax.plot(L[:, 0], 100 * L[:, 3], color=colors[cname], lw=1, label=cname)
        ax.axhline(200, color="k", lw=0.5, ls=":"); ax.axhline(-200, color="k", lw=0.5, ls=":")
        ax.set_ylabel("횡 오차 [cm]"); ax.set_title(sname, fontsize=10, loc="left")
        lim = max(10, min(220, max(np.abs(100 * traces[(sname, c)][0][:, 3]).max() for c, _ in ctrls) * 1.1))
        ax.set_ylim(-lim, lim)
    axs[0].legend(loc="upper right", fontsize=8, ncol=3)
    axs[-1].set_xlabel("시간 [s]  (시험 트랙 random100)")
    fig.tight_layout()
    os.makedirs("out", exist_ok=True)
    fig.savefig("out/compare.png", dpi=100)
    print("-> rl/results.md, out/compare.png")


if __name__ == "__main__":
    main()
