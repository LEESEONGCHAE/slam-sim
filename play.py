"""센서 로그를 재생하면서 my_slam.py 의 알고리즘에 넣는다.

  python play.py intel              실제 데이터 (Intel Research Lab, data/intel.clf)
  python play.py sim                시뮬레이터 로그 (정답 위치가 있어 오차를 잰다)
  python play.py intel --headless   끝까지 돌리고 out/play_intel.png 저장
  옵션: --speed 10 (재생 배속), --slam my_slam:MySlam (다른 구현 파일), --seconds 300 (앞부분만)

조작: 스페이스 일시정지, + / - 배속
"""
import argparse
import importlib
import math
import os
import time

import matplotlib
import numpy as np

DATA_DIR = "data"
INTEL_URL = "http://ais.informatik.uni-freiburg.de/slamevaluation/datasets/intel.clf"


def get_log(which, seed=0):
    from slamsim import SensorLog, load_carmen, record_sim
    os.makedirs(DATA_DIR, exist_ok=True)
    if which == "intel":
        cache = os.path.join(DATA_DIR, "intel.npz")
        if os.path.exists(cache):
            return SensorLog.load(cache)
        clf = os.path.join(DATA_DIR, "intel.clf")
        if not os.path.exists(clf):
            raise SystemExit(f"{clf} 가 없습니다. 먼저 받으세요:\n  python get_data.py\n(원본: {INTEL_URL})")
        log = load_carmen(clf, name="Intel Research Lab")
        log.save(cache)
        return log
    if which == "sim":
        cache = os.path.join(DATA_DIR, f"sim_seed{seed}.npz")
        if os.path.exists(cache):
            return SensorLog.load(cache)
        print("시뮬레이터 로그 만드는 중 (처음 한 번)...")
        log = record_sim(seconds=240, seed=seed)
        log.save(cache)
        return log
    if which.endswith(".clf"):
        return load_carmen(which)
    return SensorLog.load(which)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="intel", help="intel | sim | 파일(.clf/.npz)")
    ap.add_argument("--slam", default="my_slam:MySlam")
    ap.add_argument("--speed", type=float, default=None)
    ap.add_argument("--seconds", type=float, default=None, help="앞부분만 재생")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from slamsim import ScanMsg, OdomMsg

    log = get_log(args.data, args.seed)
    msgs = log.messages
    if args.seconds:
        msgs = [m for m in msgs if m.t - msgs[0].t <= args.seconds]
    mod, cls = args.slam.split(":")
    slam = getattr(importlib.import_module(mod), cls)(log.info())
    speed = args.speed or (10.0 if args.data == "intel" else 3.0)
    print(f"{log.name}: {len(msgs)} messages, {msgs[-1].t - msgs[0].t:.0f} s, "
          f"lidar {len(log.angles)} beams, truth={'yes' if log.truth is not None else 'no'}")

    est, odo, tru = [], [], []
    state = {"i": 0, "t": msgs[0].t, "pause": False, "speed": speed, "last_scan": None, "proc": 0.0}

    def feed_until(t_end):
        while state["i"] < len(msgs) and msgs[state["i"]].t <= t_end:
            m = msgs[state["i"]]
            state["i"] += 1
            t0 = time.perf_counter()
            if isinstance(m, OdomMsg):
                slam.on_odom(m)
            elif isinstance(m, ScanMsg):
                slam.on_scan(m)
                state["last_scan"] = m
                e = slam.estimate()
                if e is not None:
                    est.append(e); odo.append(m.odom)
                    g = log.truth_at(m.t)
                    tru.append(g if g is not None else (np.nan,) * 3)
            state["proc"] += time.perf_counter() - t0

    def errors():
        if log.truth is None or not est:
            return None
        E, O, T = np.array(est), np.array(odo), np.array(tru)
        ate = lambda P: math.sqrt(np.mean(np.sum((P[:, :2] - T[:, :2]) ** 2, axis=1)))
        return ate(E), ate(O)

    # ---------------- 화면
    for k in list(plt.rcParams):
        if k.startswith("keymap.") and k != "keymap.quit":
            plt.rcParams[k] = []
    fig, (axs, axm) = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1, 1.6]})
    R = min(log.max_range, 20.0)
    axs.set_xlim(-R, R); axs.set_ylim(-R, R); axs.set_aspect("equal")
    axs.set_title("raw lidar scan (robot frame, x = forward)")
    axs.plot([0, 1.0], [0, 0], "k-", lw=2); axs.plot(0, 0, "ko")
    sc_ok = axs.scatter([], [], s=4, c="tab:blue", label="valid")
    sc_bad = axs.scatter([], [], s=4, c="tab:red", marker="x", label="no return (shown at 1 m)")
    axs.legend(loc="upper right", fontsize=8)

    prob, extent = slam.map()
    img = axm.imshow(prob, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
    axm.set_aspect("equal")
    axm.set_title(f"{log.name}  -  map from {args.slam}")
    ln_est, = axm.plot([], [], "b-", lw=1.2, label="your estimate")
    ln_tru, = axm.plot([], [], "g-", lw=0.8, label="truth")
    ln_odo, = axm.plot([], [], "r--", lw=0.8, label="odometry")
    axm.legend(loc="upper right", fontsize=8)
    box = dict(facecolor="white", alpha=0.85, edgecolor="none")
    info = axm.text(0.01, 0.01, "", transform=axm.transAxes, family="monospace", fontsize=8, bbox=box)
    artists = (sc_ok, sc_bad, img, ln_odo, ln_tru, ln_est, info)

    def draw():
        m = state["last_scan"]
        if m is not None:
            a, r, v = log.angles, m.ranges, m.valid
            sc_ok.set_offsets(np.c_[r[v] * np.cos(a[v]), r[v] * np.sin(a[v])])
            sc_bad.set_offsets(np.c_[np.cos(a[~v]), np.sin(a[~v])])
        p, _ = slam.map()
        img.set_data(p)
        if est:
            E, O = np.array(est), np.array(odo)
            ln_est.set_data(E[:, 0], E[:, 1]); ln_odo.set_data(O[:, 0], O[:, 1])
            if log.truth is not None:
                T = np.array(tru); ln_tru.set_data(T[:, 0], T[:, 1])
        n_scan = len(est)
        txt = (f"t={state['t'] - msgs[0].t:7.1f}s  x{state['speed']:.0f}  scans={n_scan}"
               f"  algo={state['proc'] / max(1, n_scan) * 1000:.1f} ms/scan")
        err = errors()
        if err:
            txt += f"\nposition RMSE: yours {err[0]:.2f} m / odometry {err[1]:.2f} m"
        if m is not None:
            txt += f"\nthis scan: {m.valid.sum()}/{len(m.valid)} valid"
        info.set_text(txt + ("\nPAUSED [space]" if state["pause"] else ""))

    if args.headless:
        t0 = time.perf_counter()
        feed_until(msgs[-1].t)
        state["t"] = msgs[-1].t
        draw()
        os.makedirs("out", exist_ok=True)
        name = os.path.splitext(os.path.basename(args.data))[0]
        out = f"out/play_{name}.png"
        fig.savefig(out, dpi=110)
        err = errors()
        print(f"done in {time.perf_counter() - t0:.0f} s, {len(est)} scans -> {out}")
        if err:
            print(f"position RMSE: yours {err[0]:.2f} m, odometry {err[1]:.2f} m")
        return

    def on_key(e):
        if e.key == " ":
            state["pause"] = not state["pause"]
        elif e.key in ("+", "="):
            state["speed"] *= 2
        elif e.key in ("-", "_"):
            state["speed"] = max(1.0, state["speed"] / 2)
    fig.canvas.mpl_connect("key_press_event", on_key)

    from matplotlib.animation import FuncAnimation
    last = [time.perf_counter()]

    def tick(_):
        now = time.perf_counter()
        dt = min(0.2, now - last[0])
        last[0] = now
        if not state["pause"]:
            state["t"] += dt * state["speed"]
            feed_until(state["t"])
        draw()
        return artists

    for a in artists:
        a.set_animated(True)
    plt.tight_layout()
    anim = FuncAnimation(fig, tick, interval=40, blit=True, cache_frame_data=False)  # noqa: F841
    plt.show()


if __name__ == "__main__":
    main()
