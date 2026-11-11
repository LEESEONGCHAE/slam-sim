"""직접 운전하는 시뮬레이터. 운전하는 동안 센서 데이터가 실시간으로 나오고 my_slam.py 에 들어간다.

  python run.py                         창을 띄워 직접 운전
  python run.py --headless --seconds 60 자동 주행으로 돌리고 out/run.png 저장

조작 (창을 한 번 클릭한 뒤)
  ↑ 가속   ↓ 브레이크/후진   ← → 조향   스페이스 브레이크
  A 자동 주행 켜기/끄기   R 처음부터   L 지금까지 센서 로그 저장 (play.py 로 다시 재생 가능)
"""
import argparse
import importlib
import math
import os
import time

import matplotlib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slam", default="my_slam:MySlam")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()

    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from slamsim import SimRig, ScanMsg, OdomMsg

    for k in list(plt.rcParams):
        if k.startswith("keymap.") and k != "keymap.quit":
            plt.rcParams[k] = []

    rig = SimRig(seed=args.seed)
    world, lidar = rig.sim.world, rig.sim.lidar
    mod, cls = args.slam.split(":")
    SlamClass = getattr(importlib.import_module(mod), cls)
    st = {"keys": set(), "auto": False, "slam": SlamClass(rig.info()),
          "est": [], "odo": [], "tru": [], "scan": None, "odom": None, "note": ""}

    def reset():
        rig.reset()
        st.update(slam=SlamClass(rig.info()), est=[], odo=[], tru=[], scan=None, odom=None)

    # ---------------- 화면: 실제 세계 | 라이다 원시값 | 내 SLAM 지도
    fig, (axw, axs, axm) = plt.subplots(1, 3, figsize=(17, 6.2), gridspec_kw={"width_ratios": [1.3, 1, 1.3]})
    x0, y0, x1, y1 = rig.info()["bounds"]
    for ax in (axw, axm):
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    axw.add_collection(LineCollection(world.seg.reshape(-1, 2, 2), colors="k", lw=1.5))
    axw.set_title("world (truth) - you drive here")
    car_true, = axw.plot([], [], "g-", lw=2)
    ln_true_w, = axw.plot([], [], "g-", lw=0.8)
    hits_w = axw.scatter([], [], s=3, c="tab:orange")

    R = 20.0
    axs.set_xlim(-R, R); axs.set_ylim(-R, R); axs.set_aspect("equal")
    axs.set_title("raw lidar scan (robot frame, x = forward)")
    axs.plot([0, 1.5], [0, 0], "k-", lw=2); axs.plot(0, 0, "ko", ms=4)
    sc_ok = axs.scatter([], [], s=4, c="tab:blue")
    sc_bad = axs.scatter([], [], s=6, c="tab:red", marker="x")

    prob, extent = st["slam"].map()
    img = axm.imshow(prob, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1)
    axm.set_title(f"map from {args.slam}")
    ln_est, = axm.plot([], [], "b-", lw=1.2, label="your estimate")
    ln_tru, = axm.plot([], [], "g-", lw=0.8, label="truth")
    ln_odo, = axm.plot([], [], "r--", lw=0.8, label="odometry")
    axm.legend(loc="upper right", fontsize=7)

    box = dict(facecolor="white", alpha=0.85, edgecolor="none")
    info = axw.text(0.01, 0.01, "", transform=axw.transAxes, family="monospace", fontsize=8, bbox=box)
    raw = axs.text(0.01, 0.01, "", transform=axs.transAxes, family="monospace", fontsize=7, bbox=box)
    err_t = axm.text(0.01, 0.01, "", transform=axm.transAxes, family="monospace", fontsize=8, bbox=box)
    artists = (img, car_true, ln_true_w, hits_w, sc_ok, sc_bad, ln_est, ln_tru, ln_odo, info, raw, err_t)

    def feed(msgs):
        slam = st["slam"]
        for m in msgs:
            if isinstance(m, OdomMsg):
                slam.on_odom(m)
                st["odom"] = m
            elif isinstance(m, ScanMsg):
                slam.on_scan(m)
                st["scan"] = m
                e = slam.estimate()
                if e is not None:
                    st["est"].append(e[:2]); st["odo"].append(m.odom[:2])
                    st["tru"].append(rig.scan_truth[m.t][:2])

    def controls():
        k = st["keys"]
        steer = ("left" in k) - ("right" in k)
        accel = ("up" in k) - ("down" in k)
        if " " in k:
            accel = -1 if rig.vehicle.vx > 0.1 else 0
        return steer, accel

    def draw():
        v = rig.vehicle
        car_true.set_data(*zip(*v.outline()))
        m = st["scan"]
        if m is not None:
            a, r, ok = lidar.angles, m.ranges, m.valid
            sc_ok.set_offsets(np.c_[r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])])
            sc_bad.set_offsets(np.c_[np.cos(a[~ok]), np.sin(a[~ok])])
            x, y, th = v.pose
            hits_w.set_offsets(np.c_[x + r[ok] * np.cos(a[ok] + th), y + r[ok] * np.sin(a[ok] + th)])
            rs = " ".join(f"{val:.1f}" if good else "--" for val, good in zip(r[88:93], ok[88:93]))
            o = st["odom"]
            raw.set_text(
                f"ScanMsg t={m.t:.2f}  {ok.sum()}/{len(ok)} valid (red x = no return)\n"
                f"  ranges[88:93] (front) = {rs}\n"
                f"  odom at scan = ({m.odom[0]:.2f}, {m.odom[1]:.2f}, {math.degrees(m.odom[2]):.1f}deg)\n"
                + (f"OdomMsg t={o.t:.2f}  v={o.v:.2f} m/s  w={math.degrees(o.w):.1f} deg/s" if o else ""))
        p, _ = st["slam"].map()
        img.set_data(p)
        if st["est"]:
            for ln, key in ((ln_est, "est"), (ln_tru, "tru"), (ln_odo, "odo")):
                P = np.array(st[key]); ln.set_data(P[:, 0], P[:, 1])
            ln_true_w.set_data(*np.array(st["tru"]).T)
            E, T, O = (np.array(st[k]) for k in ("est", "tru", "odo"))
            rmse = lambda P: math.sqrt(np.mean(np.sum((P - T) ** 2, axis=1)))
            err_t.set_text(f"position RMSE  yours {rmse(E):.2f} m / odometry {rmse(O):.2f} m")
        info.set_text(
            f"t={rig.sim.t:6.1f}s  v={v.vx * 3.6:5.1f} km/h  steer={math.degrees(v.steer):5.1f}deg\n"
            f"slip={math.degrees(v.slip_angle):5.1f}deg  yawrate={math.degrees(v.r):6.1f}deg/s\n"
            f"auto={'ON' if st['auto'] else 'off'} [A]  reset [R]  save log [L]  {st['note']}")

    def save_log():
        os.makedirs("data", exist_ok=True)
        path = time.strftime("data/drive_%Y%m%d_%H%M%S.npz")
        rig.to_log(name=path).save(path)
        st["note"] = f"saved {path}"
        print(f"saved {path}  ->  python play.py {path}")

    if args.headless:
        feed(rig.step(0, 0, args.seconds, auto=True))
        draw()
        os.makedirs("out", exist_ok=True)
        fig.savefig("out/run.png", dpi=100)
        print(f"t={rig.sim.t:.1f}s  {err_t.get_text()}  -> out/run.png")
        return

    def on_press(e):
        if e.key is None:
            return
        k = e.key.lower()
        if k == "a":
            st["auto"] = not st["auto"]
        elif k == "r":
            reset()
        elif k == "l":
            save_log()
        else:
            st["keys"].add(k)

    def on_release(e):
        if e.key:
            st["keys"].discard(e.key.lower())

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("key_release_event", on_release)

    from matplotlib.animation import FuncAnimation
    last = [time.perf_counter()]

    def tick(_):
        now = time.perf_counter()
        dt = min(0.1, now - last[0])
        last[0] = now
        s, a = controls()
        feed(rig.step(s, a, dt, auto=st["auto"]))
        draw()
        return artists

    for a in artists:
        a.set_animated(True)
    plt.tight_layout()
    anim = FuncAnimation(fig, tick, interval=30, blit=True, cache_frame_data=False)  # noqa: F841
    plt.show()


if __name__ == "__main__":
    main()
