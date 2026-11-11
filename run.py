"""실행: python run.py            (창을 띄워 직접 운전)
      python run.py --headless --seconds 120   (자동 주행 후 out/result.png 저장)

조작: 방향키(또는 W/S/D) 운전, 스페이스 브레이크, A 자동 주행 켜기/끄기,
      M 지도 바꾸기(오도메트리 위치 / 실제 위치), R 처음부터.
"""
import argparse
import math
import os
import sys

import matplotlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from slamsim import Sim

    for k in list(plt.rcParams):
        if k.startswith("keymap.") and k != "keymap.quit":
            plt.rcParams[k] = []

    sim = Sim(seed=args.seed)
    state = {"keys": set(), "auto": args.headless, "map": "odom"}

    fig, (axw, axm) = plt.subplots(1, 2, figsize=(14, 6))
    fig.canvas.manager.set_window_title("slam-sim") if fig.canvas.manager else None
    x0, x1, y0, y1 = sim.grid_true.extent

    # 왼쪽: 실제 세계
    axw.add_collection(LineCollection(sim.world.seg.reshape(-1, 2, 2), colors="k", lw=1.5))
    axw.set_xlim(x0, x1); axw.set_ylim(y0, y1); axw.set_aspect("equal")
    axw.set_title("world (truth)")
    ln_true, = axw.plot([], [], "g-", lw=1, label="truth")
    ln_odom, = axw.plot([], [], "r--", lw=1, label="odometry")
    car_true, = axw.plot([], [], "g-", lw=2)
    car_odom, = axw.plot([], [], "r-", lw=1)
    pts = axw.scatter([], [], s=3, c="tab:orange")
    axw.legend(loc="upper right", fontsize=8)

    # 오른쪽: 지도
    img = axm.imshow(sim.grid_odom.prob(), origin="lower", extent=sim.grid_true.extent,
                     cmap="gray_r", vmin=0, vmax=1)
    axm.set_aspect("equal")
    car_map, = axm.plot([], [], "r-", lw=1.5)
    info = fig.text(0.01, 0.01, "", family="monospace", fontsize=9)

    def on_press(e):
        if e.key is None:
            return
        k = e.key.lower()
        if k == "a":
            state["auto"] = not state["auto"]
        elif k == "m":
            state["map"] = "true" if state["map"] == "odom" else "odom"
        elif k == "r":
            sim.reset()
        else:
            state["keys"].add(k)

    def on_release(e):
        if e.key:
            state["keys"].discard(e.key.lower())

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("key_release_event", on_release)

    def controls():
        k = state["keys"]
        steer = (1 if k & {"left"} else 0) - (1 if k & {"right", "d"} else 0)
        accel = (1 if k & {"up", "w"} else 0) - (1 if k & {"down", "s"} else 0)
        if " " in k:
            accel = -1 if sim.vehicle.vx > 0.1 else 0
        return steer, accel

    def draw():
        v = sim.vehicle
        tp = list(zip(*sim.path_true)); op = list(zip(*sim.path_odom))
        ln_true.set_data(*tp); ln_odom.set_data(*op)
        car_true.set_data(*zip(*v.outline()))
        car_odom.set_data(*zip(*v.outline(sim.odom.pose)))
        if sim.scan is not None:
            pts.set_offsets(sim.lidar.points(v.pose, *sim.scan))
        grid, pose = (sim.grid_odom, sim.odom.pose) if state["map"] == "odom" else (sim.grid_true, v.pose)
        img.set_data(grid.prob())
        car_map.set_data(*zip(*v.outline(pose)))
        axm.set_title(f"map built with {'ODOMETRY' if state['map'] == 'odom' else 'TRUE'} pose  [M]")
        err, dth = sim.odom_error
        info.set_text(
            f"t={sim.t:6.1f}s  v={v.vx*3.6:5.1f}km/h  steer={math.degrees(v.steer):5.1f}deg  "
            f"slip={math.degrees(v.slip_angle):5.1f}deg  yawrate={math.degrees(v.r):6.1f}deg/s  "
            f"odom err={err:5.2f}m {dth:5.1f}deg  auto={'ON' if state['auto'] else 'off'} [A]")

    if args.headless:
        sim.step(0, 0, args.seconds, auto=True)
        draw()
        os.makedirs("out", exist_ok=True)
        fig.savefig("out/result.png", dpi=110)
        err, dth = sim.odom_error
        print(f"t={sim.t:.1f}s  odom err={err:.2f}m {dth:.1f}deg  -> out/result.png")
        return

    from matplotlib.animation import FuncAnimation
    FRAME = 0.04

    def tick(_):
        s, a = controls()
        sim.step(s, a, FRAME, auto=state["auto"])
        draw()
        return ()

    anim = FuncAnimation(fig, tick, interval=int(FRAME * 1000), cache_frame_data=False)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.show()
    return anim


if __name__ == "__main__":
    main()
