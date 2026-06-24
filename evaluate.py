"""Evaluate a controller: live animation + diagnostic plots.

Usage:
    python evaluate.py --mode drift               # live animation + plots
    python evaluate.py --mode grip --no-anim      # headless, plots only
    python evaluate.py --mode drift --track random
    python evaluate.py --controller pid --mode grip --no-anim
    python evaluate.py --instability              # open-loop sensitivity figure
Figures go to report/figures/, prefixed pid_<track> or <mode>_<track> (rl).
"""

import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from drift_env import DriftEnv, analytic_grip_limit
from controllers import make_controller

FIG_DIR = "report/figures"
CAR_L, CAR_W = 4.0, 1.8  # drawn car footprint [m]


class ControllerPolicy:
    """Adapts a controllers.Controller to the model.predict(obs) interface below."""

    def __init__(self, controller, dt):
        self.controller, self.dt = controller, dt

    def reset(self):
        self.controller.reset()

    def predict(self, obs, deterministic=True):
        return self.controller.act(obs, {"steer": 0.0, "throttle": 0.0}, self.dt), None


def run_episode(model, env, seed=0, deterministic=True):
    obs, _ = env.reset(seed=seed)
    if hasattr(model, "reset"):
        model.reset()
    log = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        info["reward"] = reward
        log.append(info)
        done = terminated or truncated
    return {k: np.array([d[k] for d in log]) for k in log[0]}


def car_corners(x, y, psi):
    base = np.array([[CAR_L / 2, CAR_W / 2], [CAR_L / 2, -CAR_W / 2],
                     [-CAR_L / 2, -CAR_W / 2], [-CAR_L / 2, CAR_W / 2]])
    c, s = np.cos(psi), np.sin(psi)
    return base @ np.array([[c, s], [-s, c]]) + np.array([x, y])


def draw_track(ax, track):
    for line in (track.left, track.right):
        pts = np.vstack([line, line[:1]]) if track.closed else line
        ax.plot(pts[:, 0], pts[:, 1], "k-", lw=0.8)
    ax.plot(track.xy[:, 0], track.xy[:, 1], "k--", lw=0.4, alpha=0.5)
    ax.set_aspect("equal")


def animate(log, env):
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_track(ax, env.track)
    ax.set_title("DriftEnv rollout (arrow = velocity)")
    car = Polygon(car_corners(log["x"][0], log["y"][0], log["psi"][0]),
                  closed=True, fc="tab:blue", ec="k")
    ax.add_patch(car)
    trail, = ax.plot([], [], "tab:blue", lw=0.8, alpha=0.5)
    arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="r", lw=1.5))
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9)

    plt.ion(); plt.show()
    for i in range(0, len(log["x"]), 2):  # every 2nd frame -> ~25 fps real-time
        x, y, psi = log["x"][i], log["y"][i], log["psi"][i]
        vx, vy, beta = log["vx"][i], log["vy"][i], log["beta"][i]
        car.set_xy(car_corners(x, y, psi))
        trail.set_data(log["x"][:i], log["y"][:i])
        vgx = vx * np.cos(psi) - vy * np.sin(psi)
        vgy = vx * np.sin(psi) + vy * np.cos(psi)
        arrow.set_position((x, y))
        arrow.xy = (x + 0.5 * vgx, y + 0.5 * vgy)
        txt.set_text(f"t = {i * env.DT:5.2f} s   v = {np.hypot(vx, vy):4.1f} m/s   "
                     f"beta = {np.degrees(beta):+5.1f} deg")
        plt.pause(0.001)
    plt.ioff(); plt.show()


def diagnostics(log, env, prefix):
    os.makedirs(FIG_DIR, exist_ok=True)
    t = np.arange(len(log["x"])) * env.DT

    fig, ax = plt.subplots(figsize=(5, 5))
    draw_track(ax, env.track)
    ax.plot(log["x"], log["y"], "tab:blue", lw=1)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title("Trajectory")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/{prefix}_trajectory.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(log["vy"], log["r"], lw=0.8)
    ax.scatter(log["vy"][0], log["r"][0], c="g", label="start", zorder=3)
    ax.scatter(log["vy"][-1], log["r"][-1], c="r", label="end", zorder=3)
    ax.set_xlabel(r"$v_y$ [m/s]"); ax.set_ylabel(r"$r$ [rad/s]")
    ax.set_title("Phase portrait"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/{prefix}_phase_portrait.pdf"); plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    axs[0].plot(t, np.degrees(log["beta"])); axs[0].set_ylabel(r"$\beta$ [deg]")
    axs[1].plot(t, log["vx"], label=r"$v_x$"); axs[1].plot(t, log["vy"], label=r"$v_y$")
    axs[1].set_ylabel("[m/s]"); axs[1].legend()
    axs[2].plot(t, log["e_y"], label=r"$e_y$ [m]")
    axs[2].plot(t, np.degrees(log["delta"]), label=r"$\delta$ [deg]")
    axs[2].plot(t, 10 * log["T"], label=r"$10\,T$")
    axs[2].set_xlabel("t [s]"); axs[2].legend(ncol=3, fontsize=8)
    for a in axs:
        a.grid(alpha=0.3)
    axs[0].set_title("Slip angle and state histories")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/{prefix}_histories.pdf"); plt.close(fig)

    print(f"[{prefix}] {len(t)} steps ({t[-1]:.1f} s), return = {log['reward'].sum():.1f}, "
          f"finished = {bool(log['finished'][-1])}")
    n4 = len(t) // 4
    print(f"  mean |beta| = {np.degrees(np.abs(log['beta']).mean()):.1f} deg "
          f"(settled {np.degrees(np.abs(log['beta'][n4:]).mean()):.1f}, "
          f"max {np.degrees(np.abs(log['beta']).max()):.1f}), "
          f"mean |e_y| = {np.abs(log['e_y']).mean():.2f} m")
    print(f"  vx settled mean = {log['vx'][n4:].mean():.2f} m/s, max = {log['vx'].max():.2f}")
    if env.mode == "grip" and env.track_type == "circle":
        print(f"  analytic traction limit at R = {env.circle_radius:.0f} m: "
              f"{analytic_grip_limit(env.circle_radius):.2f} m/s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["drift", "grip"], default="drift")
    p.add_argument("--track", choices=["circle", "random"], default="circle")
    p.add_argument("--controller", default="rl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-anim", action="store_true")
    args = p.parse_args()

    if args.no_anim:
        matplotlib.use("Agg")

    env = DriftEnv(mode=args.mode, track_type=args.track)
    controller = make_controller(args.controller, env)
    model = ControllerPolicy(controller, env.DT)
    prefix = (f"{args.mode}_{args.track}" if args.controller == "rl"
              else f"{args.controller}_{args.track}")
    log = run_episode(model, env, seed=args.seed)
    diagnostics(log, env, prefix=prefix)
    if not args.no_anim:
        animate(log, env)
