"""Evaluate the trained policy: live 2D animation + post-run diagnostic plots.

Usage:
    python evaluate.py            # live animation + plots
    python evaluate.py --no-anim  # headless: only save plots to report/figures/
"""

import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from stable_baselines3 import PPO

from drift_env import DriftEnv

FIG_DIR = "report/figures"
CAR_L, CAR_W = 4.0, 1.8  # drawn car footprint [m]


def run_episode(model, env, deterministic=True):
    obs, _ = env.reset(seed=0)
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
    R = np.array([[c, -s], [s, c]])
    return base @ R.T + np.array([x, y])


def animate(log, env):
    fig, ax = plt.subplots(figsize=(7, 7))
    th = np.linspace(0, 2 * np.pi, 200)
    for rad in (env.TRACK_R - env.TRACK_HALF_W, env.TRACK_R + env.TRACK_HALF_W):
        ax.plot(rad * np.cos(th), rad * np.sin(th), "k-")
    ax.plot(env.TRACK_R * np.cos(th), env.TRACK_R * np.sin(th), "k--", lw=0.5)
    lim = env.TRACK_R + env.TRACK_HALF_W + 5
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.set_title("DriftEnv rollout (arrow = velocity)")

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
        # velocity vector in global frame
        vgx = vx * np.cos(psi) - vy * np.sin(psi)
        vgy = vx * np.sin(psi) + vy * np.cos(psi)
        arrow.set_position((x, y))
        arrow.xy = (x + 0.5 * vgx, y + 0.5 * vgy)
        txt.set_text(f"t = {i * env.DT:5.2f} s   v = {np.hypot(vx, vy):4.1f} m/s   "
                     f"beta = {np.degrees(beta):+5.1f} deg")
        plt.pause(0.001)
    plt.ioff(); plt.show()


def diagnostics(log, env):
    os.makedirs(FIG_DIR, exist_ok=True)
    t = np.arange(len(log["x"])) * env.DT

    # trajectory
    fig, ax = plt.subplots(figsize=(5, 5))
    th = np.linspace(0, 2 * np.pi, 200)
    for rad in (env.TRACK_R - env.TRACK_HALF_W, env.TRACK_R + env.TRACK_HALF_W):
        ax.plot(rad * np.cos(th), rad * np.sin(th), "k-", lw=0.8)
    ax.plot(log["x"], log["y"], "tab:blue", lw=1)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Trajectory")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/trajectory.pdf"); plt.close(fig)

    # phase portrait v_y vs r
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(log["vy"], log["r"], lw=0.8)
    ax.scatter(log["vy"][0], log["r"][0], c="g", label="start", zorder=3)
    ax.scatter(log["vy"][-1], log["r"][-1], c="r", label="end", zorder=3)
    ax.set_xlabel(r"$v_y$ [m/s]"); ax.set_ylabel(r"$r$ [rad/s]")
    ax.set_title("Phase portrait"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/phase_portrait.pdf"); plt.close(fig)

    # slip angle + states history
    fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    axs[0].plot(t, np.degrees(log["beta"])); axs[0].set_ylabel(r"$\beta$ [deg]")
    axs[1].plot(t, log["vx"], label=r"$v_x$"); axs[1].plot(t, log["vy"], label=r"$v_y$")
    axs[1].set_ylabel("[m/s]"); axs[1].legend()
    axs[2].plot(t, log["e_y"], label=r"$e_y$ [m]")
    axs[2].plot(t, np.degrees(log["delta"]), label=r"$\delta$ [deg]")
    axs[2].set_ylabel(""); axs[2].set_xlabel("t [s]"); axs[2].legend()
    for a in axs:
        a.grid(alpha=0.3)
    axs[0].set_title("Slip angle and state histories")
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/histories.pdf"); plt.close(fig)

    print(f"Episode: {len(t)} steps ({t[-1]:.1f} s), return = {log['reward'].sum():.1f}")
    print(f"mean |beta| = {np.degrees(np.abs(log['beta']).mean()):.1f} deg, "
          f"mean |e_y| = {np.abs(log['e_y']).mean():.2f} m")
    print(f"Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/best_model")
    p.add_argument("--no-anim", action="store_true")
    args = p.parse_args()

    if args.no_anim:
        matplotlib.use("Agg")

    env = DriftEnv()
    model = PPO.load(args.model, device="cpu")
    log = run_episode(model, env)
    diagnostics(log, env)
    if not args.no_anim:
        animate(log, env)
