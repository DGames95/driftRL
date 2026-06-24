"""Lateral step-input test: steady speed, straight line, step the reference.

Unlike evaluate.py (full track rollouts), this isolates the steering loop's
transient response: drive at a constant speed with zero track curvature and
apply a step change in the lateral reference (left, then back right), then
plot the commanded steering angle and the car's positional reaction.

DriftEnv (track_type="free") supplies vehicle dynamics only; the track-frame
error fed to the controller is synthesized directly from the step reference
each tick, bypassing env._get_obs()'s track lookup.

Usage:
    python controller_evaluation.py --controller pid
    python controller_evaluation.py --controller pidref --no-show
    python controller_evaluation.py --controller rl --model models/grip_circle/best_model
"""

import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from drift_env import DriftEnv

FIG_DIR = "report/figures"
SETTLE_TOL = 0.1  # [m] band for settle-time reporting


def build_controller(name, env, speed, model_path):
    if name == "pid":
        from controllers.pid import PIDController
        return PIDController(env, v_max=speed)
    if name == "pidref":
        from controllers.reference_pid import PIDREF
        return PIDREF(env, init_v=speed, v_min=speed, v_max=speed)
    if name == "rl":
        from controllers.rl import RLController
        return RLController(model_path)
    raise ValueError(f"unknown controller: {name}")


def run_step_test(controller, env, speed, amplitude, step1_time, step2_time, duration):
    env.reset(seed=0)
    env.state[3] = speed  # force steady forward speed at t=0
    controller.reset()

    n = int(duration / env.DT)
    log = {k: np.zeros(n) for k in
           ("t", "y_ref", "y", "e_y", "e_psi", "delta", "T", "vx")}
    for k in range(n):
        t = k * env.DT
        y_ref = amplitude if step1_time <= t < step2_time else (
            -amplitude if t >= step2_time else 0.0)

        x, y, psi, vx, vy, r = env.state
        e_y = y - y_ref
        e_psi = psi
        raw = np.array([vx, vy, r, e_y, e_psi, 0.0, 0.0, 0.0], dtype=np.float32)
        obs = raw / env.OBS_SCALE

        action = controller.act(obs, {"steer": 0.0, "throttle": 0.0}, env.DT)
        env.step(action)

        log["t"][k] = t
        log["y_ref"][k] = y_ref
        log["y"][k] = y
        log["e_y"][k] = e_y
        log["e_psi"][k] = e_psi
        log["delta"][k] = action[0]
        log["T"][k] = action[1]
        log["vx"][k] = vx
    return log


def settle_time(t, e_y, win_start, win_end, tol=SETTLE_TOL):
    """Time after win_start at which |e_y| first stays within tol for the
    rest of [win_start, win_end)."""
    mask = (t >= win_start) & (t < win_end)
    t_win, e_win = t[mask], np.abs(e_y[mask])
    inside = e_win <= tol
    for i in range(len(inside)):
        if inside[i:].all():
            return t_win[i] - win_start
    return float("nan")


def diagnostics(log, controller_name, step1_time, step2_time):
    t, e_y = log["t"], log["e_y"]
    duration_end = t[-1] + (t[1] - t[0] if len(t) > 1 else 0.0)
    windows = (("step 1 (left)", step1_time, step2_time),
               ("step 2 (right)", step2_time, duration_end))
    for label, win_start, win_end in windows:
        mask = (t >= win_start) & (t < win_end)
        seg = e_y[mask]
        peak = np.abs(seg).max() if len(seg) else float("nan")
        ts = settle_time(t, e_y, win_start, win_end)
        print(f"[{controller_name}] {label}: peak |e_y| = {peak:.2f} m, "
              f"settle time (+-{SETTLE_TOL} m) = {ts:.2f} s")
    if controller_name == "pidref":
        print("[pidref] note: PIDREF has no e_y/e_psi feedback, so its "
              "steering is expected to show ~no reaction to the step.")


def plot_step_response(log, controller_name, step1_time, step2_time):
    os.makedirs(FIG_DIR, exist_ok=True)
    t = log["t"]

    fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    axs[0].plot(t, log["y_ref"], "k--", lw=1, label=r"$y_{ref}$")
    axs[0].plot(t, log["y"], lw=1.2, label=r"$y$")
    axs[0].set_ylabel("lateral pos. [m]")
    axs[0].legend()

    axs[1].plot(t, log["e_y"], lw=1.2)
    axs[1].axhline(0, color="k", lw=0.5)
    axs[1].set_ylabel(r"$e_y$ [m]")

    axs[2].plot(t, np.degrees(log["delta"]), label=r"$\delta$ [deg]")
    axs[2].plot(t, 10 * log["T"], label=r"$10\,T$")
    axs[2].set_xlabel("t [s]")
    axs[2].set_ylabel("control")
    axs[2].legend(ncol=2, fontsize=8)

    for a in axs:
        a.grid(alpha=0.3)
        a.axvline(step1_time, color="r", lw=0.8, alpha=0.5)
        a.axvline(step2_time, color="r", lw=0.8, alpha=0.5)

    title = f"Lateral step response: {controller_name}"
    if controller_name == "pidref":
        title += " (no e_y feedback)"
    axs[0].set_title(title)
    fig.tight_layout()
    path = f"{FIG_DIR}/{controller_name}_step_response.pdf"
    fig.savefig(path)
    print(f"Saved {path}")
    return fig


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--controller", choices=["pid", "pidref", "rl"], default="pid")
    p.add_argument("--model", default="models/grip_random/best_model")
    p.add_argument("--speed", type=float, default=15.0)
    p.add_argument("--amplitude", type=float, default=2.0)
    p.add_argument("--step1-time", type=float, default=2.0)
    p.add_argument("--step2-time", type=float, default=6.0)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    env = DriftEnv(mode="grip", track_type="free")
    controller = build_controller(args.controller, env, args.speed, args.model)

    log = run_step_test(controller, env, args.speed, args.amplitude,
                         args.step1_time, args.step2_time, args.duration)
    diagnostics(log, args.controller, args.step1_time, args.step2_time)
    plot_step_response(log, args.controller, args.step1_time, args.step2_time)
    if not args.no_show:
        plt.show()
