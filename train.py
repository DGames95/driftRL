"""Train PPO on DriftEnv.

Usage:
    python train.py --mode drift            # drift agent (default)
    python train.py --mode grip             # grip-limit baseline
    python train.py --mode drift --track random
    python train.py --no-plots              # skip the post-training figures (for unattended runs)
Models are saved to models/<mode>_<track>/best_model.zip.
"""

import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_latest_run_id

from drift_env import DriftEnv


def plot_training(env, eval_cb, log_dir, tag, show=True):
    """Plot reward/length/PPO-diagnostic curves, save to models/<tag>/."""
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)

    rewards = np.array(env.get_episode_rewards())
    lengths = np.array(env.get_episode_lengths())
    ep_timesteps = np.cumsum(lengths)

    ax = axes[0, 0]
    ax.plot(ep_timesteps, rewards, alpha=0.3, label="episode reward")
    if len(rewards) >= 20:
        window = max(len(rewards) // 50, 20)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(ep_timesteps[window - 1 :], smoothed, label=f"rolling mean ({window})")
    if eval_cb.evaluations_timesteps:
        eval_ts = np.array(eval_cb.evaluations_timesteps)
        eval_mean = np.array([np.mean(r) for r in eval_cb.evaluations_results])
        eval_std = np.array([np.std(r) for r in eval_cb.evaluations_results])
        ax.plot(eval_ts, eval_mean, color="black", marker="o", label="eval mean reward")
        ax.fill_between(eval_ts, eval_mean - eval_std, eval_mean + eval_std, color="black", alpha=0.15)
    ax.set_ylabel("reward")
    ax.set_xlabel("timesteps")
    ax.legend()
    ax.set_title("Training / eval reward")

    axes[0, 1].plot(ep_timesteps, lengths, alpha=0.5)
    axes[0, 1].set_ylabel("episode length (steps)")
    axes[0, 1].set_xlabel("timesteps")
    axes[0, 1].set_title("Episode length")

    # PPO optimizer diagnostics, read back from the csv logger SB3 wrote during training
    log = pd.read_csv(f"{log_dir}/progress.csv")
    x = log["time/total_timesteps"]

    axes[1, 0].plot(x, log["train/explained_variance"])
    axes[1, 0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1, 0].set_ylabel("explained variance")
    axes[1, 0].set_xlabel("timesteps")
    axes[1, 0].set_title("Value function fit (1 = perfect, 0 = no better than baseline)")

    axes[1, 1].plot(x, log["train/approx_kl"])
    axes[1, 1].set_ylabel("approx. KL")
    axes[1, 1].set_xlabel("timesteps")
    axes[1, 1].set_title("Policy update size per rollout")

    fig.suptitle(f"Training diagnostics — {tag}")
    fig.tight_layout()
    fig.savefig(f"models/{tag}/training_curves.png", dpi=150)
    print(f"Saved training curves to models/{tag}/training_curves.png")
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["drift", "grip"], default="drift")
    p.add_argument("--track", choices=["circle", "random"], default="circle")
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--no-plots", action="store_true", help="skip showing/saving training figures (for unattended runs)")
    args = p.parse_args()

    tag = f"{args.mode}_{args.track}"
    env = Monitor(DriftEnv(mode=args.mode, track_type=args.track))
    eval_env = Monitor(DriftEnv(mode=args.mode, track_type=args.track))

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"models/{tag}",
        log_path=f"models/{tag}",
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        gamma=0.999,  # ~20 s horizon; default 0.99 made early termination invisible
        verbose=1,
        device="cpu",  # MLP policies train faster on CPU than GPU in SB3
    )
    # add a csv format alongside the usual stdout/tensorboard logging so PPO's
    # per-rollout diagnostics (explained_variance, approx_kl, ...) can be
    # re-read and plotted after training, not just streamed to tensorboard.
    run_id = get_latest_run_id("logs", tag) + 1
    log_dir = f"logs/{tag}_{run_id}"
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    model.learn(total_timesteps=args.steps, callback=eval_cb)
    model.save(f"models/{tag}/final_model")
    print(f"Done. Best model: models/{tag}/best_model.zip")

    plot_training(env, eval_cb, log_dir, tag, show=not args.no_plots)
