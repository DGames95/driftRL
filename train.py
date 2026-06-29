"""Train PPO on DriftEnv.

Usage:
    python train.py                          # see defaults below
    python train.py --mode grip             # grip-limit baseline
    python train.py --mode drift --track random
    python train.py --no-plots              # skip the post-training figures (for unattended runs)
    python train.py --init-from models/grip_circle/best_model.zip   # warm-start
Models are saved to models/<mode>_<track>/best_model.zip.

All CLI args default to the DEFAULT_* constants below -- edit those instead
of retyping flags every run; pass the flag to override for one run.
"""

import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_latest_run_id

from drift_env import DriftEnv

# --- CLI defaults, edit these directly rather than retyping flags ---
DEFAULT_MODE = "drift"
DEFAULT_TRACK = "random"
DEFAULT_STEPS = 500_000
DEFAULT_NO_PLOTS = False
DEFAULT_INIT_FROM = "models/drift_random_longer_safer_more_short/best_model"  # e.g. "models/grip_circle/best_model" to warm-start
DEFAULT_TAG = "_longer_safer_more_short_short"
N_ENVS = 8  # parallel envs: decorrelates samples -> lower-variance gradients

def plot_training(eval_cb, log_dir, tag, show=True):
    """Plot reward/length/PPO-diagnostic curves, save to models/<tag>/.

    All four panels read the csv SB3 wrote during training. With parallel envs
    the per-env Monitor episode streams interleave, so the rollout rolling means
    (ep_rew_mean / ep_len_mean) are the correctly time-aligned training curves.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)

    log = pd.read_csv(f"{log_dir}/progress.csv")
    x = log["time/total_timesteps"]

    ax = axes[0, 0]
    if "rollout/ep_rew_mean" in log:
        r = log[["time/total_timesteps", "rollout/ep_rew_mean"]].dropna()
        ax.plot(r["time/total_timesteps"], r["rollout/ep_rew_mean"],
                label="train ep_rew_mean (roll 100)")
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

    axl = axes[0, 1]
    if "rollout/ep_len_mean" in log:
        ln = log[["time/total_timesteps", "rollout/ep_len_mean"]].dropna()
        axl.plot(ln["time/total_timesteps"], ln["rollout/ep_len_mean"])
    axl.set_ylabel("episode length (steps)")
    axl.set_xlabel("timesteps")
    axl.set_title("Episode length (rolling mean)")

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
    p.add_argument("--mode", choices=["drift", "grip"], default=DEFAULT_MODE)
    p.add_argument("--track", choices=["circle", "random"], default=DEFAULT_TRACK)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--no-plots", action="store_true", default=DEFAULT_NO_PLOTS,
                   help="skip showing/saving training figures (for unattended runs)")
    p.add_argument("--init-from", default=DEFAULT_INIT_FROM,
                   help="path to a saved model .zip to warm-start the policy/value weights from")
    p.add_argument("--tag", default=None,
                   help="override the models/<mode>_<track><tag> output suffix (default DEFAULT_TAG)")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="PPO discount factor; higher = more far-sighted (values finishing the lap)")
    args = p.parse_args()

    tag = f"{args.mode}_{args.track}"
    if args.tag is not None:
        tag += args.tag
    elif args.init_from:
        tag += DEFAULT_TAG
    # N_ENVS parallel training envs (DummyVecEnv; the env step is cheap NumPy so
    # in-process beats Subproc IPC overhead). Each re-attempts a crashed random
    # track until finished; the seed offset gives every env different layouts.
    # Eval stays a single env on fresh tracks for an unbiased difficulty estimate.
    env = make_vec_env(
        lambda: DriftEnv(mode=args.mode, track_type=args.track, retry_on_crash=True),
        n_envs=N_ENVS, seed=0,
    )
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

    if args.init_from:
        print(f"Warm-starting from {args.init_from}")
        model = PPO.load(
            args.init_from,
            env=env,
            device="cpu",  # MLP policies train faster on CPU than GPU in SB3
            custom_objects={"learning_rate": 3e-4, "gamma": args.gamma},
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            # discount horizon. 0.99 (~2 s) keeps return variance low so the value
            # net fits well; raise (--gamma) to make the agent far-sighted enough to
            # value finishing the lap over grabbing near-term progress and crashing.
            gamma=args.gamma,
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

    # rollout overview: how the agent fared, summed across the parallel envs
    bases = [e.unwrapped for e in env.envs]
    total = sum(b._ep_count for b in bases)
    if total:
        agg = lambda attr: sum(getattr(b, attr) for b in bases)
        n_finish, n_crash, n_timeout = agg("_n_finish"), agg("_n_crash"), agg("_n_timeout")
        n_tracks, n_repeat = agg("_n_tracks"), agg("_n_repeat")
        pct = lambda n: f"{n} ({100 * n / total:.1f}%)"
        print("\n--- training rollout summary ---")
        print(f"episodes:        {total}")
        print(f"finished:        {pct(n_finish)}")
        print(f"crashed:         {pct(n_crash)}")
        print(f"timed out:       {pct(n_timeout)}")
        if n_tracks:
            print(f"tracks drawn:    {n_tracks}  "
                  f"(avg {total / n_tracks:.1f} attempts/track)")
            print(f"repeat attempts: {pct(n_repeat)}")

    plot_training(eval_cb, log_dir, tag, show=not args.no_plots)
