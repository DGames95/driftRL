"""Warm-start PPO on the domain-randomized DriftEnv for sim-to-real transfer.

Takes a policy already good on the nominal dynamics (default:
models/drift_random_better) and continues training it on
DomainRandomizedDriftEnv, which redraws the vehicle/actuation/sensing
parameters every episode (see domain_rand.py). The result stays fast and
stable across the whole parameter distribution, so the nominal BeamNG ETK 800
sits well inside its training support and transfers without per-car tuning.

Usage:
    python train_dr.py                       # defaults below
    python train_dr.py --steps 1500000
    python train_dr.py --init-from models/drift_random_longer_safer_more/best_model
Output: models/drift_dr_sim2real/{best_model,final_model}.zip

Only the env and a few warm-start hyperparameters differ from train.py; the
logging / eval / plotting machinery is imported from it unchanged.
"""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_latest_run_id

from domain_rand import DomainRandomizedDriftEnv
from train import plot_training

# --- defaults, edit here rather than retyping flags ---
DEFAULT_MODE = "drift"
DEFAULT_TRACK = "random"
DEFAULT_STEPS = 1_200_000
DEFAULT_INIT_FROM = "models/drift_random_better/best_model"
DEFAULT_TAG = "drift_dr_sim2real"
# Gentler LR than a from-scratch run: we are adapting a good policy to a wider
# distribution, not learning the task, so big steps would forget what works.
DEFAULT_LR = 1.5e-4
DEFAULT_GAMMA = 0.99
N_ENVS = 8


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["drift", "grip"], default=DEFAULT_MODE)
    p.add_argument("--track", choices=["circle", "random"], default=DEFAULT_TRACK)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--init-from", default=DEFAULT_INIT_FROM,
                   help="saved PPO .zip to warm-start from")
    p.add_argument("--tag", default=DEFAULT_TAG,
                   help="models/<tag> output directory")
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    p.add_argument("--no-randomize", action="store_true",
                   help="disable domain randomization (debug: behaves like train.py)")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    tag = args.tag
    randomize = not args.no_randomize

    def make_env():
        return DomainRandomizedDriftEnv(
            mode=args.mode, track_type=args.track,
            retry_on_crash=True, randomize=randomize,
        )

    # parallel randomized training envs; eval on a single randomized env so the
    # reported reward reflects performance across the reality-gap distribution,
    # not just the nominal car.
    env = make_vec_env(make_env, n_envs=N_ENVS, seed=0)
    eval_env = Monitor(DomainRandomizedDriftEnv(
        mode=args.mode, track_type=args.track, randomize=randomize))

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"models/{tag}",
        log_path=f"models/{tag}",
        eval_freq=10_000,
        n_eval_episodes=10,   # more episodes: eval variance is higher under DR
        deterministic=True,
        verbose=1,
    )

    print(f"Warm-starting from {args.init_from}  (randomize={randomize})")
    model = PPO.load(
        args.init_from,
        env=env,
        device="cpu",  # MLP policies train faster on CPU than GPU in SB3
        custom_objects={"learning_rate": args.lr, "gamma": args.gamma},
    )

    run_id = get_latest_run_id("logs", tag) + 1
    log_dir = f"logs/{tag}_{run_id}"
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    model.learn(total_timesteps=args.steps, callback=eval_cb)
    model.save(f"models/{tag}/final_model")
    print(f"Done. Best model: models/{tag}/best_model.zip")

    # rollout outcome summary across the parallel envs (same as train.py)
    bases = [e.unwrapped for e in env.envs]
    total = sum(b._ep_count for b in bases)
    if total:
        agg = lambda attr: sum(getattr(b, attr) for b in bases)
        n_finish, n_crash, n_timeout = agg("_n_finish"), agg("_n_crash"), agg("_n_timeout")
        pct = lambda n: f"{n} ({100 * n / total:.1f}%)"
        print("\n--- DR training rollout summary ---")
        print(f"episodes:  {total}")
        print(f"finished:  {pct(n_finish)}")
        print(f"crashed:   {pct(n_crash)}")
        print(f"timed out: {pct(n_timeout)}")

    plot_training(eval_cb, log_dir, tag, show=not args.no_plots)
