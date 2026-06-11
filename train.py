"""Train PPO on DriftEnv.

Usage:
    python train.py --mode drift            # drift agent (default)
    python train.py --mode grip             # grip-limit baseline
    python train.py --mode drift --track random
Models are saved to models/<mode>_<track>/best_model.zip.
"""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from drift_env import DriftEnv

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["drift", "grip"], default="drift")
    p.add_argument("--track", choices=["circle", "random"], default="circle")
    p.add_argument("--steps", type=int, default=500_000)
    args = p.parse_args()

    tag = f"{args.mode}_{args.track}"
    env = Monitor(DriftEnv(mode=args.mode, track_type=args.track))
    eval_env = Monitor(DriftEnv(mode=args.mode, track_type=args.track))

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"models/{tag}",
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
        tensorboard_log="logs",
        device="cpu",  # MLP policies train faster on CPU than GPU in SB3
    )
    model.learn(total_timesteps=args.steps, callback=eval_cb, tb_log_name=tag)
    model.save(f"models/{tag}/final_model")
    print(f"Done. Best model: models/{tag}/best_model.zip")
