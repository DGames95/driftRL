"""Train PPO on DriftEnv and save the best model to models/best_model.zip."""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from drift_env import DriftEnv

TOTAL_TIMESTEPS = 500_000

if __name__ == "__main__":
    env = Monitor(DriftEnv())
    eval_env = Monitor(DriftEnv())

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models",
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
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_cb)
    model.save("models/final_model")
    print("Done. Best model: models/best_model.zip")
