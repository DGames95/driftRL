"""RL controller: a trained PPO model drives. Ignores the human intent."""

from .base import Controller


class RLController(Controller):
    name = "rl"

    def __init__(self, env, model_path=None, **kwargs):
        from stable_baselines3 import PPO
        if model_path is None:
            model_path = f"models/{env.mode}_{env.track_type}/best_model"
        self.model = PPO.load(model_path, device="cpu")

    def act(self, obs, intent, dt):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
