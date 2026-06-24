"""RL controller: a trained PPO model drives. Ignores the human intent."""

from .base import Controller


class RLController(Controller):
    name = "rl"

    def __init__(self, model_path):
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path, device="cpu")

    def act(self, obs, intent, dt):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
