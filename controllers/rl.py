"""RL controller: a trained PPO model drives. Ignores the human intent."""

from .base import Controller


class RLController(Controller):
    name = "rl"

    def __init__(self, env, model_path=None, **kwargs):
        from stable_baselines3 import PPO
        if model_path is None:
            raise ValueError("RLController requires an explicit model_path "
                              "(pass --model-path; there is no mode/track-based default)")
        print(f"loaded model: {model_path}")
        self.model = PPO.load(model_path, device="cpu")

    def act(self, obs, intent, dt):
        action, _ = self.model.predict(obs, deterministic=True)
        return action
