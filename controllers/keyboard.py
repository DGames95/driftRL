"""Manual keyboard controller: first-order lag on the human steer/throttle
intent, reproducing the original game.py feel."""

import numpy as np

from .base import Controller

TAU_STEER, TAU_THR = 0.12, 0.25   # first-order input lag time constants [s]


class KeyboardController(Controller):
    name = "keyboard"

    def __init__(self, env, **kwargs):
        self.delta = 0.0
        self.T = 0.0

    def reset(self):
        self.delta = 0.0
        self.T = 0.0

    def act(self, obs, intent, dt):
        steer_t = 0.5 * intent["steer"]      # full lock is 0.5 rad
        thr_t = intent["throttle"]
        self.delta += (steer_t - self.delta) * dt / TAU_STEER
        self.T += (thr_t - self.T) * dt / TAU_THR
        return np.array([self.delta, self.T])
