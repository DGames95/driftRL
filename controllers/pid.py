"""PID baseline: grip-style path follower.

Steering = curvature feedforward + feedback on lateral and heading error.
Throttle = P-control toward a grip-limited target speed from the curvature
preview. Ignores the human intent. Gains are starting points; expect to
tune them in-game.
"""

import numpy as np

from drift_env import analytic_grip_limit
from .base import Controller


class PIDController(Controller):
    name = "pid"

    def __init__(self, env,
                 kp_ey=0.10, kp_epsi=0.8, kd_epsi=0.05,
                 kv=0.15, safety=0.8, v_max=22.0, **kwargs):
        self.scale = env.OBS_SCALE
        self.L = env.LF + env.LR
        self.kp_ey, self.kp_epsi, self.kd_epsi = kp_ey, kp_epsi, kd_epsi
        self.kv, self.safety, self.v_max = kv, safety, v_max
        self.prev_epsi = 0.0

    def reset(self):
        self.prev_epsi = 0.0

    def act(self, obs, intent, dt):
        # recover physical units (obs == raw / OBS_SCALE)
        vx, vy, r, e_y, e_psi, k0, k10, k25 = obs * self.scale

        # --- steering: feedforward turn-in + error feedback
        d_epsi = (e_psi - self.prev_epsi) / dt
        self.prev_epsi = e_psi
        delta = (self.L * k0
                 - self.kp_ey * e_y
                 - self.kp_epsi * e_psi
                 - self.kd_epsi * d_epsi)

        # --- throttle: target the grip limit for the tightest upcoming corner
        kappa = max(abs(k0), abs(k10), abs(k25))
        if kappa < 1e-3:
            v_target = self.v_max
        else:
            v_target = min(self.safety * analytic_grip_limit(1.0 / kappa), self.v_max)
        T = self.kv * (v_target - vx)

        return np.array([np.clip(delta, -0.5, 0.5), np.clip(T, -1.0, 1.0)])
