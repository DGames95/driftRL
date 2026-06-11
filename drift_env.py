"""DriftEnv: 3-DOF single-track (bicycle) model on a circular or random track.

Observation: [v_x, v_y, r, e_y, e_psi, k0, k10, k25]  (normalized)
             k* = track curvature 0 / 10 / 25 m ahead along the centerline
Action:      [delta, T]  steering in [-0.5, 0.5] rad, throttle in [-1, 1]
Integration: explicit Euler, dt = 0.02 s
Tire model:  F_y = -Fy_max * tanh(C_alpha * alpha / Fy_max), with a
             friction-ellipse coupling on the (driven) rear axle:
             longitudinal force use reduces the lateral saturation limit.

Reward modes:
  "drift"  rewards slip angle  -> agent should drift
  "grip"   penalizes slip angle -> agent should lap at the traction limit
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from track import Track


class DriftEnv(gym.Env):
    metadata = {"render_modes": []}

    # --- vehicle parameters (mid-size RWD car) ---
    M = 1300.0        # mass [kg]
    IZ = 1800.0       # yaw inertia [kg m^2]
    LF = 1.2          # CoG -> front axle [m]
    LR = 1.4          # CoG -> rear axle [m]
    CA_F = 90000.0    # front cornering stiffness [N/rad]
    CA_R = 90000.0    # rear cornering stiffness [N/rad]
    MU = 0.9          # friction coefficient
    G = 9.81
    F_DRIVE_MAX = 8000.0   # max rear longitudinal force demand (|T| = 1) [N]
    C_DRAG = 1.0           # quadratic drag coeff [N s^2/m^2]

    DT = 0.02
    MAX_STEPS = 1000       # 20 s episode

    # reward weights
    W_BETA_DRIFT = 3.0     # drift mode: reward |beta|
    W_BETA_GRIP = 50.0     # grip mode: penalize beta^2
    W_EY = 0.5
    W_DDOT = 0.002
    TERM_PENALTY = 400.0   # must outweigh speed reward accumulated before dying

    OBS_SCALE = np.array([20.0, 10.0, 2.0, 4.0, np.pi, 0.05, 0.05, 0.05],
                         dtype=np.float32)

    def __init__(self, mode="drift", track_type="circle", circle_radius=30.0):
        super().__init__()
        assert mode in ("drift", "grip") and track_type in ("circle", "random")
        self.mode = mode
        self.track_type = track_type
        self.circle_radius = circle_radius
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0], dtype=np.float32),
            high=np.array([0.5, 1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        # axle vertical loads (static) set the tire saturation limits
        L = self.LF + self.LR
        self.FY_MAX_F = self.MU * self.M * self.G * self.LR / L
        self.FY_MAX_R = self.MU * self.M * self.G * self.LF / L
        self.track = None
        self.state = None       # [x, y, psi, vx, vy, r] global pose + body velocities
        self.prev_delta = 0.0
        self.track_idx = 0
        self.steps = 0

    def _get_obs(self):
        x, y, psi, vx, vy, r = self.state
        e_y, e_psi, kprev, self.track_idx = self.track.frame(x, y, psi, self.track_idx)
        raw = np.concatenate([[vx, vy, r, e_y, e_psi], kprev]).astype(np.float32)
        return raw / self.OBS_SCALE, e_y, e_psi

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.track_type == "circle":
            if self.track is None:
                self.track = Track.circle(self.circle_radius)
        else:
            self.track = Track.random_track(self.np_random)  # new layout each episode
        v0 = 8.0 + self.np_random.uniform(-1.0, 1.0)
        psi0 = self.track.psi[0] + self.np_random.uniform(-0.05, 0.05)
        x0, y0 = self.track.xy[0]
        self.state = np.array([x0, y0, psi0, v0, 0.0, 0.0])
        self.prev_delta = 0.0
        self.track_idx = 0
        self.steps = 0
        obs, _, _ = self._get_obs()
        return obs, {}

    def step(self, action):
        delta = float(np.clip(action[0], -0.5, 0.5))
        T = float(np.clip(action[1], -1.0, 1.0))
        x, y, psi, vx, vy, r = self.state

        # slip angles (vx kept > 0.5 inside arctan to avoid blow-up near standstill)
        vx_safe = max(vx, 0.5)
        alpha_f = np.arctan2(vy + self.LF * r, vx_safe) - delta
        alpha_r = np.arctan2(vy - self.LR * r, vx_safe)

        # rear longitudinal tire force, capped at the friction limit (wheelspin)
        Fx_r = np.clip(T * self.F_DRIVE_MAX, -self.FY_MAX_R, self.FY_MAX_R)
        # friction ellipse: longitudinal use shrinks the rear lateral limit,
        # so full throttle kicks the tail out (power-oversteer)
        fy_max_r_eff = self.FY_MAX_R * np.sqrt(max(1.0 - (Fx_r / self.FY_MAX_R) ** 2, 1e-3))

        # lateral tire forces: linear with tanh saturation
        Fyf = -self.FY_MAX_F * np.tanh(self.CA_F * alpha_f / self.FY_MAX_F)
        Fyr = -fy_max_r_eff * np.tanh(self.CA_R * alpha_r / fy_max_r_eff)

        # total longitudinal force: rear tire force + quadratic drag
        Fx = Fx_r - self.C_DRAG * vx * abs(vx)

        # 3-DOF body-frame dynamics
        vx_dot = (Fx - Fyf * np.sin(delta)) / self.M + r * vy
        vy_dot = (Fyf * np.cos(delta) + Fyr) / self.M - r * vx
        r_dot = (self.LF * Fyf * np.cos(delta) - self.LR * Fyr) / self.IZ

        # global kinematics
        x_dot = vx * np.cos(psi) - vy * np.sin(psi)
        y_dot = vx * np.sin(psi) + vy * np.cos(psi)
        psi_dot = r

        # explicit Euler
        self.state = self.state + self.DT * np.array([x_dot, y_dot, psi_dot, vx_dot, vy_dot, r_dot])
        self.steps += 1
        vx, vy = self.state[3], self.state[4]

        obs, e_y, e_psi = self._get_obs()
        beta = np.arctan2(vy, max(vx, 0.5))
        delta_dot = (delta - self.prev_delta) / self.DT
        reward = (vx * np.cos(e_psi)
                  - self.W_EY * e_y ** 2
                  - self.W_DDOT * delta_dot ** 2)
        if self.mode == "drift":
            reward += self.W_BETA_DRIFT * abs(beta)
        else:
            reward -= self.W_BETA_GRIP * beta ** 2
        self.prev_delta = delta

        terminated = bool(abs(e_y) > self.track.half_width or vx < 1.0)
        if terminated:
            reward -= self.TERM_PENALTY
        finished = self.track.at_end(self.track_idx)
        truncated = self.steps >= self.MAX_STEPS or finished

        info = {"x": self.state[0], "y": self.state[1], "psi": self.state[2],
                "vx": vx, "vy": vy, "r": self.state[5],
                "e_y": e_y, "e_psi": e_psi, "beta": beta,
                "delta": delta, "T": T, "finished": finished}
        return obs, float(reward), terminated, truncated, info


def analytic_grip_limit(radius, env_cls=DriftEnv):
    """Max steady-state cornering speed on a circle of given radius.

    At the limit both axles saturate (static loads make this moment-balanced);
    the rear must also supply the drag force through the friction ellipse:
        (m v^2 / R)^2 = (mu m g)^2 - (c_d v^2 L / l_f)^2
    """
    e = env_cls
    L = e.LF + e.LR
    num = (e.MU * e.M * e.G) ** 2
    den = (e.M / radius) ** 2 + (e.C_DRAG * L / e.LF) ** 2
    return (num / den) ** 0.25
