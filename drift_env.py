"""DriftEnv: 3-DOF single-track (bicycle) model on a circular track.

State (observation): [v_x, v_y, r, e_y, e_psi]  (normalized before output)
Action:              [delta, T]  steering in [-0.5, 0.5] rad, throttle in [-1, 1]
Integration:         explicit Euler, dt = 0.02 s
Tire model:          F_y = -Fy_max * tanh(C_alpha * alpha / Fy_max), with a
                     friction-ellipse coupling on the (driven) rear axle:
                     longitudinal force use reduces the lateral saturation limit.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


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
    F_DRIVE_MAX = 8000.0   # max rear longitudinal force [N] (|T| = 1)
    C_DRAG = 1.0           # quadratic drag coeff [N s^2/m^2]

    # --- track / episode parameters ---
    TRACK_R = 30.0         # track centerline radius [m]
    TRACK_HALF_W = 4.0     # half track width [m]; |e_y| beyond this terminates
    DT = 0.02
    MAX_STEPS = 1000       # 20 s episode

    # --- reward weights:  R = vx*cos(e_psi) + W1*|beta| - W2*e_y^2 - W3*delta_dot^2 ---
    W1 = 3.0
    W2 = 0.5
    W3 = 0.002

    # observation scales (raw / scale -> roughly [-1, 1])
    OBS_SCALE = np.array([20.0, 10.0, 2.0, 4.0, np.pi], dtype=np.float32)

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0], dtype=np.float32),
            high=np.array([0.5, 1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        # axle vertical loads (static) set the tire saturation limits
        L = self.LF + self.LR
        self.FY_MAX_F = self.MU * self.M * self.G * self.LR / L
        self.FY_MAX_R = self.MU * self.M * self.G * self.LF / L
        self.state = None       # [x, y, psi, vx, vy, r] global pose + body velocities
        self.prev_delta = 0.0
        self.steps = 0

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _wrap(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def _track_errors(self):
        """Signed lateral error and heading error w.r.t. CCW circular centerline."""
        x, y, psi = self.state[0], self.state[1], self.state[2]
        rho = np.hypot(x, y)
        e_y = rho - self.TRACK_R                       # >0: outside the centerline
        psi_track = np.arctan2(y, x) + np.pi / 2.0     # CCW tangent direction
        e_psi = self._wrap(psi - psi_track)
        return e_y, e_psi

    def _get_obs(self):
        vx, vy, r = self.state[3], self.state[4], self.state[5]
        e_y, e_psi = self._track_errors()
        raw = np.array([vx, vy, r, e_y, e_psi], dtype=np.float32)
        return raw / self.OBS_SCALE

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # start on the centerline at angle 0, moving CCW (tangent = +y), small noise
        v0 = 8.0 + self.np_random.uniform(-1.0, 1.0)
        psi0 = np.pi / 2.0 + self.np_random.uniform(-0.05, 0.05)
        self.state = np.array([self.TRACK_R, 0.0, psi0, v0, 0.0, 0.0])
        self.prev_delta = 0.0
        self.steps = 0
        return self._get_obs(), {}

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

        # reward
        e_y, e_psi = self._track_errors()
        beta = np.arctan2(vy, max(vx, 0.5))
        delta_dot = (delta - self.prev_delta) / self.DT
        reward = (vx * np.cos(e_psi)
                  + self.W1 * abs(beta)
                  - self.W2 * e_y ** 2
                  - self.W3 * delta_dot ** 2)
        self.prev_delta = delta

        # termination
        terminated = bool(abs(e_y) > self.TRACK_HALF_W or vx < 1.0)
        if terminated:
            # must clearly outweigh the speed reward accumulated before dying,
            # otherwise PPO learns full-throttle-until-ejected
            reward -= 400.0
        truncated = self.steps >= self.MAX_STEPS

        info = {"x": self.state[0], "y": self.state[1], "psi": self.state[2],
                "vx": vx, "vy": vy, "r": self.state[5],
                "e_y": e_y, "e_psi": e_psi, "beta": beta, "delta": delta, "T": T}
        return self._get_obs(), float(reward), terminated, truncated, info
