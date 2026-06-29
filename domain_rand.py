"""Domain-randomized DriftEnv for sim-to-real transfer to BeamNG.drive.

The base DriftEnv uses a single fixed parameter set. A policy trained on it
overfits to those exact dynamics and transfers poorly to BeamNG, where the
real vehicle's mass, inertia, tire stiffness, grip, drive force, steering
calibration and sensing all differ — and are only known approximately.

DomainRandomizedDriftEnv samples a fresh parameter set every episode from
ranges centred on a BeamNG RWD sedan (the ETK 800, `VEHICLE_MODEL="etk800"`
in sil_beamng.py) and adds the actuation/sensing imperfections the SIL bridge
exhibits. A policy that stays fast and stable across this whole distribution is
robust to the reality gap, so the *nominal* BeamNG car sits comfortably inside
what it was trained on.

Randomized each reset:
  * physics      M, IZ, CoG split (LF/LR), CA_F, CA_R, MU, F_DRIVE_MAX, C_DRAG
  * actuation    first-order steering & throttle lag (BeamNG steering rack /
                 driveline are not instantaneous), and a steering-scale factor
                 modelling the uncalibrated MAX_STEER_ANGLE in sil_beamng.py
  * sensing      per-step Gaussian observation noise (added to the policy's
                 obs only — true state still drives reward/termination, exactly
                 as a real estimator would feed a noisy obs to the controller)

Everything else (reward shaping, track generation, obs layout/scaling) is
inherited unchanged, so OBS_SCALE stays identical and the warm-started weights
keep their meaning.
"""

import numpy as np

from drift_env import DriftEnv


class DomainRandomizedDriftEnv(DriftEnv):
    # --- nominal parameters, centred on a BeamNG ETK 800 (RWD sedan) ---
    # Ranges are (low, high), sampled uniformly each reset. They are wide on
    # the quantities we know least (tire stiffness, grip, drive force) and
    # tighter on the geometry we can measure (mass, wheelbase).
    PARAM_RANGES = {
        "M":           (1300.0, 1750.0),   # mass [kg]            (etk800 ~1500)
        "IZ":          (1900.0, 2900.0),   # yaw inertia [kg m^2]
        "WHEELBASE":   (2.70, 2.95),       # LF+LR [m]           (etk800 ~2.85)
        "WEIGHT_FR":   (0.50, 0.58),       # front weight fraction -> CoG split
        "CA_F":        (70000.0, 140000.0),# front cornering stiffness [N/rad]
        "CA_R":        (70000.0, 140000.0),# rear cornering stiffness [N/rad]
        "MU":          (0.80, 1.10),       # tire-road friction
        "F_DRIVE_MAX": (6500.0, 11000.0),  # max rear drive force at |T|=1 [N]
        "C_DRAG":      (0.6, 1.8),         # quadratic drag coeff [N s^2/m^2]
    }

    # actuation / sensing imperfection ranges (sampled per episode)
    TAU_STEER    = (0.04, 0.16)   # steering first-order lag [s] (rack dynamics)
    TAU_THROTTLE = (0.08, 0.30)   # throttle/driveline first-order lag [s]
    STEER_SCALE  = (0.85, 1.15)   # applied/commanded road-wheel angle ratio
                                  # (uncalibrated MAX_STEER_ANGLE in sil_beamng)

    # base per-channel obs noise std in *physical* units, layout
    # [vx, vy, r, e_y, e_psi, k0, k10, k25]. Scaled by a per-episode factor.
    OBS_NOISE_BASE = np.array(
        [0.20, 0.20, 0.03, 0.12, 0.015, 0.0, 0.0, 0.0], dtype=np.float32
    )
    OBS_NOISE_SCALE = (0.0, 1.5)  # per-episode multiplier on OBS_NOISE_BASE

    def __init__(self, *args, randomize=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.randomize = randomize
        # actuation filter state and per-episode samples
        self._delta_filt = 0.0
        self._T_filt = 0.0
        self.tau_steer = self.TAU_STEER[0]
        self.tau_throttle = self.TAU_THROTTLE[0]
        self.steer_scale = 1.0
        self.obs_noise = None

    # ------------------------------------------------------------ sampling
    def _sample_params(self):
        """Draw one vehicle/actuation/sensing parameter set and apply it."""
        rng = self.np_random
        u = lambda lo, hi: float(rng.uniform(lo, hi))
        pr = self.PARAM_RANGES

        self.M = u(*pr["M"])
        self.IZ = u(*pr["IZ"])
        L = u(*pr["WHEELBASE"])
        fr = u(*pr["WEIGHT_FR"])
        # CoG split: a heavier front (larger fr) sits the CoG closer to the
        # front axle, i.e. LF is the *rear* weight fraction times wheelbase.
        self.LF = (1.0 - fr) * L
        self.LR = fr * L
        self.CA_F = u(*pr["CA_F"])
        self.CA_R = u(*pr["CA_R"])
        self.MU = u(*pr["MU"])
        self.F_DRIVE_MAX = u(*pr["F_DRIVE_MAX"])
        self.C_DRAG = u(*pr["C_DRAG"])

        # axle vertical loads set the tire saturation limits (as in __init__)
        self.FY_MAX_F = self.MU * self.M * self.G * self.LR / L
        self.FY_MAX_R = self.MU * self.M * self.G * self.LF / L

        # actuation / sensing
        self.tau_steer = u(*self.TAU_STEER)
        self.tau_throttle = u(*self.TAU_THROTTLE)
        self.steer_scale = u(*self.STEER_SCALE)
        self.obs_noise = self.OBS_NOISE_BASE * u(*self.OBS_NOISE_SCALE)

    # ------------------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        # seed the RNG first (super().reset does this) so sampling is reproducible
        super().reset(seed=seed, options=options)  # sets self.state, obs, etc.
        if self.randomize:
            self._sample_params()
        # reset actuation filters to the policy's first command-free state
        self._delta_filt = 0.0
        self._T_filt = 0.0
        # re-derive the (now possibly noisy) first observation under new params
        obs, _, _ = self._get_obs()
        return obs, {}

    def step(self, action):
        # First-order actuation lag: the physics sees a filtered command, not
        # the instantaneous policy output. alpha = dt/(tau+dt) is the stable
        # discrete low-pass coefficient (well-behaved even when tau < dt).
        a = np.asarray(action, dtype=np.float64)
        a_steer = float(np.clip(a[0], -0.5, 0.5))
        a_T = float(np.clip(a[1], -1.0, 1.0))

        alpha_s = self.DT / (self.tau_steer + self.DT)
        alpha_t = self.DT / (self.tau_throttle + self.DT)
        self._delta_filt += alpha_s * (a_steer - self._delta_filt)
        self._T_filt += alpha_t * (a_T - self._T_filt)

        # steering miscalibration: applied road-wheel angle is scaled
        applied = np.array(
            [np.clip(self._delta_filt * self.steer_scale, -0.5, 0.5), self._T_filt]
        )
        return super().step(applied)

    def _get_obs(self):
        """Same obs as DriftEnv, but the policy's copy carries sensor noise.

        Noise is added in physical units *before* scaling and only to the
        returned observation vector — the e_y / e_psi handed back for reward
        and termination stay the true values, mirroring a real setup where a
        noisy estimate feeds the controller while ground truth decides crashes.
        """
        x, y, psi, vx, vy, r = self.state
        e_y, e_psi, kprev, self.track_idx = self.track.frame(x, y, psi, self.track_idx)
        raw = np.concatenate([[vx, vy, r, e_y, e_psi], kprev]).astype(np.float32)
        if self.obs_noise is not None:
            raw = raw + self.np_random.normal(0.0, self.obs_noise).astype(np.float32)
        return (raw / self.OBS_SCALE).astype(np.float32), e_y, e_psi
