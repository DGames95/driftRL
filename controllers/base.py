"""Controller base class.

A controller maps the *same* observation the RL agent sees to the *same*
action the env expects, plus an abstracted keyboard intent so a controller
can keep a human in the loop.

Contract
    act(obs, intent, dt) -> np.ndarray([delta, T])
      obs    : normalized 8-vector from DriftEnv._get_obs()
               [vx, vy, r, e_y, e_psi, k0, k10, k25] / OBS_SCALE.
               Recover physical units with  obs * DriftEnv.OBS_SCALE.
      intent : {"steer": float in [-1, 1], "throttle": float in [-1, 1]}
               human input, decoupled from pygame. May be ignored.
      dt     : timestep [s] (env.DT), for controllers that filter/integrate.
    reset() : clear internal state (called on env reset / restart).
"""


class Controller:
    name = "base"

    def reset(self):
        pass

    def act(self, obs, intent, dt):
        raise NotImplementedError
