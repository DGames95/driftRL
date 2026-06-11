# Project: 2D Drifting RL Controller

Minimal, self-contained RL project (~30-hour scope): train a PPO agent to
drift a 3-DOF single-track bicycle model around a circular track. Keep
everything lite and simply written — no extra abstraction layers.

## Layout

```
drift_env.py        DriftEnv (gymnasium.Env): physics, track errors, reward
train.py            PPO training, EvalCallback -> models/best_model.zip
evaluate.py         rollout + live matplotlib animation + diagnostic plots
report/report.tex   minimal LaTeX report (pdflatex; figures from evaluate.py)
report/figures/     trajectory.pdf, phase_portrait.pdf, histories.pdf
models/             saved SB3 models (generated)
logs/               tensorboard logs (generated)
train_log.txt       stdout of the last training run (generated)
```

## Environment

Run everything in the `gncnet` conda environment
(gymnasium 1.2.2, stable-baselines3 2.7.1, torch 2.5.1, numpy 2.2.6).
PPO uses `device="cpu"` — MLP policies are faster on CPU in SB3.

## Spec

- Observation: `[v_x, v_y, r, e_y, e_psi]`; action `[delta, T]` with
  delta in [-0.5, 0.5] rad, T in [-1, 1] (asymmetric Box kept per spec,
  despite the SB3 symmetric-action warning).
- Tire model: linear-with-tanh-saturation (no Pacejka), plus a
  friction-ellipse coupling on the rear axle (user-approved extension):
  rear longitudinal force is capped at mu*Fz_r and shrinks the rear lateral
  saturation limit. This is what enables power-oversteer.
- Integration: explicit Euler, dt = 0.02 s.
- Reward: `vx*cos(e_psi) + w1*|beta| - w2*e_y^2 - w3*delta_dot^2`
  with w1=3.0, w2=0.5, w3=0.002, plus a -400 terminal penalty.
- Early termination: |e_y| > 4 m (half-width) or vx < 1 m/s; 1000-step cap.
- PPO MlpPolicy, lr 3e-4, gamma=0.999, otherwise SB3 defaults (no tuning).
  gamma raised from the 0.99 default deliberately: with 0.99 and a small
  terminal penalty, PPO converged to full-throttle-until-ejected (left the
  track at 3.7 s with |beta| < 5 deg). Do not lower gamma back.
- models/v1_no_ellipse/ holds the degenerate first-run models for reference.
- Report must only describe the code and results — no conclusions or
  hypotheses about behaviour.

## Conventions

- Observations are scaled elementwise by `OBS_SCALE = [20, 10, 2, 4, pi]`
  inside `_get_obs`; the report documents this.
- Track is a counter-clockwise circle, radius 30 m, centred at the origin;
  e_y > 0 means outside the centerline.
- `evaluate.py` must be run before compiling the report (it writes the
  figures and prints the episode statistics quoted in the Results section).
