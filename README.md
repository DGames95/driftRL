# 2D Drifting RL Controller

A minimal reinforcement-learning project that trains a PPO agent
(Stable-Baselines3) to drift a vehicle around a circular track. The vehicle
is a 3-DOF single-track (bicycle) model with a tanh-saturated tire model,
integrated with explicit Euler at dt = 0.02 s.

## Files

| File | Purpose |
|---|---|
| `drift_env.py` | `DriftEnv` — custom `gymnasium.Env` (physics, track errors, reward) |
| `train.py` | Trains PPO for 500k steps; saves `models/best_model.zip` via `EvalCallback` |
| `evaluate.py` | Rolls out the trained policy: live animation + diagnostic plots |
| `report/report.tex` | Minimal LaTeX report (model, environment, results) |

## Usage

All commands assume the `gncnet` conda environment:

```bash
conda activate gncnet

# train (≈10 min on CPU; logs to logs/, models to models/)
python train.py

# evaluate with live animation; also saves figures to report/figures/
python evaluate.py

# headless: only generate the figures and episode statistics
python evaluate.py --no-anim
```

## Compiling the report

`evaluate.py` must be run first so that `report/figures/` contains
`trajectory.pdf`, `phase_portrait.pdf`, and `histories.pdf`. Then:

```bash
cd report
pdflatex report.tex
pdflatex report.tex   # second pass resolves figure references
```

Output: `report/report.pdf`.
