# Drift Toy: RL + a playable 2D drifting game

A small RL project around a 3-DOF single-track (bicycle) car model whose
rear tire has a friction-ellipse coupling — throttle reduces rear lateral
grip, so power-oversteer (and therefore drifting) is real in this model.
Two PPO agents are trained: a **grip** agent that should lap at the
analytic traction limit (a physics sanity check), and a **drift** agent
rewarded for slip angle. A pygame chase-view game lets you drive the same
physics yourself.

## Files

| File | Purpose |
|---|---|
| `track.py` | Track geometry: circle or random bounded-curvature tracks, track-frame errors |
| `drift_env.py` | `DriftEnv` (`gymnasium.Env`): physics, reward modes `grip`/`drift`, + `analytic_grip_limit()` |
| `train.py` | PPO training; saves `models/<mode>_<track>/best_model.zip` |
| `controllers/` | Swappable drivers sharing one interface: `keyboard`, `rl` (trained PPO), `pid` (classical path follower) |
| `evaluate.py` | Rollout of a controller: live animation, diagnostic figures, instability demo |
| `game.py` | Playable pygame game (`--controller keyboard/rl/pid`, or `--demo` for rl) |
| `report/report.tex` | LaTeX report (model → grip baseline → drift agent → game) |

## Setup

Everything runs in the `gncnet` conda environment
(gymnasium, stable-baselines3, torch, matplotlib, pygame):

```bash
conda activate gncnet
```

## Play the game

```bash
python game.py                          # circular track, keyboard
python game.py --track random           # random track (new layout each restart)
python game.py --track random --seed 7  # reproducible layout
python game.py --controller pid         # watch the PID baseline
python game.py --demo                   # watch the trained drift agent
```

Arrow keys drive (left/right steer, up/down throttle/brake) through a
first-order lag, so holding a key ramps the input rather than snapping it.
`R` restarts, `ESC` quits. The camera points along the velocity vector, so
the visual angle between the car body and "up" is the slip angle. A summary
(return, slip angle, e_y, vx) prints to the console each time a run ends.

## Train and evaluate

```bash
python train.py --mode grip      # traction-limit baseline (~10 min, CPU)
python train.py --mode drift     # drift agent
python evaluate.py --mode grip --no-anim    # figures + stats, headless
python evaluate.py --mode drift             # with live animation
python evaluate.py --controller pid --mode grip --no-anim   # PID baseline
python evaluate.py --instability            # open-loop sensitivity figure
```

`evaluate.py` prints the analytic traction limit next to the grip agent's
achieved speed, and writes all report figures to `report/figures/`.

## Compiling the report

Generate the figures first (`--instability` plus both `evaluate.py` modes),
then:

```bash
cd report
pdflatex report.tex && pdflatex report.tex
```

Output: `report/report.pdf`.
