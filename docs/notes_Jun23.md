## 1. Diagnosis — it was never the physics

`sanity.py` drives the random tracks with a hand-coded P controller
(curvature feedforward + `e_y`/`e_psi` feedback + speed hold): **10/10 tracks,
full 20 s at 12 m/s.** So the dynamics and tracks are drivable; every failure
was the RL setup. Four issues, in order of impact:

1. **Observation scaling used physical *max ranges*, not *operating spread*.**
   Measured operating std vs. the old `OBS_SCALE` divisor:

   | obs | op. std | old scale | normalized std (old) |
   |-----|--------:|----------:|---------------------:|
   | vx    | 0.5  | 20   | 0.025  ← invisible |
   | vy    | 0.11 | 10   | 0.011  ← invisible |
   | r     | 0.28 | 2    | 0.14 |
   | e_y   | 0.13 | 4    | 0.03   ← invisible |
   | e_psi | 0.019| π    | 0.006  ← invisible |
   | κ×3   | 0.024| 0.05 | ~0.48  (only sane one) |

   Most inputs arrive at the MLP with ~0 variance, so the network effectively
   cannot see speed or tracking error. This is the dominant defect.

2. **Returns are huge.** Reward ≈ `vx·cos(e_psi)` ≈ 12 / step × 1000 steps and
   `gamma = 0.999` ⇒ value targets ~12 000. The value network learns these
   slowly/unstably. (PPO normalizes *advantages*, so this mostly hurts the
   *value* fit, but that still stalls learning.)

3. **The reward prefers fast-and-crash.** Surviving at 9 m/s scores
   `1000×9 = 9000`; flooring it to 14 and crashing at step 700 scores
   `700×14 − 400 = 9400`. So a naive optimum *leaves the track*.

4. **PPO overshoots.** Even after it learns to drive, it keeps pushing speed
   (more `vx` = more reward) until it exceeds what it can hold, and survival
   *degrades with more training*.

---

## 2. The recipe (verified, self-contained — `test_simple.py`)

**(a) Fixed a-priori observation normalization** — the key fix. Pick scales so
each obs is ~unit variance over the *driving regime*, and an offset for the one
input that is never near zero (`vx`). You get these once, before training,
either from domain knowledge or from a known-good rollout (we measured the hand
controller, then widened a bit to leave room for RL exploration):

```python
OBS_MEAN  = [11., 0.,  0.,  0.,  0.,   0.,   0.,   0.]   # only vx is offset
OBS_SCALE = [ 4., 1., 0.5, 1.5, 0.3, 0.05, 0.05, 0.05]
obs = (raw - OBS_MEAN) / OBS_SCALE
```
Rationale: vx cruise ~11 ± 4; vy ≤ ~1 (grip); r ≤ ~0.5; e_y ≤ ~1.5 of the 4 m
half-width; e_psi ≤ ~0.3 rad; |κ| ≤ 1/22 ≈ 0.045. These are objective and the
**same for every policy** — that is the whole point.

**(b) Fixed reward scale** `reward_scale = 0.01` ⇒ returns ~O(100). A constant
is enough; it only affects the value fit (PPO already normalizes advantages).

**(c) Best-by-survival checkpointing.** Evaluate on survival *steps*, not
reward, and save the best checkpoint (PPO can still drift later). With proper
obs scaling the policy stayed at 100% and even self-moderated speed down to
~10 m/s, so overshoot was no longer the problem it was under the old scaling.

**(d) Otherwise stock PPO:** `lr 3e-4, gamma 0.999, n_steps 2048, batch 256,
ent 0, net [64,64], 8 envs`. Reaches 100% survival by ~65k steps.

Optional / minor: the over-speed penalty (`v_reward_cap`, `w_overspeed`) and a
larger `term_penalty` are in the code, but in practice they were **nearly
inert** — runs with and without were almost identical, because the penalty only
fires above the cap and stochastic rollouts rarely go that fast. Don't rely on
them; (a)–(c) do the work.


### To adopt the fixed scaling as the project default
Set `OBS_MEAN`/`OBS_SCALE` in `drift_env.py` to the values above and train with
`center_obs=True`. **Coupling:** `../CLAUDE.md` notes that BeamNG's
`track.py:obs()` scaling must equal driftRL's `OBS_SCALE`. If you change these
constants you must update the BeamNG side to match and retrain any policy that
runs there. (We left the defaults alone this session to avoid breaking the
existing drift model.)

---

## 5. Expected changes for drift mode

Drift reward is `+3·|beta|` (reward slip) instead of grip's `−50·beta²`, so the
agent deliberately runs **large slip angles** — the operating regime is very
different and the obs scaling above will be wrong for it:

- **Re-derive `OBS_SCALE` for the drift regime.** `vy`, `r` (and hence `beta`)
  are much larger when sliding. Expect `vy` up to ~5–8 m/s and `r` up to
  ~1–1.5 rad/s, vs ~1 and ~0.5 in grip. Measure them from a drift rollout (or a
  drifting hand/open-loop input) and set e.g. `vy_scale ≈ 5`, `r_scale ≈ 1`.
  `e_y`, `e_psi`, `κ` scales should be similar to grip. Keep the `vx` offset.
  **Use a separate drift `OBS_SCALE`/`OBS_MEAN`** (drift and grip need different
  ones — this is the legitimate, objective kind of per-*regime* scaling, still
  fixed and known before training, not per-policy).
- **Stability is harder.** Drift dynamics are open-loop unstable (a throttle
  stab spins the car — `evaluate.py --instability`). Expect: more timesteps,
  possibly lower `lr`, and the termination penalty + survival checkpointing
  matter more. Some entropy (`ent_coef` ~1e-3) may help exploration without
  wrecking determinism.
- **Speed control.** The fast-and-crash optimum is worse in drift. A cruise
  target (over-speed penalty) is more justified here than it was in grip; tune
  `v_reward_cap` to a drift-feasible speed.
- **Keep** `gamma 0.999` and `reward_scale` (drift adds ≤ ~1.5/step from
  `+3·|beta|` on top of the speed term — similar magnitude, same scaling works).
- **Watch the limit cycle.** The drift agent tends to a pulsed limit cycle with
  steering saturating at ±0.5 rad; raise `W_DDOT` to smooth if needed (per the
  existing project notes).

First test: confirm a drift-tuned `OBS_SCALE` still lets the agent at least
*survive* (drive without leaving the track) before chasing high slip. If it
can't survive with the drift reward, lower the slip weight until it can, then
ramp it back up — survival first, drift second.
