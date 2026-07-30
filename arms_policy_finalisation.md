# Arms policy finalisation — handoff (2026-07-30)

Pick-up document for finishing the arm policy from a fresh session. Context: the
2026-07-29 static-torque-ceiling fix (`G1-Arm-Left-Integrated-v0`) solved reaching —
what remains is one reward-design retrain, and the deployment wiring. Full history:
`policy_status.md` (arm section, entries dated 2026-07-29/30).

## Where things stand

Checkpoint: `logs/rsl_rl/arms/integrated/2026-07-29_20-27-48/model_7999.pt`
(8000 iters, fresh weights, real 40/10 hardware gain; evals in `arm_eval/` and
`arm_eval_iter2000/` in that run dir).

| Metric (final checkpoint, no_wobble / with_wobble) | Value |
|---|---|
| Reach rate (ever within 2cm) | **100.00% / 100.00%** |
| Min distance: mean / p90 | 0.14 / 0.26 cm |
| Final distance: mean | 1.26 / 1.22 cm |
| Episodes ending < 2cm ("settled") | 81.1% / 81.8% |
| Episodes ending < 3cm | **100% / 100%** (worst of 1093 episodes: 2.76 cm) |
| Per-step time inside the 2cm zone | 79% |
| Legacy "success rate" (15-consecutive-step hold + early termination) | 6.49% / 9.39% — **artifact, see step 1** |

Why the legacy number is low and NOT a capability gap: `terminate_on_success` makes
completing the 15-consecutive-step hold *reward-negative* — termination ends the
50/step `goal_reached_bonus` stream and respawns a distant goal — so the
reward-optimal policy hovers at ~1.1 cm, flickering across the 2 cm line, and rarely
"succeeds." Proof it's the incentive: 2k→8k training improved every real metric
(min dist 0.22→0.14 cm, final dist 1.40→1.26 cm, in-zone 0.71→0.79) while the success
flag FELL 10.8%→6.5%.

Root-cause background (one line): the pre-fix pipeline (`targets = current + delta`,
`|delta| ≤ 0.06`) capped static torque at kp×0.06 = 2.4 Nm vs the 4.5–5.7 Nm gravity
needs — the entire 25–30% plateau era was downstream of that. Probe:
`testing/general_testing/check_arm_static_torque_ceiling.py`.

## Step 1 — make holding reward-optimal, retrain (~one evening)

Goal: the policy should *want* to settle deep and still, instead of dithering at the
2 cm boundary. Two options, either alone is probably enough; (a) is simpler:

- **(a) Train without early termination on success.** New env cfg subclassing
  `G1ArmLeftIntegratedEnvCfg` with `terminate_on_success = False` (field already
  exists — used by the LongHold eval cfg). Episodes always run to timeout, the bonus
  keeps paying while in-zone, so max reward = get in-zone fast and stay. Note
  `Train/mean_episode_length` then stops being the progress signal (always ~max) —
  watch `Episode_Reward/goal_reached_bonus` (max 50/step) and
  `Metrics/frac_envs_reached` instead.
- **(b) Escalate the bonus with the hold counter.** In `_get_rewards`, scale
  `goal_reached_bonus` by e.g. `(1 + hold_counter / goal_hold_steps)` so deeper,
  stiller holds strictly dominate boundary-dithering even with termination kept.
- Optional sharpener for either: pay the in-zone bonus on a slightly tighter radius
  (e.g. 1.5 cm) than the 2 cm success/eval threshold, so the settled position sits
  well inside the zone instead of at its edge.

Retrain fresh, 8000 iters (same as the current run — it was still improving on real
metrics at 8k), then eval:

```bash
python scripts/rsl_rl/train.py --task <NewTaskId> --max_iterations 8000 --headless --seed 42
python validation/eval_arm.py --checkpoint logs/rsl_rl/arms/<exp>/<run>/model_7999.pt --integrated --headless
```

Acceptance bar: **Settled <2cm ≥ 95%** with mean final dist ≤ ~1 cm (current
checkpoint: 81% / 1.26 cm) **and Tail-settle (5s) ≥ ~90%** (the hold-quality check —
see step 2b; not yet measured on the current checkpoint, worth running once as a
baseline before the retrain for comparison). If option (a) alone doesn't push the
settled tail in, add (b). The current model_7999 is already usable as a fallback if
the retrain misbehaves.

**Expected side effect of option (a), checked in the code (2026-07-30) — don't
mistake this for a regression:** with `terminate_on_success=False`, the CSV's
legacy `success` column (`metrics_wrappers.py:1148`) reads
`bool(terminated[env_id])`, and `_get_dones()` only ever sets `terminated =
self.successes` when `terminate_on_success` is True — otherwise it's always
`False`. So this retrain's eval will show **legacy success rate = 0.00% by
construction**, not a rise, regardless of how well the policy actually holds — the
internal hold-counter still runs, it just never reaches the CSV. This is fine and
expected: judge the retrain by the **Settled <2cm/<3cm** and **Tail-settle (Ws)**
columns (step 2b) plus `Episode_Reward/goal_reached_bonus` /
`Metrics/frac_envs_reached` during training.

## Step 2 — eval metrics: DONE (2026-07-30)

`validation/eval_arm.py` now reports two families of metric that don't depend on the
legacy hold-counter/termination path at all:

- **Settled <2cm / Settled <3cm** — fraction of episodes *ending* that close to the
  goal, from the existing `final_dist_to_goal_cm` column. One-instant measure (the
  very last step of the episode).
- **Tail-settle (Ws)** (2b, added same day) — fraction of episodes that stayed under
  `goal_threshold` for the *entire* trailing `W` seconds (`--tail_window_s`, default
  5.0), computed from `ArmMetricsCsvWrapper`'s existing per-second
  `dist_to_goal_cm_t*s` snapshot columns. This is the real "hold capability" answer —
  it catches a policy that's still oscillating right up to the last instant (which
  Settled-<2cm alone wouldn't), without needing the 15-consecutive-*step*
  hold-counter or termination at all, so it works identically whether
  `terminate_on_success` is True or False. Episodes shorter than the requested
  window are excluded from the rate (reported separately as `excl=N` in the cell),
  not silently counted either way. Both are pure CSV post-processing — no env or
  `metrics_wrappers.py` change.

The legacy `success_rate` column is kept (renamed "legacy" in the table, with a
header note explaining the artifact) for comparison against historical numbers.
Judge checkpoints by Settled + Tail-settle + distance stats until step 1's retrain
makes the legacy definition meaningful again.

```bash
# default 5s tail window; raise/lower with --tail_window_s if 5s isn't the right bar
python validation/eval_arm.py --checkpoint <ckpt> --integrated --headless --tail_window_s 5.0
```

## Step 3 — g1_rl_control integration (deployment spec)

The critical rule: **replicate the integrated-target pipeline, NOT the old
`current + delta`** — re-anchoring the target to measured position on hardware
reinstates the 2.4 Nm ceiling and the sag. Everything below is per control tick at
**30 Hz**, gains **kp=40, kd=10, tau_ff=0**, joint order
`[shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]`.

**Action pipeline** (state carried between ticks: `filtered` (7,), `target` (7,);
init at policy engage: `filtered = 0`, `target = measured joint positions`):

```
a_raw    = policy(obs)                          # (7,)
filtered = 0.25 * a_raw + 0.75 * filtered       # action_filter_alpha = 0.25
delta    = clip(filtered * 0.5, -0.06, +0.06)   # action_scale = 0.5, max_delta = 0.06 rad
target   = clip(target + delta, q_lo, q_hi)     # INTEGRATE, clamp to joint limits
send q_des = target with kp=40, kd=10, tau=0
```

**Observation vector — 46-D, in this exact order** (matches
`g1_arm_env.py::_get_observations` with `include_action_feedback=True`,
`integrate_action_targets=True`, `env_local_obs=True`):

| # | Block | Dim | Hardware source |
|---|---|---|---|
| 1 | base_lin_vel | 3 | pelvis linear velocity, base frame (state estimator; zeros acceptable while standing — training saw a synthetic ±0.3 m/s wobble, i.e. it's noise-tolerant here) |
| 2 | base_ang_vel | 3 | IMU gyro (rad/s, base frame) |
| 3 | projected_gravity | 3 | gravity unit vector rotated into base frame from IMU quaternion |
| 4 | joint_pos | 7 | encoders, joint order above (absolute rad, NOT default-relative) |
| 5 | joint_vel | 7 | encoder velocities (rad/s) |
| 6 | ee_pos | 3 | FK position of `left_wrist_yaw_link` in the **local frame** below |
| 7 | goal | 3 | target position, same frame |
| 8 | error | 3 | goal − ee_pos |
| 9 | action_fb | 7 | the `filtered` state from the action pipeline (before scaling) |
| 10 | target_fb | 7 | `target − joint_pos` (the accumulated PD bias) |

**Local frame** (`env_local_obs`): a frame anchored where the robot stands, z=0 at
the floor, axes aligned with the robot's spawn yaw — in sim it's "world minus env
origin". On hardware: an odometry/pelvis-anchored ground frame is the equivalent;
for a standing robot, pelvis-x/y at engage time with z measured from the floor is
fine. Goals were trained in the box x∈[0.20,0.42], y∈[0.08,0.40], z∈[0.90,1.15] (m)
in that frame — command goals inside it.

**Export/config checklist for `g1_rl_control`:**

- `PolicyConfig` expected obs dim: **46** (was 39 for best_combined, 32 for
  pre-action_fb) — update the ONNX wrapper's dim check.
- The actor was trained with `actor_obs_normalization=True` — confirm the ONNX
  export bakes the normalizer in (rsl_rl's exporter wraps it; verify output stats on
  a recorded sim obs trace, not just shape).
- Add the two new pipeline states (`target` integration + `target_fb` obs) — the
  existing EMA/`action_fb` logic carries over unchanged.
- kp/kd are set at deploy time via the SDK; 40/10 is what this checkpoint trained
  against — deploy with exactly that.
- Sanity check before the real robot: run the exported ONNX against
  `G1-Arm-Left-Integrated-Play-v0` obs (sim-in-the-loop) and confirm identical
  behavior to the .pt checkpoint.

## Known secondary items (not blockers, in priority order)

1. **Passive waist/torso lean** — ~10–14 cm of ee displacement from waist joints
   sagging under the arm's weight (probe-measured; same ~20° waist_pitch creep as
   the 2026-07-27 finding). The policy compensates since it sees true ee position,
   but a torso-hold term or gravity compensation on the waist would remove wasted
   workspace. On hardware the walking policy holds the waist — behavior will differ
   from the fixed-base sim here; check early.
2. **Symmetry mirror validation** — `mdp/symmetry.py`'s 7-DOF sign convention was
   adapted by pattern, never empirically mirror-tested; required before trusting a
   mirrored right arm (`policy_status.md`, "Symmetry mirror convention").
3. **Right arm / both arms** — no native checkpoints exist; mirror-driven only.
4. **Orientation goals (6-DOF)** — position-only today; wrist DOF are ready for it.
5. **Distance-adaptive speed for hardware UX** — deferred item 12; can be done
   deployment-side (rate shaping in `g1_rl_control`) without retraining.

## Pointers

- `policy_status.md` — arm section, 2026-07-29 "ROOT CAUSE FOUND" + 2026-07-30
  outcome entries (full evidence chain and numbers).
- `testing/general_testing/check_arm_static_torque_ceiling.py` — the no-RL probe
  that confirmed the torque ceiling; rerun it if the action pipeline ever changes.
- Lesson worth carrying to any future task: before ablating RL hyperparameters,
  verify the action interface can physically produce the torques the task needs —
  reward curves cannot distinguish "hard to learn" from "physically impossible."
