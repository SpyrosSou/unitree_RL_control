# Training Regimes

Current training-process snapshot for the three active policy families in this repo.
Date: 6 July 2026.

This is the single source of truth for *current* per-policy configuration (reward
weights, command ranges, curricula). For the algorithm/architecture that doesn't
change between tuning passes, see `algorithm_explanation.md`. For a running record
of *why* things changed phase-to-phase, see `phase_logs/`. For known limitations and
deferred improvements (not yet acted on), see `known_issues.md`.

## Common Training Setup (All Policies)

- Environment manager: `conda`.
- Conda environment: `isaac_g1_control`.
- Training entrypoint: `scripts/rsl_rl/train.py`.
- Recommended working directory: project root (`~/Elm/Code/g1_locomotion`).
- Checkpoints are written under `logs/rsl_rl/...` with timestamped run folders.

Typical command shape:

```bash
conda activate isaac_g1_control
cd ~/Elm/Code/g1_locomotion
python scripts/rsl_rl/train.py \
  --task <TASK_NAME> \
  --headless \
  --max_iterations <N>
```

## 1. Walking Policy Regime

Primary role:

- Learn velocity-command tracking for locomotion.
- Used as the movement policy in stand/walk switching and locomotion demos.

Main walking training regime (`G1-Locomotion-Flat-v0`):

- Commanded motion is sampled as `[v_x, v_y, w_z]`.
- Command timing is now faster (`resampling_time_range = 0.9 to 2.6 s`) to increase stop/start exposure.
- Forward command range now includes reverse samples (`lin_vel_x = -0.5 to 1.0`).
- Lateral and yaw command ranges are widened (`lin_vel_y = -0.6 to 0.6`, `ang_vel_z = -1.2 to 1.2`).
- A small fraction of walking episodes now uses near-zero/zero commands (`rel_standing_envs = 0.10`) so the policy learns low-speed behavior without becoming a standing policy.
- Reward emphasis remains command tracking plus smooth, stable motion.
- **Left-right symmetry-augmented training (Phase 1 addition)**: `use_data_augmentation=True` via `RslRlSymmetryCfg`, mirroring every observation/action left-right during PPO updates (see `mdp/symmetry.py`). Added after a first Phase-1 walking run converged cleanly by every CSV metric but visually showed one foot lifting much higher than the other — nothing in the reward function constrains gait *shape*, only step timing, so plain PPO was free to settle into an asymmetric habit. See `phase_logs/phase_1.md` for the full investigation.

Transition-focused walking refinement (`G1-Locomotion-Flat-Transition-v0`):

- Adds stronger exposure to start/stop/reverse transitions.
- Uses very fast command resampling windows (`0.5 to 1.6 s`).
- Uses stronger near-zero/zero exposure (`rel_standing_envs = 0.30`).
- Uses wider lateral and yaw ranges (`lin_vel_y = -0.9 to 0.9`, `ang_vel_z = -1.2 to 1.2`).
- Improves stand-to-walk and walk-to-stand handoff quality.

Output/checkpoints:

- Base walking logs: `logs/rsl_rl/legs/g1_locomotion_flat/<run>/...`
- Transition walking logs: `logs/rsl_rl/legs/g1_locomotion_flat_transition/<run>/...`
- Before promoting a checkpoint, run `validation/eval_walking.py` against it (see `validation/README.md`) — a deterministic, fixed-command sweep including a drift check, decoupled from the noisy training curve.

## 2. Standing Policy Regime

Primary role:

- Learn low-motion balance and posture maintenance at (or near) zero command.
- Used as the idle/stop policy in stand/walk switching.

Main standing training regime (`G1-Locomotion-Standing-Flat-v0`):

- Training command distribution is mostly zero velocity, with a small near-zero pulse slice.
- Designed to suppress bounce, sway, and jitter rather than produce gait.
- Reward shaping is intentionally different from walking: stronger penalties on vertical body velocity, roll/pitch body rates, and abrupt action changes.
- Current standing reward weights:
  - `lin_vel_z_l2 = -2.2`
  - `ang_vel_xy_l2 = -0.12`
  - `action_rate_l2 = -0.008`
  - `dof_acc_l2 = -1.5e-7`
  - `feet_air_time = 0.0`
  - `joint_deviation_torso = 0.0` (relaxed from the inherited `-0.1` — experimental, phase-1: the torso needs to be free to move as a balance-compensation DOF when the arm-motion disturbance below shifts the CoG. Watch for excessive torso wobble; this is the first thing to re-tighten if it looks bad.)

Standing arm-motion disturbance curriculum (same task, `G1-Locomotion-Standing-Flat-v0`) — **this is the mechanism that teaches corrective stepping, and it's a phase-1 requirement, not a later add-on**: even though phase 1 doesn't involve a live arm-IK policy, the standing policy needs *some* CoG-shifting stimulus during training or it never has a reason to learn that stepping is sometimes the right recovery action instead of just standing still.

- The standing policy trains with procedurally-generated arm trajectories that shift the center of gravity, as a stand-in for "something arm-like is moving."
- Arm targets are generated as smooth bounded trajectories, not instantaneous jumps.
- Arm gains are reduced during standing training (`stiffness=25`, `damping=8`) so disturbances evolve over realistic durations.
- Curriculum phases are applied by global step count with longer low-speed exposure:
  - Phase 0: no arm motion (`0.00 rad/step`)
  - Phase 1: mild motions (`0.03 rad/step`)
  - Phase 2: moderate motions (`0.05 rad/step`)
  - Phase 3: stronger motions (`0.10 rad/step`)
  - Phase 4: stress-test bursts (`0.25 rad/step`)
  - Phase 5: hardware-limit stress-test (`0.66 rad/step` ≈ 33 rad/s, added 2026-07-07, purely additive on top of phases 0-4 — **not yet trained or visually verified**, see `known_issues.md`)
- Shoulder joints receive larger excursion envelopes than elbow joints so arm raises become more realistic without making elbow behavior too extreme.
- Phase boundaries are in raw env-steps (`env.common_step_counter`), not training iterations — with `num_steps_per_env=24`, iteration N corresponds to env-step `N × 24`. Worth double-checking which phase a checkpoint actually reached rather than assuming from `max_iterations` alone.

**Random push disturbance has been removed** (it was briefly added, then removed before ever being trained into a checkpoint — the arm-motion curriculum above is the intended stimulus for corrective stepping, not an external shove).

Suggested workflow now:

- Train standing for `2000-2500` iterations headless first.
- Then inspect behavior with `play.py` and standing-specific demos.

Suggested walking workflow now:

- Train walking after standing converges so checkpoint updates are staggered.
- Recommended first run is the transition-focused walking task, because it is strongest for switch robustness.

Output/checkpoints:

- Base standing logs: `logs/rsl_rl/standing/g1_locomotion_flat/<run>/...`
- Each standing run also writes `standing_summary.csv` (convergence-at-a-glance: reward, fall/timeout, `step_count`, curriculum phase) and `standing_detailed.csv` (everything — max tilt, minimum height, action extremes, `max_foot_air_time_s`, etc.) in the run folder. See `logging_reference.md` for the full column reference and how to read them.
- Before promoting a checkpoint, run `validation/eval_standing.py` against it (see `validation/README.md`) — a deterministic, fixed-seed sweep across every disturbance difficulty level, including a fall-rate-when-stepped check, decoupled from the noisy training curve.

## 3. Left Arm Control Policy Regime

Primary role:

- Learn arm end-effector reaching to 3D targets.
- Left-arm policy can be mirrored at deployment time for right-arm control.

Main left-arm training regime (`G1-Arm-IK-Left-v0`):

- Arm training is isolated from locomotion (fixed-base setup) so the policy focuses on arm IK behavior.
- Observation and reward focus on target-reaching error, smooth joint behavior, and respecting joint limits.
- Legs/torso/fingers are not commanded during arm-only training and stay near their default pose. Two compute levers are applied: the robot asset uses `G1_MINIMAL_CFG` (fewer collision meshes than `G1_CFG` — same joints, cheaper collision) and the solver iteration counts are reduced (`solver_position_iteration_count=4`, `solver_velocity_iteration_count=1`, down from the locomotion-tuned 8/4) since this task never needs ground-contact resolution.
- Trained with PPO through the same training script interface.
- Arm commands are filtered with a first-order action filter (`action_filter_alpha = 0.25`) to emulate actuator lag and reduce abrupt command changes.
- Per-step arm delta commands are capped (`max_action_delta_per_step = 0.06 rad`) before target application, reducing sudden elbow/shoulder jumps.

**Observation space (28-D per arm, 56-D for `arm="both"` — see `g1_arm_env.py`).** Not to
be confused with degrees of freedom: the robot's actual controllable joints for this task
are 5 per arm (shoulder pitch/roll/yaw, elbow pitch/roll — no wrist), which is the
**action** space size (5-D, or 10-D for both arms). The observation is a separate, larger
set of *input features* the network sees every step, built from:

| Block | Size | What it is |
|---|---|---|
| `base_lin_vel` | 3 | pelvis velocity (Phase 2 addition) |
| `base_ang_vel` | 3 | pelvis rotation rate (Phase 2 addition) |
| `projected_gravity` | 3 | which way is "down" relative to the pelvis (Phase 2 addition) |
| `joint_pos` | 5 | current angle of each of the 5 arm joints |
| `joint_vel` | 5 | velocity of all 5 arm joints (was 3 — shoulders only — until 2026-07-08, see below) |
| `ee_pos` | 3 | palm position in world space |
| `goal` | 3 | target position |
| `error` | 3 | goal − ee_pos (redundant with the two above, makes the reach direction explicit) |

9 (base state) + 19 (arm state) = 28 per arm; `arm="both"` duplicates the whole 28-D
block once per arm → 56. The base-state prefix order matches walking/standing's own
observation convention (see `g1_locomotion/mdp/symmetry.py`'s layout comment).

**`joint_vel` fix (2026-07-08).** Was silently sliced to the first 3 of 5 arm joint
indices (`jt[:3]`) — a pre-existing gap that predates Phase 2, not something Phase 2
introduced. The policy could never see elbow_pitch/elbow_roll velocity. Found while
diagnosing the ~55% success plateau (see `known_issues.md`): a difficulty-bucketed
breakdown of eval failures showed a roughly uniform ~45% failure rate across easy-to-hard
goals rather than concentrated at the hard tail, and failures missed by a real margin
(median 8.5cm at timeout) rather than a near-miss — a pattern more consistent with a
broad control-quality gap than a reachability/control-authority one. Fixed to observe all
5 joints, bumping the per-arm observation 26-D → 28-D (52 → 56 for `arm="both"`).

**Phase 2 additions (2026-07-07 — see `known_issues.md` for what's still unverified):**

- **Base awareness.** Observation grew from 17-D to 26-D per arm (52-D for `arm="both"`, was 34-D): added `base_lin_vel(3) | base_ang_vel(3) | projected_gravity(3)` as a prefix, matching the same convention walking/standing already use. Previously the arm policy had zero awareness of the base at all.
- **Root stays physically fixed — the wobble is synthetic, observation-only.** `_apply_root_wobble` in `g1_arm_env.py` computes a small scripted roll/pitch signal (up to ~5.7° each axis, per-env randomized amplitude/frequency/phase, sampled fresh each reset) and feeds it into what `base_ang_vel`/`projected_gravity` *report*, without moving the robot at all — this is what the new base-awareness observation terms are actually for. **The arm policy itself still never controls the torso/pelvis** — this only changes what the training *environment* tells it about the base, so the arm's own control scope (5/10 joints, unchanged) stays exactly as narrow as before.
  - **Reverted from an earlier, real-motion version** (`fix_root_link=False`, kinematically driving the actual root pose) after testing revealed it turned this into an unintended balance task: with no active balance controller, the robot was standing on nothing but passive leg PD stiffness, and random arm actions (real momentum, unlike zero actions) reliably tipped it over forward. See `known_issues.md`. The synthetic-only version carries zero fall risk since nothing physically moves, at some cost to fidelity (a real tilting base would also affect the arm's own gravity-compensation dynamics slightly; judged an acceptable trade for how small these tilts are).
  - **On a curriculum, not constant-on**: `root_wobble_enable_step` (default 30,000 env-steps, ~25% of the default 5000-iteration budget) — no wobble at all before that, so the policy learns the core reaching skill on an easy/static base first, then generalizes to a seemingly moving one. Same reasoning as standing's arm-disturbance curriculum, just two phases instead of five (this disturbance is far milder).
  - Along the way, also fixed one bug in the (now-reverted) real-motion version found via `scripts/zero_agent.py`: `default_root_state` is in each env's *local* frame, not world frame, so writing it directly without adding `env_origins` snapped every cloned robot to the same world position (all 16 stacked on top of each other). Not relevant to the current synthetic-only design, but worth remembering if real root motion is ever revisited (e.g. with an actual balance controller in the loop).
- **Legs/waist held explicitly, fingers fixed.** Legs+torso ("legs" actuator group) and ankles ("feet" group) are left at `G1_MINIMAL_CFG`'s own stock gains — deliberately verified rather than silently inherited. Separately, a real bug was found and fixed: the old arm-actuator override replaced the "arms" dict key wholesale, and stock `G1_MINIMAL_CFG` bundles every finger joint into that same "arms" group — so fingers had silently lost all actuation (no PD hold at all, fully limp under gravity) since whenever that override was first added. Fingers now get their own modest PD hold.
- **Sim2real.** Previously zero observation noise and zero domain randomization anywhere in this task (a hand-written `DirectRLEnv`, so it inherited none of `G1FlatEnvCfg`'s stock treatment). Added: `Unoise` on every observation term (same magnitudes as `G1FlatEnvCfg`), plus startup randomization of arm actuator gains (`randomize_actuator_gains`, ±20%) and arm joint friction/armature (`randomize_joint_parameters`). Friction/mass/CoM randomization (used by walking/standing) is deliberately *not* included — this task has no contact interactions at all, so it wouldn't reach anything the policy experiences.
- **Symmetry-augmented training.** Every training experience is now also trained on its left-right mirror (`RslRlSymmetryCfg` via `g1_arm/mdp/symmetry.py`, same mechanism as walking's gait-symmetry fix). This doesn't replace the runtime mirror deployment path below — it makes the assumption that path relies on (`policy(mirror(obs)) == mirror(policy(obs))`) a trained-for property instead of an untested one.
- **Self-collision.** `enabled_self_collisions` changed from `False` to `True` (was `False` on every G1 asset variant, project-wide — this fixes the "arm entering torso" interpenetration reported during testing). Applied to all three tasks (arm/standing/walking), not just this one, since it's a shared asset property. Visually confirmed via `scripts/random_agent.py` (2026-07-07): no interpenetration, no explosions. A cheap, safe complement ships alongside it regardless: a soft distance penalty (`torso_proximity_margin_m = 0.12`) discouraging the end-effector from getting within 12 cm of `torso_link`.
- **Joint range restriction — tried, broke reachability, reverted.** `random_agent.py` (2026-07-07) showed the arm occasionally reaching hardware-safe-but-useless poses (arm behind the body, forearm rotated most of the way around) — confirmed against a real Unitree G1 URDF (`g1_29dof.urdf`, found locally) that these are genuine hardware ranges (shoulder_yaw ~300°, elbow_roll/wrist_roll ~226°), not a bug. First fix (`_ARM_JOINT_TASK_RANGES_RAD`, hand-reasoned tighter per-joint bounds applied to reset randomization *and* the action-target clamp) was never verified against actual forward kinematics, and a 1500-iteration test run confirmed it broke reachability: 0% success the entire run, and a distance-to-goal distribution (p0=1.3cm, p50=17.7cm, p90=29cm, flat across all of training) that's the signature of unreachable goals, not slow learning. **Reverted 2026-07-07**: the action-target clamp and joint-limit reward are back to the real hardware range (`self._arm_hw_limits`) — restores the reachability this task had and converged under before. Only reset randomization keeps a narrower spread now (`cfg.reset_range_fraction`, 0.3 → 0.15, same formula as before, smaller fraction) — safe regardless of the exact value since it only affects the starting pose, never what's reachable via actions. The 1500-iteration checkpoint from the broken version should not be used or resumed — see `known_issues.md`.

**Phase 2 continued (2026-07-08/09) — reachability, redundancy, and the final
consolidated policy.** Full blow-by-blow with exact numbers lives in `known_issues.md`
and `phase_logs/phase_2.md`; this is the condensed version:

- **Goal workspace reshaped twice, based on measured kinematic reachability, not
  guessed.** Built `validation/check_arm_reachability.py` (samples random joint configs
  within real hardware limits, no RL involved) and found only ~47% of the original
  `_GOAL_BOUNDS` box was actually reachable within the 2cm threshold. Reshaped the box
  (`x:(0.1,0.5)→(0.20,0.42) y:(0.05,0.45)→(0.08,0.40) z:(0.9,1.2)→(0.9,1.15)`, mirrored
  for the right arm) to remove a genuinely-unreachable far corner and a near corner that
  conflicted with the torso safety margin. **Success rate went from ~55% to ~85.6%** —
  confirms reachability, not policy quality, was the dominant factor the whole time.
- **`elbow_pitch`'s joint-limit reward margin made per-joint, per-bound** — that joint's
  hardware range is heavily asymmetric (barely goes past straight, folds a lot), and a
  flat 5%-of-range margin was penalizing full extension, a normal pose needed for far
  reaches. Fixed to 1% on that joint's lower bound only, 5% everywhere else unchanged.
- **Null-space regularization added**, later made per-joint-weighted (`elbow_pitch` 3x) —
  a real, reproducible improvement to mean/p90 distance, but does not fix a separate,
  deeper issue found via goal-position + joint-config logging: manipulator redundancy
  causes the policy to land on one of two valid solutions for the same goal roughly 20%
  of the time, one meaningfully less reliable (~59% vs ~91% success) than the other, with
  no dependency on goal position. Three targeted reward-shaping fixes all failed to move
  this — likely needs a bigger architectural change (e.g. a more expressive action
  distribution) to actually resolve, not more reward tuning.
- **Final consolidated policy**: wide network (`[512,256,128]` vs. the original
  `[256,128,64]`) was the one change (of three tested — entropy, wide-net, PPO
  hyperparameter tuning) that showed a clear, reproducible improvement (best p90/tail,
  shown twice now on two different baselines) — folded into a final 5000-iteration
  training run (`G1-Arm-IK-Left-WideNet-v0`, `run_name=final_consolidated`). This is the
  checkpoint integration work should use going forward — see `phase_logs/phase_2.md`.

Mirroring / right-arm usage:

- Right-arm behavior can be obtained by mirroring the trained left-arm policy at inference time (`g1_arm_mirror_test.py`) — the mirror math now lives in `g1_arm/mdp/symmetry.py` (imported, not duplicated) so the deployment-time transform can't silently drift out of sync with what training actually used, which is exactly what happened to this script's old hardcoded indices when the observation layout changed above.
- This avoids mandatory retraining for right-arm-only validation.
- A dedicated right-arm policy (`G1-Arm-IK-Right-v0`) is optional when mirrored performance is not sufficient.

Optional extended arm regimes:

- Both-arm task exists (`G1-Arm-IK-Both-v0`) for simultaneous dual-arm control.
- Can be continued/resumed to improve reaching precision over longer training horizons.

Output/checkpoints:

- Left-arm logs: `logs/rsl_rl/arms/g1_arm_ik_left/<run>/...`
- Optional right-arm logs: `logs/rsl_rl/arms/g1_arm_ik_right/<run>/...`
- Optional both-arm logs: `logs/rsl_rl/arms/g1_arm_ik_both/<run>/...`

## How Policies Are Combined at Runtime

- Standing + walking are combined via policy switching for stop/move behavior.
- Left-arm policy runs alongside locomotion stack for manipulation commands.
- Left-arm checkpoints may be mirrored for right-arm commands when needed.
- Integrated demos load policy paths from checkpoint YAML files:
  - `testing/walking_testing/checkpoints.yaml`
  - `testing/arm_testing/checkpoints.yaml`
  - `testing/general_testing/checkpoints.yaml`

## Notes

- This document captures the current process description and the new standing arm-motion update.
- Further robustness experiments can be layered on top of this baseline.
