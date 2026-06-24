# Training Regimes

Current training-process snapshot for the three active policy families in this repo.
Date: 24 June 2026.

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

Transition-focused walking refinement (`G1-Locomotion-Flat-Transition-v0`):

- Adds stronger exposure to start/stop/reverse transitions.
- Uses very fast command resampling windows (`0.5 to 1.6 s`).
- Uses stronger near-zero/zero exposure (`rel_standing_envs = 0.30`).
- Uses wider lateral and yaw ranges (`lin_vel_y = -0.9 to 0.9`, `ang_vel_z = -1.2 to 1.2`).
- Improves stand-to-walk and walk-to-stand handoff quality.

Output/checkpoints:

- Base walking logs: `logs/rsl_rl/legs/g1_locomotion_flat/<run>/...`
- Transition walking logs: `logs/rsl_rl/legs/g1_locomotion_flat_transition/<run>/...`

## 2. Standing Policy Regime

Primary role:

- Learn low-motion balance and posture maintenance at (or near) zero command.
- Used as the idle/stop policy in stand/walk switching.

Main standing training regime (`G1-Locomotion-Standing-Flat-v0`):

- Training command distribution is mostly zero velocity, with a small near-zero pulse slice.
- This teaches minimal corrective stepping under disturbance while preserving standing behavior.
- Designed to suppress bounce, sway, and jitter rather than produce gait.
- Reward shaping is intentionally different from walking: stronger penalties on vertical body velocity, roll/pitch body rates, and abrupt action changes.
- Current standing reward weights:
  - `lin_vel_z_l2 = -2.2`
  - `ang_vel_xy_l2 = -0.12`
  - `action_rate_l2 = -0.008`
  - `dof_acc_l2 = -1.5e-7`
  - `feet_air_time = 0.0`

Standing with arm-motion disturbance (same task name: `G1-Locomotion-Standing-Flat-v0`):

- The standing policy now trains with random arm trajectories to expose center-of-gravity shifts.
- Arm targets are generated as smooth bounded trajectories, not instantaneous jumps.
- Arm gains are reduced during standing training (`stiffness=25`, `damping=8`) so disturbances evolve over realistic durations.
- Curriculum phases are applied by global step count with longer low-speed exposure:
  - Phase 0: no arm motion (`0.00 rad/step`)
  - Phase 1: mild motions (`0.03 rad/step`)
  - Phase 2: moderate motions (`0.05 rad/step`)
  - Phase 3: stronger motions (`0.10 rad/step`)
  - Phase 4: stress-test bursts (`0.25 rad/step`)
- Shoulder joints receive larger excursion envelopes than elbow joints so arm raises become more realistic without making elbow behavior too extreme.

Standing with random push disturbance (same task name: `G1-Locomotion-Standing-Flat-v0`):

- Training now includes interval push disturbances implemented as random root-velocity perturbations.
- Push intensity is intentionally decoupled from arm phase.
- Each push interval samples a mixed distribution (mostly easy/moderate, some hard outliers), so policy training includes all arm/push combinations.
- Push direction is a random planar vector (not fixed perpendicular), with optional yaw perturbation.
- A warm-up period is kept before pushes activate.

Suggested workflow now:

- Train standing for `2000-2500` iterations headless first.
- Then inspect behavior with `play.py` and standing-specific demos.

Suggested walking workflow now:

- Train walking after standing converges so checkpoint updates are staggered.
- Recommended first run is the transition-focused walking task, because it is strongest for switch robustness.

Output/checkpoints:

- Base standing logs: `logs/rsl_rl/standing/g1_locomotion_flat/<run>/...`
- Each standing run also writes `standing_metrics.csv` in the run folder with per-episode rows for fall / timeout, max tilt, minimum height, reward return, action extremes, and push disturbance metrics (`max_push_lin_speed_m_s`, `max_push_yaw_speed_rad_s`).

## 3. Left Arm Control Policy Regime

Primary role:

- Learn arm end-effector reaching to 3D targets.
- Left-arm policy can be mirrored at deployment time for right-arm control.

Main left-arm training regime (`G1-Arm-IK-Left-v0`):

- Arm training is isolated from locomotion (fixed-base setup) so the policy focuses on arm IK behavior.
- Observation and reward focus on target-reaching error, smooth joint behavior, and respecting joint limits.
- Legs remain in a neutral/support role during arm-only training.
- Trained with PPO through the same training script interface.
- Arm commands are now filtered with a first-order action filter (`action_filter_alpha = 0.25`) to emulate actuator lag and reduce abrupt command changes.
- Per-step arm delta commands are capped (`max_action_delta_per_step = 0.06 rad`) before target application, reducing sudden elbow/shoulder jumps.

Mirroring / right-arm usage:

- Right-arm behavior can be obtained by mirroring the trained left-arm policy at inference time (`g1_arm_mirror_test.py`).
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
  - `walking_testing/checkpoints.yaml`
  - `arm_testing/checkpoints.yaml`
  - `general_testing/checkpoints.yaml`

## Notes

- This document captures the current process description and the new standing arm-motion update.
- Further robustness experiments can be layered on top of this baseline.
