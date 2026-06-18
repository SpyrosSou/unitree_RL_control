# Training Regimes

Current training-process snapshot for the three active policy families in this repo.
Date: 18 June 2026.

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

- Training command distribution is centered on zero velocity.
- Designed to suppress bounce, sway, and jitter rather than produce gait.
- Reward shaping is intentionally different from walking: stronger penalties on vertical body velocity, roll/pitch body rates, and abrupt action changes.
- Feet-air-time gait reward is removed for standing behavior.

Standing with arm-motion disturbance (same task name: `G1-Locomotion-Standing-Flat-v0`):

- The standing policy now trains with random arm trajectories to expose center-of-gravity shifts.
- Arm targets are generated as smooth bounded trajectories, not instantaneous jumps.
- Arm gains are reduced during standing training (`stiffness=25`, `damping=8`) so disturbances evolve over realistic durations.
- Curriculum phases are applied by global step count:
  - Phase 0: no arm motion (baseline standing behavior)
  - Phase 1: moderate motions
  - Phase 2: large motions
  - Phase 3: large + faster (still bounded) + asymmetric + occasional reversals
- Shoulder joints receive larger excursion envelopes than elbow joints so arm raises become more realistic without making elbow behavior too extreme.

Suggested workflow now:

- Train standing for `1000` iterations headless first.
- Then inspect behavior with `play.py` and standing-specific demos.

Suggested walking workflow now:

- Train walking after standing converges so checkpoint updates are staggered.
- Recommended first run is the transition-focused walking task, because it is strongest for switch robustness.

Transition-aware standing refinement (`G1-Locomotion-Standing-Transition-Flat-v0`):

- Mostly zero-command environments plus a small fraction of micro-command environments.
- Current setup is approximately `85%` zero-command and `15%` micro-command exposure.
- Improves robustness when decelerating from walking into a full stop.

Output/checkpoints:

- Base standing logs: `logs/rsl_rl/standing/g1_locomotion_flat/<run>/...`
- Transition standing logs: `logs/rsl_rl/standing/g1_locomotion_flat_transition/<run>/...`

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
