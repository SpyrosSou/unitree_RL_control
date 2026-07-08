# Quick Run Reference

All commands run from `~/Elm/Code/g1_locomotion` with `conda activate isaac_g1_control`.

---

## Task IDs at a Glance

| ID | Terrain | Use |
|---|---|---|
| `G1-Locomotion-Flat-v0` | Flat | Walking — training |
| `G1-Locomotion-Flat-Play-v0` | Flat | Walking — evaluation/visualisation |
| `G1-Locomotion-Flat-Transition-v0` | Flat | Walking (transition-focused) — training |
| `G1-Locomotion-Flat-Transition-Play-v0` | Flat | Walking (transition-focused) — evaluation |
| `G1-Locomotion-Rough-v0` | Rough | Walking — training |
| `G1-Locomotion-Rough-Play-v0` | Rough | Walking — evaluation |
| `G1-Locomotion-Standing-Flat-v0` | Flat | Standing-only — training |
| `G1-Locomotion-Standing-Flat-Play-v0` | Flat | Standing-only — evaluation |
| `G1-Arm-IK-Left-v0` / `-Play-v0` | — | Left-arm reaching |
| `G1-Arm-IK-Right-v0` / `-Play-v0` | — | Right-arm reaching |
| `G1-Arm-IK-Both-v0` / `-Play-v0` | — | Both-arm reaching |

## Sanity-Check / Utility Scripts

```bash
# List every registered task ID
python scripts/list_envs.py

# Zero-action agent — verifies an env loads without error, no training
python scripts/zero_agent.py --task G1-Locomotion-Flat-v0 --num_envs 4 --headless

# Random-action agent
python scripts/random_agent.py --task G1-Locomotion-Flat-v0 --num_envs 4 --headless
```

---

## Arms — G1 Arm IK Reaching

### Train (visual, small scale — watch the robots)
```bash
python scripts/rsl_rl/train.py --task G1-Arm-IK-Left-v0  --num_envs 4
python scripts/rsl_rl/train.py --task G1-Arm-IK-Right-v0 --num_envs 4
python scripts/rsl_rl/train.py --task G1-Arm-IK-Both-v0  --num_envs 4
```

### Train (headless, full scale)
```bash
python scripts/rsl_rl/train.py --task G1-Arm-IK-Left-v0  --num_envs 4096 --headless
python scripts/rsl_rl/train.py --task G1-Arm-IK-Right-v0 --num_envs 4096 --headless
python scripts/rsl_rl/train.py --task G1-Arm-IK-Both-v0  --num_envs 4096 --headless
```

Logs land in `logs/rsl_rl/arms/g1_arm_ik_left/`, `.../g1_arm_ik_right/`, or `.../g1_arm_ik_both/` respectively.
Each run is timestamped: `YYYY-MM-DD_HH-MM-SS/`. Checkpoints are saved every 100 iterations as `model_N.pt`.

### Resume training from a checkpoint

Pass `--resume` together with `--checkpoint` pointing to any `.pt` file. Training picks up the
weights from that file **and, by default, keeps writing into that same run folder** — new
`model_N.pt` checkpoints, the same `<task>_summary.csv`/`<task>_detailed.csv` pair,
and the same TensorBoard event file all continue in place. `rsl_rl` restores the iteration
counter from the checkpoint (`model_N.pt`'s `"iter"` field), so checkpoint filenames continue
counting up from `N` — nothing from the earlier segment gets overwritten.

```bash
# Resume left arm from a specific checkpoint
python scripts/rsl_rl/train.py --task G1-Arm-IK-Left-v0 --num_envs 4096 --headless \
    --resume --checkpoint logs/rsl_rl/arms/g1_arm_ik_left/<run>/model_3000.pt

# Resume both arms
python scripts/rsl_rl/train.py --task G1-Arm-IK-Both-v0 --num_envs 4096 --headless \
    --resume --checkpoint logs/rsl_rl/arms/g1_arm_ik_both/<run>/model_3000.pt

# Resume legs
python scripts/rsl_rl/train.py --task G1-Locomotion-Flat-v0 --headless \
    --resume --checkpoint logs/rsl_rl/legs/g1_locomotion_flat/<run>/model_1500.pt
```

> **`--max_iterations` on resume is additive, not absolute** — it's how many *more*
> iterations to run from the restored point (`rsl_rl` internally does
> `tot_iter = start_iter + num_learning_iterations`), not a total to stop at. Resuming
> from iteration 150 with `--max_iterations 500` runs to iteration 650.
>
> Want the old behavior (a fresh timestamped folder instead of continuing in place)?
> Add `--resume_new_dir`. `params/env.yaml`/`agent.yaml` are archived with a timestamp
> suffix (not overwritten) each time you resume in place, in case that resume changed
> `--num_envs`/`--max_iterations`/etc. Use `--run_name my_label` as before if you do
> want a distinctly-named new folder.

### Evaluate (generic play — random goal positions)
```bash
python scripts/rsl_rl/play.py --task G1-Arm-IK-Left-Play-v0 \
    --checkpoint logs/rsl_rl/arms/g1_arm_ik_left/<run>/model_5000.pt

python scripts/rsl_rl/play.py --task G1-Arm-IK-Right-Play-v0 \
    --checkpoint logs/rsl_rl/arms/g1_arm_ik_right/<run>/model_5000.pt

python scripts/rsl_rl/play.py --task G1-Arm-IK-Both-Play-v0 \
    --checkpoint logs/rsl_rl/arms/g1_arm_ik_both/<run>/model_5000.pt
```

### Test with specific targets — see `testing/quickrun_tests.md` for full details

```bash
# Left arm
python testing/arm_testing/g1_arm_reach_test.py \
    --arm left \
    --checkpoint logs/rsl_rl/arms/g1_arm_ik_left/<run>/model_5000.pt \
    --targets 0.3 0.2 1.0

# Right arm
python testing/arm_testing/g1_arm_reach_test.py \
    --arm right \
    --checkpoint logs/rsl_rl/arms/g1_arm_ik_right/<run>/model_5000.pt \
    --targets 0.3 -0.2 1.0

# Both arms (y is auto-mirrored for the right arm)
python testing/arm_testing/g1_arm_reach_test.py \
    --arm both \
    --checkpoint logs/rsl_rl/arms/g1_arm_ik_both/<run>/model_5000.pt \
    --targets 0.3 0.2 1.0

# Single-arm checkpoint mirrored to the opposite arm (no retraining)
python testing/arm_testing/g1_arm_mirror_test.py \
    --trained_arm left \
    --active_arm right \
    --checkpoint logs/rsl_rl/arms/g1_arm_ik_left/<run>/model_5000.pt \
    --targets 0.3 -0.2 1.0
```

**Target coordinate reference (robot-local frame, base at origin):**
- x: forward from robot (0.1 – 0.5 m is reachable)
- y: left for left arm (0.05 – 0.45 m), right for right arm (-0.05 – -0.45 m)
- z: height above ground (0.9 – 1.2 m is the trained workspace)

> **Note:** Left, right, and both-arm policies must be trained separately — they have different
> observation/action dimensions and cannot share checkpoints.
>
> **Optional shortcut:** You can train one single-arm policy (e.g. left) and run it on the
> opposite arm with bilateral y-mirroring via `testing/arm_testing/g1_arm_mirror_test.py`.

---

## Legs — G1 Locomotion

### Interactive demo (pre-trained Isaac Lab policy, keyboard control)
```bash
python testing/walking_testing/g1_locomotion_demo.py
# Controls: W=forward  S=stop  A=left  D=right  Q=strafe-left  E=strafe-right  C=camera
```

### Interactive demo with a locally trained checkpoint
```bash
python testing/walking_testing/g1_locomotion_demo.py \
    --checkpoint logs/rsl_rl/legs/g1_locomotion_flat/<run>/model_1500.pt
```

### Train (headless)
```bash
python scripts/rsl_rl/train.py --task G1-Locomotion-Flat-v0 --headless
python scripts/rsl_rl/train.py --task G1-Locomotion-Rough-v0 --headless
```

### Train (visual, small scale)
```bash
python scripts/rsl_rl/train.py --task G1-Locomotion-Flat-v0 --num_envs 4
```

### Evaluate
```bash
python scripts/rsl_rl/play.py --task G1-Locomotion-Flat-Play-v0 \
    --checkpoint logs/rsl_rl/legs/g1_locomotion_flat/<run>/model_1500.pt
```

### Integrated standing + arm control

The full demo keeps locomotion and arm control together, but only activates the
arm policy when you explicitly provide a target. Targets are entered in
robot-local coordinates, not world coordinates.

```bash
# Default: load checkpoints from testing/general_testing/checkpoints.yaml
python testing/general_testing/g1_full_demo.py

# Override checkpoints on the CLI if needed
python testing/general_testing/g1_full_demo.py \
    --standing_checkpoint logs/rsl_rl/standing/g1_locomotion_flat/<run>/model_1499.pt \
    --walking_checkpoint logs/rsl_rl/legs/g1_locomotion_flat/<run>/model_3149.pt \
    --arm_checkpoint logs/rsl_rl/arms/g1_arm_ik_left/<run>/model_4200.pt \
    --arm left

# Start with an explicit robot-local target for the active arm
python testing/general_testing/g1_full_demo.py \
    --arm left \
    --target 0.3 0.2 1.0
```

Controls:
- `W/A/D/Q/E` move the robot
- `S` stops walking
- `T` opens a prompt for a new arm target
- `L/R` switch the active arm when `--arm both`
- `C` toggles the camera

### Standing-only policy (isolated logs)

Uses a separate task family and stores logs under `logs/rsl_rl/standing/`.
This leaves the baseline `legs/` experiments untouched.

```bash
# Train (headless)
python scripts/rsl_rl/train.py --task G1-Locomotion-Standing-Flat-v0 --headless

# Train (visual, small scale)
python scripts/rsl_rl/train.py --task G1-Locomotion-Standing-Flat-v0 --num_envs 4

# Evaluate in play env
python scripts/rsl_rl/play.py --task G1-Locomotion-Standing-Flat-Play-v0 \
    --checkpoint logs/rsl_rl/standing/g1_locomotion_flat/<run>/model_1500.pt

# Interactive demo (auto-picks latest standing checkpoint)
python testing/walking_testing/g1_standing_demo.py

# Stand/walk switch demo (uses dedicated standing + walking checkpoints)
python testing/walking_testing/g1_stand_walk_switch_demo.py

# Transition-focused retraining (better stop/start and reversals)
python scripts/rsl_rl/train.py --task G1-Locomotion-Flat-Transition-v0 --headless

# Play with transition-trained checkpoints
python scripts/rsl_rl/play.py --task G1-Locomotion-Flat-Transition-Play-v0 \
    --checkpoint logs/rsl_rl/legs/g1_locomotion_flat_transition/<run>/model_1500.pt

# Use transition-trained walking checkpoint in switch demo
python testing/walking_testing/g1_stand_walk_switch_demo.py \
    --walking_checkpoint logs/rsl_rl/legs/g1_locomotion_flat_transition/<run>/model_1500.pt \
    --standing_checkpoint logs/rsl_rl/standing/g1_locomotion_flat/<run>/model_1500.pt
```

---

## TensorBoard (both arms and legs)
```bash
tensorboard --logdir logs/rsl_rl/
# open http://localhost:6006
```
