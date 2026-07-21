# Quickrun — 23dof branch

Fast commands to get the full interactive demo running. Everything here assumes:

```bash
conda activate isaac_g1_control
cd ~/Elm/Code/g1_locomotion
```

For the full eval-script suite (gate/native/integration evals, arm-only eval, etc.) see
each script's own `--help` — this file is deliberately just the interactive demo.

## Full demo — stand/walk/arm, one command

```bash
python3 testing/general_testing/g1_full_demo.py \
    --standing_checkpoint <path> \
    --walking_checkpoint chosen_checkpoints/walking_latest.pt \
    --arm left --arm_checkpoint chosen_checkpoints/arm_left_latest.pt
```

### Which `--standing_checkpoint` to use

| Checkpoint | Command value | Notes |
|---|---|---|
| Best visual/idle posture | `chosen_checkpoints/best_checkpoints/height_reward_model_5999.pt` | No squat, ~20° torso. **Do not** use this for a live arm-reach demo — collapses 55-70% of the time under real reach. Idle/walking-only is fine. |
| Best for real deployment | `chosen_checkpoints/best_checkpoints/gainmatch_model_5999.pt` | Low fall rate under real reach (0-12.5%). Visible torso twist (60-70°) at rest — known, described, not a stability issue. |
| Torso-clip illustration | `chosen_checkpoints/best_checkpoints/torsoclip30_model_5999.pt` | Needs `--torso_clip_deg 30` (see below) or it's running under the wrong action space. |
| Default pointer | `chosen_checkpoints/standing_latest.pt` | Currently = height_reward. Change intentionally, not by accident — every script reaches for this by default. |

See `chosen_checkpoints/README.md` for the full table and why none of these are a clean
"both good posture and low falls" answer yet.

### `--torso_clip_deg` — only for clip/lock-trained checkpoints

`g1_full_demo.py` builds its env from the walking lineage, which does NOT automatically
inherit a standing checkpoint's own action-space clip. Omitting this flag for a
clip/lock-trained checkpoint silently runs it under an unclipped action space it was
never trained on.

```bash
--torso_clip_deg 30   # for torsoclip30 (best_checkpoints/ or archive/)
--torso_clip_deg 0    # for torsolock (archive/torsolock_model_5999.pt)
                       # omit entirely for height_reward, gainmatch, or anything else
```

### `--wait_arm_rest` — walk-transition behavior

Off by default: switching to walking starts the arm's homing move but doesn't wait for
it to finish (usually fine, homing is fast). Add this flag to instead delay the
stand→walk switch until the arm has fully homed — makes walking feel less responsive but
is the safer choice if you're specifically testing the walk-transition-while-reaching
failure mode:

```bash
--wait_arm_rest
```

### `--arm` — which arm(s) are controllable

`--arm left` (default in practice), `--arm right`, or `--arm both`. Only `left` has a
natively-trained policy; `right` is driven by mirroring the left-trained policy
(known ~8% deviation from true equivariance, see `definitive_next_steps.md`).

### Setting a reach target once it's running

Press **T**, type `x y z` (robot-local frame), Enter:

- left arm: `x ∈ [0.20, 0.42]  y ∈ [0.08, 0.40]  z ∈ [0.9, 1.15]`
- right arm: `x ∈ [0.20, 0.42]  y ∈ [-0.40, -0.08]  z ∈ [0.9, 1.15]`

**Y** = mirrored right-arm target (only when `--arm left`), **U** = one target per arm
simultaneously (only when `--arm left`). **W/A/D/Q/E** = walk, **S** = stop.

## Example: full deployment-candidate demo, both arms tested

```bash
python3 testing/general_testing/g1_full_demo.py \
    --standing_checkpoint chosen_checkpoints/best_checkpoints/gainmatch_model_5999.pt \
    --walking_checkpoint chosen_checkpoints/walking_latest.pt \
    --arm left --arm_checkpoint chosen_checkpoints/arm_left_latest.pt
```
