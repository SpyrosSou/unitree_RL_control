# `g1_full_demo.py` — integrated arms + walking demo

Interactive, keyboard-controlled demo combining the unified stand+walk locomotion
policy with a trained arm-reaching policy on the same robot — the closest thing in this
repo to "watch the actual deployment target." WASD/QE drive locomotion, T sets an arm
target, arm reaching runs unconditionally (no mode gating — see below).

**Isolated variants (2026-07-31)**: `arms_full_demo.py` and `walking_full_demo.py`,
same directory — see their own section near the bottom of this file. Use these when
you want to test one half without the other interacting (e.g. is a wobble in the arm
demo coming from the arm policy itself, or from the walking policy's own base motion
feeding into it?).

## Running it

```bash
conda activate isaac_g1_control
cd ~/Elm/Code/g1_locomotion

python testing/visual_testing/full_demo/g1_full_demo.py \
    --loco_checkpoint chosen_checkpoints/walking_latest.pt \
    --arm_checkpoint logs/rsl_rl/arms/best_combined/2026-07-26_13-09-32/model_1999.pt \
    --arm left \
    --target 0.3 0.2 1.0
```

Omit `--loco_checkpoint`/`--arm_checkpoint`/`--arm` to fall back to `checkpoints.yaml`
in this directory instead.

**Obs-dim/noise-parameterization are inferred from the checkpoint itself** (2026-07-27
fix — previously hardcoded to the pre-`action_fb` 32-D/scalar-noise format, which would
silently only load checkpoints from before 2026-07-26). This means the same script loads
both older 32-D checkpoints (e.g. the 200/20-gain reference) and newer 39-D
`action_fb`-era ones (e.g. `best_combined`) correctly, automatically.

## Controls

- **W/A/S/D, Q/E** — locomotion (forward/strafe/turn).
- **T** — type a new arm target at the console (robot-local frame: x forward
  0.20-0.42m, y lateral left-arm 0.08-0.40m/right-arm -0.40--0.08m, z height 0.9-1.15m).
- **L / R** — select which arm to address when `--arm both`.
- **Y** (`--arm left` only) — type a target for the *right* arm, driven by mirroring the
  left-trained policy (see `mdp/symmetry.py`) — no separate right-arm checkpoint needed.
  Shown as a blue marker vs. the native target's red.
- **C** — toggle camera follow off/on (off lets you orbit freely with the mouse).
- **V** — reset camera to the default chase view (re-enables follow).

## `--reset_arm_on_walk` (2026-07-27)

By **default**, the arm keeps reaching/holding its target regardless of the locomotion
command — this is deliberate (see the script's own module docstring): the 23dof-era demo
gated arm reaching to "standing only," but the current arm-motion-disturbance curriculum
doesn't distinguish standing from walking in principle, so gating this demo the same way
would test a narrower condition than training actually covers. Confirmed
2026-07-27: the arm genuinely holds position through a walk command, and the combined
system tolerates it (including a stress test where the robot kept walking backward to
balance under an unreachable target before eventually falling) — but this is **untrained
generalization, not a validated capability**. The arm policy has only ever been trained
against a synthetic base-motion "wobble" signal, and the walking policy's own
`ArmMotionDisturbance` curriculum is explicitly gated to standing-commanded envs only
(arms relax to default once a walk command is given, in training) — "arm disturbance
while walking" is still a real, un-done future training item.

Pass `--reset_arm_on_walk` for the safer/more conservative behavior instead: the arm
clears its target and homes to default as soon as the commanded velocity magnitude
crosses 0.1 (same threshold `mdp.ArmMotionDisturbance` itself uses), i.e. walking always
means "arms go to default first."

## `g1_full_demo_legacy32.py`

Exact pre-2026-07-27 copy of this script, kept for one reason only: a guaranteed-working
fallback if the checkpoint-inference fix above ever needs debugging, or if you
specifically want the old hardcoded-32-D behavior. For normal use, prefer
`g1_full_demo.py` — it handles both old and new checkpoint formats already, this file
doesn't add any capability the main script lacks.

## `arms_full_demo.py` — arm control only, lower body physically fixed

Derived from `g1_full_demo.py` 2026-07-31: every arm-control code path (checkpoint
loading, `--integrated` auto-detection, observation construction, the integrated
action-target pipeline, mirror-testing, target prompting, camera control) is copied
verbatim — only the locomotion half is removed. The robot's root is rigidly pinned
(`fix_root_link=True`, the same convention `g1_arm_env.py`'s own training task uses)
instead of being driven by a walking policy, and legs/waist are held at their default
pose every step. No `--loco_checkpoint` needed — there's no locomotion policy at all.

```bash
python testing/visual_testing/full_demo/arms_full_demo.py \
    --arm_checkpoint chosen_checkpoints/arm_left_latest.pt \
    --arm left --integrated --target 0.3 0.2 1.0
```

Controls: **T** (new target), **L/R** (select arm, `--arm both` only), **Y/U**
(mirror-testing, `--arm left` only), **C/V** (camera). No WASD — there's nothing to
walk. `--integrated` is required for a `G1-Arm-Left-Integrated(-NoTerm)-v0` checkpoint,
same as `g1_full_demo.py` (see that flag's own `--help` text for why).

Use this when you want to know whether something you're seeing in `g1_full_demo.py`
is really the arm policy's own behavior, or an interaction with the walking policy's
base motion (a moving/tilting torso the arm was only ever trained against a synthetic
version of).

## `walking_full_demo.py` — locomotion only, arms held rigid

Derived from `g1_full_demo.py` 2026-07-31: the locomotion half (WASD/QE, loco-policy
loading, the velocity-command term, camera control) is copied verbatim; every
arm-related code path (target prompting, mirror-testing, goal markers, arm-policy
loading) is removed, and both arms are pinned to their default joint positions every
step instead of being driven by an arm policy. Built from the plain (non-
`ArmDisturbance`) locomotion task deliberately, so no scripted arm-motion disturbance
fires either — see the script's own module docstring for why that's safe (the
checkpoint's action/observation space is identical between task families).

```bash
python testing/visual_testing/full_demo/walking_full_demo.py \
    --loco_checkpoint chosen_checkpoints/walking_latest.pt
```

Controls: **W/A/D/Q/E** (locomotion), **S** (stop), **C/V** (camera). No arm-related
keys — arms never move.

Use this when you want to check walking-specific behavior (drift, gait, fall recovery)
without any possibility of the arm's presence or motion confounding what you're
watching.
