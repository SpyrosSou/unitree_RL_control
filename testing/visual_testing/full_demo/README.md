# `g1_full_demo.py` — integrated arms + walking demo

Interactive, keyboard-controlled demo combining the unified stand+walk locomotion
policy with a trained arm-reaching policy on the same robot — the closest thing in this
repo to "watch the actual deployment target." WASD/QE drive locomotion, T sets an arm
target, arm reaching runs unconditionally (no mode gating — see below).

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
