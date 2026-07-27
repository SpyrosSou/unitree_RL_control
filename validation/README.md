# Validation

Batch evaluation scripts for answering "is this checkpoint actually good?" with a
deterministic, fixed-seed rollout — decoupled from the noisy training reward curve, and
repeatable whenever you retrain. Run these before promoting a checkpoint to
`chosen_checkpoints/`. For *watching* a policy instead of just measuring it, see
`testing/visual_testing/` instead.

Rewritten 2026-07-27 — the previous version was stale (dead links to removed docs,
described a since-deleted `eval_standing.py`). All scripts here reuse the same
per-episode CSV wrappers training uses (`g1_locomotion.utils.metrics_wrappers`), so
numbers are directly comparable to a run's own `*_detailed.csv`. Nothing here trains
anything — every script only loads a checkpoint and runs inference.

## `eval_walking.py` — the single comprehensive walking eval

One script, one Isaac Sim launch, one `summary.md`, three sections:

1. **Command-bucket sweep** — a fixed command envelope (forward at a few speeds,
   backward, strafe, turn, a combined forward+turn case), each held constant for the
   whole rollout. Reports fall rate, tracking error, foot slip, and — for the
   zero-lateral/zero-yaw "straight" buckets — heading/lateral drift.
2. **Arm-disturbance phase sweep** — fall rate while standing still, under each of the
   4 disturbance phases pinned individually (only with `--arm_disturbance`; skip with
   `--skip_phases`). **Important**: the disturbance's phase is sampled once per episode
   at reset based on elapsed training step count — an *unpinned* check in a short eval
   session will only ever show the easiest phase. Always pin `--phases` for a
   meaningful disturbance check.
3. **Raw world-frame displacement check** — reads `root_pos_w` directly over a
   fixed-speed rollout, independent of section 1's reward-derived tracking error
   (guards against misreading a tracking metric as achieved velocity — a real past bug
   in this project). Skip with `--skip_displacement`.

```bash
conda activate isaac_g1_control
python validation/eval_walking.py --checkpoint chosen_checkpoints/walking_latest.pt --arm_disturbance --headless
```

Output: `<checkpoint_dir>/walking_eval/summary.md` plus raw per-episode CSVs per
bucket/phase in their own subdirectories. Pass `--arm_disturbance` for any checkpoint
trained on `G1-Locomotion-Velocity-ArmDisturbance-v0` (every current walking
checkpoint); omit only for a pure base-recipe checkpoint.

**Reading the numbers** (current reference: `walking_latest.pt`, 2026-07-27): fall rate
0% across every bucket/phase is the bar to clear. `mean_lin_vel_track_err_m_s` around
0.10-0.13 m/s at 0.3-0.6 m/s commanded is the current baseline. Heading drift is a known
open issue — see `policy_status.md`'s "Known gap" section for current numbers and
context before treating any specific drift value as surprising.

## `eval_arm.py` — can it actually reach the goal?

```bash
conda activate isaac_g1_control
python validation/eval_arm.py --checkpoint logs/rsl_rl/arms/<run>/model_<iter>.pt --headless
```

Output: `<checkpoint_dir>/arm_eval/summary.md` — success rate, mean/median/p90 distance
to goal, joint-range utilization (did it use a genuine range of motion or just tiny
movements), and a per-reward-term breakdown, across two buckets (synthetic base-tilt
"wobble" forced off vs. on).

**Flags for older checkpoints** (needed or `runner.load()` fails on a shape/key
mismatch, not a silent corruption):
- `--legacy32` — checkpoints trained before 2026-07-26's `action_fb` observation
  addition (32-D obs instead of 39-D) — e.g. the 200/20-gain reference.
- `--log_std` — checkpoints trained with `noise_std_type="log"` instead of the default
  `"scalar"` (e.g. `best_combined`).
- `--locked_wrist` — a `G1-Arm-Left-LockedWrist-v0` checkpoint (5 controlled joints).
- `--hidden_dims` — only if the checkpoint used a non-default network size.

**Important caveat, confirmed 2026-07-27**: this eval's methodology (many quick
episodes per env, goals resample rapidly on success) does **not** measure true
single-attempt reliability — see `eval_arm_long_hold.py` below and
`policy_status.md`'s "Critical finding" section. A high `success_rate` here does not by
itself mean a fresh single reach attempt will usually succeed.

## `eval_arm_long_hold.py` — does a single, fresh attempt actually succeed?

Built 2026-07-27 after finding `eval_arm.py`'s aggregate success rate can look
dramatically better than true single-shot reliability. Gives each env exactly **one**
fixed goal from a fresh reset, for a much longer window (45s default) than the training
episode length, and reports both a strict (15-consecutive-step hold) and lenient
("ever within threshold at all") success definition, plus how much of the
post-first-reach time is spent actually in-threshold.

```bash
conda activate isaac_g1_control
python validation/eval_arm_long_hold.py --checkpoint logs/rsl_rl/arms/left/2026-07-22_06-20-55/model_2999.pt --headless
```

Output: `<checkpoint_dir>/long_hold_summary.txt`. Same `--legacy32`-style checkpoint
compatibility as `eval_arm.py` — currently only supports the legacy 32-D format
directly (see the script for details if evaluating a newer checkpoint).

## `check_arm_reachability.py` — is the goal workspace actually reachable? (no RL)

Samples a large number of random joint configurations within real hardware limits and
records where the palm ends up — pure kinematics, no policy or reward involved at all —
then checks how much of the goal box (`_GOAL_BOUNDS` in `g1_arm_env.py`) is covered
within a few distance tolerances, plus a per-octant breakdown of which region (if any)
is hardest. Existing results (`validation/arm_reachability/*.md`, current as of
2026-07-27): ~97-97.5% coverage within 2cm — reachability is not the bottleneck for
current arm performance.

```bash
conda activate isaac_g1_control
python validation/check_arm_reachability.py --arm left --headless
```

Output: printed coverage table + `validation/arm_reachability/<arm>_summary.md`. Cheap
(a couple of minutes, no training) — worth re-running any time `_GOAL_BOUNDS` changes,
before spending a training run finding out the hard way whether it's solvable.

## `integration_validation/eval_full_demo.py` — combined loco+arm, headless, vectorized

Runs the unified locomotion policy and an arm-reaching policy together across many
parallel envs, scripted (not interactive) — the batch-metrics counterpart to
`testing/visual_testing/full_demo/g1_full_demo.py`. **Not re-run against the current
`walking_latest.pt`/`best_combined` pairing as of 2026-07-27** — see
`policy_status.md`'s "Not yet done" note. Also has a known `debug_vis` remote-asset-hang
fix (present in `eval_walking.py`) not yet ported here — check before relying on it
unattended.

## `check_ik_accuracy.py` — likely stale, 23dof-era

References a class (`StandingArmIKReachDisturbance`) and a doc
(`definitive_next_steps.md`) that don't exist in the current 29dof codebase — this
script predates the pivot and hasn't been verified against current code. Flagging
rather than fixing or deleting, since it wasn't part of this pass's scope.

## Why this lives here instead of just watching the training curve

The training reward curve tells you the policy is converging under the *training*
command/disturbance distribution, which resamples constantly and mixes easy and hard
cases together. It can't answer "how does this specific checkpoint behave under *this*
specific, held-fixed condition" — exactly the question worth asking before trusting a
checkpoint for a demo or promoting it to `chosen_checkpoints/`. Every script above
exists to answer that directly, with the same fixed seed every time so results are
comparable run-to-run.
