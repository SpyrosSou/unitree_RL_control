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
   zero-lateral/zero-yaw "straight" buckets — heading/lateral drift. **Fixed
   2026-07-30**: the turn/strafe/combo buckets previously commanded velocities well
   outside what any recipe actually trains on (ang_vel_z curriculum was never wired
   in — see `policy_status.md`) — default buckets now stay in-distribution; the old
   out-of-distribution commands are kept as opt-in `*_stress` buckets (`--buckets
   turn_left_stress ...`). Numbers from summaries generated before this date used the
   old commands for turn/strafe/combo — compare those against `*_stress`, not the
   same-named defaults.
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

**Reading the numbers** (current reference: `walking_latest.pt`, the 2026-07-30
"standing package" promotion — see `chosen_checkpoints/README.md`): fall rate 0%
across every bucket/phase (including in-distribution turning) is the bar to clear.
`mean_lin_vel_track_err_m_s` around 0.10-0.13 m/s at 0.3-0.6 m/s commanded is the
baseline. This checkpoint trades better standing (step count ~2, lateral drift ~3cm)
for worse straight-line heading drift and untrained in-place turning — see
`policy_status.md`'s Walking section for current numbers and context before treating
any specific drift value as surprising.

## `eval_arm.py` — can it actually reach the goal, and can it hold it?

```bash
conda activate isaac_g1_control
python validation/eval_arm.py --checkpoint logs/rsl_rl/arms/<run>/model_<iter>.pt --headless
```

Output: `<checkpoint_dir>/arm_eval/summary.md` — mean/median/p90 distance to goal,
joint-range utilization (did it use a genuine range of motion or just tiny
movements), and a per-reward-term breakdown, across two buckets (synthetic base-tilt
"wobble" forced off vs. on). Headline columns as of 2026-07-30:
- **Settled <2cm / <3cm** — fraction of episodes *ending* that close to the goal.
- **Tail-settle (Ws)** — fraction of episodes that stayed under `goal_threshold` for
  the *entire* trailing `W` seconds (`--tail_window_s`, default 5.0) — the direct
  hold-capability measure, distinguishing genuine settling from a policy that's still
  oscillating right up to the last instant. Computed purely from existing per-second
  distance snapshots; works on any checkpoint.
- **Success rate (reach+hold, legacy)** — the original 15-consecutive-step-hold +
  early-termination definition. Kept for historical comparison only — it's a known
  **incentive artifact** for a well-trained policy under `terminate_on_success=True`
  (completing the hold ends the per-step bonus stream, so the policy learns to dither
  at the boundary instead) and reads a **structural 0%** for any checkpoint trained
  with `terminate_on_success=False`. Do not judge a checkpoint by this column alone —
  see `policy_status.md`'s 2026-07-30 arm entries and `arms_policy_finalisation.md`.

**Flags for older/other checkpoints** (needed or `runner.load()` fails on a shape/key
mismatch, not a silent corruption):
- `--integrated` — a `G1-Arm-Left-Integrated-v0` checkpoint (2026-07-29 static-
  torque-ceiling fix: integrated action targets + env-local ee/goal obs, 46-D) — the
  current best arm candidate, see `chosen_checkpoints/README.md`.
- `--integrated_no_term` — a `G1-Arm-Left-IntegratedNoTerm-v0` checkpoint (2026-07-30:
  same 46-D layout as `--integrated`, but `terminate_on_success=False` — the fix for
  the dithering/hold-quality problem, see `arms_policy_finalisation.md`). Selects a
  matching eval env that's also `terminate_on_success=False`, so every episode runs
  full length and the Tail-settle-rate metric's trailing window is never cut short.
  Do not combine with `--integrated`.
- `g1_full_demo.py` (`testing/visual_testing/full_demo/`) needs the equivalent
  `--integrated` flag for either of the above two checkpoints — see that script's
  own `--help`/module docstring; fixed 2026-07-30, not yet live-tested.
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
