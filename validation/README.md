# Validation

Three scripts for answering "is this checkpoint actually good?" with a deterministic,
fixed-seed rollout — decoupled from the noisy training reward curve, and repeatable
whenever you retrain. Run them before promoting a checkpoint to `chosen_checkpoints/`.

All three reuse the same per-episode CSV wrappers training uses
(`g1_locomotion.utils.metrics_wrappers`), so the numbers here are directly comparable to
what you'd see in a run's own `*_detailed.csv` — see `logging_reference.md` for the full
column reference. Nothing here trains anything; all three scripts only load a checkpoint
and run inference.

`eval_standing.py`/`eval_walking.py` build on Isaac Lab's `_PLAY` env configs, which
disable observation noise (IMU/encoder `Unoise`) by default for clean visualization —
that would make the eval meaningfully easier than real training/deployment, so both
scripts explicitly restore the training-time noise config
(`observations.policy.enable_corruption = True`) before running. `eval_arm.py` doesn't
need this workaround: the arm task applies its observation noise manually in code
(`g1_arm_env.py`'s `_get_observations`), not through the same declarative mechanism the
`_PLAY` configs toggle off, so it's already on. See `known_issues.md` for the other
sim2real gaps not yet addressed anywhere in this repo (actuator gain randomization for
walking/standing, action/observation latency, etc.).

## `eval_arm.py` — can it actually reach, and does the synthetic base-tilt signal hurt it?

Sweeps two buckets — the wobble curriculum (Phase 2's synthetic base-tilt observation
signal, see `training_regimes.md`) forced off vs. forced on — against a fixed-seed
rollout.

```bash
conda activate isaac_g1_control
python validation/eval_arm.py --checkpoint chosen_checkpoints/arm_left_latest.pt --headless
```

Output: `<checkpoint_dir>/arm_eval/summary.md`.

**Two flags added 2026-07-08, for comparing checkpoints fairly:**

- `--hidden_dims 512 256 128` (or whatever the checkpoint's actual actor/critic shape
  is) — required if the checkpoint wasn't trained with the default `[256,128,64]`
  network (e.g. the wide-net variant), or `runner.load()` fails on a shape mismatch.
- `--goal_x_range X_MIN X_MAX` — evaluates against a restricted x-range of the goal box
  instead of the full thing, so two different checkpoints (or the same checkpoint before/
  after a change) can be compared on exactly the same sub-region. Remember to clear the
  checkpoint's existing `arm_eval/` folder first if re-running with a different range —
  the CSV writer appends, so mixing full-box and restricted-range episodes into the same
  file silently corrupts the comparison (this happened once — see `known_issues.md`).

**Why this exists, concretely**: this eval would have caught a real regression the day it
happened. A first attempt at restricting the arm's joint ranges (to stop it wandering into
hardware-safe-but-unnatural poses) made a large chunk of the goal workspace physically
unreachable — but that only became visible after actually training 1500 iterations and
noticing the training CSV had plateaued. A fixed-seed eval against a *specific* checkpoint
gives a fast, repeatable answer without needing to train first.

**What "good" looks like** (as of the 2026-07-08 reachability fix, ~85% success is the
reference point — see `known_issues.md` / `phase_logs/phase_2.md` for how it got there):

| Metric | Healthy | Watch out for |
|---|---|---|
| `success_rate` | High, ideally >80-90% for a converged policy | Stuck near 0% no matter how long training runs — the unreachable-goal signature (see below), not "just needs more training" |
| `mean_dist_cm` / `p50_dist_cm` | Converging toward the 2cm goal threshold | A `p50_dist_cm` that's stopped improving and sits well above the threshold |
| **The actual unreachable-goal tell**: compare `p50_dist_cm` to `p90_dist_cm`. A healthy policy has both low and close together. A wide gap (e.g. p50 near the threshold, p90 still 20cm+) with neither improving over training time means some goals are being solved fine and others are stuck at a hard floor — check whether `_GOAL_BOUNDS` in `g1_arm_env.py` is actually reachable within `self._arm_hw_limits` before assuming it's a training problem. ||
| `no_wobble` vs `with_wobble` | Some degradation under `with_wobble` is expected (harder observation signal) | A collapse in success rate specifically in `with_wobble` — means the base-tilt observation is confusing the policy rather than just adding difficulty |

**Note on `success`/`outcome`**: these were broken (always `False`/`"timeout"`, a
pre-existing bug fixed 2026-07-07 — see `known_issues.md`) for every run before that
date. If you're looking at an older `arm_summary.csv`, use `min_dist_to_goal_cm <
goal_threshold` (2cm) as a proxy instead — `eval_arm.py` and any training run from after
the fix report `success`/`success_rate` correctly.

**Joint range utilization** (also in `summary.md`, per bucket): how much of the *real
hardware* range each arm joint actually used during the eval, as a percentage — the
direct answer to "did it learn genuine reaching motion, or just tiny movements near the
default pose that happen to land on nearby goals." No baseline yet; a joint sitting at
~0% while others are well-exercised is the thing to look twice at, not a specific target
percentage (a joint might genuinely not need much range for this particular goal
workspace).

## `check_arm_reachability.py` — is `_GOAL_BOUNDS` actually reachable? (no RL involved)

Added 2026-07-08 after a real regression this would have caught immediately: `_GOAL_BOUNDS`
had been hand-picked and never checked against the arm's actual kinematics, and only ~47%
of it turned out to be reachable within the 2cm success threshold — a hard ceiling on
success rate no amount of training could ever cross, mistaken for weeks for a policy-
quality problem. See `known_issues.md` / `phase_logs/phase_2.md` for the full story.

Samples a large number of random joint configurations within the real hardware limits
(`self._arm_hw_limits` — the same limits training/eval already clamp actions to) and
records where the palm ends up, building a point cloud of the arm's true reachable
workspace. No RL policy or reward involved at all — pure kinematics via the physics sim.
Then checks how much of `_GOAL_BOUNDS` is actually covered by that point cloud, at a few
distance tolerances, plus a per-octant breakdown (which corner/region is hardest, if any).

```bash
conda activate isaac_g1_control
python validation/check_arm_reachability.py --arm left --headless
```

Output: printed coverage table + `validation/arm_reachability/<arm>_summary.md`.

**Run this before ever trusting a new `_GOAL_BOUNDS` definition again** — cheap (no
training needed, a couple of minutes), and it's the only way to know whether a workspace
change is actually solvable before spending a training run finding out the hard way.

## `eval_standing.py` — does it stay upright, and does it actually step when it needs to?

Sweeps the arm-motion-disturbance curriculum's difficulty phases (0 = no motion .. 4 =
stress-test) and runs a fixed-seed rollout at each, forcing that phase for the whole
rollout regardless of how far training has actually progressed.

```bash
conda activate isaac_g1_control
python validation/eval_standing.py --checkpoint chosen_checkpoints/standing_latest.pt --headless
```

Output: `<checkpoint_dir>/disturbance_eval/summary.md` (one row per phase) plus the raw
per-episode CSVs per phase.

**What "good" looks like**, calibrated against the last fully-converged run (6000
iterations, `phase_logs/phase_1.md`):

| Metric | Healthy | Watch out for |
|---|---|---|
| `fall_rate` | ~1-2.5%, roughly flat across phases 0-4 | Rising sharply with phase (disturbance is beating balance), or not falling at all even at phase 0 (something more broadly broken) |
| `stepped_at_least_once_rate` | Low is *expected* (2-8% seen historically) — standing's default mode is "don't move," corrective stepping is meant to be a rare/last-resort behavior, not the common case | N/A — a very high rate here would actually suggest wobbliness, not a positive sign |
| `fall_rate_when_stepped` vs. `fall_rate_when_no_step` | Stepped episodes falling *less* than no-step episodes — stepping is the escalation response for disturbance too strong for leaning alone, so it should work when it's actually needed | **Red flag**: stepped-episode fall rate *higher* than no-step fall rate (28-83% vs. 1-2.5% is what we've actually measured) — means stepping only gets triggered once things are already going wrong, and doesn't reliably save the episode. This is a known, currently-open issue — see `phase_logs/phase_1.md`'s "Decision" section for the two remediation ideas on the table (push disturbance back in; a 5th curriculum phase at 33 rad/s) |
| `mean_max_tilt_deg` | Single digits to low teens | Approaching/exceeding the `recovery_tilt_threshold_deg` (12°) routinely without falling is fine (that's `recovery_success`); tilt *and* fall rate both high together means it's not really recovering |

If a phase reports `episodes: 0`, increase `--steps_per_phase` — episodes just didn't
finish in the rollout window.

## `eval_walking.py` — does it track commands cleanly, and does it drift?

Sweeps a fixed command envelope (forward at a few speeds, backward, strafe, turn, and a
combined forward+turn case) and holds each command constant for the whole rollout —
training's command resamples every ~1-2.5s, so no single training episode isolates "how
well does it track *this* command" the way this eval does. Random pushes are disabled so
the numbers reflect gait/tracking quality, not push recovery.

```bash
conda activate isaac_g1_control
python validation/eval_walking.py --checkpoint chosen_checkpoints/walking_latest.pt --headless
```

Output: `<checkpoint_dir>/command_eval/summary.md` (one row per command bucket) plus the
raw per-episode CSVs per bucket.

**What "good" looks like**, calibrated against the last fully-converged run (3000
iterations, symmetry-augmented, `phase_logs/phase_1.md`):

| Metric | Healthy | Watch out for |
|---|---|---|
| `fall_rate` | Near 0 (~0.5% seen historically) at all speeds within the trained range (`lin_vel_x` -0.5 to 1.0 m/s) | Rising at the high end of the speed range, or nonzero even at `forward_slow`/`stand_still` |
| `mean_lin_vel_track_err_m_s` | ~0.15-0.2 m/s in the historical run | Meaningfully higher, or rising with commanded speed (not keeping up at speed) |
| `mean_foot_slip_speed_m_s` | Low and roughly flat across buckets | Spiking specifically during `turn_left`/`turn_right`/`strafe_*` (feet dragging through direction changes rather than lifting cleanly) |

**Drift (new metric — no historical baseline yet, this is the first run it'll be measured
on)**: reported only for the `forward_slow`/`forward_medium`/`forward_fast`/`backward`
buckets (zero lateral velocity, zero yaw rate commanded — a "go straight" test). Two
numbers:

- `heading_drift_deg` — how much the robot's actual heading rotated *beyond what the
  ang_vel_z command asked for* (0 for these buckets, so in practice this is just "did it
  turn even though nothing told it to"). A few degrees over a ~20s episode is probably
  fine; a persistent drift of 10-20°+ in one direction (not just noise scattering both
  ways) is the failure mode you originally noticed on the undertrained policy — check the
  sign is consistent across episodes (systematic bias) vs. scattered (just noise).
- `lateral_drift_m` — how far sideways the robot ended up relative to where its own
  commanded forward speed, integrated against however it was actually facing, said it
  should be. This is the direct "walked forward but ended up somewhere off to the side"
  number. A few centimeters over a 20s episode is noise; tens of centimeters or more,
  especially growing with episode length, means there's a real asymmetry (this is exactly
  the kind of thing gait-symmetry training, `mdp/symmetry.py`, is meant to fix — if this
  looks bad on a policy trained *with* the symmetry augmentation, that's worth flagging
  as a real regression, not expected variance).

Since there's no historical baseline for drift yet, run this once on the current
`chosen_checkpoints/walking_latest.pt` first and treat that as the reference point for
judging future retrains — don't wait for a specific number to look "wrong" in isolation.

## Why this lives here instead of just watching the training curve

The training reward curve (and even the native `Episode_Termination/*` TensorBoard
scalars) tells you the policy is converging under the *training* command/disturbance
distribution — which resamples constantly and mixes easy and hard cases together. It
can't answer "how does this specific checkpoint behave under *this* specific, held-fixed
condition" — which is exactly the question worth asking before trusting a checkpoint for
a demo or promoting it to `chosen_checkpoints/`. Both scripts here exist to answer that
question directly, with the same fixed seed every time so results are comparable
run-to-run.
