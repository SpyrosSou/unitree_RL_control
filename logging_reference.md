# Logging Reference

What gets logged during training, where it lives, and what each metric means. Written
so this can be handed to a colleague without them having to read the wrapper code.

## Where everything lives

Every training run writes to `logs/rsl_rl/<experiment_name>/<run_timestamp>/`, e.g.
`logs/rsl_rl/standing/g1_locomotion_flat/2026-07-06_14-22-33/`. Inside that one folder:

```
model_0.pt, model_50.pt, ...          # checkpoints, saved every save_interval iterations
params/env.yaml, params/agent.yaml    # the fully-resolved config actually used for this run
events.out.tfevents.*                 # TensorBoard log
standing_summary.csv                  # convergence-at-a-glance — start here (name depends on task family)
standing_detailed.csv                 # everything, for when the summary raises a question
exported/                             # policy.pt / policy.onnx, written by play.py
```

**Two CSVs per run, not per policy or per metric.** `scripts/rsl_rl/train.py` picks exactly
one *pair* based on the `--task` name (`"Standing"` → `standing_summary.csv` +
`standing_detailed.csv`, `"Locomotion"` but not standing → `walking_*.csv`, `"Arm-IK"` →
`arm_*.csv`); the checks are mutually exclusive (`if`/`elif`, "Standing" checked first
since e.g. `G1-Locomotion-Standing-Flat-v0` contains both words), so a run only ever
produces one pair. Both files share the same `episode_index`/`env_id`/`env_step` values
per row, so they can be joined back together later (e.g. in pandas) if a detailed column
turns out to matter for something the summary flagged.

**Resuming a run** (`--resume --checkpoint ...`) continues appending to both files in the
same folder by default (see `quickrun.md`'s "Resume training" section) — it does not
create new files. Episode numbering (`episode_index`) continues from the highest index
already in the file rather than restarting at 0.

## How to read the summary file (you don't need an RL background for this)

Open `<task>_summary.csv` and look at these, roughly in this order of importance:

1. **`episode_return` (and `mean_reward`) over time.** Sort/plot against `env_step`
   (or just `episode_index`, since episodes finish roughly in step order). Rising then
   flattening = it's learning, then has settled into a stable behavior. Still rising =
   not done yet, keep training. Wildly oscillating with no clear trend, even late in
   training = something's unstable and worth flagging (reward weights fighting each
   other, learning rate too aggressive, etc.) rather than just "needs more iterations."
2. **The outcome rate.** For standing/walking, that's `fell` (1/0) — plot its rolling
   average; it should trend toward 0 and stay there. For arm, it's `success` — should
   trend toward 1. **This is the one that actually matters, not just reward** — reward
   can plateau at "avoids the worst penalties" before the behavior you actually want
   (not falling, or reaching the goal) is solid. Always check both.
3. **The one task-specific column that says "is it doing its actual job":**
   - Standing → `step_count`. Is it *stepping* when the disturbance is strong, not just
     leaning? A policy can have low fall rate and high reward while never stepping, if
     it just hasn't been tested hard enough yet — see point 4.
   - Walking → `mean_lin_vel_track_err_m_s`. Should trend down and flatten near 0.
   - Arm → `min_dist_to_goal_cm`. Should trend down toward (and below) the 2 cm goal
     threshold.
4. **Standing only — read `fell`/`step_count` *together with* `arm_disturbance_phase`.**
   The arm-motion-disturbance curriculum gets harder as training progresses (phase 0 = no
   motion, phase 4 = stress-test bursts — see `training_regimes.md`). A rising fall rate
   late in training might just mean the curriculum reached a harder phase, not that the
   policy regressed. Always check what phase a given stretch of rows is in before reading
   the fall rate as good or bad news.

If 1–3 all look converged and 4 doesn't change the story, that checkpoint is a reasonable
candidate to actually play/watch, and to run through the matching `validation/` script
(`eval_standing.py` or `eval_walking.py`) before trusting it further.

Everything in `<task>_detailed.csv` (tilt angles, action smoothness, foot-slip speed,
min base height, etc.) is real and useful, but it's diagnostic detail for *when something
in the summary looks wrong* — not what to look at first.

## Layer 0 — already there for free (TensorBoard, no CSV needed)

Isaac Lab's manager-based envs (walking, standing) log these natively; open with
`tensorboard --logdir logs/rsl_rl`:

- `Episode_Reward/<term_name>` — per-reward-term episodic average, for every term in that
  task's reward config. The per-term breakdown behind `episode_return`.
- `Episode_Termination/<term_name>` — e.g. `Episode_Termination/base_contact` (fall count)
  vs `Episode_Termination/time_out`. Fall rate vs. timeout rate, already a native scalar.
- `Curriculum/<term_name>` — would show curriculum progression automatically, *if* it
  were a real `CurriculumTermCfg`. The arm-motion-disturbance phase stepping is currently
  a hand-rolled step counter inside an `EventTerm`, so it does **not** show up here —
  that's why `arm_disturbance_phase` is logged manually in the CSV instead.
- Standard PPO scalars: mean reward, mean episode length, value loss, surrogate/policy
  loss, KL divergence, learning rate (it's adaptive — watch this move), noise/entropy std.

TensorBoard's curves are smoother/easier to eyeball for a quick "is it converging" check
(that's the same question the summary CSV answers, just as a plottable file instead).
The arm task (`DirectRLEnv`) does not get `Episode_Reward`/`Episode_Termination` — those
are manager-based-env features. Its bookkeeping is entirely in the arm CSVs below.

## Layer 1 — the summary CSVs (start here)

### `standing_summary.csv`

| Column | Meaning |
|---|---|
| `episode_index` | Running count of finished episodes for this env slot (continues across a resume). |
| `env_id` | Which of the `num_envs` parallel environments this row is from. |
| `env_step` | Global simulation step count when this episode finished — the training-progress axis (rises steadily; roughly `iteration × num_steps_per_env`). |
| `episode_steps` | Length of this episode in control steps. |
| `episode_return` / `mean_reward` | Sum of reward over the episode / that divided by length. |
| `outcome` | `"fall"` or `"timeout"`. |
| `fell` | 1 if the episode ended by hitting the ground (the `base_contact` termination). |
| `step_count` | How many genuine corrective steps (foot lifted >0.05 s, then planted) it took this episode. |
| `arm_disturbance_phase` | Which curriculum phase (0–4) was active when this episode finished. |

### `walking_summary.csv`

| Column | Meaning |
|---|---|
| `episode_index`, `env_id`, `env_step`, `episode_steps`, `episode_return`, `mean_reward`, `outcome`, `fell` | Same meaning as standing's columns above. |
| `mean_lin_vel_track_err_m_s` | Mean `|commanded_xy_velocity − actual_xy_velocity|` over the episode — the core "is it tracking commands" signal. |
| `heading_drift_deg` | How much the actual heading rotated *beyond* what the `ang_vel_z` command asked for over the episode — nonzero means real rotational drift, not commanded turning. See `validation/README.md` for how to read this. |

### `arm_summary.csv`

| Column | Meaning |
|---|---|
| `episode_index`, `env_id`, `env_step`, `episode_steps`, `episode_return`, `mean_reward` | Same meaning as above. |
| `outcome` / `success` | `"success"` if all active arm(s) reached their goal within `goal_threshold` before the episode timed out. |
| `min_dist_to_goal_cm` | Closest the (worst-performing, if "both") arm got to its goal — near 0 means it nailed it. |

## Layer 2 — the detailed CSVs (everything, for follow-up questions)

### `standing_detailed.csv` (summary columns, plus:)

| Column | Meaning |
|---|---|
| `timed_out` | 1 if the episode ended by the `time_out` termination (the complement of `fell`). |
| `recovery_success` | 1 if it didn't fall *and* tilted past `recovery_tilt_threshold_deg` at some point — i.e. it was pushed off-balance and recovered, rather than never being meaningfully disturbed. |
| `max_tilt_deg` | Peak body tilt from vertical during the episode. |
| `min_base_height_m` | Lowest pelvis height reached (a crouch/near-fall signal even without a full topple). |
| `max_lin_speed_m_s` / `max_ang_speed_rad_s` | Peak root planar speed / peak root angular speed — how much the body actually moved. |
| `max_action_abs` / `max_action_delta_abs` | Peak `|action|` and peak per-step action change — smoothness/aggressiveness sanity checks. |
| `max_foot_air_time_s` | Longest single continuous foot-airborne duration in the episode — magnitude companion to `step_count`. |
| `recovery_tilt_threshold_deg` | The threshold used for `recovery_success` (currently 12°) — stored per-row so old rows stay self-describing if the constant ever changes. |

### `walking_detailed.csv` (summary columns, plus:)

| Column | Meaning |
|---|---|
| `timed_out` | Same meaning as standing's. |
| `max_tilt_deg` | Sanity check, not the primary signal for a walking gait. |
| `mean_ang_vel_track_err_rad_s` | Mean commanded-vs-actual yaw-rate error (the rotational counterpart to `mean_lin_vel_track_err_m_s`). |
| `mean_foot_slip_speed_m_s` | Mean foot horizontal speed *while that foot is in ground contact* — high values mean the foot is sliding/dragging instead of a clean plant. |
| `lateral_drift_m` | How far sideways (relative to the episode's starting heading) actual position ended up from where the robot's own commanded velocity, integrated against its actual heading each step, implied it should be — magnitude companion to `heading_drift_deg`. See `validation/README.md`. |

### `arm_detailed.csv` (summary columns, plus:)

| Column | Meaning |
|---|---|
| `max_dist_to_goal_cm` | Furthest that arm ever was from its goal in the episode — mostly reflects the initial distance, useful for sanity-checking goal sampling ranges. |

## Layer 3 — the validation scripts (separate from training, run on demand)

Neither of these runs during training — you run them against a saved checkpoint whenever
you want a deterministic, repeatable readout instead of the noisy training curve. Both
reuse the same CSV wrappers training uses internally. Full usage and "what's a good
number" guidance lives in `validation/README.md`; this is just the file-layout reference.

`validation/eval_standing.py` sweeps the arm-motion-disturbance curriculum's phases.
Output lands next to the checkpoint, under `<checkpoint_dir>/disturbance_eval/`:

- `phase_<N>/standing_detailed.csv` (+ a `standing_summary.csv` you can ignore here) — raw
  per-episode rows for a fixed-seed rollout forced into curriculum phase `N` for its whole
  duration.
- `summary.md` — one row per phase: episode count, fall rate, `stepped_at_least_once_rate`
  (fraction of episodes where `step_count > 0`), mean step count, fall rate split by
  whether a step was taken that episode, mean/max tilt, mean peak foot air time.

`validation/eval_walking.py` sweeps a fixed command envelope (forward/backward/strafe/
turn at a few speeds), holding each command constant for the whole rollout. Output lands
under `<checkpoint_dir>/command_eval/`:

- `<bucket_name>/walking_detailed.csv` (+ `walking_summary.csv`) — raw per-episode rows
  for a fixed-seed rollout under that one held-constant command.
- `summary.md` — one row per command bucket: episode count, fall rate, tracking error,
  foot slip, and — for the straight-line buckets only — mean absolute heading/lateral
  drift.

Both are the tool for "is this checkpoint actually robust/accurate under *this specific*
condition" — a training reward curve going up doesn't answer that on its own, since
training's command/disturbance distribution keeps changing within and across episodes.
