# Phase 1 — Standing + Walking Baseline

> **Archival phase log** — frozen record. Current status lives in `definitive_next_steps.md` (repo root); the integration/debugging endgame is `phase_logs/phase_3.md`. `logging_reference.md` was removed 2026-07-16 — CSV columns are documented in `metrics_wrappers.py` itself.

Plain-language summary of what happened, plus a quick reference for every metric we
collect. Full technical detail on the metrics lives in `logging_reference.md` — this
is the condensed version. Date: 6–7 July 2026.

## What happened

- Trained walking for 3000 iterations. Looked great on paper (low fall rate, good
  velocity tracking) but visually the right foot lifted way higher than the left.
  Checked the reward, the robot's default pose, and the actuator setup — none of them
  were asymmetric. This was just PPO settling into a lazy, lopsided habit, since
  nothing in the setup was telling it to keep both legs behaving the same way. Known
  failure mode in legged-robot RL generally, not a bug specific to this project.
- Fixed it with "mirror training" (symmetry-augmented PPO): every training example is
  now shown to the network twice — once normally, once mirrored left-right — so it
  can't get away with a lopsided gait. Retrained: problem gone, walking looks good.
- Trained standing for 6000 iterations, covering the entire arm-disturbance difficulty
  curriculum, including the hardest "stress-test" level. Fall rate is low (~1–2.5%) and
  stable across every difficulty level, and it looks composed when actually watched —
  including in the combined demo alongside the arm-reaching policy.
- Dug into the "does it take a small step to catch itself" question specifically: yes,
  but rarely (2–8% of episodes at any given difficulty), and when it does step, it
  still falls a large fraction of the time (28–83%). So the skill exists in a rough
  form but isn't reliable yet — most likely because it doesn't happen often enough
  during training to be practiced well. Ruled out "just needs more iterations": the
  back half of training (iterations ~3300–6000) sat at the hardest difficulty level the
  whole time with zero further improvement in either fall rate or stepping.
- Checked real G1 hardware specs: arm joints are rated for ~37 rad/s; the current
  hardest difficulty level only asks for ~12.5 rad/s. So there's real headroom before
  hitting a hardware ceiling, if the curriculum ever needs to go harder.
- **Decision**: leave the rare/unreliable stepping as a known limitation for now, since
  overall balance is solid and this is a rare edge case. Two ideas on the table for
  later, not implemented yet: bring back a "push" disturbance (a sudden shock is more
  likely to force stepping than an arm swing can), and/or add a 5th curriculum phase at
  **33 rad/s** (agreed number — safely under the ~37 rad/s hardware limit, unlike the
  initially-floated 40 rad/s). Note for whenever this is implemented: the swing-amplitude
  clamp needs to scale up alongside the speed, or the motion will snap against the old
  amplitude limit instead of swinging smoothly.

## What changed vs. before (code / training approach)

For presenting the delta from the pre-handover state.

**Standing policy training:**
| | Before | Now |
|---|---|---|
| Disturbance during training | Two mechanisms mixed together: a random arm-swing curriculum *and* a random external push/shove | Push removed entirely (traced it — it had never actually been trained into any real checkpoint anyway); arm-swing curriculum only |
| Torso reward | Penalized any torso motion (`joint_deviation_torso = -0.1`, inherited default) | Relaxed to `0.0` so the torso can help with balance recovery instead of fighting it |
| Per-run logging | One CSV (`standing_metrics.csv`) with tilt/velocity/action stats and push-specific columns | Two CSVs (`standing_summary.csv` + `standing_detailed.csv`); push columns gone; new `step_count`/`max_foot_air_time_s` (actually detects a foot leaving the ground and landing again) and `arm_disturbance_phase` (which difficulty level was active) |
| Robustness check before trusting a checkpoint | None — only the training reward curve | `validation/eval_standing.py` (was `eval_standing_disturbance.py`): runs the checkpoint's actual deterministic (deployed) behavior against a fixed-seed sweep of every difficulty level, independent of the noisy training curve. `validation/eval_walking.py` added alongside it — same idea, sweeping a fixed command envelope instead, plus a new drift check (does it wander off a straight line even when told to go straight) |

**Walking policy training:**
| | Before | Now |
|---|---|---|
| Left-right gait symmetry | Nothing enforced it — reward only cares about step timing, not which leg does what | Symmetry-augmented PPO training (every training example mirrored left-right and trained on both) — added after this exact repo's own last training run produced a visibly lopsided gait |
| Per-run logging | None | `walking_summary.csv` + `walking_detailed.csv` added (same pattern as standing) |

**Arm policy:** not retrained yet this phase — one prep change made: switched the training asset to a lighter collision-mesh variant and reduced physics-solver precision that arm-only training doesn't need, purely for training speed. No behavioral change.

**Training workflow / infrastructure (affects all three policies):**
- `--resume` now continues writing into the *same* log folder/CSV by default. Before, every resume — even of the same run — started a brand-new timestamped folder, so a single training effort could end up scattered across several folders.
- Repo reorganized: the three testing folders merged under one `testing/` parent; redundant docs removed; a few docs that had drifted from what the code actually does were corrected.
- New `logging_reference.md` and this `phase_logs/` directory — didn't exist before; nothing like a metrics glossary or a running change log existed previously.

## Metrics glossary

### Always logged automatically (TensorBoard — `tensorboard --logdir logs/rsl_rl`)

- **Train/mean_reward** — average reward per step. Should rise, then flatten.
- **Train/mean_episode_length** — average episode length. Rising toward the max means surviving longer / failing less.
- **Episode_Reward/<term>** — the total reward broken down by individual reward term. Shows which specific behavior is driving the score.
- **Episode_Termination/<term>** — how episodes actually ended (e.g. fall vs. timeout). Direct fall-rate/success-rate signal, no CSV needed.
- **Loss/entropy**, **Policy/mean_noise_std** — how random/exploratory the policy still is. Should generally trend down over training; collapsing to near-zero *before* reward flattens suggests it stopped exploring too early.
- **Loss/learning_rate** — auto-adjusted step size. Mostly a stability sanity check, not something to act on day-to-day.
- **Loss/surrogate**, **Loss/value_function** — the raw PPO optimization losses. Mostly useful for debugging instability, not routine monitoring.

### `<task>_summary.csv` — quick health check, one row per finished episode

- **episode_index**, **env_id** — which episode, which of the parallel environments.
- **env_step** — how far into training this episode happened. The x-axis for judging convergence.
- **episode_steps** — how long the episode lasted.
- **episode_return** / **mean_reward** — total / average reward for that episode.
- **outcome**, **fell** / **success** — how the episode ended.
- **step_count** *(standing only)* — how many corrective steps it took.
- **arm_disturbance_phase** *(standing only)* — which difficulty level (0–4) was active.
- **mean_lin_vel_track_err_m_s** *(walking only)* — how well it tracked the commanded speed.
- **heading_drift_deg** *(walking only)* — how much heading rotated beyond what was commanded; the "does it drift when going straight" check.
- **min_dist_to_goal_cm** *(arm only)* — closest the end-effector got to the target.

### `<task>_detailed.csv` — everything else, for when the summary raises a question

- **timed_out** — the complement of `fell` (episode ended by hitting the time limit, not by falling/failing).
- **max_tilt_deg** *(standing, walking)* — peak body tilt from vertical during the episode.
- **min_base_height_m**, **max_lin/ang_speed**, **max_action(_delta)_abs** *(standing)* — how much the body actually moved/crouched, and how aggressive the actions were — smoothness/sanity checks.
- **max_foot_air_time_s** *(standing)* — longest single foot-airborne duration; magnitude companion to `step_count`.
- **recovery_success**, **recovery_tilt_threshold_deg** *(standing)* — whether it tilted past a threshold (currently 12°) and recovered without falling, i.e. was actually tested and passed, vs. never being seriously disturbed.
- **mean_ang_vel_track_err_rad_s** *(walking)* — commanded-vs-actual turning-rate error.
- **mean_foot_slip_speed_m_s** *(walking)* — how much a foot slides while planted, instead of a clean lift/plant.
- **max_dist_to_goal_cm** *(arm)* — furthest the end-effector ever was from the goal — mostly reflects how far the goal was sampled, a sanity check more than a quality signal.

### `validation/` scripts output — on demand, not automatic

- `eval_standing.py` → `summary.md` per checkpoint: episode count, fall rate, stepped-at-least-once rate, mean step count, fall rate split by whether a step was taken, and tilt/air-time stats, one row per difficulty phase, from a fixed-seed rollout (not the noisy training curve).
- `eval_walking.py` → `summary.md` per checkpoint: episode count, fall rate, tracking error, foot slip, and (straight-line command buckets only) heading/lateral drift, one row per held-fixed command bucket.
