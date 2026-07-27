# Training logs — what's kept and why

Living index of every run directory under `logs/rsl_rl/` (gitignored — this file isn't,
so the explanation survives even though the checkpoints don't get committed). Update
this whenever a run is added or a cleanup happens — see `policy_status.md` (repo root)
for the full experiment history, including everything that's been deleted and why.

## Arms

- `arms/left/2026-07-22_06-20-55` — the 200/20-gain reference, 99.98% success at 2999
  iterations *by the training-time metric*. Kept as a comparison point, but its true
  single-shot reliability is only ~30% (confirmed 2026-07-26/27, see `policy_status.md`'s
  "Critical finding") — don't treat the headline number as representative without that
  caveat.
- `arms/left/2026-07-23_22-54-45` — 40/10-gain 7-DOF baseline, 27.68% success at 5999
  iterations. Kept as the reference/fallback and resume source.
- `arms/best_combined/2026-07-26_13-09-32` — **current best arm candidate**
  (`privileged_critic` + `log_std` + `action_fb`, 28.20%/32.85%). Not yet promoted to
  `chosen_checkpoints/`. Flagged to extend to 8000 iters to confirm the 2000-iter
  plateau holds — not done yet.
- `arms/goal_curriculum/2026-07-26_18-01-26` — goal-distance curriculum (starts at 15%
  of the workspace, expands over training). Extended to ~12k iterations overnight
  2026-07-26/27; plateaued at 25.69%/25.42% — not better than `best_combined`.
- `arms/gain_60_kd1p5/2026-07-26_13-50-29`, `arms/gain_15_kd1/2026-07-26_16-25-42` —
  gain-search variants testing real Unitree SDK/deploy values directly (not guesses).
  Both worse than 40/10 or 200/20, in two different ways (oscillation/joint-limit
  saturation vs. near-total training stall respectively) — see `policy_status.md`'s
  "Gain search — closed" note. Kept as reference for why gain-matching isn't the fix.

## Walking

- `walking/ablation_reward_weights/2026-07-24_03-13-16` — the 4-way-ablation run that
  fixed the original "never falls, never walks" plateau, now the shared default. Kept
  because it sources `chosen_checkpoints/walking_2026-07-24_base_only_prev.pt`.
- `walking/arm_disturbance/2026-07-24_15-47-18` — warm-started continuation onto the
  real deployment recipe. Kept because it sources `chosen_checkpoints/walking_latest.pt`
  (`model_15998.pt`).
- `walking/arm_disturbance/2026-07-27_00-43-20` — drift-fix round 2
  (`HeadingDriftPenalty`, weight -0.1, fresh weights, 8000 iters). Fall rate excellent,
  heading drift worse than baseline and scales badly with speed — **not promoted**. See
  `policy_status.md`'s "Round 2" note for full numbers. Third consecutive drift-fix
  attempt to make things worse, not better — treated as a real pattern.

## Cleanup log

- **2026-07-25**: removed ~14GB of superseded/dead-end runs (pre-reward-fix-era walking
  attempts, aborted/incomplete runs, no-op reward ablations, duplicate arm baseline,
  wrist-locking, isolated hyperparameter ablations, a killed partial rate-limit run) —
  see `policy_status.md`'s "Rejected/superseded experiment logs" section for what each
  showed before deletion.
- **2026-07-27** (pre-`29dof_IK`-branch cleanup, point 2): all 19 non-aborted runs as of
  that date (13 arm + 6 walking) backed up to
  `/home/spyros/Elm/Backups/g1_locomotion/29dof/` (see that dir's `README.md` for the
  full list and what's included/excluded), then removed from local disk — 388MB total,
  down from ~2.7GB. Only the 3 runs sourcing a `chosen_checkpoints/` file were kept
  locally (`arms/left/2026-07-22_06-20-55`, `arms/best_combined/2026-07-26_13-09-32`,
  `walking/arm_disturbance/2026-07-24_15-47-18`), each trimmed to its final checkpoint
  only. Their raw per-episode CSVs (`*_detailed.csv`, `*_summary.csv` — 1.9GB combined,
  not backed up anywhere, conclusions already in `policy_status.md`) were deleted too,
  bringing local `logs/rsl_rl/` from 2.4GB to 56MB. The aborted `arms/left/2026-07-26_06-51-43`
  (only `model_1.pt`) was deleted without backup — nothing of value.
