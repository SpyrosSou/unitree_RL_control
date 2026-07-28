# Chosen Checkpoints

This folder contains the curated deployable checkpoints that are intentionally kept in-repo.

**Branch note (`ik_residuals`, 2026-07-27)**: unchanged for now — will be updated as the
IK arm integration (`ik_arm_integration_plan.md`) lands, since that replaces the arm
entry below with an IK-backed component rather than a promoted RL checkpoint.

**Current state (2026-07-27), 29dof pivot — see `policy_status.md` (repo root) for the
full, living status writeup this summary is kept in sync with:**

- `walking_latest.pt` — **working, ready for initial real-robot testing**, but has a
  known, actively-being-fixed drift gap (see `policy_status.md`). From
  `logs/rsl_rl/walking/arm_disturbance/2026-07-24_15-47-18/model_15998.pt`. Three
  drift-fix attempts since (2026-07-25/26/27) have all made drift worse, not better —
  none promoted; this checkpoint is untouched by any of them.
- `walking_2026-07-24_base_only_prev.pt` — the pure base-recipe checkpoint
  `walking_latest.pt` was warm-started from (no arm-disturbance training), kept for
  reference/rollback. From
  `logs/rsl_rl/walking/ablation_reward_weights/2026-07-24_03-13-16/model_5999.pt`.
- `arm_left_latest.pt` — **stale, not working, do not treat as current.** Predates the
  real-hardware-gain fix and every finding since. **The actual current-best arm
  candidate has NOT been promoted here yet** — it's `best_combined`
  (`logs/rsl_rl/arms/best_combined/2026-07-26_13-09-32/model_1999.pt`), still sitting
  in `logs/`. Not promoted because arm reaching's real single-shot reliability is only
  ~30% even for the best checkpoint (see `policy_status.md`'s "Critical finding") — not
  yet at a bar worth promoting to a curated default. See `policy_status.md` for the
  current plan (a pivot toward IK for precision grasping).

23dof-era checkpoints (1-DOF waist, 5-DOF arms, no wrists — wrong action space for this
hardware layout regardless) were removed from `main` as part of the pivot; the full
checkpoint set from that phase lives on the `23_dof` branch. See `policy_status.md`'s
"Lessons learned" section (repo root) for findings from that phase worth carrying into
this one.

Recommended update workflow once new checkpoints exist:

1. Train and evaluate a new candidate checkpoint (gate with `--freeze_arms` first, then
   the real integration eval — see whatever the 29dof eval scripts end up being called).
2. Copy the chosen `.pt` file into this folder using a clear filename (`standing_latest.pt`,
   `walking_latest.pt`, `arm_left_latest.pt`, or similar — keep the old one as `*_prev.pt`
   until the replacement is confirmed in the demo).
3. Commit the replacement so demos/tests continue to use stable default paths.
