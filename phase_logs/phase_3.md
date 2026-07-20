# Phase 3 — Integration Debugging → Resolution (2026-07-09 … 2026-07-16)

Condensed record of the phase that took the combined standing+walking+arm system from
"every integration eval fails catastrophically" to a verified working configuration.
Current status and next steps: `definitive_next_steps.md` (repo root) — that file is
authoritative; this is the historical record.

## The problem

All three policies passed their own native evals (~0% falls), but the integration eval
(`validation/integration_validation/eval_full_demo.py`) showed 50–100% fall rates in the
standing+arm-reach buckets for every checkpoint, across a week of standing retrains.

## What it turned out to be: deployment-side bugs, not policies

Found and fixed, in order:

1. **Per-goal arm teleport** — the eval reset the arm to default via
   `write_joint_state_to_sim` on every new goal: a nonphysical CoM/momentum shock
   training never contains. (Demo got a smooth clamped-delta homing move instead.)
2. **Arm gain mismatches** — deployment ran the reaching arm at 200/20 (the RL arm
   checkpoint's training gain; no hardware counterpart — Unitree's own SDK examples use
   60/1.5) while standing trained everything at 60/1.5. Made CLI-controlled
   (`--active_arm_gain`) and per-arm split.
3. **Instant goal resample on success** — training holds each goal 15 s (dwell);
   the eval resampled the moment the hand first crossed 2 cm, producing relentless
   back-to-back reaching. Fixed to dwell-until-timeout (`steps_to_reach` column added).
4. **Reimplemented arm drivers drift** — the eval's script-local IK driver
   (`--arm_driver ik`) never worked (28 cm misses). Solution: `--arm_driver event`
   attaches standing training's *own* disturbance term, eliminating reimplementation.
5. **Nulled buffer bug** — a bucket set `_standing_arm_motion_joint_ids = None`; the
   disturbance term only wrote it at construction, so the blend action term silently
   fell through and the arms ran on the standing policy's unshaped garbage output
   (violent 1.6 s thrash-falls). Term now re-asserts both env attrs every call.
6. Assorted: TensorDict obs normalization in the new checkpoint-width-inferring policy
   loaders; missing `obs_groups` fallback; the demo's transition arm-flail (blended
   garbage arm actions during the 10-step stand→walk blend — arm columns now come from
   the walking policy only); `enable_corruption` restored in the demo.

## Big findings along the way

- **Seed variance dominates standing training**: identical config produced 2.1% vs 100%
  reach-bucket falls (`consolidated` vs `consolidated_seed7`) → the 2-seed rule.
- **The static-arm hole**: checkpoints trained with the disturbance always active fall
  100% (within ~2.5 s) when arms are simply held still — bisected with the new
  `--freeze_arms` gate; closed for future runs by `no_reach_prob` (idle-episode slice).
- **Torso-deviation penalties are a dead end** (tried twice; didn't remove the rotation,
  produced worse policies). The **arm-intent observation**
  (`mdp.standing_arm_motion_targets`) gave the best posture ever (~8° torso) instead.
- **The analytic IK never reaches** — first-ever measurement (`event_arm_goals.csv`):
  p50 ≈ 39–54 cm to goal. Standing was therefore always trained against arm-waving,
  not reaching — which is why the genuinely-reaching RL arm destabilizes it. This is
  the open problem handed to the next phase.

## Outcome

- 8-checkpoint sweep under the fixed eval (2026-07-16 folders in
  `validation/integration_validation/`): **height_reward**
  (`2026-07-13_23-52-48_height_reward/model_5999.pt`) is the only checkpoint at 0% falls
  in both standing_still and the reach bucket, with no squat (0.72 m) and ~20° peak
  torso. Promoted to `chosen_checkpoints/standing_latest.pt`.
- Walking: untouched and 0% falls throughout.
- Next phase (see `definitive_next_steps.md`): FK-verify/fix the IK → one final standing
  retrain (2 seeds, + intent obs + idle slice) → optional RL-arm-at-hardware-gains route
  → wire user-typed targets into the demo.
