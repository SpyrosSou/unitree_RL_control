# Chosen Checkpoints

This folder contains the curated deployable checkpoints that are intentionally kept in-repo.

**Current state (2026-07-28), 29dof pivot — see `policy_status.md` (repo root) for the
full, living status writeup this summary is kept in sync with:**

- `walking_latest.pt` — **PROMOTED 2026-07-28**, from
  `logs/rsl_rl/walking/arm_disturbance/2026-07-28_11-41-33/model_20996.pt`
  (`G1-Locomotion-Velocity-ArmDisturbance-LooseHeightNoStep-v0`). Walking/standing is
  deliberately **frozen here** for now — this checkpoint has a real, large stand-still
  stepping/drift fix (the wobble affecting the arm policy) but a known, elevated
  `turn_left` fall rate under sustained turning (see `policy_status.md`'s 2026-07-28
  tradeoff note for why: a follow-up `joint_mirror` experiment fixed `turn_left` and
  forward-walking drift, but reintroduced ~2x of the stand-still regression it was meant
  to solve — not promoted here for that reason, paused pending a gating fix, not
  abandoned). Being trained a further ~5000 iterations overnight (same recipe, no reward
  changes) via `overnight_train.sh` — check `policy_status.md` for whether that changed
  anything before assuming this entry is still current.
  The previous `walking_2026-07-24_base_only_prev.pt` reference checkpoint was removed
  2026-07-28 (superseded, no longer needed for rollback).
- `arm_left_latest.pt` — **CORRECTED 2026-07-28**: previously called stale here, which
  was wrong — see `policy_status.md`'s "CORRECTED 2026-07-28" note. Real checkpoint,
  ~8000 iterations, 24.29%/24.24% success (no_wobble/with_wobble). `best_combined`
  (`logs/rsl_rl/arms/best_combined/2026-07-26_13-09-32/model_1999.pt`, currently only
  2000 iterations, 28.20%/32.85%) is somewhat ahead but not promoted here — being
  trained to 10000 iterations overnight via `overnight_train.sh` to properly test
  whether the ~25-30% plateau holds with more training; may get promoted here instead
  once that's done. Both checkpoints show the same failure pattern in per-episode data:
  converges within ~10-12s of a 20s episode to a stable point 6-10cm from the goal and
  plateaus there, rather than running out of time while still approaching — see
  `policy_status.md` for the full diagnostic.

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
