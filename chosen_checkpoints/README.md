# Chosen Checkpoints

This folder contains the curated deployable checkpoints that are intentionally kept in-repo.

**Empty as of 2026-07-21** — the 23dof-era checkpoints (`standing_latest.pt`,
`walking_latest.pt`, `arm_left_latest.pt`, `standing_2026-07-09_prev.pt`, and the
`exported/` ONNX artifacts) were removed from `main` as part of the pivot to the G1's
real 29dof hardware layout — they're built against the wrong action space
(1-DOF waist, 5-DOF arms, no wrists) and would need retraining regardless. They're not
gone: the full checkpoint set, with a curated best/archive breakdown and a
`validation/history/` of every eval this project ever ran, lives on the `23_dof` branch.
See `lessons_learned.md` (repo root) for the findings from that phase worth carrying
into this one.

Recommended update workflow once new checkpoints exist:

1. Train and evaluate a new candidate checkpoint (gate with `--freeze_arms` first, then
   the real integration eval — see whatever the 29dof eval scripts end up being called).
2. Copy the chosen `.pt` file into this folder using a clear filename (`standing_latest.pt`,
   `walking_latest.pt`, `arm_left_latest.pt`, or similar — keep the old one as `*_prev.pt`
   until the replacement is confirmed in the demo).
3. Commit the replacement so demos/tests continue to use stable default paths.
