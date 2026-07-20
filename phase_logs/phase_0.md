# Phase 0 — Infrastructure & Cleanup

> **Archival phase log** — frozen record. Current status lives in `definitive_next_steps.md` (repo root); the integration/debugging endgame is `phase_logs/phase_3.md`.

Plain-language summary of what changed and why. Date: 6 July 2026.

## Repo cleanup

- Merged the three testing folders (`walking_testing/`, `arm_testing/`, `general_testing/`) under one `testing/` parent directory, kept as separate subfolders.
- Removed redundant/stale docs (`todo.md`, `RLoperations.md`) and fixed a few others that had drifted out of sync with what the code actually does.

## Standing policy

- Removed the "push" (random shove) training feature — traced it and found it was never actually used to train the checkpoint you had, so nothing was lost by removing it.
- Kept the arm-motion disturbance curriculum (the mechanism meant to teach corrective stepping) active from the very start of training.
- Loosened a reward setting so the torso is allowed to move to help with balance recovery, instead of being penalized for any motion at all.

## Logging

- Added a real "did it actually step" measurement to standing's logs — previously we could only see tilt/velocity, which can't tell a deliberate step apart from just leaning.
- Added a difficulty-phase label so every logged episode says which curriculum level (0–4) it happened at.
- Extended per-episode logging (previously standing-only) to walking and arm training too.
- Built `eval_standing_disturbance.py` (now `validation/eval_standing.py`, see `phase_1.md`) — a tool to test any saved standing checkpoint against every difficulty level on demand, instead of only trusting the training curve.
- Later: split each run's log into a short "summary" file (quick health check) and a "detailed" file (everything else), so a glance doesn't require wading through every column.

## Arm training

- Switched to a lighter version of the robot model and cut some physics precision that arm-only training doesn't need, to speed it up a bit.
- Did not remove the (unused, but still simulated) leg/torso joints — doing that properly needs a custom-built robot model, which felt like overkill for this pass.

## Training workflow

- `--resume` now continues writing into the same log folder by default instead of starting a new one each time — checkpoints, CSVs, and TensorBoard logs all just keep growing in place.

## Known limitation at the time

None of this was tested by actually running Isaac Sim (not available in this working session) — it was verified by carefully reading the relevant source code and compiling/linting everything. The first real training run was the actual test, and it did surface real issues (see `phase_1.md`).
