# Chosen Checkpoints

This folder contains the curated deployable checkpoints that are intentionally kept in-repo.

Current contents (updated 2026-07-16):

- `walking_latest.pt`  -> primary flat walking policy (0% falls in every eval, untouched all phase)
- `standing_latest.pt` -> primary standing policy = `logs/rsl_rl/standing/g1_locomotion_flat/`
  `2026-07-13_23-52-48_height_reward/model_5999.pt` — winner of the 2026-07-16 8-checkpoint
  sweep (only checkpoint at 0% falls in BOTH standing_still and the IK-arm reach bucket,
  no squat at 0.72m height, ~20° peak torso). See definitive_next_steps.md for the full verdict
  and the caveat (it must be retrained once the arm's IK reaching is fixed — see there).
- `standing_2026-07-09_prev.pt` -> the previous standing pointer (kept as backup; its source
  run may no longer exist under logs/).
- `arm_left_latest.pt` -> primary left-arm IK policy (RL; reaches well in isolation but its
  motion destabilizes standing — see definitive_next_steps.md Step 3).

These files are the only `.pt` files expected to be committed.
All other training checkpoints under `logs/` are ignored.

Recommended update workflow:

1. Train and evaluate a new candidate checkpoint (gate with
   `validation/eval_standing_ikreach.py --freeze_arms` first, then the integration eval).
2. Copy the chosen `.pt` file into this folder using the standard filename (keep the old
   one as `*_prev.pt` until the replacement is confirmed in the demo).
3. Commit the replacement so demos/tests continue to use stable default paths.
