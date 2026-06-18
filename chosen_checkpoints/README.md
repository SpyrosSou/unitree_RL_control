# Chosen Checkpoints

This folder contains the curated deployable checkpoints that are intentionally kept in-repo.

Current convention:

- `walking_latest.pt`   -> primary flat walking policy
- `standing_latest.pt`  -> primary standing policy
- `arm_left_latest.pt`  -> primary left-arm IK policy

These files are the only `.pt` files expected to be committed.
All other training checkpoints under `logs/` are ignored.

Recommended update workflow:

1. Train and evaluate a new candidate checkpoint.
2. Copy the chosen `.pt` file into this folder using the standard filename.
3. Commit the replacement so demos/tests continue to use stable default paths.
