# Repo structure — what's here, what moved to Backups (23dof branch)

This branch (`23_dof`) was reorganized 2026-07-21 for archival: the project pivoted to
the G1's real 29dof hardware layout (see `definitive_next_steps.md`), and this branch
preserves the working 23dof pipeline without cluttering `main` going forward. This file
explains where everything ended up.

## In this git repo (tracked, ships with a clone)

```text
source/, testing/, validation/*.py     <- all code, unchanged
definitive_next_steps.md               <- full project history, the authoritative doc
known_issues.md, phase_logs/           <- frozen historical archives
quickrun.md                            <- fast commands to run the demo (this branch)

chosen_checkpoints/
    standing_latest.pt                  <- default pointer (currently height_reward)
    walking_latest.pt, arm_left_latest.pt
    README.md                            <- START HERE for checkpoint provenance/verdicts
    best_checkpoints/                    <- the 3 checkpoints worth knowing about
    archive/                              <- one final checkpoint per historical experiment

validation/history/
    standing_eval_history.csv            <- one row per (checkpoint, eval type), all
    arm_eval_history.csv                    curated experiments, compiled from every
    walking_eval_history.csv                summary.csv this project ever produced
    by_checkpoint/<name>/                <- raw summary + joint_diagnostics CSVs per
                                              curated checkpoint, same data uncompressed
```

Every `.pt` file in `chosen_checkpoints/` (root + `best_checkpoints/` + `archive/`) is
tracked — ~39MB total, no size concern. `validation/history/` is ~520KB (small CSVs
only — the schema-compiled summaries, not raw per-step logs).

## NOT in git — local disk only (`logs/`, gitignored, always has been)

Every training run's full output directory. As of this cleanup, each retained run dir
has: the final checkpoint (`model_5999.pt`), the tensorboard event file, `params/`,
`git/` (code diff snapshot at train time), and its own small eval-run subfolders
(`ikreach_eval/<timestamp>/`). **Intermediate checkpoints** (saved every 50 iterations —
~120 per run) **and the raw per-step training telemetry** (`standing_summary.csv`/
`standing_detailed.csv`, written continuously through all 6000 iterations, up to
211MB each) **were deleted** — the former because the final checkpoint is what matters
and already lives in git, the latter because it's raw noise the tensorboard event file
already captures far more usefully. This dropped `logs/` from 5.4G to ~545MB.

## Moved to `~/Elm/Backups/g1_locomotion/23_dof/` (external, not in this repo at all)

The "more detailed than what's in git" tier — full run directories for every curated
checkpoint (matching `chosen_checkpoints/`'s table), in case anyone ever wants to
re-inspect training curves or re-run an eval without retraining:

```text
~/Elm/Backups/g1_locomotion/23_dof/
    standing_runs/<run_name>/     <- tensorboard, params, git diff, ikreach_eval outputs
                                      (NOT intermediate checkpoints, NOT the giant
                                      per-step training CSVs — same exclusions as logs/)
    arm_runs/<run_name>/           <- same pattern, all 3 arm training runs
    legs_runs/<run_name>/          <- the one walking run
    integration_validation/<ts>/   <- full eval_full_demo.py output folders, one per
                                       eval run, for every curated checkpoint (excludes
                                       the 4 pre-fix runs with the torso=150deg eval bug)
```

~551MB total. Nothing here is referenced by any script — it's a reference archive, not
part of the working pipeline. If this machine's `~/Elm/Backups/` ever needs its own
backup, that's a separate concern from this repo.

## What got deleted outright, nowhere

- All intermediate `model_*.pt` checkpoints (every training run, ~120 each) — the exact
  final checkpoint is what matters and is preserved (git for curated ones).
- The raw per-step `standing_summary.csv`/`standing_detailed.csv`/`arm_summary.csv`/
  `arm_detailed.csv` training telemetry (1.7G+1.9G total) — superseded by tensorboard.
- The 4 `validation/integration_validation/` eval runs that hit the pre-fix
  `eval_full_demo.py` torso-clip bug (torso reads ~150° in those, known-invalid data).
- Non-curated `validation/integration_validation/` runs not tied to any checkpoint in
  `chosen_checkpoints/README.md`'s table.

## Quick lookup: "where is X"

| Looking for | Where |
|---|---|
| A checkpoint to load right now | `chosen_checkpoints/` (see its README) |
| Numbers for a specific checkpoint/eval combo | `validation/history/standing_eval_history.csv` (or arm/walking) |
| Per-joint diagnostics for a curated checkpoint | `validation/history/by_checkpoint/<name>/` |
| Full training curves / tensorboard for a curated run | `~/Elm/Backups/g1_locomotion/23_dof/standing_runs/<run>/` |
| A non-curated / dead-end run's raw data | Gone — only its one-line verdict survives, in `chosen_checkpoints/README.md`'s table |
| Full project narrative / why decisions were made | `definitive_next_steps.md` |
