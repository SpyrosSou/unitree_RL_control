# Chosen Checkpoints

**This is the `23_dof` branch** — the frozen, archived state of the project as it stood
before the pivot to the G1's real 29dof hardware layout (3-DOF waist, 7-DOF arms with
wrists — see `definitive_next_steps.md`'s 2026-07-21 section for the full story of why).
Everything in this branch was trained against Isaac Lab's `G1_MINIMAL_CFG` asset
(1-DOF waist, 5-DOF arms, no wrists), which does not match the physical hardware.
`main` is being adapted for 29dof going forward — this branch exists so the working
23dof pipeline, its checkpoints, and the full record of what was tried stay available for
reference (or as a fallback) without cluttering `main`.

## Start here — the 2-3 checkpoints that actually matter

- **`standing_latest.pt`** (= `best_checkpoints/height_reward_model_5999.pt`) — **best
  visual/idle posture** of everything tested: 0% falls standing still, no squat (0.72m),
  ~20° peak torso, minimal tilt. **Do not use for a live arm-reach demo** — it collapses
  55-70% of the time under real arm-policy-driven reach (it predates both the
  joint-ordering IK fix and the arm-gain matching fix, so it never trained against
  anything resembling the real reaching arm). Idle/walking-only demos are safe; reaching
  is not.
- **`best_checkpoints/gainmatch_model_5999.pt`** — **best for actual deployment**: the
  only checkpoint with a low fall rate under REAL arm-policy-driven reach (0% / 0% /
  5.2% / 12.5% across left/left-edge/right/right-edge, 0% gate, 6.0% native). Visible
  cosmetic issue: torso sits twisted 60-70° even standing still, confirmed live as
  "clearly wrong" — a known, described issue, not a stability problem.
- **`best_checkpoints/torsoclip30_model_5999.pt`** — attempted fix for GainMatch's torso
  twist via a hard ±30° action clip. Torso is properly bounded now, but fall rate got
  worse (26.3% native, up to 24.7% policy-mode) and a NEW visible issue appeared (wide
  hip stance) — the compensation moved, it didn't go away. Kept as the clearest
  illustration of the project's core recurring finding (see below), not as a
  recommendation to use.

**The one thing to know before touching any of these**: none of them combine "good
posture" and "low fall rate under real reach" at the same time. That gap was never
closed in the 23dof lineage — see the walk-through in `definitive_next_steps.md`.

## Why none of them fully worked — the recurring pattern

Every attempt to fix standing's posture ran into the same wall: `torso_joint` was
deliberately left unpenalized from day one ("let it be free for balance"), and once that
was fixed (TorsoClip), the same compensation need just moved to the next-cheapest joint
(hip, then — one experiment later — ankle, at OrientHip, driven to its literal hard
mechanical limit). Three joints in a row, same shape. Directly informed the 29dof
finding: our robot has a 1-DOF waist standing in for what's actually a 3-DOF waist on the
real hardware, so some of what kept getting displaced onto hip/ankle may never have
belonged there at all. Full writeup in `definitive_next_steps.md`.

## Folder layout

```text
chosen_checkpoints/
    standing_latest.pt        <- canonical pointer, loaded by default in eval/demo scripts
    walking_latest.pt         <- untouched all project, 0% falls always
    arm_left_latest.pt        <- see arm note below
    standing_2026-07-09_prev.pt   <- pre-height_reward pointer, kept for continuity
    best_checkpoints/          <- the 3 above, plus this file's context
    archive/                    <- one final checkpoint per experiment below, historical reference
```

**Arm checkpoint note**: `arm_left_latest.pt` currently points to the
`null_space_weighted` (2026-07-08) run, not the later `gains_fixed` (2026-07-12) run —
this looks like it could be stale but wasn't changed here since it's outside this
session's scope; worth double-checking intentionality before the 29dof arm rebuild.
Both are in `archive/` for comparison, along with the `wide_net` variant.

## Full experiment table

Numbers are from `validation/history/standing_eval_history.csv` (native/gate = own
disturbance; integration = full stand+walk+arm eval). "Kept" = full run directory backed
up to `~/Elm/Backups/g1_locomotion/23_dof/` in addition to the final checkpoint here;
"discarded" = only the checkpoint + summary CSVs were kept, no raw logs/tensorboard.

| Experiment | Gate falls | Native falls | Policy-mode falls (reach buckets) | Torso/posture | Verdict | Data kept |
|---|---|---|---|---|---|---|
| dwell_phase_fix | — | — | (static-arm hole, not tested) | 48° tilt, static-arm hole | superseded, squats | discarded |
| **height_reward** | — | — | 55-70% (collapses) | Best posture, ~20° torso | **best visual**, unsafe for real reach | **kept** |
| leg_symmetry | 100% | — | — | Best native tilt ever, but 100% falls w/ static arms | dead end | discarded |
| consolidated | 0% | — | 26% (event) | OK idle | works when seed cooperates | discarded |
| consolidated_torso | — | 0% | 100% (still, event) | reward-penalty attempt | **dead end**, don't retry this approach | discarded |
| consolidated_intent | 0% | 3.1% | 53% (event) | Best posture in this family, ~8-9° | not robust alone | discarded |
| consolidated_noreach | 0% | 4.0% | 49% (event) | closes static-arm hole | building block, not final | discarded |
| consolidated_seed7 | 0% | 30.1% | 91% (event) | same config as consolidated, diff seed | proved seed variance dominates | discarded |
| ikreach_height_intent (attempt 1) | 0% | 48.6% | 99-100% (collapse) | torso PINNED 150° (hard limit) | gain-mismatch collapse, superseded | kept |
| **gainmatch** | 0% | 6.0% | 0% / 0% / 5.2% / 12.5% | Torso twisted 60-70°, "clearly wrong" | **best for deployment** | **kept** |
| **torsoclip30** | 0% | 26.3% | 0% / 2.1% / 24.7% / 22.7% | Torso bounded ~32°, but wide hip stance | fix moved the problem, didn't solve it | **kept** |
| torsolock | 0% | 51.0% | up to 79.3% | Torso ~0.7° but visibly unstable even idle | **dead end**, worse than clip on every axis | kept |
| torsoclip30_orienthip | 90.8% | 78.1% | — | Ankles driven to hard mechanical limit | **catastrophic**, discard | kept (as the whack-a-mole reference case) |

Bold rows are the ones in `best_checkpoints/`. Full per-checkpoint eval CSVs (including
per-joint diagnostics) are in `validation/history/by_checkpoint/<name>/`.

## Recommended update workflow (if this branch is ever revisited)

1. Train and evaluate a new candidate (gate with `--freeze_arms` first, then the real
   `--arm_driver policy` integration eval — event mode alone is not sufficient, see
   `definitive_next_steps.md`).
2. Copy the chosen `.pt` into `best_checkpoints/` or `archive/` as appropriate, add a row
   to the table above and to `validation/history/standing_eval_history.csv`.
3. Only update `standing_latest.pt` itself deliberately — it's the default every script
   reaches for.
