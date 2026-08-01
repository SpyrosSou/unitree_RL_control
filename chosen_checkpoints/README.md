# Chosen Checkpoints

This folder contains the curated deployable checkpoints that are intentionally kept in-repo.

**Current state (2026-08-01), 29dof pivot — see `policy_status.md` (repo root) for the
full, living status writeup, and `policy_overview.md` for a brief reward/curriculum
summary of both current recipes:**

- `walking_latest.pt` — **PROMOTED 2026-07-30**, from
  `logs/rsl_rl/walking/arm_disturbance/2026-07-29_23-28-41/model_6999.pt`
  (`G1-Locomotion-Velocity-ArmDisturbance-StandingPackage-v0`, trained fresh). Chosen
  for the arm-integration priority: stand-still stepping/drift roughly 4x better than
  the previous promotion (step count 8.2→2.0, |lateral drift| 0.076→0.031m, 0% falls
  across every disturbance phase), and the in-distribution turn eval now reads 0%
  falls everywhere (the prior checkpoint's elevated `turn_left` fall rate was found
  2026-07-30 to be almost entirely an out-of-distribution eval artifact — `eval_walking.py`
  was commanding turns/strafes well outside the trained range; see `policy_status.md`).
  **Known costs, not yet addressed**: `forward_slow` straight-line heading drift roughly
  doubled (62°→112°; medium/fast/backward improved), and in-place turning is now
  essentially ignored (the policy stands still through a pure-turn command). Not yet
  re-checked in `g1_full_demo.py` — eval-confirmed only.
  Previous promotion (`model_20996`, `LooseHeightNoStep`) is no longer kept
  alongside it — **2026-08-01: `chosen_checkpoints/` policy changed to exactly one
  `.pt` per policy, no `*_prev.pt` rollback copies.** If a rollback to `model_20996`
  is ever needed, the source is backed up at
  `~/Elm/Backups/g1_locomotion/29dof/legs_runs/2026-07-28_11-41-33_loose_height_nostep/`
  (final + every-3rd checkpoint) and the pruned local copy (final checkpoint only)
  is still at `logs/rsl_rl/walking/arm_disturbance/2026-07-28_11-41-33/`.

- `arm_left_latest.pt` — **PROMOTED (updated since 2026-07-31): now
  `logs/rsl_rl/arms/integrated_no_term/2026-07-30_19-57-17/model_11999.pt`
  (`G1-Arm-Left-IntegratedNoTerm-v0`)**, superseding the interim `Integrated`
  (pre-NoTerm-fix, `model_7999.pt`) checkpoint this entry previously described.
  Checksum-verified identical to that log file. **Both reaching and holding are
  solved** at the real 40/10 hardware gain:
  - **Reaching**: 100% reach rate, ~1.4mm mean/2.6mm p90 distance-to-goal (fixed
    2026-07-29 — the static-torque-ceiling root cause, see `policy_status.md`'s
    "ROOT CAUSE FOUND" entry).
  - **Holding**: the earlier `Integrated` checkpoint reached fine but dithered at
    the 2cm boundary instead of settling (an incentive bug — `terminate_on_success`
    made completing the hold reward-negative). Fixed by removing it
    (`terminate_on_success=False`, `G1-Arm-Left-IntegratedNoTerm-v0`, 12000 fresh
    iters): **Settled <2cm/<3cm 100%/100%, Tail-settle (5s window) 100%, ~97% of
    every step spent inside the 2cm zone** — see `policy_status.md`'s 2026-07-31
    "RESULT" entry for the full numbers (and the eval-script bug that briefly made
    Tail-settle read 0%, since fixed).
  - **Demo compatibility fixed 2026-07-30/31**: `testing/visual_testing/full_demo/
    g1_full_demo.py` previously hand-rolled the legacy `current+delta`/world-frame
    arm control loop unconditionally, which silently ran the wrong control law for
    an Integrated-family checkpoint (and crashed outright the moment an arm command
    actually ran, from the missing `target_fb` observation block). Fixed by adding
    a `--integrated` flag that switches the demo to the integrated-target/env-local-
    frame pipeline this checkpoint needs — **the flag is required for this
    checkpoint and must NOT be passed for any other one** (the demo cross-checks
    the flag against the checkpoint's actual observation width and refuses to run
    on a mismatch). Two arm-only/walking-only isolated demo variants
    (`arms_full_demo.py`, `walking_full_demo.py`, same directory) exist too. **None
    of the three have been live-tested against Isaac Sim yet** — verify
    interactively before trusting them:
    ```
    python testing/visual_testing/full_demo/arms_full_demo.py \
        --arm_checkpoint chosen_checkpoints/arm_left_latest.pt \
        --arm left --integrated --target 0.3 0.2 1.0
    ```
  - Deployment note: `g1_rl_control` must replicate the integrated-target pipeline
    (`target += clamped_delta`), not the legacy `current + delta` — see
    `arms_policy_finalisation.md` step 3 for the full spec (action pipeline, 46-D
    observation order, local-frame convention). This is the last open item before
    calling the arm policy fully finished — see `arms_policy_finalisation.md`.

23dof-era checkpoints (1-DOF waist, 5-DOF arms, no wrists — wrong action space for this
hardware layout regardless) were removed from `main` as part of the pivot; the full
checkpoint set from that phase lives on the `23_dof` branch. See `policy_status.md`'s
"Lessons learned" section (repo root) for findings from that phase worth carrying into
this one.

Recommended update workflow once new checkpoints exist:

1. Train and evaluate a new candidate checkpoint (gate with `--freeze_arms` first, then
   the real integration eval — see whatever the 29dof eval scripts end up being called).
2. **2026-08-01: exactly one `.pt` per policy in this folder — no `*_prev.pt` rollback
   copies.** Overwrite the chosen `.pt` file directly (`walking_latest.pt`,
   `arm_left_latest.pt`, or similar). If a rollback point is ever needed, the source
   run is what's backed up under `~/Elm/Backups/g1_locomotion/` — recover from there,
   not from a local `*_prev.pt`.
3. Commit the replacement so demos/tests continue to use stable default paths.
