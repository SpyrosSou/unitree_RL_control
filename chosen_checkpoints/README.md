# Chosen Checkpoints

This folder contains the curated deployable checkpoints that are intentionally kept in-repo.

**Current state (2026-07-30), 29dof pivot — see `policy_status.md` (repo root) for the
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
  Previous promotion (`model_20996`, `LooseHeightNoStep`) kept as `walking_latest_prev.pt`
  for rollback/comparison, not deleted.

- `arm_left_latest.pt` — **STILL STALE, not the current-best candidate.** ~8000
  pre-`action_fb` iterations from 2026-07-23 — kept only because nothing has been
  promoted over it yet. The actual current-best arm checkpoint is
  `logs/rsl_rl/arms/integrated/2026-07-29_20-27-48/model_7999.pt`
  (`G1-Arm-Left-Integrated-v0`) — **deliberately NOT copied here yet**:
  - **Reaching itself is solved** at the real 40/10 hardware gain (a 2026-07-29
    static-torque-ceiling fix — see `policy_status.md`'s "ROOT CAUSE FOUND" entry):
    100% reach rate, ~1.4mm mean/2.6mm p90 distance-to-goal, every one of 1093 eval
    episodes ending within 2.76cm of goal.
  - But the env's `terminate_on_success` + per-step bonus makes completing the
    original 15-consecutive-step hold *reward-negative*, so the policy dithers at
    the 2cm boundary rather than settling deep and still — the legacy success-rate
    column reads a misleading 6.5-9.4% as a result, not a capability gap. One more
    retrain (reward/termination redesign) is needed before this is a finished
    artifact — full plan in `arms_policy_finalisation.md` (repo root, **not a status
    doc, don't fold its content back into this README** — it's the standalone
    pick-up spec for finishing this specific policy).
  - **Also not yet demo-compatible**: `testing/visual_testing/full_demo/g1_full_demo.py`
    hand-rolls its own arm action/observation pipeline rather than reusing
    `G1ArmEnv`, and doesn't yet know about the Integrated task's persistent-target
    action integration or its 46-D (`target_fb` + env-local ee/goal) observation
    layout — pointing the demo at this checkpoint today would silently run the
    wrong control law. Needs a demo-side update before visual inspection works, not
    just a checkpoint copy.
  - Deployment note for whenever this does get promoted: `g1_rl_control` must
    replicate the integrated-target pipeline (`target += clamped_delta`), not the
    legacy `current + delta` — see `arms_policy_finalisation.md` step 3 for the full
    spec (action pipeline, 46-D observation order, local-frame convention).

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
