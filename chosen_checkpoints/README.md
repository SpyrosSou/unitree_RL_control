# Chosen Checkpoints

This folder contains the curated deployable checkpoints that are intentionally kept in-repo.

**Branch note (`ik_residuals`, 2026-08-01): arm entry promoted.** The IK + residual-RL
architecture (`ik_arm_integration_plan.md` Phase 4) now has a validated checkpoint —
`arm_left_residual_latest.pt`, see its own entry below. This supersedes `arm_left_latest.pt`
(the old pure-RL checkpoint), kept only for historical/fallback reference.

**Current state (2026-08-01), 29dof pivot — see `policy_status.md` (repo root) for the
full, living status writeup this summary is kept in sync with:**

- `walking_latest.pt` — **working, ready for initial real-robot testing**, but has a
  known, actively-being-fixed drift gap (see `policy_status.md`). From
  `logs/rsl_rl/walking/arm_disturbance/2026-07-24_15-47-18/model_15998.pt`. Three
  drift-fix attempts since (2026-07-25/26/27) have all made drift worse, not better —
  none promoted; this checkpoint is untouched by any of them.
- `walking_2026-07-24_base_only_prev.pt` — the pure base-recipe checkpoint
  `walking_latest.pt` was warm-started from (no arm-disturbance training), kept for
  reference/rollback. From
  `logs/rsl_rl/walking/ablation_reward_weights/2026-07-24_03-13-16/model_5999.pt`.
- **`arm_left_residual_latest.pt` — current best arm checkpoint, promoted 2026-08-01.**
  From `logs/rsl_rl/arms/residual_left/2026-07-31_14-00-08/model_11999.pt` (11999
  training iterations, resumed across two runs from `2026-07-31_07-52-15`).
  **This is NOT a drop-in replacement for `arm_left_latest.pt`'s usage pattern** — it
  is a *residual* policy: it only produces the right behavior when combined, every
  control step, with (a) a numerical IK baseline solved once per episode/goal
  (`g1_locomotion.controllers.arm_ik.G1ArmIK`, cached as `q_ik`/`tau_ik`), (b) the
  IK solver's gravity/Coriolis feedforward torque applied via
  `set_joint_effort_target`, and (c) a persistent, slew-limited commanded target
  (`current_target += clamp(policy_output * residual_action_scale, ±0.06 rad)`, never
  re-anchored to the raw measured position — re-anchoring reinstates a real
  static-torque ceiling, see `ik_arm_integration_plan.md`'s 2026-07-31 entry). Expects
  a 46-D observation (base task's 39-D + a 7-D `q_ik - current_joint_pos` feature).
  Trained on a **genuinely fixed base** (`fix_root_link=True`, no walking policy in
  the loop) — walking integration is still deferred, see `policy_status.md`. The
  reference implementation of all of the above is
  `source/.../tasks/manager_based/g1_arm_residual/g1_arm_residual_env.py`
  (`G1ArmResidualEnv`); do not hand-roll the action pipeline independently without
  reading it first — see landmine-style caveats in `ik_arm_integration_plan.md`.
  Trained with `terminate_on_success=False` and a tiered (20%/10%/5% margin,
  escalating scale) joint-limit-proximity penalty on top of the base task's reward.
  **Eval (fixed base, trimmed goal box x∈[0.20,0.31], `validation/eval_arm_residual.py`,
  1024 episodes across wobble/no-wobble):** Settled <2cm 100%, Tail-settle (5s) 100%,
  mean min-dist-to-goal 0.67cm, mean final-dist 1.21cm — vs. the `ik_baseline` bucket
  (residual forced to zero) at 9.3cm mean / 0% settled, confirming the residual is
  doing real, substantial work. **Known open issue, not blocking promotion**: the
  tiered joint-limit penalty had a *mixed* effect — max excursion improved for some
  joints (wrist_pitch no longer exceeds its own soft limit, was 99-111% pre-fix) but
  the tightest-margin (5%) per-step dwell-time metric actually rose (~3.5%→~17%
  averaged across joints) and precision dipped slightly (mean min-dist 0.55→0.67cm
  vs. the pre-tiered-penalty checkpoint) — some joint configurations still look
  visually awkward. Deferred, not re-opened, per user decision 2026-08-01 ("sort of
  improved but not 100%, this will do for now"). See `ik_arm_integration_plan.md`'s
  2026-08-01 entries for the full before/after numbers and root-cause reasoning.
- `arm_left_latest.pt` — **superseded by `arm_left_residual_latest.pt` above, kept only
  for historical/pure-RL-fallback reference — do not treat as current.** Predates the
  real-hardware-gain fix and every finding since; standalone pure-RL policy (39-D obs,
  `current_pose + delta` action), no IK/residual machinery involved. The pure-RL
  `best_combined` candidate this once pointed to
  (`logs/rsl_rl/arms/best_combined/2026-07-26_13-09-32/model_1999.pt`) was never
  promoted either — see `policy_status.md`'s historical arm section ("Critical
  finding": ~30% true single-shot reliability even at the best pure-RL checkpoint,
  the reason this branch pivoted to IK + residual RL in the first place).

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
