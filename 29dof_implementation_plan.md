# 29dof implementation plan

## Progress — 2026-07-21 session (Phase 1 + most of Phase 2 built)

Built today, all under `source/g1_locomotion/g1_locomotion/`:

- `assets/robots/unitree.py` + `unitree_actuators.py` — `UNITREE_G1_29DOF_CFG`
  ported from `unitree_rl_lab`, plus the (unused-by-G1-but-available) delay/friction
  actuator model.
- `tasks/manager_based/g1_locomotion/` — the new task package:
  - `velocity_env_cfg.py`, `mdp/{rewards,observations,curriculums,commands}.py` — ported
    near-verbatim from `unitree_rl_lab`, only the asset import changed.
  - `mdp/symmetry.py` — adapted from the 23dof-era version. **Not a trivial port**: found
    and fixed a real bug in translation — `ObservationsCfg.PolicyCfg` uses
    `history_length=5` (every term stacked over 5 steps, not present in the 23dof task),
    which the naive port would have silently mirrored incorrectly. Rewrote the transform
    to be history-aware (block-reshape per term). Flagged in the file's own docstring as
    not yet verified against a live env — the `RuntimeError` guards will fire immediately
    if the assumed dimension math is wrong.
  - `mdp/events.py` + `mdp/actions.py` — the scripted (non-IK) arm-motion-disturbance
    curriculum, ported from `StandingArmTrajectoryDisturbance`, joint list extended to 14
    (7 per arm, wrist included). Two gym tasks: `G1-Locomotion-Velocity-v0` (pure
    base recipe) and `G1-Locomotion-Velocity-ArmDisturbance-v0` (base + curriculum).
  - `agents/rsl_rl_ppo_cfg.py` — `BasePPORunnerCfg` ported verbatim; symmetry
    augmentation wired into the base task's PPO cfg only, deliberately NOT into the
    arm-disturbance variant (matching the 23dof-era precedent's own stated reasoning —
    combining scripted asymmetric disturbance with symmetry augmentation needs dedicated
    design, not assumed safe).
- `deploy_reference/` (repo root) — `unitree_rl_lab`'s C++ deploy harness, copied in
  inert (not wired into anything). Excludes the 22MB onnxruntime prebuilt binaries and
  3.2MB of dance/mimic policy assets — see `deploy_reference/README.md` for exactly what
  was left out and why.
- CSV logging: **no code changes needed** — both new gym task ids contain
  `"Locomotion"`, so `train.py`'s existing substring dispatch already routes them to
  `WalkingMetricsCsvWrapper`, and that wrapper has no DOF-count-specific assumptions
  (unlike `StandingMetricsCsvWrapper`, which isn't used here).

**NOT done today** (explicitly out of scope, see "Today's scope" below):
- The IK-driven and policy-driven arm-reach disturbance variants
  (`StandingArmIKReachDisturbance`/`StandingArmPolicyReachDisturbance`) — blocked on the
  7-DOF arm rewrite (`_GOAL_BOUNDS` etc. don't exist yet).
- Phase 3 (7-DOF arm rewrite) — not started.
- Phase 4/5 (demo/eval script updates) — not started, nothing to point them at yet.

**Real blockers before training can actually launch** (code-complete, environment is not):
1. **`UNITREE_MODEL_DIR` is still a placeholder** (`assets/robots/unitree.py`) — the
   USD asset pack needs to be fetched (`git clone
   https://huggingface.co/datasets/unitreerobotics/unitree_model`) and the path updated.
   Not done — didn't want to pull an unknown-size external dataset without checking
   first.
2. **Gain reconciliation not yet cross-checked line-by-line** — `UNITREE_G1_29DOF_CFG`'s
   actuator gains were ported as-is (trusted from the earlier comparison against
   `deploy.yaml`), but a full line-by-line diff hasn't been done. Worth a final check
   before a long training run, not before a first smoke-test.
3. **Symmetry module unverified against a live env** — see above; will either work or
   fail loudly on the very first PPO update once training launches.

Next steps once you're ready to actually run something: get the USD assets in place,
then launch a short smoke-test run of `G1-Locomotion-Velocity-v0` (few hundred
iterations, small `--num_envs`) to confirm the env constructs and trains at all, before
committing to a long run or moving on to the arm-disturbance variant.

## Progress — 2026-07-21, later same session (assets fetched, gains fixed, Phase 3 built)

- Assets fetched (see `assets/robots/unitree.py`'s docstring), gains reconciled
  against `deploy.yaml` with a real fix (N5020-16/W4010-25 damping 1.0→10.0, traced via
  git history to a commit that changed it for an unrelated 23dof config change).
- Both 50-iteration smoke tests (`G1-Locomotion-Velocity-v0` and
  `...-ArmDisturbance-v0`) ran clean — logs verified populated and correctly wired
  (domain randomization, asset path, `bad_orientation` termination, `arm_motion_disturbance`
  event, symmetry `null` as intended — all confirmed present in the dumped `env.yaml`/
  `agent.yaml`, not just source).
- Log paths renamed to `logs/rsl_rl/walking/...` (was `g1_locomotion/...`), per
  user request — keeps this parallel with the arm policy's own `logs/rsl_rl/arms/...`.
- `train.py`'s CSV-wrapper dispatch changed from `"Arm-IK" in task` to `"Arm" in task` to
  match the new (non-"IK") arm task naming.
- **Phase 3 (7-DOF arm policy) built**: new package
  `tasks/manager_based/g1_arm/` — `g1_arm_env.py` (`G1ArmEnv`, DirectRLEnv, 7-DOF
  per arm, position-only goal per the Phase-0.2 decision, built on
  `UNITREE_G1_29DOF_CFG` fixed-base instead of `G1_MINIMAL_CFG`), `mdp/symmetry.py`
  (full rewrite for the 32-D/7-D per-arm layout), `agents/rsl_rl_ppo_cfg.py` (wide-net +
  symmetry as defaults, not opt-in experiments — both were confirmed-good 23dof-phase
  findings), gym ids `G1-Arm-{Left,Right,Both}-v0` (+ `-Play-v0`). Deliberately
  drops "IK" from every name (user request, confirmed factually correct: the deployed
  policy is pure RL joint-space control).
- `_GOAL_BOUNDS` in the new task reuses the old 5-DOF task's numeric box as a **starting
  hypothesis only** — NOT yet validated for this kinematic chain. Built
  `validation/check_arm_reachability.py` (adapted from
  `check_arm_reachability.py`) to validate it — **not yet run**, this is the next
  Isaac Sim command needed before a real arm training run.
- Simplifications vs. the 23dof-era arm task (deliberate, not oversights): dropped the
  goal-difficulty curriculum and the overnight-sweep experimental variants
  (RewardShape/GoalCurriculum/StressRegion/PPOTuning/Entropy) — those were 23dof-phase
  investigations, not needed for a fresh start. Null-space regularization and joint-limit
  margins are uniform per-joint (not the old elbow_pitch-specific asymmetric weighting,
  since that joint doesn't exist in this form) — revisit with real per-joint data once
  training results come in.

**Not yet done**: the actual arm training run, Phase 4
(integration — needs a new arm-overlay action term for the demo path, plus the
walk→stop→immediate-reach failure-mode test and the old `--wait_arm_rest`-style flag
reconsidered for the unified-policy architecture — all per user notes, deferred until
Phase 3 has a real checkpoint), and updating `validation/eval_arm.py`-equivalent for
7-DOF (explicitly deferred by user until "when we get to the validation scripts").

## Progress — 2026-07-21, reachability check run + file/directory consolidation

- **Reachability check run** (`validation/check_arm_reachability.py --arm left
  --headless`): 97.0% of the goal box covered within 2cm (the actual `goal_threshold`),
  mean nearest-reachable distance 0.61cm — better than the 5-DOF box's own best (~65% at
  2cm, after two rounds of reshaping). The reuse hypothesis held; **no reshaping
  needed**. Right arm and any future joint-limit change should still be re-checked.
- **File/directory consolidation, per explicit user request**: this repo already has the
  full 23dof-era implementation preserved on the `23_dof` git branch, so there's no
  reason to keep a second, parallel "29dof-suffixed" copy alongside it on `main` — that
  just adds confusing duplication for no safety benefit. Deleted the old 23dof
  `g1_locomotion`/`g1_arm` task packages and `eval_standing.py`/`eval_standing_ikreach.py`
  (superseded — one unified policy now, no separate standing task) from `main`, and
  moved the new 29dof implementation into their place — same paths, same file names,
  same simple `validation/<test_name>/` output-directory convention as before (e.g.
  `validation/arm_reachability/`, not `validation/g1_29dof_arm_reachability/`). Also
  dropped the "29dof"/"IK" qualifiers from gym ids and class names now that there's no
  more ambiguity to disambiguate against on this branch (`G1-Locomotion-Velocity-v0`,
  `G1-Arm-Left-v0`, `G1LocomotionEnvCfg`, etc.) The 23dof branch is untouched by any of
  this — nothing described here is destructive to that history.
- Existing checkpoints/logs from the smoke tests are unaffected — the rename only
  touched code file paths, class names, and gym ids, not the environment's actual
  observation/action space or asset, so those `.pt` files remain fully valid to load.

## Progress — 2026-07-21, Phase 4 (integration demo + eval) built

- **`validation/integration_validation/eval_full_demo.py`** rewritten in place for the
  unified policy + 7-DOF arm. Kept: the `_ArmAttemptTracker` (goal-dwell-until-timeout,
  interrupted/timeout/success semantics), the goal-frame convention, the mirror-driven
  right arm, the per-bucket CSV/summary structure. Removed: `--arm_driver ik/event`
  (depended on `StandingArmIKReachDisturbance`, not ported yet — flagged rather than
  faked). Added: `walk_stop_reach`, the new bucket testing walk→stop→immediate-reach
  per user request, no settling delay, deliberately reproducing the failure mode
  instead of avoiding it.
- **`testing/general_testing/g1_full_demo.py`** rewritten in place. The unified policy
  removes the entire mode-switch state machine; arm reaching is no longer gated to
  "standing only" (the training curriculum doesn't distinguish either, per Phase 2).
  `ArmDisturbanceBlendJointPositionAction` (already built for training) turned out to
  be directly reusable for the demo/eval arm overlay too — no new action term needed.
- **Real bug caught and fixed in both files**: `ObservationsCfg.PolicyCfg`'s
  `history_length=5` means the flat observation tensor's `velocity_commands` block
  isn't a fixed 3-wide column slice the way the 23dof-era single-step layout was — the
  23dof-era demo's `obs[:, 9:12] = cmd` technique would have silently fed the policy a
  garbled command if ported as-is. Fixed by writing the command term's own live buffer
  (`vel_command_b`) instead, which flows through the normal observation pipeline
  (including the history buffer) correctly. Same root cause, two different call sites:
  `g1_full_demo.py`'s live keyboard command, and `eval_full_demo.py`'s
  `_force_command()` when called mid-rollout with no `reset()` in between (the
  `walk_stop_reach` bucket's whole point) — `_force_command` used to only update
  `cfg.ranges` (correct when followed by a `reset()`, silently a no-op otherwise).
  Neither bug would have thrown an error — both would have just quietly tested the
  wrong thing.
- `metrics_wrappers.py`'s `StandingMetricsCsvWrapper` fixed: `find_joints("torso_joint")`
  → `find_joints("waist_yaw_joint")` (the 29dof asset has no single torso joint; waist_yaw
  is the direct analog — the 23dof-era code's own comment already called it "a waist
  YAW joint").
- `testing/general_testing/checkpoints.yaml` updated for one `loco` checkpoint instead
  of separate `walking`/`standing`.

**Known remaining gap, not done**: three individual-component interactive test scripts
still reference the old 5-DOF/"IK" arm naming and, in one case, an now-obsolete concept
— `testing/arm_testing/g1_arm_reach_test.py`, `g1_arm_mirror_test.py` (need the same
7-DOF updates `eval_arm.py` got), and `testing/walking_testing/g1_stand_walk_switch_demo.py`
(tests the two-policy mode-switch mechanism that no longer exists — fully superseded,
candidate for deletion rather than updating). Not touched this session — flagged rather
than silently expanded into.

## Future consideration — sim2real "noise" on/off toggle (documented, not acted on)

Raised by the user 2026-07-21: domain randomization (actuator gain/friction/armature
scale, observation `Unoise`) is on from the very first training run in this repo's
recipe, unlike some workflows that ramp it in gradually or add it only once a clean
baseline is confirmed. Risk flagged: training with all of it on from the start may be
more ambitious than necessary and could slow early convergence or mask how much
capability is being "spent" on robustness vs. task performance. **Decision: leave it on
for the upcoming overnight runs** (per the user) — but if tomorrow's results look good,
we may want a way to quantify how much better the policy could be *without* all this
noise, i.e. an ablation, not just a toggle. Worth designing as an explicit CLI flag or a
`_PLAY`-style low-noise config variant when that question actually comes up — not
speculatively now.

---

Written 2026-07-21, following on from `29dof_pivot_context.md` (strategic decisions/
findings — read that first if you haven't) and `lessons_learned.md` (23dof-phase
gotchas). This is the concrete, phased plan for actually doing the rebuild. Based on a
direct read of both `~/Elm/Code/unitree_rl_lab` (as of its current checkout) and this
repo's current code (`source/g1_locomotion/`, `testing/`, `validation/`).

## Key facts this plan is built on (verified by reading the actual code)

- **Physical DOF breakdown, 29dof, confirmed from `unitree_rl_lab`'s own reward code**
  (`velocity_env_cfg.py`'s `joint_deviation_arms` pattern: `.*_shoulder_.*_joint`,
  `.*_elbow_joint` — singular, `.*_wrist_.*`): each arm is **shoulder_pitch, shoulder_roll,
  shoulder_yaw, elbow (single joint), wrist_roll, wrist_pitch, wrist_yaw = 7 DOF**. This is
  a structural change from the current 5-DOF model, not just "add 2 wrist joints" — the old
  model's `elbow_pitch`/`elbow_roll` pair collapses into one `elbow_joint`, and 3 wrist
  joints are added. Waist is 3 DOF (`waist_yaw`, `waist_roll`, `waist_pitch`), replacing the
  old single `torso_joint`. Legs are unchanged (6 per side). Total: 12 + 3 + 14 = 29. ✓
- **Isaac Lab already ships a matching asset**: `G1_29DOF_CFG` in
  `isaaclab_assets/robots/unitree.py` (`~/Elm/Code/IsaacLab/source/isaaclab_assets/...`).
  Confirmed joint groups: `legs` (hip×3+knee, DCMotor), `feet` (ankle×2, DCMotor), `waist`
  (3, ImplicitActuator, stiffness 5000/damping 5 — very stiff), `arms` (shoulder×3+elbow+
  wrist×3, ImplicitActuator, stiffness 3000/damping 10 — uniform, same for every arm
  joint), `hands` (index/middle/thumb — **not part of the 29 DOF count**, a dexterous-hand
  extra; per the pivot doc, disregard, the real robot's hands aren't in scope).
- **`unitree_rl_lab` has its own separate G1-29dof asset config**,
  `UNITREE_G1_29DOF_CFG` in `unitree_rl_lab/source/.../assets/robots/unitree.py` — used by
  their actual training recipe (`velocity_env_cfg.py` imports this one, not
  `isaaclab_assets`'s). Its actuator gains differ meaningfully from
  `isaaclab_assets.G1_29DOF_CFG` (e.g. arms are grouped/gained differently, not one flat
  3000/10 block). **These two assets are not interchangeable** — training against one and
  deploying assumptions from the other is exactly the "different gain than deployment"
  trap `lessons_learned.md` #3 warns about.
- **`deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml`** is the
  ground-truth, hardware-real spec for the walking policy: 29-element arrays for
  `joint_ids_map` (PhysX↔hardware index mapping), `stiffness`, `damping`,
  `default_joint_pos`, action `scale`/`offset` per joint, observation scales, and
  `history_length: 5`. This is more trustworthy than either asset file's defaults for
  "what gain does deployment actually run" — use it as the tie-breaker whenever the two
  asset configs disagree, per lesson #3.
- **No raw rsl_rl checkpoint exists anywhere in `unitree_rl_lab`** — only the exported
  `policy.onnx`. Their recipe must be trained from scratch in this repo; there is nothing
  to warm-start from.
- **`unitree_rl_lab`'s stand+walk is a single unified policy** (`rel_standing_envs=0.02`,
  `feet_gait` reward auto-disables under a velocity threshold), not two separate
  checkpoints the way this repo currently has `standing` and `walking` as different
  policies. This is a real architectural simplification opportunity — see Phase 3.
- **This repo's existing arm-disturbance-curriculum machinery
  (`g1_locomotion/mdp/events.py`) is DOF-count-agnostic in its joint-ordering logic**
  (fixed 2026-07-20 to derive columns from `find_joints()`'s actual returned ids, never
  from an assumed block layout — see `lessons_learned.md` #1) but has **hardcoded 5-DOF
  joint name lists** (`_ARM_JOINT_NAMES`, `_LEFT_ARM_JOINTS`/`_RIGHT_ARM_JOINTS` imported
  from `g1_arm_env.py`) that need updating to the 7-DOF names throughout.
- **The walking-task symmetry code (`g1_locomotion/mdp/symmetry.py`,
  `compute_symmetric_states`) is genuinely DOF-agnostic already** — it builds its
  left/right swap-index and sign maps dynamically from `robot.data.joint_names` at
  runtime, and its `_mirror_sign()` substring match (`"_roll_"`/`"_yaw_"` → flip) already
  correctly classifies `waist_roll_joint` (flips) and `waist_yaw_joint` (flips) vs.
  `waist_pitch_joint` (doesn't) with **no code change needed** — the old special case for
  the literal name `"torso_joint"` just becomes dead code to remove. This is one of the
  few pieces of this migration that's closer to "verify" than "rewrite."
- **The arm task's mirror code (`g1_arm/mdp/symmetry.py`) is the opposite** — hardcoded,
  fixed-size per-dimension sign vectors (`_OBS_SIGN`, `_ACTION_SIGN`) sized for the 28-D/
  5-D layout. This needs a full rewrite for 7-DOF, not a tweak.
- **The per-episode CSV logging infra (`g1_locomotion/utils/metrics_wrappers.py`) is
  reusable infrastructure, not stale data** — `StandingMetricsCsvWrapper`,
  `WalkingMetricsCsvWrapper`, `ArmMetricsCsvWrapper` are wired into `train.py` by a
  task-name string match and write the dual detailed+summary CSVs you asked about
  directly answering "are logs like the old scripts available" — **yes**, this is exactly
  that system, and it survived the 23dof→29dof cut deliberately. It needs updating in
  a few DOF-specific spots (e.g. `StandingMetricsCsvWrapper` currently does a single
  `find_joints("torso_joint")` lookup for its torso-rotation columns — needs to become 3
  waist-joint columns, or a combined metric).

## Phase 0 — Decisions to lock in before writing code

These aren't things I should guess at silently; flagging each with a recommendation.

1. **Which G1-29dof asset config to build on.** Recommendation: fork
   `UNITREE_G1_29DOF_CFG` (not `isaaclab_assets.G1_29DOF_CFG`) into this repo's own
   `g1_locomotion/assets/` (or similar), since it's what the reward/training recipe you're
   adopting was actually tuned against, then reconcile its gains against `deploy.yaml`
   explicitly (they should already match — `deploy.yaml` is the export of this same
   training config — but verify, don't assume). The arm joints' gains will very likely
   need a **separate, RL-tuned value** the way this repo's current arm task already
   diverges from stock G1 gains (200/20, chosen after training experiments, not any
   spec sheet) — expect the same to be true again at 7-DOF, and expect it to interact
   with whatever gain the unified stand+walk policy trains its arm joints at (same
   gain-matching lesson as #3/#4 in `lessons_learned.md`, now with three consumers of the
   same arm joints instead of two — standing, walking, and arm-IK are one policy now on
   the loco side).
2. **Does the arm goal stay position-only (3-DOF), or become position+orientation
   (6-DOF), now that wrists exist?** The old 5-DOF arm had no way to control palm
   orientation independently of position — 3 wrist DOF exist specifically for that. Colossal
   scope difference: position-only means the wrist joints are just 3 more redundant DOF for
   reaching a point (more null-space, same reward structure); position+orientation means
   redesigning the goal representation, reward function, and observation space around a
   6-DOF pose target. **Recommendation: keep position-only for the first 29dof arm
   checkpoint** (de-risks by changing one thing at a time — DOF count — before also
   changing the task's goal semantics), explicitly note orientation-goal as the natural
   Phase 2 arm extension once position-only is validated. Flag this back to you before
   committing — it changes the shape of a lot of downstream code.
3. **Does the arm-motion disturbance curriculum apply only near-zero-velocity ("standing")
   episodes, or all episodes including active walking?** The old repo's curriculum was
   standing-only by construction (a separate standing task). The new unified policy walks
   too, and your own restated goal #4 is "expand to move arms while walking" — so the
   disturbance should probably eventually apply regardless of commanded velocity.
   Recommendation: land the unified base policy first without arm disturbance (matching
   `unitree_rl_lab`'s own recipe, to get a known-good baseline you can compare against),
   then add the arm-disturbance curriculum as a second training stage across the *whole*
   command distribution (not gated on standing-only) — this is more ambitious than the old
   standing-only curriculum but matches what deployment will actually need.
4. **Repo structure**: port `unitree_rl_lab`'s task config into this repo's own
   `source/g1_locomotion/g1_locomotion/tasks/manager_based/` tree (e.g. a new
   `g1_locomotion` package sitting alongside `g1_locomotion`/`g1_arm`), rather than
   taking a runtime dependency on the `unitree_rl_lab` Python package. Keeps this repo
   self-contained and lets the existing disturbance-curriculum/metrics-wrapper/symmetry
   layering pattern (the `G1LocomotionStandingFlatEnvCfg` → `...IKReach` → `...Height` →
   ... chain) apply the same way it already does. The old `g1_locomotion` package
   (5-DOF-arm-aware, 1-DOF-waist) stays as reference/fallback per the pivot doc's
   explicit "not deleted, reversible" decision — don't delete it, build the new one
   alongside.

## Phase 1 — Port the walking/standing base (unitree recipe → this repo)

1.1. Copy `unitree_rl_lab`'s locomotion task files into the new package: `velocity_env_cfg.py`
     (rewards/observations/actions/terminations/curriculum/scene), `mdp/rewards.py`,
     `mdp/observations.py`, `mdp/curriculums.py`, `mdp/commands/velocity_command.py`
     (`UniformLevelVelocityCommandCfg`), `agents/rsl_rl_ppo_cfg.py`
     (`BasePPORunnerCfg` — 512/256/128 net, 50000 max_iterations, standard PPO
     hyperparams). Adapt imports to this repo's package layout.

1.2. Bring in the asset config per the Phase 0.1 decision, reconciled against
     `deploy.yaml`'s literal gain/default-pose arrays.

1.3. Register the new gym task ID(s) (`__init__.py`, following the existing
     `gym.register(...)` pattern already in both repos).

1.4. **Train the unadorned base recipe first, no arm curriculum, no repo-specific
     additions** — this is the "does the ported recipe actually reproduce Unitree's
     result in our environment" sanity gate before building anything else on top of it.
     Compare against their reward-curve shape/convergence qualitatively (no numeric
     checkpoint to compare against directly, per the pivot doc — theirs is
     ONNX-only). Two seeds minimum before trusting the result, per lesson #5.

1.5. Wire in `metrics_wrappers.py` logging (`WalkingMetricsCsvWrapper`, or a merged
     "walking+standing" variant now that it's one policy — the CSV columns designed for
     a pure-standing task, e.g. `mean_max_tilt_deg`/stepping metrics, are still relevant
     to the near-zero-velocity slice of this unified task; consider whether
     `StandingMetricsCsvWrapper` and `WalkingMetricsCsvWrapper` should merge into one
     wrapper now that both regimes live in one policy/one training run, vs. keeping them
     separate and picking one by `common_step_counter`/commanded velocity per episode).
     This is a design call worth making deliberately rather than defaulting either way.

1.6. Port the `bad_orientation` hard termination (already validated as a good idea in
     this repo's own `TorsoClipBadOrientationEnvCfg` experiment, `limit_angle=0.8`,
     literally copied from this same `unitree_rl_lab` source) — you already did the
     legwork proving this is worth having; carry it forward as a first-class part of the
     base recipe rather than rediscovering the need for it.

## Phase 2 — Layer the arm-motion disturbance curriculum onto the base

2.1. Port `StandingArmTrajectoryDisturbance`/`StandingArmIKReachDisturbance`/
     `StandingArmPolicyReachDisturbance` (`events.py`) to the 29dof joint names (14
     arm joint names instead of 10) and to whatever `_GOAL_BOUNDS`-equivalent the new
     7-DOF arm task ends up using (can't reuse the old bounds directly — reachable
     workspace changes with the extra wrist reach/rotation, needs its own
     `check_arm_reachability.py`-style validation, not assumed).

2.2. Per the Phase 0.3 decision, decide the scope (standing-only vs. all commanded
     velocities) and implement accordingly — likely a bigger event-manager change than a
     simple port, since "which envs get disturbance" currently isn't conditioned on the
     velocity command at all in the old standing-only task (it didn't need to be, there
     was no velocity command).

2.3. Re-run the same layered single-variable experiment discipline that got this far
     last time (a `base_height_l2` fix, gain-matching to the arm-IK policy's real gain,
     the arm-intent observation, the torso action clip, the hard orientation
     termination) — expect this list to need re-discovering to some degree since the
     underlying dynamics (29dof, 3-DOF waist instead of 1, 7-DOF arm instead of 5) are
     different enough that the old ordering/magnitudes are a starting hypothesis, not a
     guarantee. `lessons_learned.md` #2 (price every joint from day one) is the one
     lesson most directly worth front-loading here rather than re-discovering
     reactively — consider costing waist AND hip deviation from the first config, not
     iterating into it the way the 23dof phase did.

## Phase 3 — Update the arm policy to 7-DOF

3.1. Rewrite `g1_arm_env.py`'s joint lists (`_LEFT_ARM_JOINTS`/`_RIGHT_ARM_JOINTS`, now 7
     each), observation dims (28-D → new per-arm width, recompute exactly once the
     Phase 0.2 orientation decision is locked — position-only stays 3+3+3
     goal/ee/error, adds 2 more joint-pos/joint-vel columns each = 32-D per arm instead
     of 28-D if position-only, more if 6-DOF goal), action dim 5→7.

3.2. Re-derive `_GOAL_BOUNDS` for the 7-DOF arm from scratch via
     `check_arm_reachability.py` (updated for 7-DOF) — do **not** carry the old bounds
     forward even provisionally; the reachability fix history in this repo
     (`lessons_learned.md`/`known_issues.md` on the `23_dof` branch) is a direct
     demonstration of how expensive an unverified goal box is (weeks lost to an
     unreachable ~53% of the box before the first fix).

3.3. Rewrite `g1_arm/mdp/symmetry.py`'s mirror vectors for the new 7-DOF per-arm block
     (`wrist_roll`/`wrist_yaw` flip sign under mirror, `wrist_pitch` doesn't — same
     convention as everything else, just 2 more entries to add correctly).

3.4. Re-tune actuator gains for the 7-DOF arm (don't assume 200/20 carries over — new
     joints, new mass distribution at the end of the chain) — and resolve this against
     whatever gain Phase 2's arm-disturbance curriculum ends up training the loco
     policy's arm joints at, per the Phase 0.1 gain-matching note.

3.5. Update `checkpoints.yaml`, `g1_arm_reach_test.py`, `g1_arm_mirror_test.py` for the
     new joint lists/dims.

3.6. Train, applying the same lessons: two seeds minimum (#5), watch for the
     manipulator-redundancy null-space issue this repo already found once (now with 2
     more redundant DOF, expect it to be *more* present, not less — the
     `null_space_penalty_scale`/per-joint-weighted tiebreaker mechanism this repo built
     for exactly this problem should carry over conceptually, needs re-verifying which
     joint(s) play the "two branches, same goal" role this time, likely wrist as well as
     elbow now).

3.7. Wire in an updated `ArmMetricsCsvWrapper` (7-DOF-aware — the joint-config-at-min-
     dist columns are per-joint-name-driven already, should mostly just work once the
     joint list is right, verify the `null_space_weight`-style diagnostics still make
     sense with wrist joints in the mix).

## Phase 4 — Integrate into `g1_full_demo.py`

4.1. **Simplification opportunity**: since the loco side is now one unified stand+walk
     policy instead of two, the whole mode-switch state machine in `g1_full_demo.py`
     (`MIN_MODE_STEPS`, `SWITCH_TO_WALK_THRESHOLD`/`SWITCH_TO_STAND_THRESHOLD`,
     `_start_transition`, `_maybe_switch_mode`, the standing/walking policy swap) goes
     away — replaced by just feeding the velocity command straight to the one policy.
     The arm-overlay machinery (`StandingArmBlendJointPositionAction`, the homing logic,
     per-mode arm gain switching, the mirror-testing L/R/Y/U keys) is orthogonal to that
     state machine and should port over close to unchanged, just re-pointed at the new
     joint names/dims.

4.2. Update `_LEFT_ARM_JOINTS`/`_RIGHT_ARM_JOINTS`/`_LEFT_EE_BODY`/`_RIGHT_EE_BODY`
     imports, `ARM_ACTION_SCALE`/`ARM_MAX_JOINT_DELTA_PER_STEP`/
     `ARM_ACTION_FILTER_ALPHA` to match Phase 3's actual trained values (these are
     currently hardcoded constants that must match training exactly, per lesson #4 —
     a demo/eval env built from a different lineage silently dropping a config change
     is a proven failure mode here).

4.3. Update `_GAIN_STANDING`/`_GAIN_ARM_IK`/`_GAIN_WALKING` to the new resolved gains
     from Phases 1-3.

4.4. Re-verify the observation-frame handling in `_build_arm_obs` (`quat_apply_inverse`
     rotation into body frame) still applies correctly — should, since it's
     geometry-generic, not joint-count-specific, but verify.

## Phase 5 — Update validation/eval scripts

5.1. `eval_standing.py`/`eval_walking.py` → likely merge into one `eval_locomotion.py`
     (or keep as two views into the same unified checkpoint, sweeping the same
     phase/command axes each already sweeps) — decide based on how Phase 1.5's
     logging-merge question resolves.

5.2. `eval_arm.py`, `check_arm_reachability.py`, `check_ik_accuracy.py` → update for
     7-DOF joints/dims, no other structural change needed (the methodology — fixed-seed
     rollout, bucketed sweep, CSV+summary.md — is exactly what the pivot doc says to
     keep).

5.3. `validation/integration_validation/eval_full_demo.py` → same joint/dim/gain
     updates as `g1_full_demo.py` (Phase 4), since it deliberately mirrors that script's
     env-construction logic closely (per lesson #4, these two must never drift apart on
     any config change that isn't purely reward/observation).

5.4. Re-establish baselines: none of the "what does healthy look like" tables in
     `validation/README.md` carry over numerically (different DOF count, different
     dynamics, different policy) — the first fully-converged 29dof run becomes the new
     reference point, same as how `walking`/`standing`'s current tables cite specific
     historical runs.

## Phase 6 — Sim2real groundwork (explicitly flagged as real, not incidental, effort)

Per the pivot doc: `unitree_rl_lab`'s `deploy/` C++ harness (ONNX runtime, FSM with
Passive/FixStand/Velocity states, a full C++ port of `ManagerBasedRLEnv`'s obs/action
pipeline) is a substantial, currently-nonexistent piece of engineering in this repo.
Not blocking for Phases 1-5 (everything above is sim-only), but worth scoping explicitly
once a checkpoint is ready to leave simulation:

6.1. The `deploy/robots/g1_29dof/` C++ project is the target harness — review
     `main.cpp`/`src/State_RLBase.cpp`/`include/Types.h` for the actual inference-loop
     structure once there's a checkpoint to export.

6.2. Exporting this repo's trained checkpoint to ONNX in the same input/output
     convention `deploy.yaml` documents (joint order via `joint_ids_map`, obs
     scale/history_length=5, action scale/offset per joint) — this repo doesn't
     currently have an ONNX export step at all (the removed `chosen_checkpoints/
     exported/` was 23dof-era and gone); needs building.

6.3. `unitree_actuators.py`'s `DelayedPDActuator`-based friction/delay model (added per
     `CHANGELOG.rst` 0.2.0) exists in `unitree_rl_lab`'s assets module but is **not**
     currently applied to the G1 configs there (used for Go2/other robots' actuator
     configs in that same file) — don't assume it's already "free" sim2real coverage for
     G1 just because the changelog line exists; check the actual `UNITREE_G1_29DOF_CFG`
     actuator definitions before crediting this, and consider adopting it explicitly if
     not already wired in (this is exactly the actuator-delay gap
     `29dof_pivot_context.md` flags as something the project's own deferred-items list
     had wanted).

## What I'd explicitly NOT change

- Don't touch the `23_dof` branch or `~/Elm/Backups/g1_locomotion/23_dof/` — stays the
  reversible fallback per the pivot doc's own words.
- Don't try to warm-start anything from `unitree_rl_lab`'s ONNX-only checkpoint — nothing
  to warm-start from, confirmed above.
- Don't carry forward the old `_GOAL_BOUNDS`, arm gains, or "what good looks like"
  validation tables numerically — every one of them is specific to the old 5-DOF/
  1-DOF-waist dynamics and must be re-derived, not adapted.

## Phase-0 decisions — RESOLVED 2026-07-21

1. **Asset config: `unitree_rl_lab`'s `UNITREE_G1_29DOF_CFG`, exclusively.** Confirmed by
   direct comparison that its per-joint stiffness/damping match `deploy.yaml` (the
   hardware-exported spec) — this is clearly the actual source of truth. Do not use
   `isaaclab_assets`'s `G1_29DOF_CFG` (generic defaults, e.g. a flat 3000/10 arm gain
   that isn't what any real recipe trains against).
2. **Arm goal: position-only 3-DOF for now.** Orientation (using the wrist DOF for what
   it's actually for) is an explicit, planned follow-up once position-only is validated
   — not forgotten, just sequenced later.
3. **Arm-disturbance curriculum scope: land it early, IK-driven, standing-first is fine
   as the starting point but don't treat it as optional or late-stage.** Confirmed via
   `phase_logs/phase_1.md`/`phase_2.md`: the 23dof phase's standing policy needed a real,
   multi-iteration arm-disturbance curriculum to not fall over the first time a real
   reaching arm was introduced (scripted joint-swing → real IK reach target → the
   `find_joints()` breadth-first joint-order bug that cost ~1 week → training against the
   actual frozen arm policy instead of clean IK → a gain-mismatch collapse). A unified
   stand+walk policy trained with zero arm disturbance will almost certainly fail the same
   way. The IK-driven version of this curriculum does NOT need a trained arm policy to
   exist first (it drives disturbance via real numerical IK), so it can start right after
   the base unified policy lands, in parallel with the Phase 3 arm rewrite rather than
   blocked on it — resequenced into today's scope below.

## Sim2real findings from reading `unitree_rl_lab`'s actual code (2026-07-21)

Answering "does Unitree have anything that helps with sim2real, e.g. IMU/encoder noise":

- **Observation noise**: yes — `Unoise` on `base_ang_vel` (±0.2), `projected_gravity`
  (±0.05), `joint_pos_rel` (±0.01), `joint_vel_rel` (±1.5). No noise on `base_lin_vel`
  because the *policy* observation group doesn't even include it (only the critic does,
  unnoised) — a deliberate actor/critic asymmetry matching real hardware (base linear
  velocity isn't directly measurable without extra instrumentation on a real robot).
  These numbers already match this repo's own arm task (`g1_arm_env.py`'s own docstring:
  "matching G1FlatEnvCfg's magnitudes") — consistent, not a new finding.
- **Domain randomization present**: ground friction/restitution, base mass (±1 to
  +3kg), random velocity pushes every 5s. A reset-time force/torque event exists but is
  configured with a **zero range** — present as a toggle, not active.
- **Domain randomization ABSENT from their leg/waist recipe** (real gap, and a place
  this project is already ahead): no actuator gain randomization, no joint friction/
  armature randomization. `g1_arm_env.py` already does both on the arm joints (gain
  scale 0.8-1.2×, friction 0-0.03, armature 0.8-1.2×) — worth carrying that practice to
  the new unified task's leg/waist/arm joints too, not assuming Unitree's recipe already
  covers it.
- **Actuator delay + Coulomb friction model** (`UnitreeActuator`, a `DelayedPDActuator`
  subclass: `Fs*tanh(vel/Va) + Fd*vel`) exists in `unitree_rl_lab`'s codebase as reusable
  infrastructure but is **confirmed NOT applied to G1** (verified by reading
  `UNITREE_G1_29DOF_CFG`'s actual actuator definitions — plain `ImplicitActuatorCfg`
  throughout; the delay/friction model is used for Go2/other robots' configs in the same
  file, not G1). This is real, available, not-yet-applied groundwork — worth adopting
  deliberately for G1 rather than assuming it comes free with the recipe. Directly
  answers the actuator-delay gap `29dof_pivot_context.md` flags.
- **New capability this project didn't have before**: terrain-difficulty and
  command-velocity curricula (`terrain_levels_vel`, `lin_vel_cmd_levels`) that ramp
  difficulty as training progresses, instead of fixed ranges for the whole run.

## What gets ported from `unitree_rl_lab`, exactly

**Into this repo's own task package** (new package alongside `g1_locomotion`/`g1_arm`,
not a runtime dependency on the external repo):
- `velocity_env_cfg.py` + `mdp/rewards.py`, `mdp/observations.py`, `mdp/curriculums.py`,
  `mdp/commands/velocity_command.py` (`UniformLevelVelocityCommandCfg`)
- `agents/rsl_rl_ppo_cfg.py` (`BasePPORunnerCfg`)
- `assets/robots/unitree.py`'s `UNITREE_G1_29DOF_CFG` + its `UnitreeArticulationCfg`/
  `UnitreeUsdFileCfg` base classes
- `assets/robots/unitree_actuators.py` (`UnitreeActuator`/`UnitreeActuatorCfg`)
- The `gym.register(...)` pattern

**Copied in as inert reference** (not wired into anything, per the user — no physical
robot access for another week): `deploy/robots/g1_29dof/` (C++ FSM + ONNX inference
loop), `deploy/include/` (shared FSM/`isaaclab`-port headers), `deploy.yaml`/
`config.yaml`.

**Not ported**: `mimic/` (dance-motion tasks, unrelated), go2/h1/b2 robot configs,
`docker/` (this repo has its own env setup).

## Today's scope (2026-07-21 session)

Given "next few hours," arms is a stretch goal, not a commitment:
1. Port Unitree's asset/actuator config + walking/standing task into this repo, register
   the gym task.
2. Copy the `deploy/` C++ harness in as inert reference material.
3. Adapt the symmetry-augmentation code (expected close to drop-in — it's already
   joint-name-driven, not hardcoded) and the CSV logging wrappers onto the new task.
4. Port the IK-driven arm-disturbance curriculum (joint names updated to the 14 new arm
   joint names) onto the new unified task — resequenced earlier per the Phase-0.3
   resolution above, does not need the 7-DOF arm policy to exist first.
5. Stretch goal only if time remains: start the 7-DOF arm rewrite (Phase 3) — not
   committing to finishing it today given the goal-bounds re-derivation alone was a
   multi-day effort last time.
