# DEFINITIVE NEXT STEPS — G1 stand/walk/arm integration (written 2026-07-16)

Handoff for a fresh session. Everything below was verified against code and eval CSVs on
2026-07-16. Read this fully before acting; do NOT re-derive or re-litigate — the history
is summarized here precisely so you don't repeat it. This is the ONLY maintained status
document (docs were consolidated 2026-07-16 — see "Documentation map" at the bottom).

## Hard rules (user-set, non-negotiable)

1. **The user runs all Isaac Sim / training commands** (conda env `isaac_g1_control`).
   You prepare commands/configs and hand them over.
2. **Do not read or edit .md files without asking first** (context cost). This includes
   plan.md and memory files.
3. **Never trust a single training run**: PPO on the standing task has PROVEN massive
   seed variance (identical config produced 2.1% vs 100% fall rates —
   `consolidated` vs `consolidated_seed7`). Any claim from a new training needs 2 seeds.
4. The user strongly dislikes: persistent torso rotation at spawn, squatting, and any
   "weird" resting posture. Factor this into every policy choice.
5. Every experiment = new gym ID + new config class (established pattern in
   `g1_locomotion_env_cfg.py` + `__init__.py`); checkpoints are immutable;
   `chosen_checkpoints/*.pt` is the manually-updated pointer set.

## Current state — what is SOLVED

**The week-long "integration falls" mystery is closed.** It was a stack of
deployment-side bugs (all fixed, all in the working tree), NOT a policy problem:

- per-goal arm teleports in the eval (removed),
- goal resampled instantly on success instead of training's 15s dwell (fixed),
- eval reimplemented the arm-driving instead of reusing training's code (fixed via
  `--arm_driver event`, which attaches training's own disturbance term),
- the disturbance term lost its `_standing_arm_motion_joint_ids` env attribute when
  another bucket nulled it → arms silently driven by the standing policy's unshaped
  garbage output (fixed: term re-asserts both attrs every call — `mdp/events.py`),
- policy loaders now infer input width from the checkpoint
  (`_load_loco_policy` in both deployment scripts; handles the 85-D intent checkpoint
  and 75-D ones with no flags; note loco runner cfgs lack `obs_groups` — loaders fall
  back to `{"policy": ["policy"], "critic": ["policy"]}`),
- `RslRlVecEnvWrapper` returns TensorDict obs — loaders normalize via `obs["policy"]`.

**Result (verified, 3 seeds, no training involved):** checkpoint
`logs/rsl_rl/standing/g1_locomotion_flat/2026-07-13_23-52-48_height_reward/model_5999.pt`
in the integration eval with `--arm_driver event` scores **0% falls in every bucket**,
tilt ~5–6°, height 0.72–0.73 m (no squat), peak |torso| ~20–22° (transient, not a held
rotation). Standing + walking + IK-style arm motion is a working integrated system.

**8-checkpoint sweep (2026-07-16, event mode, all in
`validation/integration_validation/`, each run's `run_meta.yaml` names its checkpoint):**
height_reward is the ONLY checkpoint passing both standing_still and the reach bucket at
0%. Each policy is robust only to its own training-style arm motion (consolidated line —
trained vs the RL arm at 200/20 — falls 26–91% under the IK disturbance; IK-trained ones
fall 92–100% with STATIC arms — the "static-arm hole", closable with the
`no_reach_prob` idle-slice param that already exists in `StandingArmIKReachDisturbance`).

## Current state — the REMAINING problem (this is the work)

**The arm never actually reaches.** New logging (`event_arm_goals.csv`, printed
per-side stats in the event bucket) revealed the training disturbance's analytic IK
misses its goals by **p50 ≈ 39–54 cm, ~0% within 3 cm** — it has been this weak in every
training run since it was introduced, unlogged until now. Consequences:

- Standing checkpoints were trained against arm-WAVING, not arm-REACHING. That's why
  the RL arm policy (which genuinely reaches: 2 cm native / ~10 cm deployed) kills them.
- The stable integrated system above "gestures toward" targets; the reach deliverable
  is NOT met.
- The user distrusts Isaac's IK — justified. Prime suspect: the floating-base Jacobian
  indexing in `StandingArmIKReachDisturbance` (`_jacobian_b`, `_jacobi_joint_ids =
  joint_ids + 6`, body idx unshifted) and its copy `_IKArmDriver` in
  `eval_full_demo.py`. A 50 cm miss over 15 s of rate-limited iteration (0.06 rad/step,
  gains 60/1.5) smells like mis-solving, not gravity droop (droop at kp=60 explains
  ~10 cm at most).

## NEXT STEPS, in order

### Step 0 — promote the winner — **DONE (2026-07-16)**
`chosen_checkpoints/standing_latest.pt` is now the height_reward checkpoint
(`logs/.../2026-07-13_23-52-48_height_reward/model_5999.pt`); the previous pointer is
preserved as `chosen_checkpoints/standing_2026-07-09_prev.pt`. See
`chosen_checkpoints/README.md`.

### Step 1 — FK-verified IK accuracy test (build it; ~3 min runtime; DECISIVE)
Plain-language for the user: FK (forward kinematics) = "given joint angles, where is
the hand" — exact, free from the sim, used to CHECK things. IK (inverse) = "given a
desired hand position, what joint angles" — the hard direction, what Isaac's
`DifferentialIKController` does, and what our measurements say is broken in our usage.

Build a small script (suggest `validation/check_ik_accuracy.py`): fixed-base arm env
(`g1_arm_env.py` machinery or the arm task's PLAY env), drive the arm with
`DifferentialIKController` using EXACTLY the same code path as
`StandingArmIKReachDisturbance._step_side` (copy it), sample goals in `_GOAL_BOUNDS`,
log FK hand-to-goal distance. Compare fixed-base vs floating-base (the walking env)
if possible.
- IK accurate on fixed base but not floating → Jacobian indexing bug for floating base
  (the `+6` offset / body row convention) → fix in `events.py`, and the fix
  automatically applies to training AND deployment (they now share the code).
- IK inaccurate everywhere → our controller settings/usage are wrong (or DLS+rate-limit
  +soft-gain combination genuinely can't track) → consider computing IK at stiff
  "virtual" tracking internally, or switch the plan to the RL-arm route (Step 3).

### Step 2 — ONE final standing retrain, only after Step 1 fixes reaching
Once IK actually reaches, the training disturbance becomes much STRONGER than what
height_reward survived — its 0% will not automatically hold. Config: height_reward's
recipe (`G1LocomotionStandingFlatIKReachHeightEnvCfg` lineage) + the fixed IK + two
proven add-ons:
- `no_reach_prob=0.15` (closes the static-arm hole — param already implemented),
- the arm-intent observation (`mdp.standing_arm_motion_targets`, 10-D appended obs —
  fully implemented incl. deployment support; it produced the best posture ever
  measured, ~8° torso, in `consolidated_intent`) — directly serves the user's
  no-rotation requirement.
**Run TWO seeds** (rule 3). Gate each checkpoint before full evals:
`validation/eval_standing_ikreach.py --checkpoint <ckpt> --env_cfg <cfg> --freeze_arms`
(pass = fall_rate ≈ 0; this catches the static-arm collapse in 90 s).
Then the integration eval (`--arm_driver event`) + check `event_arm_goals.csv` reach
stats + `mean_max_abs_torso_deg` (want < ~25° peak; the column exists in all standing
summaries since 2026-07-15).

### Step 3 — RL arm upgrade path (optional, after Steps 1–2 deliver)
Only if IK reaching quality is insufficient: retrain the arm policy at hardware gains
(60/1.5 — verified against unitree_sdk2_python's own G1 arm examples; the current
checkpoint's 200/20 has no hardware counterpart) via a gain curriculum from the existing
checkpoint, plus a stillness-at-goal reward (penalize arm joint_vel within ~3–4 cm of
goal — kills the near-goal jitter that destabilizes standing). Then a standing retrain
against it (the `consolidated` recipe — policy-driven disturbance + gain match — its
one good seed proved the approach works). 2 seeds, always.

### Step 4 — finish integration (after 2)
- Wire the event-term-style arm driving with USER-TYPED targets into
  `testing/general_testing/g1_full_demo.py` (currently it still uses the RL arm) so the
  user can visually confirm: stand, reach, walk, switch. Note the demo already has: the
  smooth homing move (no teleports), per-mode arm gains, the transition arm-flail fix
  (arm action columns from the walking policy only during blends), and the
  checkpoint-width-inferring loader.
- Then the known leftover items: batched walk↔stand transition eval (transitions are
  exercised but never measured; user saw falls there), and the inner goal-box margin
  (near-inner corner is reachable but destabilizing/elbow-limited — trim only after
  reach works).

## Key facts a new session will otherwise get wrong

- **Native evals do NOT predict integration** for checkpoints with the static-arm hole;
  always run the `--freeze_arms` gate first.
- `eval_full_demo.py` flags: `--arm_driver {policy,ik,event}` (`event` = training's own
  term = the trusted mode; `ik` = a script-local reimplementation, currently broken,
  don't trust it), `--active_arm_gain KP KD`, per-run `run_meta.yaml` records everything.
- In event mode: the 4 side/edge reach buckets collapse to one
  `standing_arm_event_reach`; the walking bucket's arms are pinned (ignore its arm
  realism); reach accuracy prints to console AND lands in `event_arm_goals.csv`.
- `eval_standing_ikreach.py` writes to `<checkpoint_dir>/ikreach_eval/<timestamp>/`
  (moved 2026-07-15) and has `--env_cfg` choices per training variant + `--freeze_arms`
  / `--no_push` bisect flags. All eval scripts write `run_meta.yaml`.
- Standing PPO runner has NO symmetry augmentation (docs are right; a grep can mislead).
- `StandingMetricsCsvWrapper` logs `max_abs_torso_deg`/`mean_abs_torso_deg` — the
  torso-rotation metric the user cares about. `arm_disturbance_phase` column is stale
  for IKReach-era runs (ignore it).
- Torso-deviation penalties were tried twice and both produced worse/collapsed policies
  — do NOT retry; posture is fixed with the intent observation instead.
- The walking policy is untouched and fine (0% falls everywhere, always).
- 8 standing checkpoints on disk under `logs/rsl_rl/standing/g1_locomotion_flat/`;
  sweep results in the eight 2026-07-16 folders of `validation/integration_validation/`.
- `overnight_train.sh` is rewritten per sweep (check contents, never assume);
  `overnight_train_2.sh` shows the wait-for-GPU-then-queue pattern for stacking runs.

## Experiment verdicts — what worked and what didn't (don't retry the dead ends)

Standing training experiments (all 6000-iter PPO, single seed unless noted; final
integration numbers from the 2026-07-16 event-mode sweep + earlier policy-mode runs):

| Experiment | Idea | Verdict |
|---|---|---|
| dwell_phase_fix | hold goals 15s in training | superseded; has the static-arm hole, squats (0.64m) |
| **height_reward** | + base_height_l2 (anti-squat) | **WINNER — chosen checkpoint.** 0% falls everywhere under training-matched arm motion, 0.72m, ~20° peak torso |
| torso_retighten / consolidated_torso | penalize torso deviation | **DEAD END, tried twice** — didn't remove rotation, produced collapse-group policies |
| leg_symmetry | penalize left-right leg asymmetry | best native tilt ever, but 100% falls with static arms; superseded by no_reach_prob |
| policy_driven / consolidated | train vs the REAL RL arm at deployment gains | right idea, works when the seed cooperates (2.1% falls once) — the recipe for Step 3 |
| consolidated_intent | + arm-intent observation | **best posture ever (~8° torso)** — adopt the obs term in Step 2; not robust alone |
| consolidated_noreach | + idle-episode slice | closes the static-arm hole — adopt in Step 2 |
| consolidated_seed7 | same config, new seed | proved seed variance dominates → the 2-seed rule |

Deployment/eval bugs found and fixed (the ACTUAL cause of the week of bad integration
results): per-goal arm teleports; instant goal resample vs training's dwell; gain
mismatches (200/20 vs 60/1.5, now CLI-controlled and hardware-verified vs
unitree_sdk2_python); the nulled `_standing_arm_motion_joint_ids` buffer (garbage arm
control); a script-local IK reimplementation that never worked (`--arm_driver ik` —
don't trust it); TensorDict obs handling in the checkpoint-width-inferring loaders; and
the transition arm-flail in the demo (blended garbage arm actions at walk onset).
What remains genuinely unsolved: the analytic IK never reaches (Step 1) and therefore
standing has never trained against true reaching (Step 2).

## Documentation map (post-cleanup, 2026-07-16)

- `definitive_next_steps.md` (this file) — the ONLY maintained status/next-steps doc.
- `README.md` — project intro (untouched).
- `chosen_checkpoints/README.md` — current checkpoint provenance (maintained).
- `known_issues.md` — frozen historical archive (banner at top; code comments point
  into it; do not act on it without checking here).
- `validation/README.md` — eval-script intro with a staleness banner; trust script
  `--help` over it.
- `phase_logs/*.md` — deep per-phase history, archival.
- `personal_development/*.md` — the user's own learning notes, leave alone.
- Deleted 2026-07-16 (superseded/stale): `plan.md`, `next_big_steps.md`,
  `logging_reference.md` (columns now documented in `metrics_wrappers.py`),
  `training_regimes.md` (config docstrings are authoritative),
  `quickrun.md` + `testing/quickrun_tests.md` (commands live here and in `--help`).

Deferred pre-hardware items (from the deleted next_big_steps, still valid): enable
push/external-force events + actuator-gain DR for legs before any real-robot deployment;
model action/obs latency; `base_lin_vel` is privileged ground truth in the obs (real G1
estimates it); batched walk↔stand transition eval; MoveIt2 is the likely real-hardware
arm-IK stack if analytic IK stays in the picture.
