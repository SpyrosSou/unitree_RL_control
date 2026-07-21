# DEFINITIVE NEXT STEPS — G1 stand/walk/arm integration (written 2026-07-16, updated 2026-07-21)

Handoff for a fresh session. Everything below was verified against code and eval CSVs on
2026-07-16, plus a 2026-07-21 section (read FIRST — most current) and two 2026-07-20
sections (read in order after that — the second corrects a framing mistake made earlier
the same day): the IK-accuracy fix (supersedes the "REMAINING problem"/Step 1 framing
below it), then the RL-arm-policy correction (supersedes `--arm_driver event`/"trusted
mode" language wherever it appears below and Step 3's "optional" framing). Read this
fully before acting; do NOT re-derive or re-litigate — the history is summarized here
precisely so you don't repeat it. This is the ONLY maintained status document (docs were
consolidated 2026-07-16 — see "Documentation map" at the bottom).

## 2026-07-21 UPDATE — Attempt 3 results, the root cause of the whole torso/hip pattern, and two queued follow-ups (read this first)

**Attempt 3 results (TorsoClip +/-30° vs TorsoLock 0°, one seed each, both retrained and
fully evaluated including a corrected integration run — see below):**

| | GainMatch (unclipped) | TorsoClip ±30° | TorsoLock 0° |
|---|---|---|---|
| Gate tilt (idle, 0% falls) | 8.42° | 37.19° | 25.80° |
| Gate torso reading | 60.27° (the exploit) | 32.17° (bounded, as designed) | 0.69° (locked, as designed) |
| Native (active reach) falls | 6.0% | 26.3% | 51.0% |
| Policy-mode integration falls (left/left_edge/right/right_edge) | 0%/0%/5.2%/12.5% | 0%/2.1%/24.7%/22.7% | 79.3%/65.9%/6.3%/12.1% |
| hip_roll frac_of_soft_limit_used (native, left/right) | 10.9%/27.1% | 60.7%/10.9% | 55.2%/30.9% |

**Verdict: Clip clearly beats Lock — Lock is DROPPED as a direction.** Confirmed two
ways: (1) numerically, Lock is worse almost everywhere despite torso reading ~0° "as
designed"; (2) a live `g1_full_demo.py` visual check (user, arms not even touched) showed
Lock standing with a visibly wide, unstable stance and obvious tilt even at rest —
locking torso removed a stabilization DOF the policy leans on even doing nothing, not
just during active-reach recovery. **But Clip's own pose is not acceptable either** —
same visual check showed legs spread far wider than GainMatch's already-reasonable idle
pose, with obviously elevated tilt.

**Bug found and fixed on the way to these numbers**: `eval_full_demo.py` builds its
shared env from the WALKING lineage (`G1LocomotionFlatEnvCfg_PLAY`), which never
inherited the standing lineage's torso clip on its own — so the FIRST integration run
against both Clip and Lock (overnight, unattended) silently tested them under an
UNCLIPPED action space, producing meaningless 100%-falls-everywhere / torso=150.03°
numbers in every bucket. Fixed by adding `--torso_clip_deg` to `eval_full_demo.py`
(pass the SAME value the checkpoint trained with — 30 for Clip, 0 for Lock; omit for
any unclipped checkpoint) — wired into `_build_env_cfg()` right after the
`class_type` swap. **The same gap existed in `testing/general_testing/g1_full_demo.py`**
(same walking-lineage env build) and got the same `--torso_clip_deg` fix. **General
lesson for future standing-lineage config changes**: anything that isn't purely
reward/observation-based (action-space clips, action term swaps) needs to be checked
against whether `_build_env_cfg()` in `eval_full_demo.py` AND `g1_full_demo.py`'s env
setup also need to carry it — both build from the walking lineage and inherit nothing
from the standing lineage automatically. The table above uses the CORRECTED numbers.

Also fixed while visually testing Clip: `g1_full_demo.py` crashed with
`AttributeError: 'G1FullDemo' object has no attribute '_mirror_homing'` when run with
`--arm right` or `--arm both` — `_mirror_homing` was only initialized inside the
`arm_mode == "left"` block, but `_update_arm_sim_targets`/`_arm_ready_for_walk` read it
unconditionally regardless of mode. Fixed by initializing it unconditionally. Isolated to
this interactive script's own homing state machine — does not affect and was never
imported by any eval script; none of the numbers above are affected by it.

**Root cause of the whole torso → hip pattern (why this keeps recurring, one joint at a
time)**: found by reading the standing task's own history from the start.
`G1LocomotionStandingFlatEnvCfg.__post_init__` (`g1_locomotion_env_cfg.py`, day one of
the standing task, well before any of this week's work) deliberately relaxes
`joint_deviation_torso` from the inherited -0.1 to **0.0**, reasoning "the torso needs to
be free to act as a balance-compensation DOF." That was a reasonable call at the time —
the disturbance it had to absorb then was `StandingArmTrajectoryDisturbance`, a curriculum
starting at literally zero arm motion. It was never revisited as the disturbance mechanism
escalated over the following two weeks: scripted joint wiggle → real Cartesian-goal IK
reach (`StandingArmIKReachDisturbance`) → the actual trained arm policy driving it
(`StandingArmPolicyReachDisturbance`) → deployment-matched 200/20 gain
(`match_deployment_arm_gains`, this week). The "free DOF" stayed free the whole time while
what it had to absorb kept growing, until it became the 60-120° torso exploit found this
week. Capping it (correctly) didn't remove the underlying compensation need — it moved to
the next-cheapest lever, hip_roll (the hip_roll frac_of_soft_limit_used jump in the table
above, and the visually-obvious wide stance). **Expect this to keep recurring on a new
joint each time one gets capped, until something addresses the pattern itself rather than
one joint at a time** — see the two queued follow-ups below, which are deliberately two
different KINDS of fix for this reason.

**Useful outside reference found**: `~/Elm/Code/unitree_rl_lab` — Unitree's own G1 RL
recipe, real/hardware-deployed. Confirmed NOT an architecture match (single unified
walk+stand velocity-tracking policy, `rel_standing_envs=0.02` — standing is a rare edge
case of the walking distribution, not a dedicated disturbance-robust standing policy like
ours; and zero arm/IK/manipulation code anywhere in the repo — only locomotion + a
`mimic` motion-imitation dance task, unrelated technique). But their reward/termination
set is a genuinely useful comparison point: `joint_deviation_waists` (their torso
equivalent) sits at -1.0 PERMANENTLY, never relaxed to 0 the way ours was; hip deviation
is -1.0 not our original -0.1; and — the structurally interesting one — they have a hard
`bad_orientation` termination (`limit_angle=0.8` rad, ends the episode on excess tilt)
that we've never had, orientation here has only ever been a soft reward cost.

**Two follow-up experiments, deliberately built as independent single-variable siblings
of TorsoClip (not stacked on each other), so each gets its own clean result:**

1. **`G1-Locomotion-Standing-Flat-IKReach-Height-Intent-GainMatch-TorsoClip-OrientHip-v0`**
   (`...TorsoClipOrientHipEnvCfg`) — TorsoClip + `flat_orientation_l2` -1.0→-3.0 +
   `joint_deviation_hip` -0.1→-0.5 (both inherited unchanged from the walking-tuned
   G1RoughEnvCfg until now; starting points, not tuned values). **LAUNCHED 2026-07-21
   night** (single seed, `overnight_train.sh` as of this update) — same joint being
   patched again, just this time the specific joint that needs pricing (hip) instead of
   letting the pattern keep surfacing reactively.
2. **`G1-Locomotion-Standing-Flat-IKReach-Height-Intent-GainMatch-TorsoClip-BadOrientation-v0`**
   (`...TorsoClipBadOrientationEnvCfg`) — TorsoClip + a hard `bad_orientation` termination
   (`limit_angle=0.8`, Unitree's own literal value). **Code ready (config + gym IDs +
   `eval_standing_ikreach.py --env_cfg ikreachintentgainmatchtorsoclipbadorientation`),
   NOT YET LAUNCHED** — queued behind OrientHip so the two don't get evaluated as a
   confounded stack. Structurally different from every other fix so far (torso clip,
   OrientHip reweight, the two dead-end reward-penalty attempts): terminates on the
   OUTCOME (excessive tilt) rather than pricing in one specific joint, so it can't be
   dodged by moving the compensation to a new DOF the way every per-joint fix so far has
   been. Flagged risk: TorsoClip's own native eval already shows mean_max_tilt_deg=48.94°
   (mean of per-episode maxes) — meaningfully above the 0.8 rad (~45.8°) threshold — so
   this will likely terminate a real fraction of episodes early; that pressure is the
   intended mechanism, but check training stability first if it looks like it's
   collapsing rather than converging (loosen `limit_angle` to 60-70° or curriculum-ramp
   it in if so).

**Launch order**: OrientHip result first (single-variable), decide whether to also run
BadOrientation, or combine the two into a fourth config if each helps independently but
neither fully solves it alone. Both remain single-seed until one direction looks right,
per the standing 2-seed-before-trusting rule.

## 2026-07-20 CORRECTION — the RL arm policy is the deployment target, not analytic IK (read this second)

**Explicit user correction, same day as the IK fix below**: this doc (and the session
that wrote the IK-fix section) had drifted into treating analytic IK / `--arm_driver
event` as "the trusted mode" / the thing to move toward for `g1_full_demo.py` and real
hardware. **That is wrong and is now corrected.** The deliverable is an RL-based arm
policy (`chosen_checkpoints/arm_left_latest.pt`, trained via `g1_arm_env.py` — direct
joint-space control, nothing IK-related in the deployed policy itself) — this is a
project requirement, not an optional upgrade path. Analytic IK
(`StandingArmIKReachDisturbance`) exists ONLY as a training-time robustness curriculum
for the standing policy — cheap, symmetric-by-construction, varied arm motion for
standing to learn to counteract — exactly what its own docstring in `events.py` always
said ("Deliberately real numerical IK, not the trained arm-IK RL policy... this decouples
standing's robustness training from whatever quality/asymmetry the RL arm policy
currently has"). The 07-16 pivot that promoted `--arm_driver event` to "the trusted
deployment mode" overreached past that original intent; today's IK fix is still real and
still valuable (standing now trains against accurate reach motion instead of scrambled
39-54cm-off motion), but its role is unchanged: make the training curriculum accurate, not
replace the arm policy as what actually drives the robot.

**Concrete consequences**:
- `g1_full_demo.py` already does the right thing (drives the arm via the RL policy,
  `_load_arm_policy`) — Step 4's old bullet about "wiring event-style arm driving... into
  g1_full_demo.py" is WRONG and should not be done; ignore it.
- The decisive integration test for Step 2's retrain is `--arm_driver policy` (the real
  RL arm policy driving reach), NOT `--arm_driver event`. Event mode is now only a cheap
  training-self-consistency sanity check (does the checkpoint survive exactly its own
  training disturbance replayed identically) — informative, but not the acceptance
  criterion. `overnight_train.sh` runs both, labeled accordingly.
- Step 3 ("RL arm upgrade path") is NOT gated on "only if IK reaching quality is
  insufficient" — analytic IK reaching quality is irrelevant to whether Step 3 is needed,
  since IK isn't deployed. Step 3 (retrain the arm policy at hardware-realistic gains,
  60/1.5 instead of the current sim-only 200/20, plus a stillness-at-goal reward) remains
  a real future item for hardware-readiness, but is NOT required to use the CURRENT arm
  policy now — "we already have something that works" (the user's words) for the full
  demo and for validating the retrained standing policy. Don't block on it.
- Historical eval-script language calling `event` "the trusted mode" (`eval_full_demo.py`
  flag help text, this doc's "Key facts" section) reflects the pre-correction framing —
  still technically accurate about what the flag DOES (replays training's own disturbance
  term), just no longer the right one to reach for when the question is "will this
  survive real deployment."

## 2026-07-20 UPDATE — Step 1 done, root cause found and fixed (read this first)

**The "arm never actually reaches" problem (see "REMAINING problem" below) is SOLVED.**
Root cause was NOT the floating-base Jacobian indexing the 07-16 write-up suspected — it
was a joint-ordering bug: `StandingArmIKReachDisturbance.__init__` (`events.py`) called
`find_joints(_ARM_JOINT_NAMES)`, which returns ids in **asset order**. The G1
articulation orders joints breadth-first, which **interleaves left/right**
(`left_shoulder_pitch, right_shoulder_pitch, left_shoulder_roll, right_shoulder_roll,
...`) — NOT `_ARM_JOINT_NAMES`'s left-block-then-right-block layout. The old code sliced
`_targets` by `_col_slice[side] = slice(i*5, i*5+5)` (0:5 for left, 5:10 for right),
silently writing each side's IK output into the WRONG columns — e.g. a right-arm reach
had its shoulder pitch/roll commands landing on left-arm joints. This one bug explains
the full 39-54cm miss and was present in every IKReach-era training run and in the
trusted `--arm_driver event` deployment mode; it plausibly also explains the historical
one-sided-bracing / asymmetric-fall findings from 07-15/07-16.

Fix (in working tree, uncommitted): `events.py` — `_col_slice` (slice per side) replaced
with `_col_idx` (per-side `torch.Tensor` of columns, derived from the joint ids
themselves via an id→column map, correct regardless of asset ordering). Applied in both
`StandingArmIKReachDisturbance._step_side` and the `StandingArmPolicyReachDisturbance`
override. Every OTHER consumer of the `_standing_arm_motion_targets`/`_joint_ids` buffers
(the blend action term, the arm-intent observation, both deployment scripts) already
paired columns to ids dynamically and was never affected — this was localized to these
two training-disturbance classes.

**Verified via `validation/check_ik_accuracy.py`** (new script, drives the real
`StandingArmIKReachDisturbance` class against a bare scene, FK-measures palm-to-goal
distance — see the script's docstring for full method):
- Before fix: p50 miss 14cm (left) / 30cm (right) even with PERFECT tracking (kinematic
  mode), ~0-10% within 3cm, strongly asymmetric left/right per-axis error — the
  asymmetry signature that identified this as a wiring bug rather than a solver/gain one.
- After fix: p50 = 0.0cm, ~80% within 3cm, ~90% within 5cm, left and right statistically
  identical, at BOTH perfect tracking and real 60/1.5 PD tracking (PD is not the
  bottleneck — matches the gravity-droop estimate of ~10cm max from the original
  07-16 write-up). Residual tail (max 49-60cm, 2-4% still slewing at timeout) matches
  the already-known ~35% sliver of `_GOAL_BOUNDS` that's kinematically unreachable
  (`validation/arm_reachability/*.md`) — not a bug, a goal-box limitation to revisit
  separately, not urgent.

**Consequence for every existing standing checkpoint**: all of them (`height_reward`
included) were trained against scrambled/wrong arm-reach commands — still arm-WAVING in
a different, more chaotic way than 07-16 realized, not the accurate reaching disturbance
the class was designed to be. This makes Step 2 (below) load-bearing rather than
optional: `height_reward`'s 0% integration falls was earned against a weaker, wrong
disturbance and needs re-verification once retrained against the NOW-correct one.
**Nothing has been retrained against the fix yet** — that's the next action: Step 2
below, unchanged in substance, just now unblocked and mandatory rather than conditional.

## Hard rules (user-set, non-negotiable)

1. **The user runs all Isaac Sim / training commands** (conda env `isaac_g1_control`).
   You prepare commands/configs and hand them over.
2. **Do not read or edit .md files without asking first** (context cost). This includes
   plan.md and memory files.
3. **Never trust a single training run**: PPO on the standing task has PROVEN massive
   seed variance (identical config produced 2.1% vs 100% fall rates —
   `consolidated` vs `consolidated_seed7`). Any claim from a new training needs 2 seeds.
4. The user strongly dislikes: persistent torso rotation at spawn, squatting, wide/spread
   leg stance, elevated idle tilt, and any "weird" resting posture generally — not just
   the specific joint most recently found exploiting it. Factor this into every policy
   choice (2026-07-21: confirmed this generalizes beyond torso — capping torso alone
   produced a new, equally-disliked wide-stance/hip-abduction posture).
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

### Step 1 — FK-verified IK accuracy test — **DONE (2026-07-20)**
Built `validation/check_ik_accuracy.py`: drives the REAL `StandingArmIKReachDisturbance`
class (not a copy) against a bare standalone scene via a duck-typed env shim, FK-measures
palm-to-goal distance. Found and fixed the joint-ordering bug — see the 2026-07-20 UPDATE
section at the top of this file for the full result. Root cause was a wiring/column bug,
not the suspected Jacobian indexing or a controller/gain problem — neither of the
decision matrix's other two branches applied.

### Step 2 — ONE final standing retrain, now unblocked by the Step 1 fix

**Attempt 1 (2026-07-20 afternoon) — DONE, diagnosed, superseded, checkpoint kept as a
documented data point, do not deploy it:**
`G1-Locomotion-Standing-Flat-IKReach-Height-Intent-v0` (`G1LocomotionStandingFlatIKReachHeightIntentEnvCfg`
— height_reward recipe + fixed IK + `no_reach_prob=0.15` + arm-intent observation), one
seed, 6000 iters, checkpoint `2026-07-20_11-32-18_ikreach_height_intent`. Results: gate
0% falls (static-arm hole closed, good), native (own disturbance) 48.6% falls, event-mode
integration 55.5% falls (self-consistent with native — no eval bug), but **policy-mode
integration (the real RL arm driving reach) collapsed to 99-100% falls** — user reported
this as "back to square one." Root cause, confirmed by cross-referencing which eval
buckets change which variable: `eval_full_demo.py`'s policy-mode reach buckets set the
actively-reaching arm's gain to 200/20 (`--active_arm_gain` default, matching the arm-IK
policy's own training gain) while event mode never changes gains (always stays at the
trained 60/1.5) — this config never enabled `match_deployment_arm_gains`, so it never
once trained against a 200/20-stiff arm's reaction torques before meeting one for the
first time at eval. This is the SAME failure mode the project's 2026-07-14 notes already
diagnosed, and exactly what `match_deployment_arm_gains` exists to fix — it just hadn't
been carried from the Consolidated lineage to this analytic-IK lineage. NOT a new
deployment/integration bug, NOT a regression from the joint-ordering fix (which remains
correct and verified). Also found: `mean_max_abs_torso_deg` read exactly 150.01° in
EVERY bucket of this checkpoint including 0%-fall ones (vs 15-27° in the 07-16 baseline)
— matches the torso joint's mechanical limit almost exactly; unclear yet whether this is
a real "torso pinned at its limit" behavior or a metrics artifact specific to this
checkpoint; needs a visual check. Unrelated to the fall-rate collapse (present even in
stable buckets).

**Attempt 2 (DONE, 2026-07-20 evening) — the gain fix, confirmed working:**
`G1-Locomotion-Standing-Flat-IKReach-Height-Intent-GainMatch-v0`
(`G1LocomotionStandingFlatIKReachHeightIntentGainMatchEnvCfg`) — attempt 1's config plus
exactly one change, `match_deployment_arm_gains=True`. One seed (42), 6000 iters,
checkpoint `2026-07-20_15-43-16_ikreach_height_intent_gainmatch`. **Results: fall-rate
collapse SOLVED.** Gate 0%, native 6.0% (was 48.6%), event-mode 13.5% (was 55.5%),
policy-mode (the decisive test): left reach 0% (was 99.4%), left edge 0% (was 100%),
right reach 5.2% (was 100%), right edge 12.5% (was 100%). Small residual left/right
asymmetry remains (0% vs 5-12.5%) — consistent with the already-known mirror-quality gap
(right arm has no dedicated training, only mirrors the left-trained policy, ~8% measured
deviation from true equivariance) — not urgent, sits under Step 3 part 2 below.

`mean_max_abs_torso_deg` improved a lot but did NOT resolve: 60.27° gate / 68.44° native
(was 150.01° pinned at the hard limit in attempt 1). A live `g1_full_demo.py` check
confirmed this numerically-improved-but-still-elevated reading is visually real and bad
("clearly wrong" resting posture) — NOT a metrics bug (unmodified code, same joint,
wildly different reading only for this checkpoint). It also explained a second symptom:
the standard reach coordinate that used to work easily became hard to reach, NOT because
of anything about how the goal coordinate is computed (root-frame-relative is correct
and intentional) but because the shoulder sits downstream of `torso_joint` in the
kinematic chain — a 60-70° twist displaces it from where the arm-IK policy (trained
against a nominal, untwisted torso) expects to find itself relative to the goal frame.
Same mechanism the project already found once before (2026-07-15, `events.py`'s
docstring: a rotated torso "displaces the shoulder... swallowed the near-inner reach
workspace"), just recurring at a smaller magnitude. **Practical implication: fixing torso
will likely also fix reach difficulty, for free — they're not two separate problems.**

**New diagnostic tool built the same day**: `StandingMetricsCsvWrapper` now writes a
`joint_diagnostics.csv` per eval run (one row per joint: max |position| hit, vs. that
joint's own soft/hard limits) — wired into `eval_standing.py`, `eval_standing_ikreach.py`,
and `eval_full_demo.py`'s three standing buckets. Deliberately does NOT correlate the raw
action tensor's column order against joint names (that would repeat the exact
join-ordering assumption mistake from Step 1) — reads only from `robot.data.joint_pos`/
`joint_names`/`soft_joint_pos_limits`/`joint_pos_limits`, three tensors guaranteed
mutually consistent by construction. Root cause of the torso pinning, found via this
tool: `max_action_abs` (already-existing column) peaked at ~45 on the attempt-2
checkpoint — wildly outside a normal PPO actor's range — and the stock
`ActionsCfg.joint_pos` (`JointPositionActionCfg(joint_names=[".*"], scale=0.5)`,
inherited unmodified from Isaac Lab's base velocity task) sets NO clip on any joint's
commanded target. Nothing in this recipe's reward stack prices in torso deviation at all,
so PPO found sitting `torso_joint` near/at its hard mechanical limit as a free,
unpenalized rest state.

**Secondary watch item, NOT urgent, found via the same diagnostic tool**: under ACTIVE
reaching (native eval) specifically — NOT present at idle (gate) — several arm joints on
BOTH sides ran at or just past their own soft limits: shoulder-pitch (~144-153°),
shoulder-roll (~119-121°, past soft into the hard-limit buffer), shoulder-yaw (~135°, at
the line), elbow-pitch (~162-186°), elbow-roll (~91-105°). `right_hip_yaw` also climbed
from 80.65° (idle) to 130.03° (reaching), still within its own soft limit. Current best
read: NOT the same pathology as torso (that showed up at idle with 0% falls; these only
appear under load) — more likely either genuine reach into the already-documented ~35%
marginal edge of the goal box (`check_arm_reachability.py`), or ordinary PD overshoot
past a target the disturbance already clamps correctly. Current logging can't
distinguish which (it only tracks worst-case position across the whole run, not which
side was active or which episode produced it) — re-check after the torso fix lands
(torso may stop "helping" via whatever it's currently doing, shifting arm/hip dynamics
again anyway) rather than chasing this now.

**On tilt/torso reward-tuning — do NOT retry a reward penalty a third time**: two prior
direct attempts (`torso_retighten`/`consolidated_torso`) both failed (see the experiment
verdicts table) — a deviation PENALTY must be learned/respected and likely fights
legitimate corrective use of the joint. This new finding is mechanistically different
(an unconstrained DOF being exploited toward its own hard limit, not a learned bracing
posture), which opens a different KIND of fix — see attempt 3, below.

**Attempt 3 (queued, `overnight_train.sh` as of 2026-07-20 night) — TWO torso-clip
widths, one seed each, not two seeds of one:**

1. `G1-Locomotion-Standing-Flat-IKReach-Height-Intent-GainMatch-TorsoClip-v0`
   (`G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoClipEnvCfg`) — attempt 2's
   config plus `self.actions.joint_pos.clip = {"torso_joint": (-radians(30), radians(30))}`.
2. `G1-Locomotion-Standing-Flat-IKReach-Height-Intent-GainMatch-TorsoLock-v0`
   (`G1LocomotionStandingFlatIKReachHeightIntentGainMatchTorsoLockEnvCfg`) — same, but
   `clip = {"torso_joint": (0.0, 0.0)}` — a full 0-width lock, target pinned to exactly
   default every step (functionally equivalent to excluding the joint from the action
   space; the "legs" actuator's own 200/5 gain holds it there, same pattern
   `g1_arm_env.py` already uses for its own non-RL-actuated joints).

Both are a hard ACTION-SPACE constraint, not a reward term — sidesteps the reward-penalty
dead end entirely rather than retrying it. `clip` on `JointPositionActionCfg` applies to
the FINAL commanded target (already scale+offset'd, real joint-angle radians — see
`isaaclab/envs/mdp/actions/joint_actions.py`'s `process_actions`), so both bounds are
exact, not an indirect scale guess. **Why two widths instead of two seeds of one**
(2026-07-20, explicit user call, corrected mid-evening from an earlier single-width
2-seed plan): +/-30° keeps meaningful recovery authority (torso_joint is a waist YAW
joint, plausible legitimate use countering reach-induced angular momentum) but a full
lock directly tests whether that concern is real rather than assumed — one seed of each
answers a different, more useful question than two seeds confirming only one width.
Sentinel for "locking it fully cost real recovery capability": the Lock variant's
ACTIVE-reach fall rates (native + policy-mode) regressing vs. the Clip variant or vs.
GainMatch's own numbers — not just whether torso itself reads near 0° (it will, by
construction, for Lock).

**MUST be a full retrain for both** (checkpoint immutability + the clip changes the
dynamics the policy is optimized against — deploying attempt 2's checkpoint under a
newly-clipped action space it never trained within would be exactly the train/deploy
mismatch this whole day has been about eliminating). Same eval sequence as attempt 2 for
each variant (gate → native → event-mode → policy-mode integration) —
`overnight_train.sh` runs both variants back to back automatically. Decisive checks on
return: (1) policy-mode fall rate must not regress attempt 2's numbers above for EITHER
variant, (2) `mean_max_abs_torso_deg`/`joint_diagnostics.csv` must show torso now bounded
(near/under 30° for Clip, near 0° for Lock) in every bucket, (3) Clip vs Lock comparison
on active-reach fall rates specifically, (4) re-glance at the secondary-watch-item joints
above. Whichever variant wins (or both, if equivalent — simpler wins ties) should get a
second seed before being trusted as final, per the project's proven seed-variance history
— this run is one seed each, a real but smaller signal than a fully seed-validated result.

**Explicitly considered and rejected: touching the ARM policy for this.** `torso_joint`
is not part of the arm task's action space at all — `g1_arm_env.py` trains with a fixed
root and only actuates the 5 shoulder/elbow joints of whichever arm; legs/waist/fingers
are held rigid, never RL-controlled there. Nothing to fix there for this specific issue.

**Explicitly considered and deferred: using the second overnight training slot for Step 3
(arm hardware-gain retrain) instead of a second torso-clip seed.** Step 3's "gain
curriculum" and "stillness-at-goal reward" are still only described in prose here, not
implemented anywhere in code — launching it would mean designing a gain-reduction
schedule and writing/validating a brand-new reward term under time pressure, not a
config tweak. Deliberately not rushed into tonight's queue; deserves its own dedicated
design session before it gets a training slot.

### Step 3 — arm-policy final optimization (real future work, not gated on Step 2's outcome)
Not "only if IK reaching is insufficient" (analytic IK isn't deployed — see the
correction section above) — this is about the ARM POLICY's own deployment-readiness, and
it has TWO required parts, not one (2026-07-20 evening, user explicit — don't drop the
second part, it's easy to forget since the repo only ever exercises the left checkpoint):

1. **Hardware-realistic gains.** Current checkpoint (`arm_left_latest.pt`) runs at
   200/20, a sim-only gain with no hardware counterpart (verified against
   unitree_sdk2_python's own G1 arm examples, which use 60/1.5). Retrain via a gain
   curriculum from the existing checkpoint, plus a stillness-at-goal reward (penalize arm
   joint_vel within ~3–4cm of goal — kills the near-goal jitter that destabilizes
   standing). Then a standing retrain against it (the `consolidated` recipe —
   policy-driven disturbance + gain match — its one good seed proved the approach
   works). 2 seeds, always.
2. **Both-arm deployability.** The arm policy is trained LEFT ONLY; the right arm is
   driven by mirroring the left-trained policy at inference (`mdp/symmetry.py`,
   `mirror_arm_actions`/`mirror_arm_obs`, exercised via `g1_full_demo.py`'s Y/U keys and
   `eval_full_demo.py`'s right-side buckets) — never a separately trained right-arm
   checkpoint. This mirror is KNOWN imperfect: `StandingArmIKReachDisturbance`'s own
   docstring in `events.py` records "~8% average deviation from true equivariance...
   concentrated on roll/yaw" measured against the actual trained checkpoint, and it got
   *worse* under a wider network attempt. Both arms must be genuinely deployable (not
   just the left, and not "right works in theory via mirroring") before the arm policy
   counts as finished — quantify the current deviation properly (not just the one
   historical 8% number), decide whether it's acceptable or needs a fix (a real
   right-arm-specific fine-tune, or a mirror-quality fix), and verify with `--arm both`.
   This has NOT been revisited since the 8% finding — treat it as open, not resolved.

Needed before real hardware, NOT needed to use the current arm policy for
`g1_full_demo.py` or to validate Step 2's standing retrain — "we already have something
that works" (explicit user call, 2026-07-20) is about LEFT-arm-only, single-checkpoint
deployment being fine as a validation vehicle right now; it is not a claim that the arm
policy is finished.

## Issues observed 2026-07-20 (afternoon/evening, live demo testing) — fix before arm-policy work proceeds

Both found by the user directly driving `g1_full_demo.py`, independent of anything the
Step 2 retrain changes (both apply to the CURRENT `chosen_checkpoints/standing_latest.pt`,
i.e. `height_reward`, and should be re-checked against whatever Step 2 produces too):

1. **Walk→stand transition + arm reach can combine to cause a fall.** User walked
   around, returned to standing, then commanded an arm reach at a normal/standard
   coordinate — the robot fell. This is a real interaction gap: the doc's existing
   "known leftover item" was a walk↔stand transition eval done with arms doing nothing
   in particular; this is the first evidence the transition ITSELF, immediately followed
   by active reaching, is its own failure mode — not something either the standing-alone
   evals (arms reaching, no transition) or the transition-alone check (no active reach)
   would catch. Needs a dedicated eval bucket: walk for N seconds → transition to stand →
   immediately command a reach → measure falls, not just "transition eval" and "reach
   eval" as separate buckets. Not yet built.
2. **Torso lift and tilt during standing is still visually present** — not resolved by
   height_reward's `base_height_l2` (anti-squat, fixes deep-squat collapse, not lift/tilt)
   nor expected to be by Step 2's arm-intent observation alone (intent reduced rotation
   from ~50° to 8-29° on the separate Consolidated lineage in the one measurement that
   exists, but was never combined with height_reward's exact recipe until Step 2, and
   even 8° may still read as "not nice visually" at this fidelity). This directly matters
   for hard rule 4 (user dislikes persistent torso rotation/weird resting posture) and is
   NOT yet solved — torso-deviation reward penalties were tried twice before and both
   FAILED (dead end, do not retry, see the experiment verdicts table) — the intent
   observation is the only lever proven to help so far, and Step 2 is the first real test
   of it combined with height_reward. If Step 2's checkpoint still shows visible tilt,
   this needs its own dedicated investigation, not another blind reward-tuning attempt.

### Step 4 — finish integration (after Step 2, informed by the two issues above)
- Visually confirm the retrained standing policy with `g1_full_demo.py` as-is — it
  ALREADY drives the arm via the real RL policy (`_load_arm_policy`), which is correct
  and should NOT be changed to analytic-IK/event-style driving (an earlier version of
  this doc said otherwise — that bullet is wrong, see the correction section above; do
  not act on it). The demo already has: the smooth homing move (no teleports), per-mode
  arm gains, the transition arm-flail fix (arm action columns from the walking policy
  only during blends), and the checkpoint-width-inferring loader. Specifically re-test
  issue 1 above (walk, stand, then reach) against whatever Step 2 produces — do not
  assume it's fixed just because Step 2 targets torso rotation, they're different
  failure modes.
- Build the walk→stand→reach eval bucket (issue 1) — batched, not just visual spot
  checks, so it's measured rather than anecdotal.
- Investigate torso lift/tilt (issue 2) if still present after Step 2 — dedicated
  investigation, not a third blind reward-penalty attempt (two already failed).
- Then the pre-existing known leftover item: the inner goal-box margin (near-inner
  corner is reachable but destabilizing/elbow-limited — trim only after reach works).

## Key facts a new session will otherwise get wrong

- **Native evals do NOT predict integration** for checkpoints with the static-arm hole;
  always run the `--freeze_arms` gate first.
- `eval_full_demo.py` flags: `--arm_driver {policy,ik,event}` — `policy` = the REAL RL
  arm policy, the actual deployment condition and the decisive one (2026-07-20
  correction: an earlier framing here called `event` "the trusted mode"; that's wrong,
  see the correction section near the top); `event` = training's own disturbance term
  replayed identically, useful only as a cheap training-self-consistency check; `ik` = a
  script-local reimplementation, currently broken, don't trust it. `--active_arm_gain
  KP KD`, per-run `run_meta.yaml` records everything.
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
- `eval_full_demo.py` and `g1_full_demo.py` both build their env from the WALKING lineage
  (`G1LocomotionFlatEnvCfg_PLAY`) and do NOT automatically inherit anything from the
  standing lineage's config chain — both now have a `--torso_clip_deg` flag (pass the
  checkpoint's own training clip width, e.g. 30 for TorsoClip, 0 for TorsoLock, omit for
  unclipped checkpoints) to carry it over manually. Forgetting this silently tests under
  the wrong action space (2026-07-21 — this produced a night of meaningless 100%-falls
  integration numbers before being caught). Any FUTURE standing-lineage change that isn't
  purely reward/observation-based needs the same check.
- The arm-IK task (`g1_arm_env.py`) was audited 2026-07-21 for the same "unpenalized free
  DOF" pattern that hit standing's torso/hip — found already well-defended: every joint
  target is hard-clamped to real hardware limits every step (`_apply_action`), plus
  nonzero `joint_limit_penalty_scale`, `null_space_penalty_scale`, and
  `torso_proximity_penalty_scale` throughout. No fix needed there.
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
| GainMatch (2026-07-20) | fixed arm-gain mismatch | fall collapse SOLVED, but torso pinned 60-70° idle (the joint-ordering/gain fixes exposed it, didn't cause it) |
| TorsoClip ±30° (2026-07-21) | hard action-space clip on torso_joint | torso bounded as designed, BUT idle tilt/native falls got worse (8.4°→37.2° gate tilt, 6.0%→26.3% native falls) — compensation moved to hip_roll, not removed. Best of the two clip widths; not yet acceptable alone |
| TorsoLock 0° (2026-07-21) | full torso lock | **DEAD END** — worse than Clip on every axis (51.0% native falls, 79.3% policy-mode left-reach falls, visually unstable wide stance even at idle with 0 arm activity) |

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
