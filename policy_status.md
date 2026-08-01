# Policy status — 29dof pivot

Living summary of what's actually validated vs. still open, for the checkpoints in
`chosen_checkpoints/`. Update this when a checkpoint gets promoted or a real gap is
found — don't let it go stale the way `known_issues.md` did on the 23dof branch.

**DEVELOPMENT FROZEN (2026-08-01)**: sim-side training/tuning is paused while the
currently-chosen checkpoints (`walking_latest.pt` = StandingPackage `model_6999.pt`;
`arm_left_latest.pt` = IntegratedNoTerm `model_11999.pt`) are attempted on the
physical robot. Both are eval-confirmed in sim but **not yet demo-verified in Isaac
Sim, let alone on real hardware** — this deployment attempt is the actual first
real-world test. Do not start new sim-side training/tuning work on this branch
without checking in first; findings from the physical attempt (what breaks, what
doesn't match sim) should be folded back into this doc before resuming.

## Walking + standing (`chosen_checkpoints/walking_latest.pt`)

**PROMOTED 2026-07-30 (current):** `logs/rsl_rl/walking/arm_disturbance/
2026-07-29_23-28-41/model_6999.pt` (`G1-Locomotion-Velocity-ArmDisturbance-
StandingPackage-v0`, trained fresh — the "standing package": exact-zero command snap,
`rel_standing_envs` 0.02→0.2, a new position-anchor penalty at standing onset; see the
2026-07-30 entries below for the full recipe and results). Chosen specifically for the
arm-integration priority — stand_still step count 8.23→1.96, |lateral drift|
0.076→0.031m, 0% falls across every disturbance phase, and the in-distribution turn
eval (see the 2026-07-30 eval-bucket-fix entry below) reads 0% falls everywhere.
**Known costs, not yet addressed:** `forward_slow` heading drift roughly doubled
(62→112 deg; medium/fast/backward improved), and in-place turning is now essentially
untrained/ignored. Superseded checkpoint below (`model_20996`) — better straight-line
walking and turning, worse standing — is **not** kept alongside the promoted one in
`chosen_checkpoints/` (2026-08-01: exactly one `.pt` per policy there, no `*_prev.pt`
copies); its source is backed up at `~/Elm/Backups/g1_locomotion/29dof/legs_runs/
2026-07-28_11-41-33_loose_height_nostep/` if ever needed again.

<details>
<summary>Superseded 2026-07-28 promotion history (model_20996, kept for context)</summary>

**FROZEN 2026-07-28** (explicit user decision — "we can spend weeks trying to improve
the walking... let's freeze it a little bit"). **Promoted**: `logs/rsl_rl/walking/
arm_disturbance/2026-07-28_11-41-33/model_20996.pt`
(`G1-Locomotion-Velocity-ArmDisturbance-LooseHeightNoStep-v0`) — good stand-still
stepping/drift (the actual thing the arm policy needed fixed), known elevated
`turn_left` fall rate under sustained turning (later found 2026-07-30 to be almost
entirely an eval artifact, not a real weakness — see below). See "2026-07-28: the
joint_mirror tradeoff" below for exactly what was tried after this and why it wasn't
promoted instead.

</details>

Superseded: the 2026-07-25 promotion (`logs/rsl_rl/walking/arm_disturbance/
2026-07-24_15-47-18/model_15998.pt`, warm-started from `walking_2026-07-24_base_only_
prev.pt`) — real walking, but the standing/stepping issue that motivated this whole
2026-07-28 push. `walking_2026-07-24_base_only_prev.pt` removed from
`chosen_checkpoints/` the same day (superseded, no longer needed for rollback).

**2026-07-28: the joint_mirror tradeoff.** After promoting model_20996 (LooseHeightNoStep)
above, one more experiment was tried on top: `joint_mirror` (ported from `unitree_rl_lab`,
never wired in before — penalizes squared left/right leg-joint-pair differences), added
because the full eval of model_20996 found a real, measured asymmetry (right knee
bending 5-7 deg more than left across every forward-walking speed) plausibly behind both
the sustained ~3.2 deg/s straight-walking heading-drift bias and `turn_left`'s elevated
fall rate. Result, trained 2999 more iterations
(`logs/rsl_rl/walking/arm_disturbance/2026-07-28_15-41-19/model_22995.pt`):

- **Worked exactly as designed** — `forward_slow`'s left/right knee-angle gap went from
  -4.70 deg to +0.02 deg (essentially symmetric).
- **Fixed two real problems**: forward-walking heading drift roughly halved across every
  speed/direction (`forward_slow` 62.1->24.7 deg, `backward` 63.8->22.1 deg, etc.), and
  `turn_left`'s fall rate dropped from 31.79% to 2.69%.
- **But reintroduced ~2x of the stand-still regression `feet_contact_without_cmd` had
  just fixed**, across every arm-disturbance phase, not just the disturbance-free
  bucket: step count roughly doubled (8-10 -> 14-19), heading drift roughly tripled
  (9-10 deg -> 20-34 deg), lateral drift roughly doubled (0.07-0.12m -> 0.13-0.26m), plus
  one new stand-still fall (1/257, in the first 3s).

Likely mechanism: `joint_mirror` penalizes left/right asymmetry unconditionally,
including during `stand_still`, but `feet_contact_without_cmd` may have settled on a
stance that relies on a slightly asymmetric weight distribution to stay still — the two
terms are in tension specifically at low/zero commanded velocity. **Not promoted** —
user explicitly chose to keep model_20996 (pre-mirror) as the frozen checkpoint instead,
prioritizing the standing win (what the arm/IK work actually needs) over the walking-drift
and turn_left wins. See item 11 under "Deferred / future items" for the proposed fix
(gate `joint_mirror` off below `command_norm < 0.1`, the same way
`feet_contact_without_cmd` is gated) — not implemented, walking is paused, not the mirror
idea abandoned.

**2026-07-29: plain continuation also regressed standing, with no reward change at
all.** Overnight, model_20996 was trained 5000 more iterations on the exact same
`LooseHeightNoStep` recipe (no new terms, no weight changes) as a control/"does more
training alone help" check, landing at `logs/rsl_rl/walking/arm_disturbance/
2026-07-29_03-06-11/model_25995.pt`. Result on `stand_still`: step count 8.23->31.72
(nearly 4x), |lateral drift| 0.076->0.773m (~10x), and a new nonzero fall rate
(0.00%->1.16%) — a bigger regression than `joint_mirror` produced, with none of that
experiment's compensating upside (no turn_left/forward-drift improvement, since nothing
about the reward changed). **Puzzling part**: `Train/mean_reward` and
`Episode_Reward/feet_contact_without_cmd`'s own tensorboard curves stay completely flat
throughout the whole continuation (episode length near-max the entire time, no visible
divergence) — so this isn't "training visibly broke," the aggregate training-time
reward doesn't clearly show it. Best guess: the policy drifted into a slightly
different, more brittle behavior specifically at the exact `(0,0,0)` command edge case,
which training rarely samples exactly (same dead-curriculum issue noted for
`turn_left`/`turn_right` — `rel_standing_envs=0.02` plus a `(-0.1,0.1)` sampled range
means literal zero-command is a narrow slice of what's actually trained against, so
nothing strongly anchors behavior there specifically) — not confirmed, just the most
consistent explanation for a regression invisible in the aggregate reward. **Not
promoted** — `chosen_checkpoints/walking_latest.pt` confirmed still exactly model_20996
(checksum-verified), this experiment lives entirely in its own separate `logs/`
directory. **General lesson**: don't assume plain continuation of a working recipe is
safe-by-default even with an unchanged reward and a flat-looking training curve — worth
an eval check before trusting any further continuation of this checkpoint, not just
reward-changing experiments.

**2026-07-29 — turn/strafe evals were out-of-distribution, plus a "standing package"
built (not yet trained).** Code review found `ang_vel_cmd_levels` exists in
`mdp/curriculums.py` but was NEVER wired into `CurriculumCfg` (only `terrain_levels` +
`lin_vel_cmd_levels` are), so every run trains yaw commands at the starting (-0.1, 0.1)
only — while `eval_walking.py` commanded turns at ±0.6 (3-6x OOD) and strafe at ±0.4
(vs the ±0.3 limit). The elevated `turn_left` fall rates above are therefore
stress-test numbers, not in-distribution performance (the left-vs-right difference
still reflects the real measured gait asymmetry). Fixed 2026-07-29: default eval
buckets now stay inside the trained envelope (turn ±0.2, strafe ±0.3, combo wz 0.2);
the old commands live on as opt-in `*_stress` buckets — pre-2026-07-29 summaries'
turn/strafe rows compare against `*_stress`, not the same-named defaults. Also fixed
`ang_vel_cmd_levels`' stale episode-length normalization (the exact bug
`lin_vel_cmd_levels` had fixed 2026-07-22) so it actually works if ever wired in.
**Standing package** (`G1-Locomotion-Velocity-ArmDisturbance-StandingPackage-v0`,
implemented, NOT yet trained — train from FRESH weights): the model_20996 recipe plus
three standing-only-gated changes targeting the unanchored-zero-command mechanism
behind both 2026-07-28/29 standing regressions — (1) sampled commands with norm < 0.1
snap to exact (0,0,0) (`UniformLevelVelocityCommand`, kills the contradictory
track-vs-feet-contact band and multiplies exact-zero exposure), (2) `rel_standing_envs`
0.02→0.2, (3) `mdp.StandingPositionDriftPenalty` weight -1.0 (anchors position at
standing onset the way `heading_drift` anchors yaw; zero whenever commanded to move).
Also noted 2026-07-29: RSL-RL resumes restore weights but NOT env curriculum state —
every warm-started run re-anneals command ranges from (-0.1, 0.1), a confound when
comparing layered experiments.

**Confirmed working, not assumed:**
- Real walking — verified via direct `root_pos_w` displacement measurement (not just
  reward/error metrics, after this project's own earlier mistake reading a track-error
  metric as achieved velocity — see git history 2026-07-23/24). Forward tracking error
  0.10-0.13 m/s at 0.3/0.6 m/s commanded, 0% fall rate.
- Standing under arm disturbance — 0% fall rate across all 4 disturbance phases
  (no/mild/moderate/max), verified with a real, working disturbance signal (found and
  fixed a bug 2026-07-24 where the phase-pin eval flag silently had no effect on an
  env's first episode — see `mdp/events.py`'s `ArmMotionDisturbance.__init__`) and
  confirmed the disturbance itself produces real, scaling joint motion when *pinned*
  (see `testing/general_testing/check_arm_disturbance_magnitude.py`).
- **Training exposure to full-amplitude disturbance — verified 2026-07-27, not just
  inferred.** Visual inspection of `walking_latest.pt` under `eval_walking.py`'s
  *unpinned* `stand_still` bucket showed no visible arm motion — traced to a real
  mechanism, not a bug: `ArmMotionDisturbance` samples each env's phase once per
  episode at reset, weighted by `common_step_counter` at that moment; a fresh eval
  session starts at 0, so its first (often only) reset always draws phase 0, and stays
  there for the whole session regardless of elapsed steps. Built
  `testing/general_testing/check_disturbance_training_exposure.py` to test the
  mechanism directly instead of guessing: forces `common_step_counter` to a
  training-representative value (624,000, matching `walking_latest.pt`'s ~26,000
  iterations × 24 steps/iter) before reset, confirming (a) phase sampling correctly
  weights toward phase 3 once past the boundaries (62.5% sampled vs. 53.3%
  theoretical), and (b) individual arm joints reach 88-90% of their phase's designed
  amplitude ceiling given a 20s window (0.21/0.40/0.63 rad vs. 0.24/0.45/0.70 rad
  ceilings for phases 1/2/3 — measured per-joint; averaging across all ~14 arm joints
  first, as an earlier pass did, understates this to ~25-50% since different joints
  peak at different moments). Given a real training run spends vastly more than 20s
  per env past the phase boundaries, this confirms the mechanism the checkpoint was
  trained under does reach real, full-range disturbance — not just a plausible
  inference. Eval sessions checking disturbance visually need `--phases N` pinned to
  see it; the unpinned/natural-cycling path is not representative in a short session.
- Reward reshape that unblocked this (`action_rate` -0.05→-0.005, `joint_deviation_legs`
  -1.0→-0.1) was validated via a 4-way isolated ablation, not a bundled guess — see
  `g1_locomotion_env_cfg.py`'s `RewardsCfg` comments for the full before/after numbers.

**Known gap — heading/position drift correction, round 2 in progress.** Baseline
(`walking_latest.pt`): 24-27° heading drift, 0.6-0.9m lateral drift over a single 20s
straight-line episode. Root cause: `track_lin_vel_xy_yaw_frame_exp` rewards velocity in
the robot's *current* (possibly already-drifted) yaw frame, and `track_ang_vel_z` has no
memory of accumulated heading error — nothing in the reward looks at absolute heading.
Confirmed NOT self-correcting with more training (a 7300-iter continuation, no drift
reward, made drift *worse*: 59-72°).

- **Round 1, attempt A (2026-07-25, reverted)**: `HeadingDriftPenalty`/
  `LateralDriftPenalty`, two new additive terms, weight -1.0 each, warm-started. Made
  drift *worse* (27-63°/1.6m) and caused broad regression across unrelated reward terms
  (`action_rate`, `base_angular_velocity`, `flat_orientation_l2`, `joint_deviation_legs`,
  and the raw `Metrics/base_velocity/error_vel_yaw` tracking error). Two suspected,
  non-exclusive causes: two new competing terms disturbing an already-converged ~20-term
  balance, and/or warm-starting into a changed reward with a critic that had never seen
  the new terms (value estimates wrong from step one, possibly compounding with the
  adaptive-KL learning-rate schedule).
- **Round 1, attempt B (2026-07-25/26, reverted)**: replaced `track_lin_vel_xy_yaw_frame_exp`
  with `TrackLinVelXYExpectedYawFrameExp` (same weight, rotates into the *expected* yaw
  frame instead of current) — a same-weight replacement, not an addition, meant to avoid
  attempt A's risks. First test was invalidated by a real, separate bug: the curriculum
  (`Curriculum/lin_vel_cmd_levels`) got stuck at its minimum tier the whole run, so it
  never practiced 0.3-0.6 m/s walking at all (fixed with
  `G1-Locomotion-Velocity-ArmDisturbance-StartFullCmdRange-v0`, confirmed reaching the
  full command range from iteration 0). Once actually tested at real speeds: heading
  drift got dramatically worse (63.6°/136.4°), while lateral drift and track error stayed
  roughly normal. Root cause understood, not just observed: this reward only constrains
  the *direction of world-frame velocity*, never the robot's actual *body orientation* —
  it removed the stock reward's accidental (but real) side effect of tying heading to
  direction of travel, without replacing it with anything that actually constrains
  heading. Reverted `track_lin_vel_xy` back to stock.
- **Round 2 (2026-07-26/27, trained, not promoted)**: `track_lin_vel_xy_yaw_frame_exp`
  back to stock, `HeadingDriftPenalty` reintroduced at weight -0.1, fresh weights, 8000
  iters, `G1-Locomotion-Velocity-ArmDisturbance-v0`
  (`logs/rsl_rl/walking/arm_disturbance/2026-07-27_00-43-20/model_7999.pt`). Fall rate:
  0% across every command bucket and all 4 disturbance phases — genuinely good. Heading
  drift: **worse than baseline and scales badly with speed** — 28.3° (forward_slow, ~=
  baseline's 24-27°) → 67.6° (forward_medium) → 94.5° (forward_fast), vs. baseline's
  24-27° roughly flat across speed. Lateral drift, by contrast, is *better* than baseline
  for the forward buckets (0.47-0.67m vs. 0.6-0.9m) — worse for backward (1.26m). Same
  failure shape as round 1 attempt B (heading drift getting dramatically worse), just
  less extreme — three different heading-correction attempts now, all making heading
  drift worse, not better. **Not promoted; `walking_latest.pt` remains the fallback.**

  **2026-07-28, tensorboard-checked (not previously done for any drift attempt)**:
  unlike baseline/round1-attemptA/round1-attemptB (all genuinely flat by the last 20% of
  training — last-10%-vs-prev-10% change all under 1%) and the round1-related run (which
  was actively *worsening*, not cut off early — monotonic rise 0.76->0.90 the whole back
  half), **round 2's `Metrics/base_velocity/error_vel_yaw` was still monotonically
  falling when training stopped at 8000 iters** (0.94->0.88->0.74->0.70->0.67,
  -2.5% in the last-10%-vs-prev-10% window — the only one of the four NOT flat). So
  round 2's "worse than baseline" verdict may reflect an under-trained checkpoint rather
  than a genuinely worse approach — worth revisiting with more iterations if walking-
  heading-drift becomes the priority again (it currently isn't — 2026-07-28 focus moved
  to standing-while-stationary specifically, see below).
  Live visual inspection (not just the metric) found a plausible mechanism: near-extended
  knees during straight walking, with occasional foot-crossing/asymmetric-length
  corrective steps appearing right around where drift kicks in — see "Potential
  improvement — base_height rigidity" below, not yet acted on.
- Whether lateral or heading drift matters more depends on the use case: lateral (does
  it end up where it's supposed to) is more directly relevant for short-distance tasks;
  heading matters through its effect on position over longer distances/durations (a 20s
  test may not have run long enough for a backloaded heading drift to fully show up as
  lateral error yet) and independently if final facing direction matters downstream
  (e.g. arm-reach alignment). Not resolved which regime this checkpoint is actually in.

**Potential improvement — `base_height` reward possibly too rigid (2026-07-27, not
acted on).** Hypothesis from the round-2 visual inspection above: `base_height`'s weight
(-10, by far the strongest single reward term in the whole recipe, target 0.78m exactly)
leaves little room for knee-flexion-based balance correction, since holding pelvis
height precisely is geometrically cheapest with near-straight legs — and a
near-locked-knee stance has less margin to absorb a perturbation without an awkward
compensating step. User is open to a few cm of height variation (not a full squat like a
pre-pivot policy) if it buys smoother, more human-like balance recovery. **Not
confirmed** — added real per-episode knee-angle tracking (`mean_knee_angle_deg`/
`min_knee_angle_deg` in `WalkingMetricsCsvWrapper`) plus a knee-angle-vs-drift
correlation section to `eval_walking.py` (section 4) to check this with data before
touching the reward; not yet run. Deliberately deferred — today's priority is the
arm-IK integration below.

**Not yet done:**
- No real-robot test at all — sim-verified only.
- `eval_full_demo.py` (the actual integration eval, loco+arm together) hasn't been
  re-run against this checkpoint.
- The `debug_vis` remote-asset-hang fix (found 2026-07-24) was applied to
  `eval_walking.py`/`check_real_displacement*.py` but not yet to `eval_full_demo.py` —
  check before relying on that script unattended.

## Arm reaching

**Status as of 2026-08-01: root cause found and fixed — reaching AND holding are both
solved. `G1-Arm-Left-IntegratedNoTerm-v0` (`model_11999.pt`) is PROMOTED to
`chosen_checkpoints/arm_left_latest.pt`.**
The 25-29%-plateau history below (everything up to and including the
"CORRECTED 2026-07-28" note) turned out to trace to one thing: the action pipeline's
`current + delta` parameterization capped static holding torque well below what
gravity requires at the real 40/10 hardware gain (see "2026-07-29 — ROOT CAUSE FOUND"
further down). Fixed via `G1-Arm-Left-Integrated-v0`: 100% reach rate, ~1-3mm typical
precision at the real gain — but that checkpoint dithered instead of holding (an
incentive bug, `terminate_on_success` made completing a hold reward-negative); fixed
by `G1-Arm-Left-IntegratedNoTerm-v0` (`terminate_on_success=False`), which is the
promoted checkpoint: 100%/100% Settled-<2cm/<3cm, 100% Tail-settle, ~97% time-in-zone
(see the 2026-07-31 "RESULT" entry below for the full numbers). Remaining open item
before calling the arm policy fully finished: `g1_rl_control` deployment wiring +
live demo verification — see `arms_policy_finalisation.md`. The history below (gain
sweeps, entropy ablations, locked
wrist, etc.) is kept for the record — all of it was, in hindsight, working around a
single control-interface detail, not the actual bottleneck.

**CORRECTED 2026-07-28** (this note was wrong): `chosen_checkpoints/arm_left_latest.pt`
was previously called "stale — do not treat it as current" here. Per the user directly:
this checkpoint was trained for ~8000 iterations (pre-2026-07-26, i.e. before the
action_fb observation addition — its own training run's logs were deleted in the
2026-07-25 cleanup, so the exact config can't be recovered, but the file's mtime,
2026-07-23, confirms it predates action_fb) and reached "the same results" as
`best_combined`'s 2000-iteration run (~25-29% band) — the user deliberately kept using
it over `best_combined` for that reason. **This is a real, meaningful data point for the
plateau question below**: two checkpoints with different recipes/training lengths
(2000 vs ~8000 iterations) landing in the same success band is evidence *against* "just
needs more iterations" being the fix on its own — consistent with the "uniform
capability shortfall across the whole box" read from the goal-curriculum work below.
**The 200/20-gain reference checkpoint (99.98% success) is intact and NOT in
chosen_checkpoints/ — it's `logs/rsl_rl/arms/left/2026-07-22_06-20-55/model_2999.pt`. No
retraining needed to re-validate it.**

**`best_combined` (`logs/rsl_rl/arms/best_combined/2026-07-26_13-09-32/model_1999.pt`)
was briefly copied into `chosen_checkpoints/arm_left_best_combined.pt` 2026-07-28 for
`g1_rl_control`'s first-implementation pass, then deleted by the user the same day** in
favor of continuing to use `arm_left_latest.pt` (see above) — `chosen_checkpoints/`
currently has no arm checkpoint in it as of this note. `best_combined`'s own weights
still exist untouched at the `logs/` path above if needed again (user plans to retrain
it overnight for more iterations to test the plateau properly).

**Known gap for `g1_rl_control` if `arm_left_latest.pt` becomes the one actually
deployed**: unlike `best_combined` (whose `params/env.yaml` gave confirmed
action_scale=0.5/action_filter_alpha=0.25/max_action_delta_per_step=0.06/
include_action_feedback=true), `arm_left_latest.pt`'s exact training config is
unrecoverable (logs deleted). Its pre-action_fb mtime means `include_action_feedback`
was almost certainly `false` for this checkpoint (32-D per-arm obs, not 39-D) —
`g1_rl_control`'s `PolicyConfig` currently defaults to `true`/39-D, matching
`best_combined`, not this checkpoint. If `arm_left_latest.pt` (rather than a
freshly-retrained `best_combined`) ends up being what's actually exported to ONNX and
deployed, `g1_rl_control`'s `PolicyConfig.include_action_feedback` needs to be set to
`False` and the ONNX wrapper's expected-obs-dim check (32 vs 39) needs to be re-verified
against that exported model — not yet done, flagging before Thursday's deployment.

**The real signal**: `Train/mean_episode_length` (this task terminates early on
success — hold goal 15 consecutive steps). The 200/20-gain reference converges to
~60-63 steps out of a 300-step max. **Every single 40/10-gain variant tried sits at
285-297 steps — essentially always timing out** — see the full list below. This has
been the single most reliable signal throughout: check it before trusting a final eval
number, though it's not infallible (the rate-limit ablation looked meaningfully better
mid-training and still ended up worse — see below).

**Ruled out or found neutral, with real evidence:**

- Reachability — 97.5% goal-box coverage within 2cm, ~= the 7-DOF baseline's 97.0%.
- Action-routing/indexing bug — direct code read confirmed no cross-arm/waist
  contamination for `arm="left"`. Right-arm/waist motion during testing confirmed real
  with actual numbers 2026-07-27 (`g1_arm_reach_test.py --debug_passive_joints`,
  `best_combined` checkpoint) — not a routing bug: `right_elbow_joint` settles fast
  (~100 steps) to a stable ~-53°, but `waist_pitch_joint` and `right_shoulder_pitch_joint`
  don't settle at all within an episode — still climbing at step 570-599 (waist_pitch
  20.2°→20.5°), then reset and climb again from near-0 toward the *same* ~20° region in
  the next episode (8.0°→18.1° by step 180) — a slow, continuous, reproducible creep
  toward a consistent equilibrium (confirms real physics, not noise/a bug), not a
  quick one-time settle. Non-trivial magnitude (~20° torso lean, -53° passive elbow
  deflection) — genuine future refinement candidate (gravity compensation or a
  passive-joint reward term), not blocking: the controlled arm still converged to
  2.2cm before the episode's natural reset interrupted it. Unrelated to the
  success-rate collapse itself.
- `entropy_coef` (0.0/0.003/0.005/0.01, adaptive or fixed LR schedule) — 0.0 collapses
  noise to ~0.09-0.12 with reward flat; every nonzero value tested explodes (0.01 fast,
  0.005/0.003 slower but still climbing, same shape regardless of schedule — ruling out
  the adaptive-KL schedule as the specific cause, since fixed schedule showed the
  identical trajectory). No value/schedule combination is stable.
- `position_reward_exp_scale=3.0` alone — no improvement; same noise-collapse pattern
  as baseline, so reward-shape wasn't the actual mechanism.
- Wrist-locking (5 DOF, `G1-Arm-Left-LockedWrist-v0`) — success collapsed to 2.66%/0.94%
  despite ~3x more training (17000 iters). The rationale that motivated this (fewer DOF
  = less redundancy = easier) was falsified by its own results.
- Rate-limiting relaxation (`action_filter_alpha` 0.25→0.6, `max_action_delta_per_step`
  0.06→0.15 rad) — looked genuinely different early (episode length 245 vs baseline's
  298 at iteration 1415, reward strongly positive vs baseline's negative), but kept
  climbing and finished at 277.9 by iteration 5000, and final eval (21-24%) was worse
  than baseline (27-30%). The one case where the early-signal heuristic was misleading —
  worth remembering before trusting a mid-training snapshot as final.

**Found harmful, with real evidence:**

- `joint_vel_noise` reduced 1.5→0.1 rad/s (theory: 1.5 rad/s is borrowed from the
  locomotion recipe, where legs swing at rad/s-scale — far too coarse for arm-reaching's
  fine-settling regime, ~0.05-0.3 rad/s) — success dropped to 2.31%/3.44%, an order of
  magnitude *worse* than baseline (24.83%/22.53%). Theory was wrong, or backfired —
  possibly the noise was acting as useful implicit regularization within a short
  training budget. Confirmed harmful, not just unhelpful — dropped from further use.

**Found modestly positive, with real evidence (isolated 2000-iteration legs, same
seed):**

- Privileged/asymmetric critic (critic reads clean, noise-free observations instead of
  the same noisy ones the actor sees — critic never deploys, no reason it needs to cope
  with sim2real noise) — 29.37%/26.83% vs baseline's 24.83%/22.53%.
- `noise_std_type="log"` instead of the default `"scalar"` (parameterization only,
  `entropy_coef` stays 0.0) — 26.61%/26.50%.
- Neither broke the underlying ~290/300 episode-length ceiling on its own.

**New structural fix (2026-07-26, not yet evaluated in isolation)**: the policy never
observed its own action-filter's internal state (`self.filtered_actions` — an EMA of
past actions, `action_filter_alpha=0.25`, i.e. ~75% memory of history each step).
`_apply_action`'s actual commanded delta depends on this hidden state, not just the raw
action, but a memoryless feedforward policy was never given it as an observation — a
real POMDP-vs-MDP gap that could plausibly make precise settling harder to learn. Added
as `action_fb` to the observation (`g1_arm_env.py`'s `_get_observations`) — baked into
the base env class, so it applies to every arm task automatically going forward.
Observation dims changed: 32→39 (7-DOF), 28→33 (locked-wrist), 64→78 (both-arms).
Symmetry mirror sign vector in `mdp/symmetry.py` updated to match (verified tensor
length 39, module's own assert passes). Live-sanity-tested (baseline, locked-wrist,
privileged-critic all load/run cleanly at the new dims) but not yet trained/evaluated on
its own merit.

**`G1-Arm-Left-BestCombined-v0`** — combines `privileged_critic` + `log_std` (both
non-harmful) + `action_fb`, fresh weights, 2000 iters: **28.20%/32.85%, best plain-40/10
result so far, currently the chosen "optimal" arm candidate** (as of 2026-07-27 — used
for the arm visual inspections, `model_1999.pt`). `frac_envs_reached`/`goal_reached_bonus`
plateaued by ~step 500 in this 2000-iter run — more iterations at this exact setup
looked like they wouldn't help. **Flagged (2026-07-27, not done yet — deferred, same as
the walking items above):** extend to 8000 iters total to actually confirm the plateau
holds, rather than relying on the 2000-iter read alone (goal_curriculum's own "still
climbing" read at 2000 iters turned out to still be true after extending — worth the
same confirmation here before fully trusting this as final).

**RESOLVED 2026-07-29**: extended to ~10000 iters total overnight
(`logs/rsl_rl/arms/best_combined/2026-07-29_00-09-42/model_9999.pt`). Plateau confirmed
real, not an artifact of stopping early: 30.15%/28.10% (no_wobble/with_wobble) vs the
2000-iter checkpoint's 28.20%/32.85% — essentially a wash, `no_wobble` up ~2pts,
`with_wobble` down ~5pts, both within noise for these episode counts. This is the
**second** independent confirmation that iteration count alone isn't the lever —
`arm_left_latest` (~8000 iters, older pre-action_fb recipe) already landed in the same
~25-30% band; now the actual `BestCombined` recipe does too at ~10000. Not promoted
over the 2000-iter checkpoint (no real improvement to justify it). Two other findings
from the same eval, using new columns ported from `ik_residuals` 2026-07-29 (see
`validation/eval_arm.py`'s `_summarize`): `reach_rate_no_hold` (30.63%/28.92%) is barely
above `success_rate` — holding once it arrives is essentially free, reaching is the
entire bottleneck, confirming the "holding is basically solved" read above; and
`mean_final_dist_cm` (7.28/7.47cm) stays close to `mean_dist_cm` (7.11/7.22cm) — same
converges-then-plateaus shape as the 2000-iter checkpoint, unchanged by more training.
**Iteration count is now ruled out twice over — the next lever has to be structural
(`goal_curriculum`, still unfinished, or the IK pivot), not more brute-force training.**

**2026-07-29 — ROOT CAUSE FOUND (probe-confirmed, fix implemented, training pending):
the action pipeline itself caps static holding torque below what gravity demands.**
`_apply_action` re-anchors the commanded target to the MEASURED joint position every
step (`targets = current + delta`, `|delta| <= 0.06 rad`), so at any static equilibrium
the PD torque is capped at `kp * 0.06` — **2.4 Nm at 40/10** vs 12 Nm at 200/20. URDF
analysis (left arm 3.52 kg incl. hand): typical goal-box reach postures need 4.5-5.7 Nm
at shoulder pitch/roll; ~72% of random arm poses exceed 2.4 Nm at some proximal joint,
~0% exceed 12 Nm, and kp=15's 0.9 Nm ceiling is exceeded by 99% — mapping cleanly onto
the observed 200/20≈99.98% / 40/10≈28-30% / 15/1≈5.5% results, the ~290/300
episode-length ceiling (can't hold 15 consecutive steps without static torque), the
~7cm converged-but-imprecise sag plateau, and why no RL-side lever ever moved it.
Confirmed with a no-RL probe (`testing/general_testing/check_arm_static_torque_ceiling.py`,
run 2026-07-29): the pipeline saturates at exactly 2.42-2.46 Nm implied torque and sags
35-54 deg on high-torque postures (but holds a 1.4 Nm control posture — falsifiable
both ways, passed both ways), while an integrated-target variant holds ALL postures at
0.00 deg error at the same 40/10 gain, sustaining 5.4-6.2 Nm. Probe side-finding: the
~11-14cm residual ee error common to ALL well-holding modes is torso/waist lean under
the arm's weight (stock-gain passive waist joints — same mechanism as 2026-07-27's
measured ~20 deg waist_pitch creep; compensable by a policy that sees true ee position).
**Fix implemented as `G1-Arm-Left-Integrated-v0`** (`G1ArmLeftIntegratedEnvCfg` +
`G1ArmLeftIntegratedPPORunnerCfg`, BestCombined agent recipe): same per-step rate
limit/EMA, but the delta accumulates into a persistent target that can hold a gravity
bias (matches real deployment's absolute-target semantics — g1_rl_control must
replicate `target += clamped delta`, NOT `current + delta`). Also in this variant:
obs 39→46 (`target_fb` = target − joint_pos, closing the new hidden-state POMDP gap;
symmetry.py extended for 46/92-D) and env-local ee/goal obs (`env_local_obs=True` —
raw world coords carried per-env origin offsets of tens of meters, making those dims
normalization-noise and the symmetry mirror on them physically inconsistent). Every
pre-existing task/checkpoint untouched (both behaviors flag-gated, default off).
Eval via `validation/eval_arm.py --integrated`. NOT yet trained — 2k-iter comparison
vs best_combined's 28.20%/32.85% is the pending decisive test.

**2026-07-30 — Integrated trained 8000 iters
(`logs/rsl_rl/arms/integrated/2026-07-29_20-27-48`): the fix works — reaching is
SOLVED; the residual low "success rate" is a termination-incentive artifact, not a
capability gap.** Headline numbers (final checkpoint, 539/554 episodes):
`reach_rate_no_hold` **100.00%** both buckets (best_combined: ~30%), min-dist mean
**0.14-0.15 cm** / p90 0.25-0.26 cm (best_combined: ~7 cm), `final_dist_to_goal_cm`
mean 1.22-1.26 cm with **100% of all 1093 episodes ending within 2.76 cm** (81% within
2.0 cm), per-step `frac_envs_reached` 0.79. Millimeter-precision reaching at the real
40/10 hardware gain, robust to wobble. BUT `success_rate` (15 CONSECUTIVE steps <2cm +
early termination) reads only 6.49%/9.39% (10.78%/16.67% at the 2000-iter checkpoint) —
diagnosis: the policy hovers at ~1.1 cm mean, flickering across the 2 cm line, so the
consecutive-hold counter keeps resetting; and `terminate_on_success` makes completing a
hold REWARD-NEGATIVE (termination ends the 50/step `goal_reached_bonus` stream and
respawns a distant goal), so the reward-optimal behavior is to dither at the boundary
and never "succeed." Evidence it's the incentive, not noise: 2k→8k training improved
every real quantity (min dist 0.22→0.14 cm, final dist 1.40→1.26 cm, in-zone frac
0.71→0.79, joints-at-limit 0.84→0.22, mean reward 33.8→38.2) while the success flag
FELL 10.8→6.5 — the optimizer is learning to avoid termination. Fix is reward/metric
design, not more capability work: (a) train without `terminate_on_success` (or keep
paying the bonus post-success) and/or scale `goal_reached_bonus` by the hold counter so
settling deeper strictly beats dithering; (b) judge checkpoints by final-window
distance (this one scores ~81% at 2 cm / 100% at 3 cm), which is what a grasp use case
actually needs. For deployment NOW this is already the best arm artifact this repo has
produced (the 200/20 reference's true single-shot rate was ~30%). Deployment reminder:
`g1_rl_control` must integrate deltas (`target += clamped_delta`), not `current+delta`.

**2026-07-30 — retrain queued to fix the dithering incentive: `G1-Arm-Left-
IntegratedNoTerm-v0`.** Implements option (a) above: `terminate_on_success=False`
(new env cfg `G1ArmLeftIntegratedNoTermEnvCfg` in `g1_arm_env.py`, same 46-D
obs/gain/reward as the Integrated task otherwise — an incentive-only change).
Queued via `overnight_train.sh` the night of 2026-07-30: fresh weights, 12000 iters,
followed automatically by `validation/eval_arm.py --integrated_no_term` on the final
checkpoint (`logs/rsl_rl/arms/integrated_no_term/<run>/model_11999.pt` once done —
**not run as of this writing**, check that directory before trusting this as
current). Judge the result by the eval's Settled-<2cm/<3cm and Tail-settle-rate
columns — the legacy success-rate column reads a **structural, expected 0%** for
this checkpoint (its `terminate_on_success=False` means the CSV's `success` column,
`bool(terminated[env_id])`, is never True by construction — see
`arms_policy_finalisation.md` step 1's own note on this). `--integrated_no_term`
also selects a matching NoTerm eval env (same reason: keeps every episode running
the full length, so the Tail-settle metric's trailing window is never cut short by
an early termination).

**2026-07-31 — RESULT: the retrain worked — holding is now genuinely solved, not
just reaching.** Trained (`logs/rsl_rl/arms/integrated_no_term/2026-07-30_19-57-17`,
12000 iters, `model_11999.pt`) and evaluated. Real numbers, both buckets: Settled
<2cm/<3cm **100.00%/100.00%**, mean dist **0.10cm**, median **0.05cm**, p90
**0.23-0.24cm**, `frac_envs_reached` (per-step time in zone) **~97%**,
`goal_reached_bonus` **48.5/step** (of a 50 ceiling) — a policy that gets in deep and
stays, exactly what removing the terminate-on-success incentive was supposed to
produce. Legacy success rate correctly reads a structural 0% as predicted (see
above).

**Bug found and fixed the same day**: Tail-settle initially reported 0% (`excl=512`,
i.e. every episode excluded) — traced to `validation/eval_arm.py`'s
`_tail_settle_stats` anchoring its trailing window to the nominal final snapshot
column (`dist_to_goal_cm_t20.0s` for a 20s episode), which is **empty in 100% of
rows for a structural reason**: the episode truncates one control step short of the
exact step-count boundary `metrics_wrappers.py`'s `_update_dist_snapshot` needs to
fire that slot's "due" check — a property of that existing (unmodified) wrapper, not
specific to this checkpoint. Anchoring to a column that's never populated excluded
every episode regardless of real hold quality — confirmed wrong immediately from the
same eval's own Settled/mean-dist/frac_envs_reached numbers, all showing
near-perfect holding. **Fixed**: drop any snapshot column empty for every row before
selecting the window (now correctly uses t15.0s-t19.0s for a 20s episode/5s window).
**Recomputed tail-settle for this run directly from its existing `arm_detailed.csv`
files (no re-run needed): 100.00% both buckets, 0 excluded** —
`logs/rsl_rl/arms/integrated_no_term/2026-07-30_19-57-17/arm_eval/summary.md`
corrected in place. Any other checkpoint evaluated with the pre-2026-07-31
`eval_arm.py` and a 20s-episode PLAY cfg has the same stale Tail-settle number —
re-run (cheap, no sim needed if the CSVs still exist) before trusting an old
Tail-settle reading.

**PROMOTED (confirmed 2026-08-01)**: `logs/rsl_rl/arms/integrated_no_term/
2026-07-30_19-57-17/model_11999.pt` is now `chosen_checkpoints/arm_left_latest.pt`
(checksum-verified), superseding the interim pre-NoTerm-fix `model_7999.pt` copied
there 2026-07-30 — reaching AND holding are both now solved at the real 40/10
hardware gain. See `arms_policy_finalisation.md` for what's still open (live demo
verification, `g1_rl_control` deployment wiring) before calling this fully finished.

**2026-07-30 — `g1_full_demo.py` didn't support the Integrated observation/action
layout at all; fixed.** Found after the user copied `model_7999.pt` into
`chosen_checkpoints/arm_left_latest.pt` and hit a runtime error the moment an arm
command actually ran — root cause: the demo hand-rolls its own arm control loop
(not a reuse of `G1ArmEnv`) using the legacy `current + delta`/raw-world-frame path
unconditionally, which is a hard 39-vs-46 observation-shape mismatch against an
Integrated checkpoint (crashes) and, even where dimensions happened to line up,
would silently apply the wrong control law (reinstating the torque ceiling) and the
wrong observation frame. Fixed via a new `--integrated` flag: switches the demo to
a persistent per-arm target buffer (mirroring `g1_arm_env.py`'s `joint_targets`,
re-anchored whenever homing completes) integrated the same way training does, plus
a torso-anchored local frame for ee_pos/goal (the moving-base analog of training's
fixed-env-origin subtraction — see `_root_anchor_pos`'s docstring in
`g1_full_demo.py`) and the `target_fb` observation block. Covers both the native-arm
path and the Y/U mirror-testing path (mdp/symmetry.py's 46-D sign convention, added
2026-07-29, already handled the mirrored `target_fb` block correctly). The flag is
cross-checked against the checkpoint's actual observation width and refuses to run
on a mismatch either way (safe failure, not silent corruption). **Implemented, NOT
yet live-tested against Isaac Sim** — verify interactively before trusting it:
`python testing/visual_testing/full_demo/g1_full_demo.py --arm_checkpoint
chosen_checkpoints/arm_left_latest.pt --arm left --integrated --target 0.3 0.2 1.0`.

**2026-07-30 — StandingPackage trained 7000 iters fresh
(`logs/rsl_rl/walking/arm_disturbance/2026-07-29_23-28-41`, model_6999): standing goal
achieved, with two honest costs.** `stand_still`: step count 8.23→**1.96**, |lateral
drift| 0.076→**0.031 m**, 0% falls, near-0% across all 4 disturbance phases (1/1025)
— the package (zero-snap + rel_standing 0.2 + position anchor) did exactly what it was
built for. Costs vs model_20996: (1) `forward_slow` heading drift 62→112 deg (medium/
fast/backward improved: 52→44, 57→43, 64→17); (2) in-place turning is now essentially
ignored — `turn_left/right` ang track err ≈ the commanded 0.2 rad/s with ~2 steps,
i.e. it stands still through pure-turn commands (turning was always undertrained —
dead ang_vel curriculum — but this policy sacrifices it further for stillness).
Promotion to `chosen_checkpoints/walking_latest.pt` is justified for the
arm-integration use case (user's stated priority); keep model_20996 as the
better-walking fallback if forward_slow drift or in-place turning ever matter.

**Gain search (2026-07-26) — closed.** Tested against Unitree's own real values, not
guesses: `Gain60Kd1p5` (60/1.5, dedicated arm-SDK example gain) — 24.43%/23.42%,
joints at-limit 86-90% of the time (underdamped oscillation, not an improvement).
`Gain15Kd1` (15/1, matches `unitree_rl_mjlab`'s actual deployed RL-policy config,
14.3-16.8/0.9-1.1) — 5.48%/6.23%, training reward flat from iteration 0 (never learned
at all at this gain, in this env's action-scale/decimation setup). 40/10 and 200/20
remain the only two gains that produce a working policy here; real-hardware kp/kd can be
set at deploy time regardless of what the policy trained on (SDK exposes per-joint kp/kd
live), so gain-matching is a deployment-config choice, not a blocker.

**Goal-distance curriculum (2026-07-26, in progress)** — `G1-Arm-Left-GoalCurriculum-v0`:
goals start at 15% of the full workspace box, expand to 100% by step 30k (~1250 iters).
2000-iter checkpoint: 22.70%/25.42%, close to settle-alone, **but still climbing at
cutoff** (`goal_reached_bonus` 0.19→0.31 in the last quarter) — the only variant tried
that hadn't plateaured by 2000 iters. Resuming tonight to 10,000 total.

**Critical finding — the 99.98% training-time success metric does not mean what it
looks like it means.** Built `validation/eval_arm_long_hold.py`: gives one env one fixed
goal from a fresh reset for 45s (vs. the standard eval's many quick resets per env).
Against the 200/20 reference: only **30.5% of fresh single attempts ever succeed**
(33.2% under a lenient "ever within 2cm" definition with no consecutive-hold
requirement — redefining success does NOT rescue this, the gap is in reaching, not
holding). Of attempts that do get close, hold quality is genuinely good (median 95%+ of
remaining time in-threshold, streaks averaging 16s+). Confirmed via a control test
(same script against a checkpoint with known 27.91% success — reported 31.8%, a close
match) that this is real, not a script bug. Root mechanism for the 99.98%-vs-30% gap
**not found** despite ruling out episode-count weighting, wobble config, cold-start
effects, num_envs/RNG artifacts, dones-before-rewards ordering (real, but a harmless
one-step lag), actuator/joint domain-randomization variance, and policy stochasticity
(policy is deterministic at inference). Practical implication: **for a true single-shot
precision-grasp use case, treat 200/20's real reliability as ~30%, not 99.98%.** For a
continuous/sequential-gesture use case the 99.98%-style number may be more
representative, but the mechanism gap means this hasn't been confirmed either way.

**Strategic pivot under consideration**: `unitreerobotics/xr_teleoperate` (official,
public) has `G1_29_ArmIK` — Pinocchio+CasADi numerical IK, matches our exact 29-DOF
variant, takes arbitrary SE(3) targets (not VR-locked), returns joint solution +
feedforward torque via inverse dynamics, needs only the G1 URDF (already have it) plus
`pinocchio`/`casadi`. Plan if pure-RL reaching doesn't clear a usable bar by Wednesday:
IK handles precise x/y/z grasp targets, RL policy (whatever curriculum produces) handles
compliant continuous-gesture motion. Not yet integrated — next step if pursued is a new
branch, arms only, legs untouched.

**Symmetry mirror convention — spot-checked, not fully validated.** `mdp/symmetry.py`'s
left-right sign convention (which joints flip sign under a mirror) was adapted "by
pattern" from the 23dof-era 5-DOF version when this 7-DOF layout was built, and has
never been empirically validated via the mirror-test script for this specific layout
(the module's own docstring points at `g1_arm_mirror_test.py` "for the pattern to
adapt," implying it wasn't actually run). Spot-checked `shoulder_pitch` by hand against
the real URDF's joint axes/origins (`g1_29dof.urdf`) — the dominant axis component has
the same sign on both sides, consistent with "pitch doesn't flip." Didn't verify all 7
joints this way (real risk of a hand-calculation error), and this is active in every
non-locked-wrist run — if wrong, it would inject physically-inconsistent transitions via
the 2x data augmentation, active in every result above except locked-wrist. Worth
running the actual mirror-test script against this layout if the current plan doesn't
pan out.

## Rejected/superseded experiment logs (deleted 2026-07-25 to reclaim space)

Conclusions above are the full extractable value from these — checkpoints themselves
(~15GB) deleted rather than kept as dead weight. All were untracked (`logs/` is
gitignored) so this has no git history impact; re-running any of them is possible if
ever needed, but none are expected to be.

**Walking, pre-reward-fix era (before 2026-07-24's `action_rate`/`joint_deviation_legs`
fix — see `RewardsCfg`'s own comments for the full ablation trail that found it):**

- `walking/base/2026-07-22_08-05-36` (2999 iters) — 0.302 m/s track err at 0.3 m/s
  commanded, 0.601 at 0.6 m/s commanded: essentially zero achieved velocity, the "never
  falls, never walks" local optimum this project was stuck in before the fix.
- `walking/arm_disturbance/2026-07-22_11-57-02` (deterministic disturbance phase, 5999
  iters) — 0%/33%/66%/84% fall rate across phases 0-3, the "before" reference for the
  later phase-mixing + `lin_vel_cmd_levels` curriculum fixes.
- `walking/arm_disturbance/2026-07-22_16-27-25` — phase-mixing-only intermediate step,
  0%/0%/7%/21% fall rate across phases 0-3.
- `walking/arm_disturbance/2026-07-22_23-30-26`, `walking/arm_disturbance/2026-07-21_21-37-42`
  — further pre-fix era runs, superseded.
- Aborted/incomplete (stopped at a few hundred iterations or less, no eval):
  `walking/base/2026-07-23_17-12-52`, `walking/base/2026-07-23_20-03-02`,
  `walking/arm_disturbance/2026-07-23_18-20-06`,
  `walking/ablation_reward_weights/2026-07-23_22-43-23`,
  `walking/arm_disturbance/2026-07-21_19-42-09`, `walking/base/2026-07-21_19-33-46`,
  `arms/left/2026-07-21_20-58-14` (all 3 latter ones ≤49 iterations — smoke tests).

**Walking, 2026-07-24 4-way reward ablation** — `ablation_term_penalty`,
`ablation_curriculum`, `ablation_all` confirmed no-ops (didn't fix the plateau);
`ablation_reward_weights` was the one that worked and is now the shared default — its
run (`2026-07-24_03-13-16`) is **kept**, not deleted, since it sources
`chosen_checkpoints/walking_2026-07-24_base_only_prev.pt`.

**Walking, `arm_disturbance/2026-07-25_02-45-43`** (7300 iters) — warm-started
continuation, no drift-penalty reward yet. Precision recovered, drift got worse (see
"Known gap" above). Not promoted; superseded.

**Walking, `arm_disturbance/2026-07-25_12-07-59`** (8000 iters, round 1 attempt A) and
**`arm_disturbance/2026-07-26_01-48-40`** (6000 iters, round 1 attempt B) — see "Known
gap" above for what each showed. Kept on disk for now (not yet deleted) since round 2 is
still in progress and these are recent reference points; candidates for cleanup once
round 2 concludes either way.

**Arms:**

- `arms/left/2026-07-23_02-28-39` — duplicate 40/10-gain baseline (27.32%), redundant
  with the kept `2026-07-23_22-54-45` (27.68%, near-identical result).
- `arms/left/2026-07-24_08-22-26` — combined `position_reward_exp_scale=3.0` +
  `entropy_coef=0.01` in one run; caused instability, motivated splitting into the two
  isolated ablations below.
- `arms/ablation_entropy_coef/2026-07-24_13-01-17`, `ablation_entropy_coef_small/2026-07-25_08-56-53`,
  `ablation_exp_scale/2026-07-24_13-30-59` — see "Ruled out" above.
- `arms/left_locked_wrist/2026-07-24_20-35-21` (2.7GB, the single largest reclaim) — see
  "Ruled out" above.
- `arms/ablation_rate_limit/2026-07-25_09-53-21` — killed partial run (3000-iteration
  target, stopped ~1300 in) when the target was changed to 5000; superseded.
- `arms/ablation_rate_limit/2026-07-25_10-24-14` (full 5000-iter run), `arms/left/
  2026-07-25_23-09-04` (baseline leg), `arms/ablation_low_vel_noise/2026-07-25_23-49-10`
  (confirmed harmful), `arms/ablation_privileged_critic/2026-07-26_00-29-08` (+5pts,
  reference point going forward), `arms/ablation_log_std/2026-07-26_01-09-05` (+2pts,
  reference point going forward) — kept for now as the most recent reference data; not
  yet cleaned up.

**Kept, not deleted:** `arms/left/2026-07-22_06-20-55` (the working 200/20-gain
reference, 99.98% success — unique, valuable comparison point) and
`arms/left/2026-07-23_22-54-45` (current best plain 40/10 baseline, 27.68%).

## Deferred / future items

Merged in from `deferred_items_2026-07-21.md` 2026-07-27 (that file deleted, this is
now the one place both status and deferred work live). Raised during the 29dof pivot
implementation, priority notes added 2026-07-22 per user review.

1. **Arm-rest-before-locomotion-transition safety mechanism.** The old (23dof-era)
   repo's `--wait_arm_rest` flag delayed a stand→walk transition until the arm finished
   homing back to default, avoiding a momentum-carryover fall. The current demo/eval
   have no equivalent — arm reaching is unconditional by default (see
   `testing/visual_testing/full_demo/README.md`'s `--reset_arm_on_walk` flag, added
   2026-07-27, for the alternative). Still an open question whether the default
   (unconditional) behavior causes real instability at speed — flagged, not resolved.
2. **Arm goal orientation (6-DOF)** — using the wrists for actual orientation control,
   not just as redundant DOF for a position-only goal. Position-only is what's trained
   now. Waits.
3. **IK-driven / policy-driven arm-disturbance training curricula**
   (`StandingArmIKReachDisturbance`/`StandingArmPolicyReachDisturbance`) — more
   sophisticated than the current scripted joint-space disturbance (real x/y/z reach
   targets instead of random joint swings). Superseded in spirit by the
   `xr_teleoperate` IK pivot above — revisit only if that path doesn't pan out.
4. **Actuator delay + Coulomb friction sim2real model** (`UnitreeActuator`, available
   infrastructure in `assets/robots/unitree_actuators.py`) — not wired into the G1
   asset's actual actuators yet. Waits.
5. **Right-arm / both-arm dedicated training** — right arm currently has no checkpoint
   of its own anywhere, purely mirror-driven from the left policy. `best_combined` is
   left-only; testing right-arm behavior at that quality requires the demo's Y-key
   mirror feature, not a native right-arm checkpoint. Waits.
6. **Arms-while-walking.** [**RESOLVED 2026-07-22, opposite direction**] — the
   arm-motion disturbance curriculum previously ran regardless of the commanded
   velocity. Per explicit user request, this is now gated off: `ArmMotionDisturbance`
   only disturbs envs currently commanded to (near-)stand
   (`mdp/events.py`'s `_STANDING_CMD_THRESHOLD` gate) — envs commanded to walk get arms
   relaxed toward default instead. "Arms while walking" stays a real future feature,
   deliberately not trained toward yet — see the 2026-07-27 `g1_full_demo.py` stress
   test (untrained generalization tolerated it for a while, then fell) for why this
   still matters.
7. ~~`testing/walking_testing/g1_stand_walk_switch_demo.py`~~ — **deleted 2026-07-27**
   (general cleanup pass). Referenced old 5-DOF/"IK" naming and tested a discrete
   standing/walking policy-switch mechanism (hysteresis + minimum dwell time to avoid
   chattering) that's superseded by the current single combined stand+walk policy — see
   the retrospective's "Why one combined policy" section for the reasoning.
8. **`base_height` reward possibly too rigid, contributing to walking drift AND
   standing balance quality.** See "Potential improvement" under Walking above —
   knee-flexion tracking was added to `eval_walking.py` to check this with data;
   not yet run. Rotating (A/D) visibly makes the near-locked-knee issue worse
   (2026-07-27 observation) — raises this item's priority for the next walking
   training session.
9. **`g1_full_demo.py`'s arm-checkpoint loader** now infers obs-dim/noise-type from
   the checkpoint (fixed 2026-07-27), but `--arm right`/`--arm both` still need their
   own natively-trained checkpoints that don't exist for the `best_combined` recipe —
   not a code gap, a missing-training-artifact gap.
10. **Mechanism for the 99.98%-vs-~30% single-shot gap (arms)** — not found despite
    extensive investigation (see "Critical finding" above). Revisit if the IK pivot
    doesn't fully replace the need for a reliable RL reaching policy.
11. **`joint_mirror` gating fix (walking)** — 2026-07-28's `joint_mirror` experiment
    (see "The joint_mirror tradeoff" above) fixed forward-walking drift and `turn_left`'s
    fall rate for real, but reintroduced ~2x of the stand-still stepping/drift
    regression `feet_contact_without_cmd` had just fixed — likely because `joint_mirror`
    penalizes left/right asymmetry unconditionally, fighting whatever
    (possibly-asymmetric) stance `feet_contact_without_cmd` had settled into for
    standing still. Proposed fix, not yet implemented: gate `joint_mirror` the same way
    `feet_contact_without_cmd` is gated — off (or reduced) when `command_norm < 0.1` —
    so it only enforces symmetry during actual walking, where the asymmetry was found,
    without fighting the stand-still-specific fix. Walking is frozen on the pre-mirror
    checkpoint for now (user decision, 2026-07-28); revisit this once there's time to
    spend on walking again.
12. **Arm: distance-adaptive movement speed for real hardware.** User's idea, 2026-07-28
    — move faster while far from the goal, slow to the current (precise) speed once
    close, so the real robot's arm doesn't look sluggish during the initial approach.
    Implemented in sim as `G1ArmLeftAdaptiveRateEnvCfg`/`G1-Arm-Left-AdaptiveRate-v0`
    (`g1_arm_env.py`, `use_adaptive_rate_limit` + related cfg fields) but NOT trained —
    deprioritized after the same day's diagnostic data (new `final_dist_to_goal_cm`/
    `dist_to_goal_cm_t*s` eval columns) showed failures are dominated by a
    converged-but-imprecise plateau, not a running-out-of-time-while-approaching
    pattern (92% of failed episodes end within 1cm of their own best-ever distance;
    65% stop improving well before the 20s episode ends) — so this isn't expected to
    fix the sim success-rate ceiling. Still explicitly wanted for the physical robot
    regardless (a real UX/naturalness concern independent of the success-rate metric,
    e.g. for a bystander watching the robot reach) — revisit training this once there's
    GPU time to spare, and/or wire the same fast-far/slow-near idea directly into
    `g1_rl_control`'s real-hardware action pipeline even if the sim policy itself isn't
    retrained with it (i.e. as a deployment-side rate shaping, not necessarily a
    training-side change).

## Lessons learned (carried from the 23dof phase)

Merged in from `lessons_learned.md` 2026-07-27 (that file deleted). Extracted
2026-07-21 before removing the 23dof-era status docs from `main` — the subset that
isn't specific to the old 1-DOF-waist/no-wrist robot model. Several of these have
already recurred verbatim during the 29dof build (noted inline) — treat this as an
active checklist, not just history.

1. **PhysX/`find_joints()` orders joints breadth-first from the articulation root, NOT
   in URDF/declaration order.** On the G1 this interleaves left/right. Any code that
   assumes a left-block-then-right-block layout from a hand-written joint name list
   will silently scramble commands between sides. Always derive column/index mappings
   from the actual returned joint ids, never from source-list order. Cost about a week
   on the 23dof arm-reach disturbance.
2. **An unpriced "free" DOF gets exploited, and capping it just moves the exploit to
   the next-cheapest one.** Relaxing a joint-deviation reward to 0 "to let it help with
   balance" is a real trap — fine while the disturbance is mild, silently becomes a
   problem once it escalates. On 23dof this went torso → hip → ankle across three
   successive fixes. **Recurred 2026-07-27**: `base_height`'s rigid -10 weight likely
   pushed the exploit toward near-locked knees instead (see Walking section above) —
   same pattern, different joint. Price in deviation/posture costs for every joint from
   day one, not reactively per-joint after each one gets discovered.
3. **Train against the exact deployment-time actuator gains, not a softer sim-only
   value.** A checkpoint trained with soft gains and deployed against stiff ones (or
   vice versa) can look perfect in its own training distribution and still collapse the
   first time it meets the real gain. **Directly relevant to this week's whole
   gain-search saga** (200/20 vs 40/10 vs 60/1.5 vs 15/1) — match explicitly or
   curriculum toward it deliberately, don't assume.
4. **A shared deployment env built from a different lineage than the one being tested
   does not automatically inherit that lineage's config changes.** An eval/demo script
   that builds its env from a "sibling" base config can silently drop a change that only
   existed on the tested lineage — symptom is catastrophic-looking numbers that are
   actually an eval bug, not a real regression. Any non-reward/observation config change
   needs an explicit check: does every script building this env from a different base
   actually carry it over?
5. **PPO seed variance on this task is large enough that a single run proves nothing.**
   An identical config produced 2.1% vs 100% fall rate on two different seeds once.
   Never trust a single training run's numbers as final.
6. **A wider network measurably hurt mirror-quality on the un-trained (mirrored) side
   of a symmetric task.** If a mirror-based approach is used for the second arm rather
   than training both natively, re-verify this rather than assuming a bigger network is
   safe.
