# Policy status — 29dof pivot

**Branch note (2026-07-27, `ik_residuals`)**: the "Walking + standing" section below is
current and still the live source of truth — that policy is untouched by the arm pivot.
The "Arm reaching" section is now **historical** — it's the pure-RL record that
motivated replacing RL-only arm reaching with IK (+ a later RL residual); see
`ik_arm_integration_plan.md` for the current plan and `retrospective.md` for the
narrative. Keep the arm data below as-is — Phase 3 of the plan validates the IK
approach against these exact numbers (200/20 reference ~30% true single-shot,
`best_combined` ~28-33% aggregate). Update the Walking section here as before; add new
IK/residual findings to `ik_arm_integration_plan.md` or a successor doc, not by
rewriting this section's RL history.

Living summary of what's actually validated vs. still open, for the checkpoints in
`chosen_checkpoints/`. Update this when a checkpoint gets promoted or a real gap is
found — don't let it go stale the way `known_issues.md` did on the 23dof branch.

## Walking + standing (`chosen_checkpoints/walking_latest.pt`)

**Status: working, ready for initial real-robot testing** (sim-verified only — no
real-robot test yet), **drift is a known gap, actively being worked on (round 2).**
Promoted 2026-07-25 from `logs/rsl_rl/walking/arm_disturbance/2026-07-24_15-47-18/
model_15998.pt` — warm-started from a validated base-recipe checkpoint
(`walking_2026-07-24_base_only_prev.pt`, kept alongside for reference), then trained
10,000 more iterations on the real deployment recipe
(`G1-Locomotion-Velocity-ArmDisturbance-v0`). **Untouched by any drift experiment below**
— every drift attempt trains into its own fresh directory, nothing here promotes over
it. This is the safe fallback if drift-fixing doesn't pan out.

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

**Status: not working, root cause still open, but two small real improvements found +
one structural gap fixed.** Every 7-DOF run at the real hardware gain (40/10
stiffness/damping, confirmed matching real deployment — see `unitree_rl_lab/deploy/
include/FSM/State_RLBase.h`, `tau()=0` on real hardware too, so this isn't a sim-only
artifact) plateaus in the 25-29% success band. `chosen_checkpoints/arm_left_latest.pt`
is stale — do not treat it as current. **The 200/20-gain reference checkpoint (99.98%
success) is intact and NOT in chosen_checkpoints/ — it's `logs/rsl_rl/arms/left/
2026-07-22_06-20-55/model_2999.pt`. No retraining needed to re-validate it.**

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
