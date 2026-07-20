# Known Issues / Backlog

> **HISTORICAL ARCHIVE (2026-07-16).** This file is kept because many code comments
> reference it, but it is NOT maintained as of 2026-07-16 and several "open" entries
> below are resolved or superseded. The authoritative current state and next steps live
> in `definitive_next_steps.md` (repo root). Do not act on this file without checking
> there first.

Running list of things discovered during actual work that were deliberately deferred
rather than fixed immediately — so they don't get lost or silently re-discovered later.
Not a duplicate of the roadmap plan; this is specifically things found *while doing* the
work, with enough context to act on them later. Update this file directly when an item
gets resolved (move it to the bottom under "Resolved", don't just delete it).

## Standing/integration saga — everything tried so far (running list, last updated 2026-07-14)

Quick-reference index for the standing + arm-integration debugging effort (2026-07-12
onward) — full narrative detail for each is in this file's other sections / conversation
history, this is just "what have we already tried, so we don't re-propose it blind."

**Training strategies** (changes to the disturbance mechanism, curriculum, reset
distribution, or reward shaping — i.e. what the policy is being asked to solve):

- Scripted joint-space arm disturbance (`StandingArmTrajectoryDisturbance`, 5-phase
  curriculum) — the original mechanism standing was first trained under. Superseded, not
  actively used for new training now.
- Real per-arm differential-IK disturbance (`StandingArmIKReachDisturbance`) — replaced
  the scripted trajectory with an actual x/y/z reach target, solved with analytic IK.
  Adopted, current baseline mechanism.
- Curriculum ramp (linear fraction 0.15→1.0 between `enable_step`/`ramp_full_step`) on top
  of the IK disturbance — avoids abrupt disturbance onset. Adopted.
- Dwell-until-timeout goal cycling — goals now held until `max_steps_per_goal` instead of
  resampled the instant they're reached, so "reach and hold" actually gets trained.
  Adopted (2026-07-12).
- Torso deviation reward re-tightened (`joint_deviation_torso.weight` 0.0 → -0.05) —
  **tried, measurably didn't help** (integration fall rate flat-to-worse). Dropped.
- Real arm-IK *policy* driving the disturbance instead of analytic IK — **tried, fixed
  `standing_still` outright (94%→0% fail) but broke `standing_arm_left_reach` badly
  (73%→96.5% fail)**, traced to an emergent deep squat. Not currently in the active line.
- Direct base-height reward (`base_height_l2`, target 0.75m) — **tried, fixed the squat
  (confirmed via height metrics) but did not improve arm-reach fall rate** (flat-to-
  slightly-worse). Kept as the current baseline (real, correct fix for a real problem —
  just not the dominant one).
- Left-right leg symmetry reward (`leg_symmetry_l2`, legs/hips only, not arms/torso) —
  **just launched (2026-07-14), results pending.**
- Widened arm-training root-wobble curriculum (roll/pitch 5.7°→20°, added yaw-rate/
  lin-vel axes) — real-motion (`fix_root_link=False`) attempt failed during play-testing
  (robot fell face-first, no active balance controller) and was reverted to
  synthetic-observation-only wobble; the wider synthetic-wobble + wide-net retrain was
  shelved (integration got worse, not better).
- Real-hardware PD gain standardization (60/1.5 stiffness/damping across the whole
  pipeline) — **tried, reverted**: helped standing (best native tilt result) but crashed
  arm-IK reach success 86%→30% and broke walking, confirmed root cause was Isaac Lab's PD
  actuator having no gravity-compensation feedforward (real SDK's low gains assume one
  exists in the real robot's own controller).
- Per-bucket/per-mode live gain switching (deployment-side, not training) — replaced the
  single global gain override with `write_joint_stiffness_to_sim` calls matched to
  whichever policy is actually in control at that moment. Adopted.
- Null-space regularization for the arm policy (uniform, then 3x-weighted on
  `elbow_pitch`) — real, measurable precision improvement (mean/p90 distance down) but
  **did not fix** the underlying redundancy/branch-selection issue either way. Kept
  (real positive effect), redundancy issue still open.
- Exponential proximity reward bonus for the arm policy — **tried, reverted**: caused a
  large regression (55%→11-14% success), a "good enough short of the goal" local optimum.
- Goal-curriculum (40%→100% workspace) for the arm policy — **tried, dropped**: worse
  success rate than no curriculum at all.
- Reachability-driven `_GOAL_BOUNDS` reshaping (pulled in the far corner, pushed the
  near-torso corner out past the collision margin) — **adopted**, confirmed the dominant
  fix for the arm policy's original ~55% plateau (74.7%/76.8% immediately, 85.6%/84.6%
  after a second refinement).
- Per-joint/per-bound joint-limit margin relaxation (`elbow_pitch` lower bound only) —
  **tried, no effect** on success rate or on the near-limit-branch mechanism it targeted.

**Policy strategies** (changes to network architecture or the PPO algorithm itself, same
task definition):

- Wide actor/critic network ([512,256,128] vs. [256,128,64]) for the arm policy — **the
  one clearly reproducible win** across two separate baselines (best p90/tail both times).
  Adopted into the consolidated arm policy.
- Entropy coefficient increase (0.0 → 0.01/0.005) for the arm policy — mixed: a real
  success-rate win at 0.01 (consistent across two buckets), but a large regression when
  bundled with the exponential reward bonus at 0.005 (see above — the two were never
  cleanly isolated in that run). Not adopted into the final consolidated policy.
- PPO hyperparameter tuning (`num_learning_epochs` 5→8) — **tried, no clear effect**,
  lowest-confidence of the arm-policy sweep experiments. Not adopted.
- Mirror-symmetric PPO data augmentation (`RslRlSymmetryCfg`, `compute_symmetric_states`)
  — **adopted for walking**, fixed an observed asymmetric-gait failure mode. **Never
  ported to standing** — deliberately, since the arm-disturbance curriculum is
  intentionally asymmetric sometimes (left/right-only reach) and a naive reuse would need
  to correctly re-map which arm is "active" under the mirror transform. Directly related
  to why `leg_symmetry_l2` above is a reward-term approach instead of data augmentation.
- Wider network + widened wobble curriculum together for the arm policy (see above) —
  shelved as a bundle, not individually attributable within that run.

## Arm policy — current state, for discussion (last updated 2026-07-08)

Short version, **now materially better than this section's original framing (2026-07-07
morning) suggested** — kept below as history, but don't read the opening paragraphs as
current status. As of today: **85.6%/84.6% success** (up from the original ~55%),
p90 ~3cm (down from ~12cm) — no more long tail of badly-missed goals, remaining failures
are near-misses. The dominant cause (a genuinely unreachable chunk of the goal workspace)
is fixed and confirmed. What's left is a smaller, well-characterized issue: manipulator
redundancy causing the policy to sometimes land on a less-reliable of two valid solutions
for the same goal — a targeted fix for that is written, training now (see the bottom of
this section for the current state and what's next).

**The original core problem** (2026-07-07): a 5-DOF single-arm reaching policy
(`G1-Arm-IK-Left-v0`), trained with PPO, plateaued at **~55% success rate**, median
distance-to-goal 1.98cm (right at the 2cm threshold) but **p90 ~12cm** — a real subset of
goals the policy just didn't solve, not noise around a good average. Confirmed via
`validation/eval_arm.py` (deterministic, fixed-seed) at two checkpoints: 1500 iterations
and 4998 iterations (full budget).

**More training does not fix it — confirmed, not assumed.** Bucketing the full
5000-iteration training curve into 20 slices: success rate reaches ~47% by iteration
~260 (about 5% of the way through), then sits flat at 47-48% for the remaining ~4700
iterations. This is a hard, early plateau, not slow convergence — training for longer is
a disproven hypothesis for this specific problem, not an open question.

**What's *not* the problem, ruled out with evidence:**
- Not unreachable goals — already fixed once (see "joint ranges" entry below);
  `eval_arm.py`'s joint-range-utilization table shows healthy, varied use of the real
  hardware range (43-100% across all 5 joints), not degenerate tiny movements.
- Not a coordinate-frame/observation bug — some episodes *do* solve precisely (best-case
  distance is consistently ~1cm), which wouldn't happen if the goal were being fed to the
  network incorrectly.
- Not obviously an implementation bug in the observation/action/reward pipeline — read
  through carefully multiple times this session, structurally sound.

**Leading explanation**: a fairly ordinary PPO premature-convergence pattern.
`entropy_coef=0.0` (no exploration incentive) plus a reward landscape that doesn't
strongly punish stopping short of the goal likely let the policy settle into a
"good enough" local optimum early and stop improving. One attempt at fixing this (adding
an exponential proximity bonus + raising entropy to 0.005, tried together) made things
significantly *worse* (success dropped to ~12%) — reverted; see the two entries below for
the full story. Both changes were bundled into one retrain, so we don't know which one
(or both) caused the regression — the next attempt needs to isolate variables.

**Is RL-for-IK the right approach at all?** For *pure* static reaching (no disturbance),
no — classical/analytical IK (Jacobian-based or closed-form) is the standard approach and
would already be exact, no training needed. RL is here because of where this policy
actually needs to work: reaching while the base is being disturbed (standing) or while
walking — a whole-body robustness problem that's genuinely hard to hand-engineer, and
where learned control is standard practice in legged-manipulation research. Not
proposing a change of approach right now; flagging that *residual RL* (a real IK solver
as the base, with a learned correction on top) is a real option if pure end-to-end RL
keeps plateauing — still RL, just removes "does it get basic kinematics right" as a
variable.

**Queued (2026-07-07 night): 4-way isolated overnight sweep**, run via `overnight_train.sh`
— 3 arm-policy changes, each vs. the same G1-Arm-IK-Left-v0 baseline (55.0% success,
median 1.98cm), 5000 iterations each, one variable per run so results stay attributable:
1. `G1-Arm-IK-Left-RewardShape-v0` — exponential proximity bonus, gentler/tighter than
   the first (bundled, regressed) attempt: `exp_scale=1.5`, `exp_sigma=2cm` (was 5.0/5cm).
2. `G1-Arm-IK-Left-GoalCurriculum-v0` — goals start confined to 40% of the workspace box
   around its centre, linearly widen to 100% by env-step 60,000 (~50% of the run).
3. `G1-Arm-IK-Left-WideNet-v0` — wider actor/critic network ([512,256,128] vs baseline
   [256,128,64]) — tests whether the plateau is a capacity limit, not an exploration one.
4. `G1-Arm-IK-Left-Entropy-v0` — `entropy_coef=0.01` alone (no reward-shaping bundled
   this time) — directly tests the leading plateau theory (premature convergence from
   zero exploration incentive).

Plus a standing retrain (`G1-Locomotion-Standing-Flat-v0`, 8000 iterations) exercising
the new phase-5 (33 rad/s) disturbance curriculum for the first time — see
`g1_locomotion/mdp/events.py`'s `StandingArmTrajectoryDisturbance`. Each training is
immediately followed by its matching eval script (`eval_arm.py`/`eval_standing.py`) in
the same overnight run, so morning review starts from `arm_eval/summary.md` /
`disturbance_eval/summary.md` per run rather than raw training CSVs. Which (if any) of
the 4 arm changes should actually be kept is an open question until those results come in.

**Results (2026-07-08): all 4 arm changes are small, and one is a net negative.** vs.
baseline (55.5%/56.4% success, no_wobble/with_wobble):

| Run | Success % | Mean dist (cm) | p90 dist (cm) | Mean steps-to-success |
|---|---|---|---|---|
| baseline | 55.5 / 56.4 | 4.93 / 4.85 | 12.06 / 11.82 | 102 / 101 |
| reward_shape | 55.4 / 58.5 | 4.73 / 4.73 | 11.29 / 11.59 | 98 / 101 |
| goal_curriculum | 54.1 / 51.9 | 4.38 / 4.73 | 10.47 / 11.52 | 119 / 120 |
| wide_net | 57.2 / 56.0 | 4.25 / 4.32 | 9.99 / 10.47 | 102 / 101 |
| entropy | 58.8 / 59.1 | 4.56 / 4.60 | 11.00 / 11.28 | 101 / 101 |

- **Goal curriculum: dropped.** Success rate went down in both buckets (worst of the
  four), and successful episodes take ~20% longer to converge. Negative result, worth
  keeping in mind — "easy goals first" didn't help this task.
- **Reward shaping: inconclusive**, flat on no_wobble, mildly better on with_wobble —
  not a consistent signal, not pursuing further on its own.
- **Entropy (0.01)**: the clearest success-rate win (+3pp both buckets, consistent
  between buckets unlike the other three) — supports the premature-convergence theory.
- **Wide network**: the clearest precision/tail win (p90 down ~15-17%, mean dist down),
  success rate roughly flat — supports a capacity-limit contribution separate from
  exploration.

None of these individually or combined were expected to be enough to meaningfully clear
the plateau — they read as things you layer onto an already-decent baseline, not fixes
for a mediocre one. Decided (2026-07-08) to hold off on combining entropy+wide_net and
look for a bigger structural lever first.

**Found: `joint_vel` observation bug, predates Phase 2 (2026-07-08).** While diagnosing
the plateau, bucketed the baseline eval's failures by difficulty (`max_dist_to_goal_cm`,
a proxy for initial reach distance) using the existing `arm_detailed.csv` — no retrain
needed. Two findings that don't fit a simple "goals are too far / not enough control
authority" story:
- Success rate is roughly flat (53-63%) across the easiest 80% of goals, only dropping to
  ~45-48% for the hardest 20%. A control-authority/reachability bottleneck would predict a
  much steeper distance-dependent cliff than this.
- Timeout failures miss by a real margin, not a near miss: median distance at timeout is
  8.5cm (goal threshold is 2cm), and 80% of timeouts never even got within 5cm.

This pattern — a roughly uniform ~45% failure rate regardless of nominal difficulty,
missing by a lot rather than a little — pointed at a broad control-quality gap rather
than a reachability one. Checked the observation builder and found: `joint_vel` has been
silently sliced to the first 3 of 5 arm joint indices (`jt[:3]`) since before Phase 2 —
the policy has never been able to see elbow_pitch/elbow_roll velocity, i.e. it's been
partially blind to its own arm dynamics in exactly the two joints that most directly
determine reach depth. Fixed to observe all 5 (`g1_arm_env.py`, `mdp/symmetry.py`,
`g1_full_demo.py`); observation grew 26-D → 28-D per arm (52 → 56 for `arm="both"`). See
`training_regimes.md` for the full breakdown.

**Decision: retrain with only this fix, in isolation, before layering entropy/wide-net
on top.** This is a real bug fix (not a hyperparameter guess), and the diagnostic data
points at it more directly than any of the 4 swept experiments did — worth seeing how far
it moves the needle on its own first. Target before reconsidering further additions: ~80%
success (up from the current ~55-59% ceiling across every variant tried so far).

**Result: `joint_vel` fix alone (1500-iteration test run, `run_name=joint_vel_fix`) —
no meaningful change.** 57.0%/56.3% success vs. baseline's 55.5%/56.4% — within noise.
Bucketing the run's own training-time episodes by env_step showed the identical shape as
before the fix: success jumps ~30% → ~48% in the first ~15% of training, then flatlines
for the rest. TensorBoard's loss curves looked improved, but that's expected regardless
of task performance (the observation change alone shifts what the value function has to
fit) — this is exactly why `eval_arm.py`'s success-rate metric, not loss, is treated as
ground truth here. Correct bug, real fix, worth keeping — but not the plateau's cause.

**Next isolated test (2026-07-08): null-space regularization.** This is a 5-DOF arm
reaching a 3-DOF (position-only) goal — manipulator redundancy means most goals have a
whole family of valid joint configurations, and PPO's unimodal Gaussian policy is a poor
structural fit for a target with multiple equally-valid solutions (it can waver between
them rather than consistently committing to one), which fits the observed failure
pattern (uniform ~45% failure rate regardless of difficulty, missing by a real margin —
not a near-miss). Added a small, deliberately weak reward penalty
(`null_space_penalty_scale=0.05`) on the arm's joint angles deviating from
`default_joint_pos` (the asset's own rest pose, already used elsewhere as "home") — a
soft tiebreaker among valid solutions, not a constraint on the goal itself; doesn't touch
end-effector orientation (goal stays position-only). One variable, on top of the
joint_vel-fixed baseline, nothing else changed.

**Result (2026-07-08): real effect, but not on success rate.** 56.0%/57.7% success —
still flat, same training-time plateau shape/timing as every other variant. But mean
distance and p90 both improved meaningfully and consistently in both buckets (mean dist
down ~8-11%, p90 down ~12-13% vs. baseline) — a real, non-noise effect (unlike the
joint_vel fix, which was flat on every metric), so the earlier concern about the scale
being too weak to matter is resolved — it clearly changed behavior, just not on the
specific discrete threshold. Reading: redundancy has *some* real effect (episodes that
fail land closer on average) but isn't the dominant cause of the plateau.

**Found: a large fraction of the goal workspace is not actually reachable within the
success threshold (2026-07-08).** Built `validation/check_arm_reachability.py` — samples
random joint configs within the real hardware limits (no RL/reward involved at all, pure
kinematics) and checks how much of `_GOAL_BOUNDS` is covered by the arm's actual reach.
Result (`validation/arm_reachability/left_summary.md`): only 47.1% of the goal box is
within 2cm of anything reachable, and the *median* distance from an arbitrary box point
to the nearest reachable point is 2.65cm — already past the 2cm success threshold. The
per-octant breakdown is physically coherent: hardest region is far-forward + far-to-the-
side + high-up simultaneously (mean 7.48cm, max 20.38cm — the natural edge of the arm's
reach), easiest is close-in + to-the-side + high-up (mean 1.42cm). The raw 47.1% number
understates true reachability (random joint sampling under-explores boundary
configurations that need several joints simultaneously near their limits — a curse-of-
dimensionality effect that RL's gradient-guided search doesn't suffer from, which is why
every trained variant actually lands at 54-59%, above the naive 47% estimate) — but the
qualitative conclusion looks solid: a meaningful, physically-explicable chunk of the goal
box is at or beyond the edge of what the arm can reach, and this is very plausibly the
dominant reason every arm-policy variant tried so far (baseline, joint_vel fix,
null-space reg, entropy, wide-net, reward-shape, goal-curriculum) has landed in the same
~55-59% band regardless of what else changed — no amount of retraining can cross a
success-rate ceiling set by the task definition itself.

**Fixed 2026-07-08: reshaped `_GOAL_BOUNDS`, two distinct problems on opposite edges of
the box.** Tried to get a live Isaac Sim reading of `torso_link`'s position to check the
"close-in, low" octant's poor score (6.44cm mean) directly, but two consecutive sim
launches hung (one with an "X connection broken" error mid-startup — looked like a
transient environment issue, not a script bug — both needed SIGKILL, `timeout`'s SIGTERM
didn't work, same as Isaac Sim processes elsewhere in this project). Fell back to the
real G1 URDF (`g1_29dof.urdf`, the same source used for joint limits earlier) instead of
retrying — `torso_link`'s collision box (`<box size="0.13 0.20 0.30"/>`, converted through
the URDF's joint-origin chain into the same local frame `_GOAL_BOUNDS` uses) works out to
roughly `x:[-0.056,0.070] y:[-0.10,0.10] z:[0.785,1.085]`. Two separate findings, not one:

- **The far corner (max x, max y, max z simultaneously) was genuinely unreachable** — a
  pure arm-length limit, nothing to do with collision. Needs max forward + max lateral +
  max height reach all at once, beyond what a ~50-60cm arm can do.
- **The near corner (min x, min y, min z) is not unreachable, but sits only ~3cm from
  the torso's surface** — well inside the 12cm safety margin `torso_proximity_penalty_scale`
  already enforces elsewhere. Not a collision (boxes don't literally intersect), but the
  policy was being asked to violate its own anti-collision incentive to succeed there.

Reshaped both arms' bounds accordingly (`x: (0.1,0.5)→(0.15,0.42)`,
`y: (0.05,0.45)→(0.05,0.40)` / mirrored for right, `z: (0.9,1.2)→(0.9,1.15)`) — pulled the
far corner in, pushed the near-x edge out past the torso margin. **This makes every prior
arm eval number (baseline through null-space, all ~55-59% band) non-comparable to
whatever trains next** — that's expected and intended, not a regression; a smaller,
better-clipped box should raise the achievable ceiling, but part of any jump is "the task
got fairer," not purely "the policy got better," worth remembering when reading the next
results.

**Result (2026-07-08): confirms reachability was the dominant factor.** 74.7%/76.8%
success (no_wobble/with_wobble), up from ~55-59% across every variant tried on the old
box. Failure profile changed character, not just rate: median timeout distance dropped
8.5cm -> 4.9cm, max dropped 20.0cm -> 12.5cm, ~51% of remaining timeouts are now near-misses
(<5cm) vs. 20% before — no more long tail of "nowhere close" failures.

**Re-ran the reachability check against the new bounds — still a residual gap, precisely
localized.** Coverage improved (2cm tolerance: 47.1% -> 65.3%, median nearest-reachable
distance 2.65cm -> 0.76cm) but the low-x/low-y octant is still clearly the worst (mean
4.1-4.6cm vs. <1cm for high-x/low-y) — the first x push (0.10->0.15) helped but didn't
fully clear the torso margin. Per-axis octant data isolated x as the dominant lever
(holding y low, x alone: low->high dropped that corner's mean distance ~4.6cm -> ~0.9cm)
with y a smaller secondary contributor (~4.6cm -> ~1.8cm). Pushed further:
`x: (0.15,0.42)→(0.20,0.42)`, `y: (0.05,0.40)→(0.08,0.40)` (mirrored for right). Not yet
retrained/evaluated on this second refinement.

Also worth a plain note on the >100% joint-range-utilization numbers seen in these evals
(e.g. left_elbow_pitch at 100.7% under `with_wobble`) — not a bug. `_arm_hw_limits` is
actually the *soft* limit (`soft_joint_pos_limit_factor=0.9` on the G1 asset, already 90%
of the true mechanical range) — targets are clamped to it, but the PD controller isn't
infinitely stiff, so momentum can carry the actual joint briefly past that soft boundary
into the buffer zone before the true hardware limit. That buffer is what soft limits are
for; a small transient overshoot there is expected, not a safety concern.

**Result (2026-07-08): second box refinement (`reachability_v2`) — 85.6%/84.6% success.**
Mean dist 2.15/2.17cm, median 1.89/1.90cm (essentially at the 2cm threshold), p90 down to
2.93/3.16cm — no more long-tail failures, everything is a near-miss now. Confirmed via
`g1_arm_reach_test.py` that motion quality at both the new box's far and near corners
still looks like genuine full-range reaching (joint-range utilization 65-87% across all 5
joints), not something clipped/trivial — some residual shaking near the target reported
visually, see the joint-config finding below for the likely cause.

**Clarification: null-space regularization was never removed.** It was added directly to
the base `G1ArmIKEnvCfg` (not a toggleable opt-in like the sweep experiments), so it's
been silently active in every run since — including `reshaped_goal_bounds` and
`reachability_v2`. Nothing to "reintroduce."

**Found: joint-config correlation check (2026-07-08) — elbow_pitch's asymmetric range
conflicts with the joint-limit penalty at exactly the pose needed for far reaches.**
Extended `ArmMetricsCsvWrapper` to log each joint's angle at the episode's *closest
approach* (not episode end, which for a timeout is just wherever the policy drifted to
afterward) — reused the already-trained `reachability_v2` checkpoint, no retrain needed.
Result: `left_elbow_pitch_joint` was within 5% of a hardware limit at closest approach in
56.8%/51.5% of failures vs. only 17.5%/15.2% of successes (no_wobble/with_wobble) — every
other joint showed ~0% near-limit involvement in either group. All of the near-limit
failures (100%) were at the *lower* bound (-2.5°), none at the upper (185.5°).
Mechanism: elbow_pitch's range is heavily asymmetric — 0° is roughly a straight, fully-
extended arm, and the joint barely goes past straight (-2.5°) but folds a lot (up to
185.5°). Full extension is a normal, necessary pose for far reaches (not a dangerous edge
case), but the flat 5%-of-range joint-limit penalty applied uniformly to every joint
doesn't know that, and was fighting the policy exactly when it needed to confidently
commit to full extension — plausibly also the cause of the shaking near the target
reported after the last visual check.

**Refined before training** (your own good instinct, not silly at all): the first version
reduced elbow_pitch's margin uniformly on *both* ends, but the data only supports it at
the lower bound (100% of near-limit failures there, 0% at the upper). Reworked
`_joint_limit_margin_fraction` to be per-joint *and* per-bound (`g1_arm_env.py`
`__init__`, shape `(n_joints, 2)`): elbow_pitch's lower-bound margin is 0.01, its upper
bound stays 0.05, and both bounds on the other 4 joints stay 0.05 (unchanged — no
near-limit involvement in either success or failure episodes). Same fix where there's
evidence, zero change everywhere there isn't. Doesn't touch the actual hard clamp either
(`_arm_hw_limits`, already 90% of the true mechanical range via
`soft_joint_pos_limit_factor=0.9`) — this only changes the reward-shaping margin on top
of that already-conservative bound, not what the policy can ever physically command.
Isolated, single change on top of the reachability-fixed baseline.

**Result: no improvement, and the mechanism itself didn't move.** 84.5%/83.3% vs.
85.6%/84.6% before — flat to very slightly down (likely noise). More importantly: even
with the reward penalty relaxed 5x in that zone, elbow_pitch was *still* near its lower
limit in 57.8%/56.0% of failures at closest approach — essentially unchanged from before
the fix (56.8%/51.5%). If the penalty had actually been fighting the policy out of a pose
it needed, relaxing it should have changed *behavior* even before it moved success rate.
It didn't move at all — so the correlation between "elbow near full extension" and
failure isn't caused by the reward penalty. More likely a genuine control-precision
characteristic of that pose (small angle errors near full extension may translate to
larger position errors, or it needs finer control than other configurations).

**Follow-up checked: is this actually a rare edge case, safe to just exclude from
training?** No — checked before assuming. Across all episodes, elbow_pitch sits within
10° of the extension limit in 22.6% of them, within 20° in 35.1%, and the 10th percentile
is already at the boundary. This is a substantial, regularly-occurring part of normal
usage, not a corner case — excluding it from the goal box would be trading away real,
mostly-working capability (only ~15-17% of episodes landing there fail) to inflate the
success-rate number, not fixing anything. Decided against.

**Next: 1b, isolated diagnostic — does focused training on the stress region actually
improve precision there, before building a full curriculum (1a)?** Added
`goal_bounds_x_override` (`g1_arm_env.py`) so a cfg variant can restrict just the x-range
of `_GOAL_BOUNDS` without touching y/z or the module-level default. New opt-in variant
`G1ArmIKLeftStressRegionEnvCfg` / task `G1-Arm-IK-Left-StressRegion-v0`: trains
exclusively with `x` restricted to (0.35, 0.42) — the outer ~30% of the box, tracing the
whole "far face" (all its corners and everything between them, not one isolated point) —
y/z untouched, since x (forward reach) was already established as the dominant lever for
reach difficulty. Also added `--goal_x_range` to `eval_arm.py` so *any* checkpoint
(baseline or stress-region-trained) can be evaluated against exactly the same restricted
region, for a fair before/after comparison.

**1b result: no improvement (78.2%/76.4% vs. 78.4%/76.3% baseline scored on the same
region) — but this uncovered a real bug in how the region was defined, not a genuine
negative result.** Added goal-position logging (`goal_x_m/y_m/z_m` columns,
`ArmMetricsCsvWrapper`) to check directly, and found: in ~3000 combined episodes
restricted to `x∈[0.35,0.42]`, **zero** had elbow_pitch anywhere near its limit. The
x-only restriction was based on misremembering the *original* reachability octant
finding (which needed x AND y AND z all high *simultaneously*) as "x alone is the
dominant lever" — leaving y/z free meant most sampled points never actually needed near-
full extension at all. 1b never touched the real hard condition, so its negative result
doesn't mean "more training doesn't help" — it means the experiment tested the wrong
thing.

**Re-diagnosed properly with goal-position data — the real finding is bigger than a
wrong region.** Cross-referencing goal position against the near-limit condition across
the *whole* box (not just the mis-defined slice): near-limit and non-near-limit episodes
have statistically identical goal positions (same mean, same full range, every axis).
Bucketing by forward distance and by distance-from-near-corner, the near-limit fraction
stays flat (19-25%) with no trend toward the extreme edges — if anything the farthest
bucket has the *lowest* rate. **This isn't a goal-difficulty problem at all — it's
manipulator redundancy.** For any given goal, the policy sometimes lands on one solution
branch (elbow bent ~42°, ~91% success) and sometimes on a different one for the *same*
goal (elbow pinned near -0.1°, its lower limit, ~59% success) — a free, roughly goal-
independent ~20%-of-episodes choice. This retroactively explains why both earlier fixes
failed: the joint-limit margin fix didn't help because no goal *requires* the bad
branch (relaxing the penalty didn't change anything to relax toward); 1b's misdefined
region didn't help because there's no region where the bad branch concentrates.

**Fix: null-space penalty is now per-joint-weighted, not just present.** The existing
`null_space_penalty_scale` (0.05, unweighted L2 across all 5 joints) doesn't actually
discriminate between the branches — total deviation-from-default is *similar* for both
(~79° bad vs ~82° good), because `shoulder_pitch` is far from default in both branches,
diluting the one joint that actually differs (`elbow_pitch`: ~50° off in the bad branch,
~8° off in the good one). Weighting `elbow_pitch`'s contribution 3x in the null-space
distance (`arm["null_space_weight"]`, `g1_arm_env.py` `__init__`/`_get_rewards`)
separates the branches clearly (~162° bad vs ~85° good) — a targeted signal, not a blind
scale increase (which would pressure all 5 joints toward default without specifically
discouraging the one branch that's the problem, and cost general reaching flexibility
for no reason). Isolated, single change on top of the reachability-fixed baseline.

**Result: no effect, mechanism didn't move either.** 84.1%/85.2% success — noise around
the last two checkpoints (84.5-85.6%), not a real change. Checked the actual mechanism
directly, not just the aggregate: bad-branch fraction 21.1%/21.8% (was 21.8%/23.4% before
this fix — unchanged), success on the bad branch 59.3%/61.2% (was 58.9%/60.1% —
unchanged), success on the good branch 90.8%/91.9% (was 91.6%/90.4% — unchanged). The 3x
elbow_pitch weighting didn't shift which branch the policy lands on at all, despite being
specifically designed to. Joint-range utilization shows no overcorrection sign either
(shoulder ranges didn't creep up, elbow's didn't shrink abnormally) — the fix simply had
no effect in either direction.

**This is the third targeted attempt at the redundancy/branch issue to cleanly fail**
(joint-limit margin, then weighted null-space). Leading theory now: this may be a
structural mismatch between PPO's single (unimodal) Gaussian policy and a genuinely
bimodal solution space — a reward-shaping nudge can bias which solution *looks* better on
paper, but may not be enough leverage to actually collapse the policy's tendency to
stochastically land on either mode. A real fix would likely need a more expressive action
distribution (e.g. a mixture policy) — a bigger architectural change, out of scope for
now. Not chasing this further today; the ~85% success rate itself is still a large,
confirmed improvement over the original ~55%, and the branch issue accounts for a bounded
(~15-20% of episodes), well-characterized, but not-yet-solved remainder.

**Queued (2026-07-08): overnight sweep v2, once the weighted-null-space result is
in.** 3 isolated experiments — entropy (`G1-Arm-IK-Left-Entropy-v0`), wide network
(`G1-Arm-IK-Left-WideNet-v0`), and a new PPO hyperparameter tuning variant
(`G1-Arm-IK-Left-PPOTuning-v0`, `num_learning_epochs` 5→8, lowest confidence of the
three — no specific prior evidence points at it, unlike the other two). All three pair
with the *current* `G1ArmIKLeftEnvCfg` (every fix above already baked in), not the old
~55% baseline, via `overnight_train.sh` (rewritten for this sweep — same file gets
reused/edited per sweep rather than accumulating separate scripts). 1500 iterations
each, sequential, eval after each. Script shuts the machine down once the queue
finishes — see the script's own comments for the passwordless-shutdown-permission
caveat to verify before relying on it unattended.

**Overnight sweep v2 result: none of the three fixes the branch-selection issue
(bad-branch frequency stayed 20-21% across all three, same as baseline), but wide-net
reproduces its earlier-shown strength.** vs. `null_space_weighted` baseline
(84.1%/85.2%, p90 3.70/3.61):

| Run | Success % | p90 dist (cm) | Success on bad branch |
|---|---|---|---|
| baseline | 84.1 / 85.2 | 3.70 / 3.61 | 59.3% |
| entropy_v2 | 86.3 / 85.1 | 3.33 / 3.43 | 64.0% |
| wide_net_v2 | 85.3 / 84.5 | **3.02 / 3.19** | 63.5% |
| ppo_tuning | 84.2 / 85.4 | 3.20 / 3.18 | 60.7% |

Interesting secondary effect: entropy and wide-net both modestly improve *precision
within* the bad branch (~59%→~64% success on those episodes) without changing how often
the policy lands there — a different, complementary improvement to mode-selection, not a
fix for it. wide-net gives the best p90/tail by a clear margin, and this is the *second*
time it's shown this exact strength (first on the old ~55% baseline, now on the current
~85% one) — reproducible, not a one-off. **Decision: fold wide-net alone into the final
consolidated arm policy** (entropy was mixed, PPO-tuning showed the least effect —
neither adopted).

**Queued (2026-07-09): final consolidation pass.** `overnight_train.sh` rewritten again
(same file, reused per sweep as established) for 2 sequential steps: (1) arm —
`G1-Arm-IK-Left-WideNet-v0` at the full 5000-iteration budget (already bundles
everything else confirmed good — reachability-fixed box, weighted null-space,
asymmetric elbow_pitch margin — since those live in the shared `G1ArmIKLeftEnvCfg`, not
a separate toggle); (2) standing — resumes the phase-5-curriculum run manually stopped
at iteration ~5650 back up to its original 8000-iteration target, into the same run
folder. Estimated ~2.5 hours total. Not yet run.

**Noted, not urgent — arm-vs-arm collision for `arm="both"` training.** Self-collision
(`enabled_self_collisions=True`) is a whole-robot property, so it already physically
prevents literal left-arm/right-arm interpenetration, same as it does for arm-vs-torso.
What's missing: torso got *both* the structural fix (self-collision) *and* a soft reward
penalty discouraging the policy from approaching it in the first place
(`torso_proximity_penalty_scale`) — there's no equivalent arm-to-arm proximity penalty
yet. So today, two simultaneous goals pulling the arms close together would rely entirely
on physical contact response (abrupt, reactive) rather than a learned, smooth avoidance.
Check/add before ever training `G1-Arm-IK-Both-v0` for real — not needed for single-arm
work.

**Status of the three items below, as of 2026-07-08 (superseding the earlier version of
this list, which is now stale):**

1. **Null-space regularization — confirmed real, kept, and now sharpened.** Originally a
   uniform (unweighted) penalty — confirmed real but modest (mean/p90 distance improved
   ~8-13%, success rate unchanged). Root-caused *why* it wasn't enough (see the
   joint-config correlation entry below — it's a redundancy/branch-selection problem, and
   the unweighted version doesn't discriminate between branches) and reworked it to be
   per-joint-weighted (`elbow_pitch` 3x). Training now — see the bottom of this section.
2. **Joint-config correlation check — done, and it found something bigger than
   expected.** Not "does failure correlate with a joint near its limit" in the abstract —
   concretely: ~20-23% of episodes land on a less-reliable of two valid solution branches
   for the *same* goal (elbow pinned near its lower limit, ~59% success, vs. elbow bent
   ~42°, ~91% success), independent of goal position. Full story is the last several
   entries in this section. This is *the* current lead, not a deferred item anymore.
3. **PPO hyperparameter tuning — queued, lowest confidence of the three, not yet run.**
   `num_learning_epochs` 5→8 is the one concrete variant defined so far (see the overnight
   sweep v2 entry below); learning rate schedule/`num_mini_batches`/`clip_param` remain
   untouched. Still genuinely lower-priority than the other two — no specific evidence
   points at optimizer settings the way the redundancy finding points at null-space.

**1a (curriculum oversampling a "hard" geometric region) — superseded, not just
deferred.** The original plan was: if 1b (isolated training on a hard region) helps,
build a proper curriculum (1a) that biases sampling toward it during normal training. 1b
turned out to be testing the wrong thing (see below — the region was misdefined, and once
corrected, it turned out there's no hard geometric region at all, just a goal-independent
redundancy issue). A curriculum over goal *position* can't fix a problem that doesn't
depend on goal position — 1a doesn't make sense as stated anymore. If the weighted
null-space fix doesn't fully resolve the redundancy issue, the right next lever is
something that acts on the redundancy directly (e.g. a stronger/differently-shaped
null-space term), not a goal-sampling curriculum.

**Option 2 (base repositioning for out-of-envelope targets) — real idea, scoped for
integration, not today's work.** User proposal: rather than trying to make the arm alone
handle 100% of the theoretical workspace, treat a smaller region as the arm's reliable
envelope and have the *robot* (via walking) reposition itself when a requested target
falls outside it, then reach once close enough. Assessed as more tractable than it first
sounds: it only needs *sequential* walk-then-reach (check reachability against the
trained envelope, compute a base offset, walk there, switch to standing+arm mode — all of
which either already exists or is straightforward geometry), not simultaneous
arm+walking coordination (genuinely hard, unstarted, real Phase 3 territory). Not
avoidance/overfitting in the way blindly shrinking the goal box would have been — it
relocates capability to a subsystem well-suited to it (walking's whole job is
repositioning the body) rather than deleting it. Explicitly deferred to the integration
work (tomorrow+), not touched today.

**Verification to do once the weighted null-space training/eval comes back**: user's own
good methodological question — does weighting `elbow_pitch` more heavily in the
null-space penalty risk over-correcting into *avoiding* elbow bend entirely, pushing
everything onto shoulder rotation instead? Reasoning why this is unlikely (the reference
pose is `default_joint_pos`'s ~49.8° elbow bend, a moderate bend, not 0° — the term
discourages deviating from that, not bending in general — and the overall scale is
unchanged, still deliberately weak relative to `position_reward_scale`), but reasoning
isn't verification. Check directly: joint-range-utilization table (did shoulder ranges
creep up while elbow's shrank abnormally?) and the near-limit-branch population size (did
it actually shrink, confirming the mechanism, without a new failure mode appearing
elsewhere?).

## Open

### Standing: corrective stepping is unreliable

Rare (2-8% of episodes across difficulty phases) and, when it does happen, still falls a
large fraction of the time (28-83%) — see `phase_logs/phase_1.md` for the full
investigation. Root cause: not enough repetitions during training for the skill to be
practiced reliably (ruled out "just needs more iterations" — the back half of a 6000-
iteration run sat at the hardest difficulty with zero further improvement).

Confirmed still open as of 2026-07-07: re-ran `validation/eval_standing.py` with real
observation noise enabled (see below) — 0% fall rate held across all phases, which is
good news for basic balance, but stepping was *never* triggered even once (0% at every
phase), so this eval still can't exercise the stepping behavior at all. The reliability
question remains genuinely untested by this tool, not resolved by the clean fall rate.

Two remediation ideas on the table:
- Bring back a "push"/shock disturbance (removed in Phase 0) — a sudden shock is more
  likely to force stepping than an arm-swing curriculum alone. **Not implemented.**
- Add a 5th arm-disturbance curriculum phase at **33 rad/s** (agreed number, safely under
  the ~37 rad/s real G1 hardware limit), with the amplitude clamp scaled up alongside it
  so the motion doesn't just clip against the old limit. **Code change made 2026-07-07**
  (`StandingArmTrajectoryDisturbance` in `mdp/events.py`, purely additive — phases 0-4
  untouched): `_PHASE_STEP_BOUNDARIES` extended to `(12000, 30000, 50000, 80000, 120000)`,
  new phase 5 at `_MAX_DELTA_RAD=0.66` (33 rad/s @ 50Hz), `_MAX_AMPLITUDE_RAD=1.10`,
  `_ASYMMETRY_PROB=0.85`, `_REVERSAL_PROB=0.22` (amplitude/asymmetry/reversal extrapolated
  from the existing phase-to-phase increment, not verified against real joint limits or
  visually confirmed via `play.py` — do that before a real training run, not just before
  trusting the result). **Not yet trained** — this needs an actual training run before
  it can be considered a fix rather than a hypothesis.

### Standing eval: `base_external_force_torque` stays disabled by design

`validation/eval_standing.py` restores observation noise (IMU/encoder `Unoise`, see
below) but deliberately leaves the base random force/torque event disabled — reintroducing
it would mix two disturbance sources (arm-motion curriculum + random push) into the same
sweep, making it impossible to tell which one caused a given fall/step. If a "does it
survive noise *and* an actual push, simultaneously" check is ever wanted, that needs a
separate eval mode, not folding into the existing phase sweep.

### Walking: backward locomotion is comparatively undertrained

Training's `lin_vel_x` command range is `-0.5 to 1.0` — backward gets roughly a third of
the exposure forward does. Confirmed in `validation/eval_walking.py`'s `backward` bucket,
which has the worst heading drift of the four straight-line buckets, consistently across
two eval runs (12.4° without observation noise, 13.1° with it — see below). Low priority —
backward walking isn't currently a target behavior — but if that changes, widen the
backward end of `lin_vel_x` and retrain.

### Walking: rare fall at the top of the trained speed range, under noise only

With observation noise enabled in the eval (see below), `forward_fast` (0.9 m/s, near the
top of the `-0.5 to 1.0` trained range) showed a 0.19% fall rate (1/513 episodes) — 0%
without noise. Vanishingly rare and not a blocker, but a real, new data point: the fastest
trained speed isn't perfectly robust to realistic sensor noise. Worth re-checking if this
grows on a future retrain rather than assuming it's just sampling noise.

### Sim2real toolbox — available in Isaac Lab, not used anywhere in this repo yet

Already in use: `Unoise` on IMU/encoders (walking/standing via stock `G1FlatEnvCfg`; the
arm task too as of 2026-07-07, previously had none at all), startup randomization of
friction/mass/CoM (walking/standing), randomized reset pose/velocity — and, as of
2026-07-07, `validation/eval_*.py` restoring that same observation noise for eval too
(previously the `_PLAY` configs silently disabled it, making the eval easier than real
deployment). Also as of 2026-07-07: `randomize_actuator_gains` and
`randomize_joint_parameters` on the arm task's arm joints (±20% gain jitter, friction/
armature randomization) — see `training_regimes.md`'s Phase 2 section.

Not yet used anywhere, in rough order of likely value:
- `randomize_actuator_gains` / `randomize_joint_parameters` for **walking/standing** (only
  the arm task has these so far).
- Actuator network models (`isaaclab/actuators/actuator_net.py`) instead of the ideal PD
  actuators used everywhere now — models real motor lag/backlash.
- Action/observation latency (`isaaclab.utils.buffers.DelayBuffer` or the modifiers
  system) — nothing currently models real control-loop communication delay.
- `base_lin_vel` is in the observation for walking/standing as ground truth, but a real
  G1 doesn't have that directly (it's estimated — IMU + kinematics, often a learned state
  estimator — and the estimate is noisier/biased vs. sim). Options not yet decided:
  heavier noise/bias specifically on this term, an asymmetric actor/critic split
  (privileged `base_lin_vel` for the critic only), or a learned estimator.

### Self-collision enabled project-wide, not yet verified (Phase 2, 2026-07-07)

Root cause of the reported "arm entering torso" bug: every G1 asset variant in
`isaaclab_assets/robots/unitree.py` ships with `enabled_self_collisions=False`, so links
on the *same* robot never generate contact forces against each other, project-wide (not
arm-specific). Fixed by overriding it to `True` in this project's own env cfgs for all
three tasks (`g1_arm_env.py`, `g1_locomotion_env_cfg.py`'s three walking/standing
variants) — the same pattern already used for `fix_root_link`/actuator overrides, no
change to the shared Isaac Lab package.

**Real, unverified risk**: unlike deformable bodies (which have a
`self_collision_filter_distance` safeguard for rest-pose overlap), rigid articulations —
what G1 actually uses — have no such safeguard. If any collision meshes overlap at the
default/rest pose (common at simplified-mesh joints), this could cause unstable contact
forces the moment simulation starts. Cost is also unverified — self-collision meaningfully
increases collision-pair checks, particularly at `num_envs=4096`. **Verify via `play.py`
on a small `num_envs` (watch for jitter/instability, rough step-time comparison) before
any real training run for any of the three tasks** — this has not been done yet (no
Isaac Sim access in the session that made this change).

A cheap, safe complement ships regardless of that check's outcome: `g1_arm_env.py` has a
soft distance penalty (`torso_proximity_margin_m=0.12`, `torso_proximity_penalty_scale=2.0`)
discouraging the end-effector from getting within 12 cm of `torso_link`. It only covers
the end-effector, not the forearm/elbow — a real gap if the forearm itself clips the torso
even when the palm is clear.

### Arm root wobble — real-motion version tried, failed, reverted to synthetic-only (Phase 2)

Goal throughout: the arm policy previously had zero exposure to a moving/tilting base,
the single biggest sim2real gap for using it under standing's arm-disturbance curriculum
or while walking. Three iterations to get here, in order, since each one only surfaced
its problem once actually run:

1. **First version**: `fix_root_link=False` (a genuinely free root), kinematically
   driven every step with a scripted roll/pitch wobble (`_apply_root_wobble`), on a
   curriculum (`root_wobble_enable_step`, default 30,000 env-steps).
2. **Bug found via `scripts/zero_agent.py --task G1-Arm-IK-Left-v0 --num_envs 16`**:
   `default_root_state` is in each environment's *local* frame (same `[0,0,~0.75]` for
   every clone, doesn't include `env_origins`) — the pose was written directly to sim
   without adding that offset back in, so every cloned env snapped to the same world
   position (all 16 robots stacked on top of each other). Fixed by adding
   `env.scene.env_origins` once, at the point the reference pose was captured.
3. **Bigger problem found via `scripts/random_agent.py --task G1-Arm-IK-Left-v0
   --num_envs 16`** (after fix #2, wobble curriculum not even active yet since a fresh
   run starts at env-step 0): all 16 robots fell face-first forward. Root cause: with a
   genuinely free root and no active balance controller (legs are just PD-held to a
   fixed pose — no hip/ankle/stepping strategy like the real standing policy has), the
   robot was only standing on passive leg stiffness + feet touching the ground. `zero_agent`
   (no actions) barely disturbed the center of mass, so it held — that "slight coupled leg
   movement" noted after step 2 was actually this margin being thin, not harmless settling.
   `random_agent`'s real, continuous random arm motion (unlike zero actions) was enough to
   exceed that thin passive margin and tip it over. This was a real design flaw (giving the
   arm task a free base implicitly turns it into an unintended balance task), not a tuning
   problem — reverted rather than patched.
4. **Current design**: `fix_root_link=True` again (root physically fixed, exactly the
   original, already-proven-safe setup). `_apply_root_wobble` still runs on the same
   curriculum, but now only computes a *synthetic* signal fed into what
   `base_ang_vel`/`projected_gravity` report — nothing physically moves, so there's no
   fall risk. Trade-off: doesn't capture how a real tilting base would also affect the
   arm's own gravity-compensation dynamics; judged acceptable given how small these tilts
   are (a few degrees).

**Not yet re-verified after this redesign** — no Isaac Sim access in the session that made
this change. Since nothing physically moves anymore, the main things worth checking are
functional rather than physics-stability: does `common_step_counter` genuinely gate the
observation values as expected (compare an episode before vs. after `root_wobble_enable_step`,
check `base_ang_vel`/`projected_gravity` actually change), and does the arm still reach
goals normally (nothing here should affect that, but worth a glance).

**Self-collision (`enabled_self_collisions=True`) confirmed 2026-07-07** via
`scripts/random_agent.py --task G1-Arm-IK-Left-v0 --num_envs 16` on the reverted,
fixed-root code: arm swings near the torso, no interpenetration, no explosions.

### Arm joint ranges were hardware-wide, not task-appropriate (found and fixed, 2026-07-07)

Same `random_agent.py` check that confirmed self-collision also surfaced this: the arm
would occasionally end up in poses that are hardware-safe but useless for reaching —
shoulder rotated so the arm pointed behind the body, forearm (`elbow_roll`) rotated most
of the way around. Confirmed via a real Unitree G1 URDF found locally
(`g1_29dof.urdf`, same arm hardware) that these are genuine hardware ranges, not a bug:
shoulder_yaw has ~300° of travel, elbow_roll/wrist_roll ~226°. This task's goal workspace
(`_GOAL_BOUNDS`) never needs anywhere near that much rotation, so nothing was keeping
reset randomization or action targets within a sensible subset of it.

**First fix attempt (reverted) — broke reachability, confirmed by an actual training
run.** Added `_ARM_JOINT_TASK_RANGES_RAD` in `g1_arm_env.py` — a tighter, per-joint range
for each arm joint, hand-reasoned from the real hardware limits + the goal workspace
geometry, applied to reset randomization, the action-target clamp, and the joint-limit
reward. Never verified against actual forward kinematics (no Isaac Sim access to check
reachability directly). A 1500-iteration test run confirmed the risk was real: plateaued
at 0% success, median 17.7cm from goal (2cm threshold), and — the actual tell — a wide,
bimodal-looking distance distribution (p0=1.3cm, p50=17.7cm, p90=29cm, unmoving across
the whole run). Some goals solved well, most stuck at what looks like a geometric floor:
the signature of unreachable goals, not slow learning.

**Fix, 2026-07-07**: reverted the action-target clamp and the joint-limit reward back to
the real hardware range (`self._arm_hw_limits`) — restores exactly the reachability this
task had before any of this, and reaching converged fine under it previously. Only reset
randomization keeps a narrower spread (`cfg.reset_range_fraction`, 0.3 → 0.15, same
center-of-hardware-range formula as before, just a smaller fraction) — safe regardless of
the exact value since it only affects where an episode *starts*, never what the policy
can reach via its own actions. This does mean a trained policy could still choose to
visit hardware-safe-but-unnatural poses (nothing hard-blocks it) — the original "arm
behind the body" observation came from `random_agent.py`'s literal uniform-random
actions, not from any trained, reward-seeking behavior, so a real trained policy has no
incentive to wander there; worth confirming this reasoning holds once a full training run
is watched via `play.py`, not just assumed.

**Retraining needed**: the reverted 1500-iteration run's checkpoint was trained under the
broken (too-tight) ranges and should not be used or resumed — start a fresh run.

**Retrained 2026-07-07, confirmed fixed**: fresh 1500-iteration run under the reverted
code shows `min_dist_to_goal_cm` converging to a mean of 5.4cm and a *median* of 2.3cm by
the end of training (right at the 2cm goal threshold), vs. the broken run's flat ~17.5cm
the whole time. Reachability is clearly restored.

### Arm `success`/`outcome` columns always wrong — separate, older bug (found and fixed, 2026-07-07)

Found while sanity-checking the retrained run above: `outcome` was `"timeout"` for **100%
of ~710k episodes**, despite ~46% of them recording `min_dist_to_goal_cm` under the 2cm
success threshold at some point. Root cause in `ArmMetricsCsvWrapper._flush_finished_episodes`
(`g1_locomotion/utils/metrics_wrappers.py`): it read `env.successes` *after*
`self.env.step()` returned — but `DirectRLEnv` auto-resets any env that just terminated
*inside* `step()` itself, before returning, and `_reset_idx` explicitly zeroes
`self.successes` as part of that reset. So by the time the wrapper read it, a just-
succeeded episode's flag had already been wiped back to `False`. Predates Phase 2 entirely
(this code was untouched since Phase 0) — nobody had looked at the `success`/`outcome`
columns closely before, since `min_dist_to_goal_cm` trending down was the metric actually
being watched.

Fixed by using `terminated[env_id]` (the value already returned by this step, captured
before any reset happened) instead of re-reading the live, already-reset attribute — same
value semantically, just read at the right time. Any existing `arm_summary.csv`/
`arm_detailed.csv` from before this fix has unusable `success`/`outcome` columns; use
`min_dist_to_goal_cm < goal_threshold` as a proxy for old data if needed, or just re-run.

### Arm success rate plateaued at ~54% after 1500 iterations — two changes tried, not yet retrained (2026-07-07)

`validation/eval_arm.py` against the retrained (reachability-fixed) 1500-iteration
checkpoint: 54.1% success (no_wobble), 55.0% (with_wobble), median distance 1.98cm (right
at the 2cm threshold) but p90 ~12cm — a real tail of harder goals, not just "hasn't
converged everywhere yet." Joint-range utilization was healthy (43-88% across all 5
joints, nothing near 0%) — ruled out "degenerate small-movement policy" as the cause.
Training-curve bucketing (10 bins across the run) showed `mean_dist_cm` roughly flat from
~10% of the way through training onward (4.85cm → 5.40cm across bins 1-9) — suggests the
plateau isn't purely an iteration-count problem.

Two changes tried, both reverted after confirmation below:
- **Reward shaping** (`g1_arm_env.py`): added an exponential proximity bonus
  (`position_reward_exp_scale=5.0`, `position_reward_exp_sigma=0.05`) on top of the
  existing linear distance penalty. The linear term alone has the same gradient
  everywhere, giving no extra pull to close the last few cm beyond the one-time
  `goal_reached_bonus` jump — same idea as walking's `track_lin_vel_xy_exp`.
- **Entropy coefficient** (`agents/rsl_rl_ppo_cfg.py`): `0.0 → 0.005`, to keep exploration
  alive longer in case premature convergence (zero entropy bonus) contributed to the
  early plateau.

**Retrained 2026-07-07 — confirmed harmful by both training CSV and `eval_arm.py`.**
Training CSV: `mean_dist_cm` roughly unchanged (~4.9-5.1cm throughout), but `success`
(now trustworthy — see the bug fix above) peaked early at 35.5% (bin 1) then steadily
declined to 10.9% by the last 5000 episodes — a within-run regression, not slow
convergence. `eval_arm.py` against the resulting checkpoint confirmed it wasn't just
noisy training-time measurement: **11.5%/13.8% success** (no_wobble/with_wobble) vs. the
pre-change baseline's **54.1%/55.0%** — a real, large regression, not a wash. Leading
hypothesis: the exponential bonus created a "good enough" plateau just short of the 2cm
threshold — at 3-5cm it already pays out a meaningful chunk of reward
(`exp(-0.04/0.05)×5 ≈ 2.2`), reducing the marginal incentive to push the last bit for the
discrete `+50` bonus. The higher entropy likely compounded this (more residual action
noise late in training hurts precise threshold-crossing specifically). **Both reverted**
to the confirmed-good baseline (`position_reward_exp_scale=0.0`, `entropy_coef=0.0`) —
since they were bundled into one retrain, there's no way to isolate which one actually
caused the regression, or whether both did. If revisited, change one at a time with an
`eval_arm.py` comparison in between, not bundled.

### Interactive arm-testing scripts never reset the arm between targets (found and fixed, 2026-07-07)

Reported via `g1_arm_reach_test.py` and `g1_full_demo.py`: after one out-of-range target
made the arm "act weird," the *next* target — a reasonable, in-range one — also produced
bizarre motion (shoulder rotated ~180°, elbow ~90°, arm ending up parallel to the ground
pointing backward past the torso). Root cause: neither script ever resets the arm's
physical joint state between target changes — `g1_arm_reach_test.py` calls `env.reset()`
exactly once at the very start; `g1_full_demo.py`'s `_set_arm_target` never resets at
all. So the arm just keeps going from wherever it physically ended up chasing the
*previous* target. If that target was out of range, the arm can end up in an extreme,
never-seen-in-training pose — and the policy then has to try to reach the *new* target
starting from that bad, out-of-distribution state, producing motion that looks broken but
has nothing to do with whether the new target itself is reasonable. This likely also
explains at least part of the earlier "same motion regardless of target" full-demo
report, if an early target in that session happened to be out of range too.

Fixed in both scripts: the arm's joints are now explicitly reset to their default pose
(`write_joint_state_to_sim`, arm joints only) every time a new target is set, so every
attempt starts clean regardless of how the previous one went. In `g1_full_demo.py`, only
the *targeted* arm resets (relevant for `arm_mode="both"` — doesn't interrupt the other
arm mid-reach).

Also added, since this exact confusion showed there was no way to tell a good target from
a bad one at the prompt: `g1_arm_reach_test.py` now prints the reachable x/y/z range at
every prompt (previously only `g1_full_demo.py` did), and both scripts now detect an
out-of-range target and ask for confirmation (`... may behave strangely. Send anyway?
[y/N]`) before sending it, instead of silently accepting it.

**Not yet re-verified** — no Isaac Sim access in the session that made this fix.

### Arm finger joints were silently unactuated (found and fixed, 2026-07-07)

While implementing explicit PD-hold verification for the arm task's non-arm joints
(Phase 2), found that finger joints had **no actuator group at all** — the existing
`self.robot.actuators["arms"] = ImplicitActuatorCfg(...)` override replaced that dict key
wholesale, and stock `G1_MINIMAL_CFG` bundles every finger joint into the same "arms"
group alongside shoulder/elbow. Overriding "arms" to list only shoulder/elbow silently
dropped finger actuation entirely — they'd have gone fully limp under gravity for as long
as that override existed (predates Phase 2). Fixed by adding a dedicated "fingers"
actuator group with a modest PD hold. Not otherwise consequential (fingers aren't
controlled or observed in this task) beyond looking visually wrong in any demo/render.

### Walking's PLAY config silently never got project customizations, including self-collision (found and fixed, 2026-07-07)

Reported via `g1_full_demo.py`: arm visibly colliding with the hips during walking,
despite `enabled_self_collisions=True` supposedly being enabled project-wide since
earlier in Phase 2. Root cause: `G1LocomotionFlatEnvCfg_PLAY` inherited directly from
Isaac Lab's stock `G1FlatEnvCfg_PLAY`, not from this project's own customized
`G1LocomotionFlatEnvCfg` — so *every* project-level walking customization (command
ranges, and critically the self-collision fix) silently never applied to this PLAY
variant, only to real training runs. `g1_full_demo.py` and any other PLAY-based
interactive testing were therefore always running under stock Isaac Lab settings.
`G1LocomotionFlatTransitionEnvCfg_PLAY` and `G1LocomotionStandingFlatEnvCfg_PLAY` were
already correctly inheriting from their customized parent classes — only the base
walking PLAY config had this bug. Fixed by inheriting from `G1LocomotionFlatEnvCfg`
instead and re-applying the same PLAY-specific tweaks (fewer envs, no observation
corruption, no random pushes) the stock class made, matching the pattern the other two
PLAY configs already used correctly.

**Re-verified 2026-07-07 — collision still visibly happens, but that's expected, not a
failed fix.** The config fix is real (self-collision detection is genuinely active now),
but it can't retroactively make an *already-trained* policy avoid a contact it never had
to worry about — `chosen_checkpoints`-era walking was trained entirely before self-
collision existed, so its natural gait's arm-swing trajectory was learned with zero
incentive to avoid the hip. Self-collision being on now means that overlap produces real
contact forces (a visible bump/deflection) instead of silently clipping through — this
can easily *look* like "still colliding" even though the fix is doing its job correctly
one layer down. Actually eliminating the visible collision needs a **walking retrain**
with self-collision active from the start, so the policy can learn a gait that avoids it
— not something to expect from a config-only fix. Bigger task, not done yet.

### Arm-to-walking transition causes a visible vibration/jerk (reported, not investigated)

Reported via `g1_full_demo.py`: when switching from standing (arm actively reaching) to
walking, the arm falling back to a neutral pose causes a noticeable vibration in the
system. Not urgent, not yet investigated — flagged here so it doesn't get lost. Likely
genuinely Phase 3 territory (arm+walking integration) rather than something to chase
down in isolation right now; worth a smoother arm-retraction profile during the
transition rather than an instant drop, whenever this gets picked up.

### Standing: leans/drifts and takes small stationary corrective steps (phase-5-curriculum checkpoint, 2026-07-09)

Noticed while testing `g1_full_demo.py` (see `phase_logs/phase_2.md`'s integration-
testing section) after swapping `chosen_checkpoints/standing_latest.pt` to the
phase-5-curriculum run: the robot leans to one side during normal standing and takes
small "stationary" corrective steps (feet lifting/placing with no real disturbance).
Reverting to the pre-phase-5 checkpoint resolved both for now. The stepping behavior
specifically is consistent with — not necessarily a new bug on top of — the existing
"corrective stepping is unreliable" item below (phase-5 was trained specifically to try
to improve stepping, so seeing it actually manifest for the first time on the first
checkpoint trained under it isn't surprising). The lean is unconfirmed as
checkpoint-specific vs. a broader issue. **Being picked up in a separate, dedicated
debugging session**, not part of the integration-testing work above — noted here so it
isn't lost between sessions.

### Multi-arm mirror generalization — implemented, not yet exercised (2026-07-09)

`g1_full_demo.py` gained `Y`/`U` keys to drive the right arm — and both arms
simultaneously — by mirroring the left-trained policy (`mdp/symmetry.py`'s
`mirror_arm_obs`/`mirror_arm_actions`, the same transform `g1_arm_mirror_test.py`
already validates one-arm-at-a-time in isolation, and what the policy is actually
*trained* to satisfy via symmetry-augmented PPO). This is the first time the transform
has been exercised driving *both* arms at once against the real, standing robot rather
than one arm in isolation — known_issues.md's "arm-vs-arm collision for `arm=\"both\"`"
gap (below) is exactly the kind of thing this could surface. Not yet tested — next
session should actually run `U` and report whether it holds up, which directly informs
whether training a real `G1-Arm-IK-Both-v0` policy is worth prioritizing.

### Arm actuator gains: walking's gait may be affected by a stiffness change (2026-07-09)

`g1_full_demo.py` now overrides the shared walking/standing env's arm actuator gains to
match the arm-IK task's own training config (`stiffness=200, damping=20`, up from the
walking task's stock `stiffness=40, damping=10`) — see `phase_logs/phase_2.md`'s
integration-testing section, item 6. Necessary for the arm-IK policy to track its commanded targets with the precision it
was trained to expect, but the walking policy's own natural arm-swing gait was trained
under the *softer* stock gains — not yet independently re-verified that walking still
looks/behaves the same under the new, stiffer gains.

## Resolved

**2026-07-09 (integration testing) — five real bugs in `g1_full_demo.py`, found by
actually running the combined demo:** reset-anchor using a stale pre-spawn pose
(initial `--target` only), an `inference_mode` crash on the `T`-key prompt, a z-height
convention mismatch that placed every target far too high to reach, the untouched arm
running on the standing policy's meaningless raw output whenever only one arm had a
target, and a camera whose transform was re-locked every frame (making manual orbit
impossible). Full detail, root causes, and fixes in `phase_logs/phase_2.md`'s
integration-testing section. Also refreshed
`chosen_checkpoints/arm_left_latest.pt`/`standing_latest.pt` (both stale since before
this phase) and fixed six goal-box-bounds references across
`testing/*.py`/`testing/quickrun_tests.md` that still quoted the pre-reachability-fix
numbers (display-only, never affected actual validation logic).

*(the items below this line predate this phase — nothing yet, the above are code changes
made 2026-07-07, not yet trained/verified, so they stay under Open until an actual
training run or play.py check confirms them.)*
