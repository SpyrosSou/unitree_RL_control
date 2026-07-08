# Known Issues / Backlog

Running list of things discovered during actual work that were deliberately deferred
rather than fixed immediately — so they don't get lost or silently re-discovered later.
Not a duplicate of the roadmap plan; this is specifically things found *while doing* the
work, with enough context to act on them later. Update this file directly when an item
gets resolved (move it to the bottom under "Resolved", don't just delete it).

## Arm policy — current state, for discussion (2026-07-07)

Short version: **not yet practically usable.** Reachability and self-collision are fixed
and confirmed. The core reaching accuracy is not — it's stuck at ~55% success (2cm
threshold) with a real, unaddressed tail of harder goals. Full detail below; this section
is meant to be skimmable on its own.

**The core problem**: a 5-DOF single-arm reaching policy (`G1-Arm-IK-Left-v0`), trained
with PPO, plateaus at **~55% success rate**, median distance-to-goal 1.98cm (right at the
2cm threshold) but **p90 ~12cm** — a real subset of goals the policy just doesn't solve,
not noise around a good average. Confirmed via `validation/eval_arm.py` (deterministic,
fixed-seed) at two checkpoints: 1500 iterations and 4998 iterations (full budget).

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

## Resolved

*(nothing yet — the above are code changes made 2026-07-07, not yet trained/verified, so
they stay under Open until an actual training run or play.py check confirms them.)*
