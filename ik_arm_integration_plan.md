# IK arm integration plan — `ik_residuals` branch

**Audience: a fresh Claude chat (or human) starting work on this branch with zero
session context.** Read in this order: `README.md`'s "Overview / up to speed" section →
`policy_status.md` (full current status) → `retrospective.md` (what was tried the week
of 2026-07-21..27 and why) → this file (what to build next). Written 2026-07-27, at the
moment `ik_residuals` was branched off `main` (commit `760c696`) — the two branches are
identical at time of writing; `main` stays frozen as the pure-RL fallback.

## Why this branch exists (the one-paragraph version)

The walking+standing RL policy works (`chosen_checkpoints/walking_latest.pt` — 0% fall
rate, known-but-tolerable drift gap). The RL arm-reaching policy does not clear the bar:
the best 40/10-gain candidate (`best_combined`) evals at ~28-33%, and even the 200/20
"99.98%" reference turned out to have only **~30% true single-shot reach reliability**
(see `policy_status.md`'s "Critical finding" — the root mechanism of that gap was never
found). For precision grasping we need near-deterministic reaching. Decision: replace
the pure-RL arm reaching with **numerical IK** for precise x/y/z targets, keeping the
RL walking policy untouched. The branch is named `ik_residuals` because the intended
end state keeps RL in the arm loop: **IK provides the precise joint-space baseline, and
a (later-phase) RL residual policy on top provides compliance/disturbance
robustness/naturalness** — an RL arm component remains a real deliverable, it just
stops being responsible for raw metric accuracy.

## Source material

`unitreerobotics/xr_teleoperate` (official Unitree repo, public on GitHub) contains
`G1_29_ArmIK` — a Pinocchio + CasADi numerical IK class for exactly our 29-DOF G1
variant (3-DOF waist, 7-DOF arms with wrists). Key properties (from prior research this
session — **verify all of this against the actual code after cloning, don't trust this
paragraph blindly**):

- Lives around `teleop/robot_control/robot_arm_ik.py`; solves **both arms
  simultaneously** (14 DOF) for arbitrary SE(3) targets — it is NOT VR-hardware-locked,
  the solver itself just takes end-effector target transforms.
- Builds a *reduced* Pinocchio model with non-arm joints locked, optimizes a cost of
  position error + orientation error + regularization/smoothness, solved via CasADi.
- Returns the joint solution **and** a feedforward torque via inverse dynamics (useful
  later for the gravity-compensation item in `policy_status.md`'s deferred list).
- Dependencies: `pin` (Pinocchio ≥3.x), `casadi`, optionally `meshcat` for its own
  visualizer. **None of these are installed in the `isaac_g1_control` conda env yet.**
- The repo ships its own G1 URDF assets — compare against ours before choosing which to
  use (frame conventions must match whatever end-effector frame we measure success at).

**Local assets already on disk:**

- Full 29-DOF URDF: `~/Elm/Code/g1_simulation/ros2_ws/src/g1_navigation/description_files/urdf/g1_29dof.urdf`
  (verified 2026-07-27: all 7 arm joints per side present, plus fixed
  `left_hand_palm_joint`/`right_hand_palm_joint` frames — **correction**: this file
  originally claimed the palm was the RL env's end-effector; `g1_arm_env.py`'s actual
  `_LEFT_EE_BODY`/`_RIGHT_EE_BODY` constants are `left_wrist_yaw_link`/
  `right_wrist_yaw_link`, not the palm. Moot in practice, since hands/fingers are out of
  scope for this integration anyway and the IK's end-effector is the wrist link too —
  see landmine #8 below.).
- Sim asset (USD): `~/Elm/Assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd`
  (what Isaac Lab actually simulates — the IK never touches this, but joint naming must
  agree).
- Real-deploy reference: `deploy_reference/` (C++ FSM + ONNX pipeline, ground-truth
  joint order/gains in `robots/g1_29dof/config/`).

## Target architecture

```
walking/standing:  RL policy (walking_latest.pt)  → leg + waist joint targets   [UNCHANGED]
arm reaching:      G1_29_ArmIK (pinocchio/casadi) → arm joint position targets  [NEW]
                     └─ later phase: + RL residual policy on top of IK targets
```

Integration point in sim: the demo/eval already has exactly the right override
mechanism — the arm policy's targets replace the arm-joint columns of the locomotion
action (see `ArmDisturbanceBlendJointPositionAction` in the arm env, and how
`testing/visual_testing/full_demo/g1_full_demo.py` drives arm joints). The IK
controller slots into the same seam: instead of querying an RL actor for arm deltas,
query the IK solver for arm joint targets, rate-limit/interpolate them, and write them
through the same `set_joint_position_target` path. Legs/waist stay with the walking
policy.

## Known landmines (all learned the hard way — see `policy_status.md` "Lessons learned")

1. **Joint ordering.** PhysX orders joints breadth-first from the articulation root and
   interleaves left/right; Pinocchio orders by URDF traversal. NEVER map by list
   position — always build an explicit name→index map on both sides and translate
   through joint *names*. This exact bug cost a week in the 23dof phase.
2. **Waist ownership.** The walking policy owns the waist; the IK reduced model must
   lock the waist (at its *current* measured position, or at default — decide and
   document) rather than solving with it, or the two controllers will fight.
3. **Gains.** Arm PD gains in sim are 40/10 (matches real deployment,
   `deploy_reference`). IK gives positions, PD does the tracking — expect steady-state
   gravity sag at 40/10; the IK's feedforward torque output is the natural fix if sag
   exceeds tolerance (measure first).
4. **Target jumps.** A fresh IK solve on a new goal can be far from the current pose —
   never write a raw solution directly; interpolate/rate-limit (the arm env's existing
   `max_action_delta_per_step`/EMA machinery is the in-repo precedent for sane limits).
5. **Passive-reaction creep is real physics, not a bug**: arm motion induces ~20° torso
   lean and passive right-arm deflection when those joints are only PD-held (measured
   2026-07-27, `policy_status.md` "Ruled out" section). Expect it during standing-arm
   tests; it's a refinement item, not a blocker.
6. **Env for heavy runs**: this branch uses a dedicated conda env, `isaac_g1_ik`
   (cloned 2026-07-27 from `isaac_g1_control`, which remains untouched/`main`'s env).
   `casadi` 3.7.2 is installed there alongside the pre-existing `pinocchio` 2.7.0 —
   confirmed both import cleanly together with the env's numpy (1.26.4). Any further
   new pip installs for this branch's work go into `isaac_g1_ik`, not
   `isaac_g1_control`. User confirms Isaac still launches from the new env before it's
   trusted for Isaac-side phases.
7. **Our URDF's root joint is an active floating joint — the reduced IK model must lock
   it, or silently gain 6 bogus DOF.** `g1_29dof.urdf`'s `floating_base_joint` (type
   `floating`, world→pelvis) is NOT commented out, unlike the equivalent block in
   `xr_teleoperate`'s own URDF copy (which is commented out there). Building the
   Pinocchio model straight from our URDF the way `G1_29_ArmIK.__init__` does would
   silently add 6 pelvis-pose configuration variables (quaternion+xyz) to `var_q` that
   the reference implementation's `mixed_jointsToLockIDs`-based index bookkeeping never
   anticipates — a repeat of landmine #1's failure shape (silent index scramble) if
   ported blindly. Fix: lock/exclude the floating joint in our reduced model too (same
   mechanism as locking legs/waist) — this is purely an internal-bookkeeping fix for the
   standalone Pinocchio/CasADi model and has **no effect on Isaac Sim or the walking
   policy**: the actual pelvis pose in the world keeps being driven freely by the
   walking policy exactly as today. The IK never simulates or constrains pelvis
   pose — it only ever sees arm targets already expressed *relative to* the pelvis
   (computed dynamically, every control step, from the pelvis's live current pose in
   sim — not a one-time snapshot), so the pelvis's real-world position/orientation is
   irrelevant to the arm-joint solve either way. Single URDF is used (no separate copy
   needed) — the floating joint is just excluded/locked at the reduced-model-build step
   in our vendored code, not edited out of the URDF file itself. Flagging here in case
   the lock step is accidentally dropped in a future edit.
8. **End-effector frame decision (2026-07-27): wrist, not palm.** Since precision
   grasping targets only the wrist's x/y/z (fingers/hand articulation out of scope for
   this integration), the IK's end-effector frame is `left_wrist_yaw_link`/
   `right_wrist_yaw_link` directly — no palm-joint offset needed. This also happens to
   exactly match `g1_arm_env.py`'s own `_LEFT_EE_BODY`/`_RIGHT_EE_BODY` constants (which
   are the wrist-yaw links, not the palm, despite this doc's superseded "the palm is
   what the RL env used" claim above — corrected here) — so Phase 1/3's IK-vs-RL
   accuracy comparison is apples-to-apples with no frame-conversion needed.
9. **`solve_ik`'s smoothing filter is stateful and persists across unrelated calls —
   reset it on any discontinuous target change.** Found 2026-07-27 chasing a
   Phase-0-smoke-test result that looked like a solver accuracy bug: a target only
   ~3.5cm from neutral showed an 8.5cm error. The raw IPOPT solve was actually accurate
   to 0.2cm (`Solve_Succeeded`) — the reported error came entirely from
   `WeightedMovingFilter`, which keeps a 4-call rolling window of *raw* solutions inside
   the `G1ArmIK` instance and blends them (weights `[0.4, 0.3, 0.2, 0.1]`). That's
   correct, desirable behavior across a continuous stream of nearby targets (smooths
   jitter, matches upstream's real-time teleop use case) — but wrong across a
   discontinuous jump (new episode, teleop operator picking a far-away point): for up
   to 3 calls afterward it keeps blending in stale poses from the *previous*, unrelated
   target. Fix: `G1ArmIK.reset(q)` clears the filter's window and re-seeds the warm
   start — call it whenever the target changes discontinuously, before the first
   `solve_ik` call for the new target. Phase 2/3 must call this on every fresh RL-style
   goal reset, or early post-reset reach measurements will be silently contaminated by
   the previous episode's arm pose.
10. **Upstream's rotation-cost weight (1.0) actively fights position accuracy for a
    position-only use case — set it to 0.0.** Found 2026-07-27 during Phase 1's dense
    coverage sweep: an initial full sweep (grid_res=14, both arms) came in at only
    38.6% within 2cm — far below the ~97% kinematic ceiling `check_arm_reachability.py`
    already measured for this goal box, and below the 95% gate. Diagnosed with three
    controlled tests before concluding anything was "miswired" in the frame/URDF sense:
    (a) tightening IPOPT's tolerance/iteration cap by 100x changed nothing — rules out a
    loose-convergence artifact; (b) 20 random warm-start restarts on the same failing
    points found no better solution — rules out a local-optimum/redundancy trap; (c)
    sweeping the rotation-cost weight from 1.0 down to 0.0 on a representative
    mid-box point dropped its error from 3.25cm to 1.26cm (crossing the gate threshold),
    while a genuinely-out-of-reach corner point stayed ~12cm regardless of rotation
    weight. Conclusion: forcing the wrist toward the target transform's identity
    *rotation* (upstream's real teleop use case cares about this) was measurably
    competing with the *position* objective for many points — appropriate for upstream,
    wrong for this integration (position-only per the user's explicit scope decision,
    2026-07-27 — hands/fingers/wrist-orientation out of scope). Fixed in `arm_ik.py`:
    rotation weight set to 0.0 (rotational-error machinery kept, not deleted, for the
    future 6-DOF-orientation phase). Re-swept: coverage improved to 58.6%/58.5%
    (left/right) within 2cm — a real, substantial gain, but still short of the 95% gate;
    remaining failures concentrate cleanly in the far-reach (`x=high`) half of the box
    (5-49% coverage there vs. 78-100% in the near half) and did NOT respond to the same
    random-restart/regularization-weight tests that ruled out the first two hypotheses —
    see `policy_status.md` for the open decision on how to proceed from here.

## Phases (each has a "done" gate; don't start the next before the gate passes)

### Phase 0 — setup + standalone IK smoke test (no Isaac, no GPU)
- Clone `xr_teleoperate` (outside this repo, e.g. `~/Elm/Code/xr_teleoperate`), read
  `G1_29_ArmIK` for real, correct any wrong assumption in this file.
- `pip install pin casadi` (+ `meshcat` if wanted) into `isaac_g1_control`; confirm
  Isaac still launches afterward (user runs that check).
- Copy/adapt the IK class into this repo (e.g.
  `source/g1_locomotion/g1_locomotion/controllers/arm_ik.py`) — vendor it in rather
  than importing across repos; keep upstream attribution in the docstring.
- Pick the URDF (theirs vs ours) and pin the choice + reason in the module docstring.
- **Gate**: a pure-Python script solves IK for a handful of hand-picked palm targets
  from the RL goal box; solutions respect joint limits; solve time per call measured
  and printed (matters for control-loop rates later).

### Phase 1 — offline accuracy sweep (no Isaac, no GPU)
- Sample the RL goal box (`_GOAL_BOUNDS` in `g1_arm_env.py`) densely; for each target,
  solve IK and report FK-recomputed palm position error, solve success rate, joint-limit
  saturation, solve time. Reuse the sampling/reporting style of
  `validation/check_arm_reachability.py` (~97.5% of the box is reachable within 2cm —
  that's the ceiling any solver can hit).
- **Gate**: IK reaches ≥95% of the goal box within 2cm *kinematically*. If it can't,
  something is miswired (frames/URDF) — stop and fix, don't proceed to sim.

### Phase 2 — sim integration, standing only (Isaac, GPU — **user runs all of these**)
- Wire the IK controller into `g1_full_demo.py` as the arm-mode backend (new flag, e.g.
  `--arm_backend ik|rl`, default `ik` on this branch; keep the RL path working for A/B).
- Rate-limit interpolation from current arm pose to IK solution; goal set via the same
  T/Y-key interface.
- **Gate**: user visually confirms — robot standing (walking policy active, zero
  command), left arm reaches a commanded target smoothly, no fights with the
  locomotion policy, no fall. Then measure actual palm error in sim (print it — expect
  PD sag vs the kinematic solution; decide if tau_ff is needed).

**Status (2026-07-27): code written, gate not yet run (needs Isaac Sim — user's turn).**
`--arm_backend ik|rl` added, default `ik`. Pelvis-relative conversion is dynamic (every
call, from the robot's live root pose — see `_compute_arm_targets_ik`'s docstring), no
homing-first phase for IK (valid from any starting pose, unlike the RL policy — see
landmine #9's `reset()` call sites: `_set_arm_target`, `_handle_env_resets`). Mirror
testing (Y/U) disabled for `--arm_backend ik` (RL-network-generalization test, no
meaning without an RL network). Goal markers recolor live every frame — green ≤2cm /
yellow ≤5cm / red >5cm — so the far-reach gap Phase 1 found is visible directly on the
robot, per the user's explicit request, not just in the offline sweep plot.

Run (from `isaac_g1_ik`):
```
python testing/visual_testing/full_demo/g1_full_demo.py \
    --loco_checkpoint chosen_checkpoints/walking_latest.pt \
    --arm left --target 0.3 0.2 1.0
```
Check: does the left arm reach smoothly with no fight against the walking policy /
no fall (standing, zero command)? Does the marker color match what you see (green when
the arm looks like it's really at the target, red when it's visibly short)? Try a target
from Phase 1's known-bad far-reach region too (e.g. `--target 0.40 0.35 0.95`) to see
what a real miss looks like on the robot, not just as a number.

**First real bug found and fixed (2026-07-27): PD gravity sag stalls the rate-limiter
itself, not just droops a few cm.** First live test (`--target 0.3 0.2 1.0`, a target
well inside the known-good near half of the box) plateaued at ~19cm error and stayed
red for 540+ steps. The three-way diagnostic print (added specifically for this)
localized it cleanly: `kinematic_err` stayed ~1cm the whole time (solver/frame math is
correct), but `commanded_vs_solved` and `actual_vs_solved` tracked each other closely
(within 3-4°) and **neither shrank toward 0** over hundreds of steps — the signature of
the per-step rate limit (0.06 rad ≈ 3.4°) being fully consumed by gravity sag before it
can produce net progress, a genuine stalled equilibrium, not slow convergence. Fix:
apply the feedforward torque `solve_ik` already computes (`sol_tauff`) — previously
discarded — via `Articulation.set_joint_effort_target()`, confirmed from IsaacLab's own
`ImplicitActuator.compute()` source to add additively on top of the PD position term
(`stiffness*error + damping*error_vel + joint_efforts`), not replace it. Explicitly
zeroed whenever the IK backend isn't actively driving the arm, so no stale torque lingers
into the held/default pose. **Confirmed working by the user** (`actual_vs_solved` dropped
from a stuck 15-25° plateau to converging toward single digits on re-test).

**Second finding (2026-07-27, autonomous session while user was away, explicitly
authorized for this window only — see below): sequential live re-targeting is
meaningfully noisier than Phase 1's offline sweep, and can cause real falls.** Built
`validation/eval_arm_ik_standing.py` — an early Phase-3-style script combining the
walking policy (standing, zero command) with the tau_ff-fixed IK backend, scripted to
sweep many random goal-box targets unattended, recording both the real achieved error
and the pure kinematic error (solver's own FK-recomputed answer, Phase 1's metric) for
direct comparison. Two runs (30 targets each, same seed) surfaced a real, reproducible
gap: kinematic error for the *identical* target differed substantially between runs
(e.g. target `[0.389, 0.184, 1.041]`: 26.3cm one run, 34.0cm the next), and one run hit
an 87cm kinematic error (vs. Phase 1's worst-case ~16cm) immediately before an actual
fall. Root cause understood, not just observed: Phase 1's sweep always warm-started
`solve_ik` from a fixed, reasonable default pose before every target; this live script
(matching real sequential-use — reach target A, then B, then C) warm-starts each new
target from *wherever the arm actually ended up* after the previous one, which can be an
extreme/strained configuration if the previous reach was itself poor. IPOPT (a local
solver) can land in a meaningfully worse local optimum from a bad warm start — the same
mechanism Phase 0's landmine already flagged as a possibility, now confirmed to bite in
a realistic multi-target sequence, not just a synthetic corner case.
Mitigation added (partial, not a full fix): a safety guard in the eval script rejects
any per-step solve with kinematic error > 20cm and holds position instead of acting on
it — this did not fully prevent a second fall (12th of 30 targets on the second run),
meaning single-step rejection alone isn't sufficient; the bad-solve state can still
develop momentum before crossing the threshold. **Decision (2026-07-27, user): build the
hybrid.** Implemented in both `g1_full_demo.py`'s `_compute_arm_targets_ik` and
`validation/eval_arm_ik_standing.py`: normal operation still warm-starts from the arm's
live current pose (smooth for a multi-waypoint gesture — each good reach feeds a good
warm start into the next), but if that solve's kinematic error exceeds
`ARM_IK_RETRY_KINEMATIC_ERR_M` (0.20m), the exact same target is retried once from the
neutral default pose (`arm_ik.reset()` first) before anything is acted on or rate-limited
toward. Whichever solve is used, `_ik_debug`/the eval CSV reflect the final (possibly
retried) result. Tested interactively by the user against `--target 0.3 0.2 1.0`: no
retry fired at all (kinematic_err stayed ~0.3-1.6cm throughout — this specific target
never needed the fix), and the achieved error plateaued around 7-12cm — a separate,
already-understood mechanism (PD/gravity tracking, not warm-start instability; see the
tau_ff entry above). Correctly confirms the retry doesn't do anything harmful when it
isn't needed.

**Second finding + fix (2026-07-27, same session, user out of time / explicit
"do what you think is best" authorization for a ~20 minute window): the once-per-target
retry alone is not sufficient — found via `eval_arm_ik_standing.py` re-runs.** Two
problems surfaced: (1) checking the retry condition every step (the original
implementation) was itself found to be actively harmful, comparing before/after on the
identical 14-target sequence — several targets got WORSE after adding the "safety" retry
(e.g. one target: 0.3cm kinematic error before, 45.1cm after), because a persistently
borderline target keeps re-triggering the retry every single step, repeatedly wiping the
solver's warm-start continuity before it can settle — a fresh reach should get exactly
one correction chance, not a fight every frame. Fixed: retry now gated behind a
`_ik_retry_pending`/`retry_pending` flag set only at the actual discontinuity
(`_set_arm_target`, `_handle_env_resets`, or the sweep script's per-target loop) and
cleared after its one check. (2) Even with that fixed, a *different* failure mode
appeared: a target's kinematic error can be fine on the first solve but drift to a
catastrophic value (~221cm observed) *mid-hold*, as continuous live-warm-start
re-solving wanders into a bad basin over many steps — the once-per-target retry can't
catch this since it only checks once. Added a second, independent, always-on safety net
(`ARM_IK_HARD_REJECT_M`/`_KINEMATIC_HARD_REJECT_M`, 0.30m, looser than the retry
threshold, checked every step but never resets solver state): if the current solve is
this far off, hold position and zero tau_ff for that one step rather than act on it —
recoverable next step, not a latch. Both fixes are in `g1_full_demo.py` and
`validation/eval_arm_ik_standing.py`.

**Result (2026-07-27, user's own run, 15 targets, left arm, `--hold_steps 350`):
falls eliminated on this sample, but real reach accuracy is still poor in absolute
terms.**
```
any fall during sweep: False  (targets that fell: 0/15)
retried (once-per-target, >20cm initial): 0/15
  within  2cm  real=  0.0%  kinematic= 33.3%
  within  5cm  real=  0.0%  kinematic= 46.7%
  within 10cm  real= 13.3%  kinematic= 60.0%
  within 15cm  real= 20.0%  kinematic= 66.7%
  real error:      mean=32.84cm  median=27.31cm  max=80.10cm
  kinematic error: mean=13.01cm  median=7.58cm  max=45.53cm
```
Before today's two fixes, the same-sized test had 2 falls out of 15 (and, pre-hybrid,
falls with kinematic errors up to ~87-221cm). **The safety objective — don't destabilize
the robot — looks achieved on this sample.** The accuracy objective is not: 0% within
2cm/5cm, only 20% within 15cm, mean real error 32.8cm. Part of this gap is the *designed*
cost of the new hard-reject safety net (it holds position rather than committing to a
risky solve, which necessarily leaves the arm short of the target on exactly the targets
it's protecting against) — but even the *kinematic* numbers here (33.3% within 2cm) are
below Phase 1's ~58% box average, consistent with this small random sample (n=15)
happening to draw more than its share from the known-bad far-reach half of the box.
**Bottom line: today's work fixed the falling problem, not the accuracy problem** — the
latter is still open and is where Phase 3's fuller validation / the goal-box-trimming
option / the RL-residual phase come in.

Two infrastructure bugs found and fixed along the way, both confirmed by direct
observation (not guessed): (1) a silent process termination recurring across runs at
almost the same point, traced to Isaac Sim/Kit's `env.close()` hanging after the sweep
finishes — fixed by moving all reporting before that call. (2) even after removing
`env.close()`, the process still hung — confirmed directly by the user (full SUMMARY
printed and both output files written to disk, then the process just sat idle) — the
remaining hang was `simulation_app.close()` itself; fixed by skipping it entirely and
calling `os._exit(0)` immediately once `main()` returns, since every side effect that
matters is already on disk by that point.
**Authorization note**: all Isaac Sim runs in this Phase 2 section were run
autonomously by Claude, under two separate explicit one-time authorizations (once while
the user was away, once during a ~20-minute window with explicit "do what you think is
best, get creative" instructions) — neither is a standing exception to "user runs all
Isaac Sim commands."

**Fixed-base isolation test (2026-07-27, user's own terminal — resolves the "is walking
the problem, or the solver?" question from earlier in this section): walking is a real,
major contributor, not a red herring.** Built `validation/eval_arm_ik_fixed_base.py` —
same `G1ArmIK`, same targets, same retry/hard-reject logic, but the robot's root is
genuinely bolted to the world (`fix_root_link=True`) and no walking policy runs at all —
the closest match in this repo to how `xr_teleoperate`'s own arm-only usage would look.
Result, same 15-target sample, base fixed vs. the walking-integrated run:

| | walking (Phase 2 result above) | base fixed |
|---|---|---|
| retries needed | some targets | 0/15 |
| kinematic within 2cm | 33.3% | 46.7% |
| kinematic mean / max | 13.01cm / 45.53cm | 3.16cm / 10.18cm |
| worst kinematic ever seen | ~221cm (catastrophic) | 10.18cm |

Root cause understood, not just observed, and corrects an earlier claim in this doc:
`kinematic_err` is NOT independent of the walking policy the way it was reasoned to be —
the target passed to `solve_ik` is computed *relative to the live pelvis pose*,
recomputed every frame. With the walking policy actively swaying to balance even at
zero commanded velocity, that pelvis-relative target itself jitters slightly every
frame — IPOPT is a local optimizer being asked to continuously re-solve a moving
target, not a static one, which is a plausible and now-evidenced mechanism for landing
in bad local optima over time. Locking the base removes that source of disturbance
entirely and the catastrophic failures disappear.

**However, a real, separate gap remains even with the base fully fixed**: kinematic
error is small (mean 3.16cm) but achieved (real, PD-tracked) error is not (mean
14.47cm, 0% within 2cm or 5cm even here). With zero torso motion possible, this can no
longer be attributed to walking — it matches the PD-tracking/gravity-compensation
ceiling discussed earlier (soft 40Nm/rad gain, an imperfect static feedforward torque
model) and looks like a genuine, separate limitation. **So there are now two distinct,
separately-confirmed problems, not one**: (1) walking-induced solver instability
(now demonstrated, not just theorized — candidate fixes: reduce/filter the pelvis-frame
target's sensitivity to torso sway, or don't recompute the pelvis-relative target every
single frame once a solve has converged) and (2) a PD/gravity tracking ceiling
independent of walking (candidate fixes: better feedforward modeling, higher gain if
deployment allows it, or the RL residual as originally planned). Neither is fully solved
yet; next step is deciding which to tackle first.
- Build/adapt a `validation/eval_arm_ik.py` mirroring `eval_arm_long_hold.py`'s
  definition of success (**one fresh attempt, one fixed goal, strict hold**) — that is
  THE metric this pivot exists to fix. Compare directly against the RL numbers
  (~30% single-shot for 200/20, ~28-33% aggregate for `best_combined`).
- **Gate**: single-shot success (2cm, hold) dramatically above the RL baseline —
  the working target is >90%; if sim-side PD tracking is the limiter, quantify the gap
  between kinematic solution and achieved pose before tuning anything.
- Update `policy_status.md` + `chosen_checkpoints/README.md` with the outcome either way.

### Phase 4 — the "residuals" part (later; design before building)
- Only after Phase 3 passes: RL residual policy on top of IK targets (small bounded
  corrections) for disturbance robustness / smoother motion / eventually
  arms-while-walking (deferred item #6). This keeps an RL arm deliverable in the final
  architecture. Scope deliberately not designed yet — do that as its own planning pass
  with the user, informed by Phase 2/3 findings.

## Working conventions on this branch (non-negotiable, carried from `main`)

- **Never run Isaac Sim / GPU / training commands yourself. Give the user the exact
  command and wait.** This includes "quick checks" and retries after a crash. Phases
  0-1 are deliberately CPU-only so most of the work needs no permission round-trips.
- Verify, don't infer — this week's history is a parade of plausible assumptions that
  were wrong (see `retrospective.md`). Prefer writing a check script over asserting.
- Offer visual/GUI verification for "does it actually work" questions, not just metrics.
- Keep `policy_status.md` current as findings land; commit/push only when the user asks.
- Training-run backups (if any training happens here) follow the pattern in
  `/home/spyros/Elm/Backups/g1_locomotion/29dof/README.md`.

## Quick file map for the new chat

| What | Where |
|---|---|
| Current arm RL env (goal box, palm frame, action filter, override action) | `source/.../tasks/manager_based/g1_arm/g1_arm_env.py` |
| Walking env cfg + reward recipe | `source/.../tasks/manager_based/g1_locomotion/g1_locomotion_env_cfg.py` |
| Joint order / gains ground truth | `source/.../assets/robots/unitree.py`, `deploy_reference/robots/g1_29dof/config/` |
| Interactive combined demo (integration seam for Phase 2) | `testing/visual_testing/full_demo/g1_full_demo.py` (+ its README) |
| Single-shot eval to mirror in Phase 3 | `validation/eval_arm_long_hold.py` |
| Reachability sweep to mirror in Phase 1 | `validation/check_arm_reachability.py` |
| Kept checkpoints (walking + arm references) | `chosen_checkpoints/`, `logs/rsl_rl/` (see its README) |
| 29-DOF URDF (local) | `~/Elm/Code/g1_simulation/ros2_ws/src/g1_navigation/description_files/urdf/g1_29dof.urdf` |
| Sim USD asset | `~/Elm/Assets/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/` |
