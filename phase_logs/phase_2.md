# Phase 2 — Arm Policy Improvements

> **Archival phase log** — frozen record. Current status lives in `definitive_next_steps.md` (repo root); what happened after this log (the integration debugging that resolved the fall problem) is `phase_logs/phase_3.md`. `quickrun_tests.md` was removed 2026-07-16.

Plain-language summary of what happened, plus a handoff section for whoever picks up
integration next (may be a fresh chat with no memory of this one). Full blow-by-blow
technical detail, with exact numbers for every finding, lives in `known_issues.md`'s
"Arm policy — current state" section — this is the condensed, oriented version. Date:
7–9 July 2026.

## What happened

- Started the phase with the arm policy plateaued at **~55% success** (2cm threshold),
  with a real, unaddressed tail of goals it just couldn't solve (p90 ~12cm). Several
  early hypotheses were tested and ruled out or found harmful: reward shaping + higher
  entropy bundled together (made things much worse, reverted), a `joint_vel` observation
  bug (real bug, fixed, but zero effect on the plateau), a first null-space regularization
  attempt (real but modest effect on precision, no effect on success rate).
- **The big find**: built `validation/check_arm_reachability.py` — samples random joint
  configs within real hardware limits (no RL involved, pure kinematics) to measure how
  much of the goal workspace (`_GOAL_BOUNDS`) is actually reachable. Result: only ~47% of
  the box was within 2cm of anything reachable. A large, physically real chunk of the
  workspace was being asked of the policy but was never actually solvable — a hard
  ceiling no amount of training could cross, unrelated to policy quality.
- Reshaped `_GOAL_BOUNDS` twice (once for a genuinely-unreachable far corner needing
  full-arm-extension-plus-max-lateral-plus-max-height simultaneously; once for a near
  corner that conflicted with the torso anti-collision safety margin). **Success rate
  jumped from ~55% to ~85.6%** — confirms reachability was the dominant factor.
- Found and fixed a second real bug along the way: `elbow_pitch`'s joint-limit reward
  margin was penalizing full arm extension (a normal pose needed for far reaches) because
  that joint's hardware range is heavily asymmetric (can barely go past straight, folds a
  lot). Fixed to be per-joint, per-bound. Turned out **not** to be the cause of what
  looked like a related problem (see next point) — relaxing it made no difference.
- **The remaining ~15-20% gap turned out to be manipulator redundancy, not goal
  difficulty.** Added goal-position logging and found: for any given goal, the policy
  sometimes lands on one valid joint-configuration solution (~91% reliable) and sometimes
  on a different, equally-valid one for the *same* goal (~59% reliable) — a free,
  goal-independent choice, not tied to any region of the workspace. Three different
  targeted fixes were tried and **all three failed to move it**: the joint-limit margin
  relax, a first (unweighted) null-space penalty, and a second (per-joint-weighted,
  specifically targeting the joint that differs between the two solutions) null-space
  penalty. Leading theory: PPO's single (unimodal) Gaussian policy may be structurally
  unable to reliably resolve a genuinely bimodal solution space with just reward shaping
  — a real fix would likely need a more expressive action distribution, out of scope for
  this phase.
- Ran a 3-way isolated overnight sweep (entropy, wider network, PPO hyperparameter
  tuning) on top of the reachability-fixed baseline. None fixed the redundancy issue
  either (branch-selection frequency unchanged across all three), but **wide network
  reproducibly improved precision/tail** (best p90 of any variant, and this is the
  *second* time it's shown this exact strength — same result shown once on the old ~55%
  baseline, once on the current ~85% one). Entropy and wide-net both also modestly
  improved *how well* the policy does even when it lands on the less-reliable solution
  branch, without changing how *often* it lands there.
- **Decision**: fold wide-net into a final consolidated arm policy (5000 iterations,
  `G1-Arm-IK-Left-WideNet-v0`) — queued/running as this phase closes out. Entropy and PPO
  tuning were not adopted (mixed / minimal effect respectively).
- User proposed a good architectural idea for the redundancy/reach-envelope problem that
  reward-shaping couldn't solve directly: rather than forcing the arm alone to cover
  100% of the theoretical workspace, treat a smaller region as its *reliable* envelope
  and have the robot reposition itself (via walking) for anything outside it. Assessed as
  more tractable than it sounds — doesn't need simultaneous arm+walking coordination
  (hard, unstarted), only sequential walk-then-reach on top of machinery that mostly
  already exists. Scoped for the integration phase, not started yet.

## What changed vs. before (code / config delta)

|                                             | Before this phase                                                               | Now                                                                                                                                                                                                                                    |
| ------------------------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Goal workspace (`_GOAL_BOUNDS`, left arm) | `x:(0.1,0.5) y:(0.05,0.45) z:(0.9,1.2)` — ~47% actually reachable within 2cm | `x:(0.20,0.42) y:(0.08,0.40) z:(0.9,1.15)` — reshaped twice based on measured reachability, not guessed                                                                                                                             |
| Observation                                 | 26-D per arm;`joint_vel` silently only covered 3 of 5 joints                  | 28-D per arm; all 5 joint velocities observed                                                                                                                                                                                          |
| Joint-limit reward margin                   | Flat 5% of range, same for every joint/bound                                    | Per-joint, per-bound (`elbow_pitch`'s lower bound only: 1%, everything else unchanged at 5%)                                                                                                                                         |
| Null-space regularization                   | Didn't exist                                                                    | Exists, per-joint-weighted (`elbow_pitch` 3x) — real, reproducible improvement to mean/p90 distance, but doesn't fix the redundancy/branch issue                                                                                    |
| Network size                                | `[256,128,64]` (baseline)                                                     | `[512,256,128]` (wide-net, adopted into the final consolidated policy)                                                                                                                                                               |
| Success rate (fixed-seed eval)              | ~55%                                                                            | ~85% (before the branch-selection issue is addressed, if ever)                                                                                                                                                                         |
| Reachability verification                   | None — box bounds were hand-picked, never checked against real kinematics      | `validation/check_arm_reachability.py` — samples real joint limits, measures coverage, run before trusting any future box change                                                                                                    |
| Joint-config diagnostics                    | None — only`min_dist_to_goal_cm` was logged                                  | `ArmMetricsCsvWrapper` also logs each joint's angle at closest approach (`*_deg_at_min_dist`) and the goal's own position (`goal_x_m/y_m/z_m`), so failures can be correlated with pose *and* goal location, not just distance |
| `eval_arm.py`                             | Fixed network shape, fixed goal box                                             | `--hidden_dims` (evaluate any checkpoint's actual architecture) and `--goal_x_range` (evaluate against a restricted region for apples-to-apples comparisons) added                                                                 |

## Current state (as of this handoff)

- **Best confirmed checkpoint so far**: `logs/rsl_rl/arms/g1_arm_ik_left/2026-07-08_17-00-18_null_space_weighted/model_1499.pt` — ~84-85% success, p90 ~3.6cm. Trimmed (training-time CSVs removed) but checkpoint/eval/tensorboard kept.
- **In progress / queued**: a final consolidated run (`G1-Arm-IK-Left-WideNet-v0`, 5000 iterations, `run_name=final_consolidated`) is expected to supersede the checkpoint above — check `logs/rsl_rl/arms/g1_arm_ik_left_wide_net/` for the newest run once it's done, and use *that* checkpoint (with `--hidden_dims 512 256 128` wherever it's loaded — its network shape differs from the baseline) for integration, not the one above, if it exists and its eval looks sane.
- **Standing**: also being resumed to its original 8000-iteration target (phase-5, 33 rad/s curriculum) in the same overnight run — see `phase_logs/phase_1.md` for that policy's own history. Not otherwise touched this phase.
- **Known, accepted, open limitation**: ~15-20% of episodes land on a less-reliable
  solution branch (see above) — not fixed, 3 attempts failed, next real fix (if pursued)
  needs a bigger architectural change, not more reward tuning. The current plan is to
  work around this via the base-repositioning idea below rather than keep chasing it with
  reward shaping.

## For whoever picks up integration next

1. **Use the newest checkpoint under `logs/rsl_rl/arms/g1_arm_ik_left_wide_net/`** (check
   its eval summary first — success rate should be in the 84-88% range; if it's much
   lower, something went wrong overnight and the `null_space_weighted` checkpoint above
   is the fallback). Remember `--hidden_dims 512 256 128` whenever loading it outside
   `eval_arm.py` (e.g. in `g1_full_demo.py`-style integration code) — the network shape is
   `[512,256,128]`, not the older `[256,128,64]`.
2. **The redundancy/branch-selection issue is real and not fixed.** Expect ~15-20% of
   reach attempts to occasionally be less precise than the rest, even on comfortably
   reachable goals — this isn't a bug to chase, it's a documented, accepted
   characteristic for now (see `known_issues.md` for the full diagnosis if you want to
   revisit it).
3. **Build the "walk closer if not in reach" fallback** — this was the explicit next
   step, not yet started. Suggested approach, from earlier discussion: (a) reuse
   `check_arm_reachability.py`'s method to define/check a *reliable* envelope (may not
   need to be the same as `_GOAL_BOUNDS` — could be tighter), (b) if a requested target
   falls outside it, compute a base offset that would bring it inside, (c) command
   walking to that offset, (d) once settled, switch to standing+arm mode and reach — this
   only needs sequential walk-then-reach, not simultaneous coordination, and the mode-
   switch machinery already exists in `testing/general_testing/g1_full_demo.py`.
4. **Only after (3) is working**, if edge-case goals (e.g. the box's far corners) are
   still a practical problem, consider shrinking `_GOAL_BOUNDS` further and retraining —
   this is legitimate *now* (capability moves to the walking fallback, doesn't
   disappear), where it would have been overfitting-by-avoidance earlier in this phase
   when no fallback existed. Don't shrink the box without a working fallback in place.
5. **Arm-vs-arm collision for `arm="both"`** is a known, unaddressed gap if two-arm
   integration is ever needed (see `known_issues.md`) — self-collision physically
   prevents interpenetration already, but there's no learned smooth-avoidance reward for
   it yet.
6. **`overnight_train.sh` is reused/edited per sweep**, not a fixed pipeline — expect it
   to have different content each time depending on what was queued most recently; check
   its current contents rather than assuming.

## Reference — key tools built this phase

- `validation/check_arm_reachability.py` — kinematic-only reachability checker, no RL
  involved. Run this before ever trusting a new goal-box definition again.
- `validation/eval_arm.py` — gained `--hidden_dims` (match checkpoint architecture) and
  `--goal_x_range` (restrict eval to a sub-region for fair comparisons).
- `ArmMetricsCsvWrapper` (`g1_locomotion/utils/metrics_wrappers.py`) — gained per-joint
  angle-at-closest-approach columns and goal-position columns; use these for any future
  "does failure correlate with X" diagnostic rather than re-deriving from scratch.

## Integration testing (standing + walking + arm, combined) — 2026-07-09

Picking up item 3 from "For whoever picks up integration next" above — actually
combining all three trained policies in one session (`g1_full_demo.py`) instead of
testing each in isolation, per the plan already laid out.

**What happened:**

- Started by refreshing `chosen_checkpoints/` — `arm_left_latest.pt` and
  `standing_latest.pt` were both stale (June 18, predating everything above in this
  phase). Updated to the best confirmed checkpoints (`null_space_weighted` for arm, the
  phase-5-curriculum run for standing) so `g1_full_demo.py` — the one script that
  exercises all three policies together — actually reflects this phase's work. Verified
  all six interactive `testing/` scripts already default to `chosen_checkpoints/*.pt`
  via their local `checkpoints.yaml` when no explicit checkpoint flag is given (a
  couple of `quickrun_tests.md`'s descriptions were stale about this — code was already
  correct, only the docs were wrong).
- The phase-5-curriculum standing checkpoint (trained to test whether a 33 rad/s
  arm-disturbance phase improves corrective stepping — see phase_1.md) turned out to
  still be unreliable enough in practice (visible leaning, occasional falls,
  "stationary stepping" during normal standing) that the user reverted to the
  pre-phase-5 standing checkpoint for actual demo use. Standing's own
  stepping-reliability issue remains open and is being picked up in a **separate,
  dedicated session** — not part of this integration-testing work.
- Found and fixed a real chain of bugs in `g1_full_demo.py`, each one surfaced by
  actually running the combined demo (none of these were visible from isolated
  per-policy testing in `testing/arm_testing/` or `testing/walking_testing/`):

  1. **Standing env action-term mismatch (investigated, not shipped as a fix here).**
     While first exploring an automated integration-eval approach (see "Approach
     change" below), found that `G1LocomotionStandingFlatEnvCfg` swaps in a custom
     action term (`StandingArmBlendJointPositionAction`) that *unconditionally*
     overrides arm joints with a scripted disturbance target — the standing policy's
     own raw action output for arm joints is never actually used during training. This
     doesn't affect `g1_full_demo.py` (it runs on the *walking* env, which uses a plain
     action term), but is the reason a from-scratch automated standing+arm eval harness
     is nontrivial — noted for whoever revisits `validation/eval_integration.py`.
  2. **Reset-anchor bug (initial `--target` only).** `G1FullDemo.__init__` anchored a
     CLI-supplied `--target` to the robot's pose *before* `main()`'s `env.reset()` ran
     the real randomized spawn (`reset_base` randomizes yaw over the full ±180° and
     position by ±0.5m — confirmed in `velocity_env_cfg.py`, and no PLAY config disables
     it). Anchoring before that reset meant the target could end up anywhere relative to
     where the robot actually spawned. Fixed by re-anchoring in `main()` right after the
     real reset, if a target is already active.
  3. **`inference_mode` crash on `T`.** `_set_arm_target`'s `write_joint_state_to_sim`
     call ran outside `torch.inference_mode()` (it's called from `main()`'s loop before
     entering the `with torch.inference_mode():` block), but the tensors it writes into
     were already tagged as inference tensors by the step loop — same restriction
     `validation/eval_standing.py` already has a comment about. Crashed with `RuntimeError:
     Inplace update to inference tensor outside InferenceMode`. Fixed by scoping the
     write in its own `inference_mode()` block.
  4. **Z-height convention mismatch (the "target is way too high, never reachable" bug).**
     `g1_arm_reach_test.py` and `g1_arm_env.py` both treat target z as height *above
     ground* (added straight onto `env_origins`, whose z is 0). `g1_full_demo.py`
     instead anchored z to the robot's *root link* position, which for a standing G1 is
     already ~0.75-0.8m off the ground — so a z=1.0 target actually landed at world
     height ~1.75-1.8m, far out of reach. Fixed by keeping x/y and orientation anchored
     to the root (needed for walking-to-a-new-spot correctness) but pinning z to ground
     level (`env_origins`'s z) instead.
  5. **Right arm moving unexpectedly while only the left arm has a target.**
     `select_action` only called `_hold_arms_at_default` in the `elif` branch — when an
     arm target was active it skipped holding the *other* arm, leaving it running on the
     standing policy's raw arm-column output. That output is never meaningful (see bug
     1 — standing training never lets the policy's own arm action affect anything), so
     the untouched arm produced visible, uncommanded motion. Fixed by always holding
     *both* arms at default first when standing, then overlaying the arm-IK policy only
     for the arm(s) with an active target.
  6. **Arm actuator stiffness mismatch.** The arm-IK policy trains under
     `stiffness=200, damping=20` (`g1_arm_env.py`); the shared walking/standing env
     `g1_full_demo.py` runs in never overrides "arms" at all, so it was running on the
     walking task's stock `stiffness=40, damping=10` (`G1_MINIMAL_CFG`) — a much softer
     response to the same commanded targets. Matched to the arm task's own training
     gains. Known trade-off: the walking policy's natural arm-swing gait was trained
     under the *softer* stock gains, so its arm motion during walking may look/feel
     different now — not yet independently re-verified.
  7. **Camera couldn't be manually rotated.** `update_camera()` re-locked the chase
     camera's transform every single frame, so any manual drag got overwritten before
     the next frame. Replaced the old "toggle between two camera prims" design with a
     single camera plus a `_camera_follow` flag: `C` toggles follow on/off (off = free
     mouse orbit, since nothing overwrites the transform anymore), `V` resets to the
     default chase framing and re-enables follow.
- Added **multi-arm manipulation testing** (`Y` / `U` keys, arm_mode=left only): reuses
  `mdp/symmetry.py`'s `mirror_arm_obs`/`mirror_arm_actions` (the same transform
  `g1_arm_mirror_test.py` already validates in isolation, and what the arm policy is
  actually *trained* with via symmetry-augmented PPO) to drive the right arm from the
  same left-trained checkpoint, live, in the combined demo:
  - `Y` — target for the right arm only, via mirror (blue marker, vs. the native left
    target's red).
  - `U` — one target, sent to both arms at once (left direct, right mirrored) — tests
    whether the mirror-generalized policy holds up controlling both arms
    simultaneously, not just one at a time. This is the first time that combination has
    been exercised against the real, standing robot rather than in isolation.
- Found the goal-box bounds documented in several places (`g1_full_demo.py`,
  `g1_arm_reach_test.py`, `g1_arm_mirror_test.py`, `arm_testing/checkpoints.yaml`,
  `quickrun_tests.md` ×3 tables) still quoted the pre-reachability-fix numbers
  (`x:0.1-0.5, y:0.05-0.45, z:0.9-1.2`) — never updated after this phase's reachability
  fix reshaped the box (`x:0.20-0.42, y:0.08-0.40, z:0.9-1.15`). Didn't affect actual
  behavior anywhere (all the real validation/prompt logic reads `_GOAL_BOUNDS`
  dynamically) but was actively misleading to read. Fixed all six locations.

**Approach change: automated eval harness attempted, then dropped for the interactive
demo.** Early in this round, started building `validation/eval_integration.py` — a
headless, vectorized standing+real-arm-policy eval mirroring `eval_arm.py`/
`eval_standing.py`'s conventions, meant to answer "does the real arm policy destabilize
standing" with a number instead of a visual impression. Got as far as a working harness
and a real, diagnosed bug (see item 1 above), but this consumed a lot of session budget
on Isaac-Sim-launch-heavy diagnostic loops before checking in on direction.
**Redirected per explicit user feedback**: fix the interactive `g1_full_demo.py`
directly instead — it was already close to working, cheaper to iterate on with a human
watching it, and tests the actual deployment path. `validation/eval_integration.py` is
left in the repo, unused/not wired into anything — worth revisiting later if an
automated, statistical version of this testing is wanted again, but the action-term
issue (item 1 above) needs solving first (either compensate for
`StandingArmBlendJointPositionAction`'s blend math, or revert it in a dedicated eval cfg
the same way this demo doesn't need to).

**Current state (as of this handoff):**

- `g1_full_demo.py` is confirmed working for single-arm (left) reaching while standing,
  including at the harder end of the goal box (`0.4 0.3 1.1` reached, slowly).
- Multi-arm mirror testing (`Y`/`U`) is implemented but **not yet exercised/confirmed by
  the user** — next session should actually test it and report back whether the mirror
  transform holds up driving both arms at once against the real robot, not just in the
  isolated `g1_arm_mirror_test.py` setting it was validated in before.
- Standing's own stepping-reliability issue (known, pre-existing per phase_1.md) is
  being debugged in a separate session — not blocking on anything here.
- The genuine "arm moving *while walking*" / simultaneous arm+walking coordination
  question is **not** what this round covered (this was sequential: reach while
  standing, walk with arms held, nothing simultaneous) — that's its own, harder,
  not-yet-started topic (see known_issues.md's "arm-to-walking transition vibration"
  item and the "Option 2" base-repositioning discussion above).

**Additional follow-ups from this round** (in the same spirit as "For whoever picks up
integration next" above, not a continuation of that exact numbered list):

1. **Get a read on the mirror-testing result** (`Y`/`U` in `g1_full_demo.py`) — does the
   left-trained policy generalize to the right arm, and to both arms simultaneously,
   against the real standing robot? This directly informs whether a real two-arm
   training run (`G1-Arm-IK-Both-v0`) is actually necessary or whether mirroring is
   good enough for now.
2. **Re-verify walking's arm-swing gait** after the actuator stiffness change (item 6
   above) — it was trained under softer gains than it's now running with.
3. **`validation/eval_integration.py`** is unused and has a known unresolved issue
   (the standing action-term override) — don't resume it without addressing that first,
   and confirm with the user whether an automated harness is even wanted before
   investing more time there (last time cost more than expected relative to its value).

**Individual-policy issues noticed during this round** (not fixed here — separate
track): these came up while testing the *combination*, but are properties of one policy
in isolation, not an integration bug — logged for visibility, being picked up in a
separate, dedicated debugging session per the user's own plan:

- **Standing leans/drifts to one side** during normal (no-arm-disturbance) standing —
  noticed on the phase-5-curriculum checkpoint specifically; reverting to the
  pre-phase-5 checkpoint resolved it for now. Whether this is specific to that
  checkpoint's training or a broader issue is unconfirmed.
- **Standing takes small "stationary" corrective steps** even with no real disturbance
  — also noticed on the phase-5-curriculum checkpoint; matches the known, documented
  stepping-reliability issue in `phase_1.md`/`known_issues.md` (phase-5 was specifically
  trained to try to improve this, so seeing it manifest for the first time on the first
  checkpoint actually trained under it is not surprising) — not a new issue, being
  tracked in the existing backlog item.
- **Arm reach is slow at the harder end of the goal box** (`0.4 0.3 1.1` took noticeably
  longer than easier positions) — consistent with the redundancy/branch-selection issue
  already documented earlier in this phase, not a new finding, but worth keeping in mind
  if it's still slow after any future arm retrain.
