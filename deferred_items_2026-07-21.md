# Deferred items — 2026-07-21

Raised during the 29dof pivot implementation session. Priority notes added 2026-07-22
per user review. See `29dof_implementation_plan.md` for full context on each.

1. **Arm-rest-before-locomotion-transition safety mechanism.** [**First thing tomorrow**]
   The old repo's `--wait_arm_rest` flag delayed a stand→walk transition until the arm
   finished homing back to default, avoiding a momentum-carryover fall. The new demo/eval
   have no equivalent — arm reaching is unconditional, and the new `walk_stop_reach` eval
   bucket deliberately runs *without* any mitigation, specifically to measure the failure
   mode raw before deciding if/how to reintroduce a fix. User will test the real (tonight's)
   checkpoints tomorrow for transition instability before deciding on a fix.
2. **Arm goal orientation (6-DOF)** — [waits] using the wrists for actual orientation
   control, not just as redundant DOF for a position-only goal. Position-only is what's
   trained now.
3. **IK-driven / policy-driven arm-disturbance training curricula**
   (`StandingArmIKReachDisturbance`/`StandingArmPolicyReachDisturbance`) — [**priority,
   depends on tomorrow's results**] more sophisticated than the current scripted
   joint-space disturbance (real x/y/z reach targets instead of random joint swings).
   If tomorrow's arm policy is good: drive the disturbance with the real trained policy
   (`StandingArmPolicyReachDisturbance`). If not: fall back to analytic IK
   (`StandingArmIKReachDisturbance`) — but per the user, IK has "a different problem"
   to discuss tomorrow before going that route. Both were blocked on having a trained
   arm checkpoint + re-derived goal bounds; both now exist.
4. **Actuator delay + Coulomb friction sim2real model** (`UnitreeActuator`, ported as
   available infrastructure in `assets/robots/unitree_actuators.py`) — [**first thing
   to add if tomorrow's results are good**] not wired into the G1 asset's actual
   actuators yet.
5. **Right-arm / both-arm dedicated training** — [waits] right arm currently has no
   checkpoint of its own anywhere, purely mirror-driven from the left policy (same as
   the old repo, never resolved there either).
6. **Arms-while-walking.** [**RESOLVED 2026-07-22, opposite direction**] — the
   arm-motion disturbance curriculum previously ran regardless of the commanded
   velocity (some incidental arm-motion-while-walking exposure). Per explicit user
   request, this is now gated off: `ArmMotionDisturbance` only disturbs envs currently
   commanded to (near-)stand (`mdp/events.py`'s `_STANDING_CMD_THRESHOLD` gate) — envs
   commanded to walk get arms relaxed toward default instead. "Arms while walking"
   stays a real future feature, deliberately not trained toward yet.
7. **Three untouched interactive test scripts** (`testing/arm_testing/g1_arm_reach_test.py`,
   `testing/arm_testing/g1_arm_mirror_test.py`, `testing/walking_testing/
   g1_stand_walk_switch_demo.py`) and their `checkpoints.yaml` files still reference
   old 5-DOF/"IK" naming; the switch-demo script tests a mode-switch mechanism that no
   longer exists at all (delete candidate, not update candidate). Purely visualization
   tools — agreed to fix before overnight or tomorrow, not blocking training.
