# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 0 smoke test for ``g1_locomotion.controllers.arm_ik.G1ArmIK`` (see
``ik_arm_integration_plan.md``, repo root). Deliberately CPU-only, no Isaac Sim / GPU —
pure Pinocchio + CasADi. Gate for this phase: solves converge for a handful of
hand-picked pelvis-relative wrist targets, every solution respects joint limits, and
solve time per call is measured. This is a solver sanity check only — it does NOT
attempt Phase 1's real accuracy sweep against the RL goal box (`_GOAL_BOUNDS`), since
that requires first deciding the ground-height-vs-pelvis-relative frame conversion (see
plan doc); targets here are picked directly around the arm's own neutral-pose FK instead.

Targets are position-only (rotation held at identity) — the IK's rotation-error cost
term (weight 1.0, vs. translation's 50.0) then acts as a weak regularizer toward a
neutral wrist orientation rather than a real 6-DOF goal; matches this project's
position-only convention (see ``policy_status.md``'s deferred item #2 — real wrist
orientation control is future work).

**Standalone-import note**: this script deliberately does NOT ``import g1_locomotion``
directly — see ``_standalone_arm_ik_import.py`` (same directory) for why, and Phase 1's
``check_arm_ik_coverage.py`` for the sibling script sharing this same loader.

Usage: ``python3 validation/check_arm_ik_solver.py`` (isaac_g1_ik conda env; no
``--headless``/app-launcher flags — there is no sim in this script at all).
"""

import time

import numpy as np

from _standalone_arm_ik_import import load_arm_ik

arm_ik_module = load_arm_ik()
G1ArmIK = arm_ik_module.G1ArmIK
ARM_JOINT_NAMES = arm_ik_module.ARM_JOINT_NAMES


def _se3(position: np.ndarray, rotation: np.ndarray | None = None) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, 3] = position
    if rotation is not None:
        tf[:3, :3] = rotation
    return tf


def main():
    ik = G1ArmIK()
    print(f"[check_arm_ik_solver] reduced model: nq={ik.model.nq}, nv={ik.model.nv}")
    print(f"[check_arm_ik_solver] joint order: {ARM_JOINT_NAMES}")
    print(f"[check_arm_ik_solver] lowerPositionLimit: {ik.model.lowerPositionLimit}")
    print(f"[check_arm_ik_solver] upperPositionLimit: {ik.model.upperPositionLimit}")

    q_neutral = np.zeros(ik.model.nq)
    l_neutral = ik.fk_wrist(q_neutral, "left")
    r_neutral = ik.fk_wrist(q_neutral, "right")
    print(f"[check_arm_ik_solver] neutral L wrist (pelvis frame): {l_neutral}")
    print(f"[check_arm_ik_solver] neutral R wrist (pelvis frame): {r_neutral}")

    # Hand-picked left-arm targets, offsets from the neutral FK pose (x=fwd, y=left+,
    # z=up), spanning forward/up/down/across-body reach. Right arm is held at its own
    # neutral FK pose each time (near-zero error term, doesn't perturb the left solve —
    # see arm_ik.py's docstring on solving both arms as one problem).
    left_targets = {
        "forward":      l_neutral + np.array([0.15, 0.00, 0.00]),
        "up":           l_neutral + np.array([0.05, 0.00, 0.20]),
        "down":         l_neutral + np.array([0.05, 0.00, -0.20]),
        "across_body":  l_neutral + np.array([0.10, -0.25, 0.05]),
        "small_move":   l_neutral + np.array([0.02, 0.02, 0.02]),
        "far_forward":  l_neutral + np.array([0.35, 0.00, 0.05]),  # likely near/past reach limit
    }

    tol_limit = 1e-4  # numeric slack on the hard joint-limit constraint
    q_neutral14 = np.zeros(ik.model.nq)
    solve_times = []
    all_within_limits = True

    # Each target solved FRESH from neutral, with an explicit ik.reset() beforehand:
    # chaining across unrelated targets without resetting showed a real artifact — the
    # solver's own WeightedMovingFilter keeps a 4-call rolling window of RAW solutions
    # *inside the G1ArmIK instance*, so reusing one instance across independent,
    # unrelated targets silently blends in stale poses from the previous target for the
    # next few calls (this is correct, desirable behavior for a continuous control loop
    # — smooths jitter across nearby consecutive targets — but wrong across a
    # discontinuous jump; see G1ArmIK.reset()'s docstring). Resetting before each
    # independent sweep target isolates solver correctness; the separate warm-start demo
    # below deliberately does NOT reset, to show the representative (rate-limited,
    # small-step, filter-smoothed) case Phase 2 will actually use.
    print("\n[check_arm_ik_solver] === Left-arm reach sweep (fresh solve from neutral each time) ===")
    for name, target_pos in left_targets.items():
        left_tf = _se3(target_pos)
        right_tf = _se3(r_neutral)

        ik.reset(q_neutral14)
        t0 = time.perf_counter()
        sol_q, sol_tauff = ik.solve_ik(left_tf, right_tf, current_q=q_neutral14)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        solve_times.append(dt_ms)

        achieved = ik.fk_wrist(sol_q, "left")
        err = np.linalg.norm(achieved - target_pos)

        below = sol_q >= (ik.model.lowerPositionLimit - tol_limit)
        above = sol_q <= (ik.model.upperPositionLimit + tol_limit)
        within_limits = bool(np.all(below) and np.all(above))
        all_within_limits &= within_limits

        print(f"  {name:12s} target={target_pos.round(3)} achieved={achieved.round(3)} "
              f"err={err * 100:5.2f}cm  solve={dt_ms:6.2f}ms  within_limits={within_limits}")

    print("\n[check_arm_ik_solver] === Warm-start demo: small consecutive steps toward "
          "'forward' (representative of Phase 2's rate-limited operation) ===")
    n_steps = 8
    step_target = l_neutral + np.array([0.15, 0.00, 0.00])
    current_q = q_neutral14.copy()
    for i in range(1, n_steps + 1):
        waypoint = l_neutral + (step_target - l_neutral) * (i / n_steps)
        t0 = time.perf_counter()
        sol_q, _ = ik.solve_ik(_se3(waypoint), _se3(r_neutral), current_q=current_q)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        solve_times.append(dt_ms)
        achieved = ik.fk_wrist(sol_q, "left")
        err = np.linalg.norm(achieved - waypoint)
        print(f"  step {i}/{n_steps}  waypoint={waypoint.round(3)}  err={err * 100:5.2f}cm  solve={dt_ms:6.2f}ms")
        current_q = sol_q

    print("\n[check_arm_ik_solver] === Simultaneous both-arm reach ===")
    both_left = l_neutral + np.array([0.15, 0.05, 0.10])
    both_right = r_neutral + np.array([0.15, -0.05, 0.10])
    ik.reset(q_neutral14)
    t0 = time.perf_counter()
    sol_q, _ = ik.solve_ik(_se3(both_left), _se3(both_right), current_q=q_neutral14)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    solve_times.append(dt_ms)
    ach_l = ik.fk_wrist(sol_q, "left")
    ach_r = ik.fk_wrist(sol_q, "right")
    print(f"  left  target={both_left.round(3)} achieved={ach_l.round(3)} "
          f"err={np.linalg.norm(ach_l - both_left) * 100:5.2f}cm")
    print(f"  right target={both_right.round(3)} achieved={ach_r.round(3)} "
          f"err={np.linalg.norm(ach_r - both_right) * 100:5.2f}cm")
    print(f"  solve={dt_ms:6.2f}ms")

    print("\n[check_arm_ik_solver] === Gate summary ===")
    print(f"  solve time per call: mean={np.mean(solve_times):.2f}ms  "
          f"min={np.min(solve_times):.2f}ms  max={np.max(solve_times):.2f}ms  "
          f"(n={len(solve_times)} calls, includes IPOPT warm-up on first call)")
    print(f"  all solutions within joint limits: {all_within_limits}")


if __name__ == "__main__":
    main()
