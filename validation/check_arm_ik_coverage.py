# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 1 of ``ik_arm_integration_plan.md``: dense accuracy sweep of
``g1_locomotion.controllers.arm_ik.G1ArmIK`` against the RL arm policy's own goal
workspace (``_GOAL_BOUNDS`` in ``g1_arm_env.py``). Deliberately CPU-only, no Isaac Sim /
GPU. Gate: IK reaches >=95% of the goal box within 2cm *kinematically* (matching
``check_arm_reachability.py``'s own tolerance ladder, whose already-measured result is
the ceiling any solver — including ours — can hit: 97.0% for the left arm, see that
script's docstring). If this sweep comes in well below that ceiling, something in the
solver setup is miswired (frames/URDF/weights) — the plan says stop and fix, don't
proceed to sim integration.

**Frame conversion** (the piece Phase 0 deliberately deferred): ``_GOAL_BOUNDS`` is
defined relative to ``env.scene.env_origins`` (see ``check_arm_reachability.py`` line
~149: ``ee_pos_local = ee_pos_world - env.scene.env_origins``) — i.e. **ground-relative**,
not pelvis-relative. The G1's default standing pelvis height is 0.8m
(``unitree.py``'s ``init_state.pos=(0.0, 0.0, 0.8)``, x/y offset zero, no root rotation),
so converting a goal-box point to this controller's pelvis-relative frame is: same x/y,
``z_pelvis = z_ground - 0.8``. ``_GOAL_BOUNDS`` and the height constant are copied below
rather than imported (importing the real ``g1_arm_env.py`` pulls in ``isaaclab`` -> ``pxr``,
exactly what this phase avoids needing) — **keep these two values in sync with their
source of truth by hand** if either ever changes.

Method: for each arm side, sample a dense grid over that side's goal box, convert each
point to the pelvis frame, and solve IK fresh (``ik.reset()`` beforehand — see
``arm_ik.py``'s landmine #9 on why chaining across unrelated points corrupts the result)
from the sim's actual default arm pose (not all-zero — see ``unitree.py``'s
``init_state.joint_pos``), holding the *other* arm at its own default-pose FK position.
Records FK-recomputed position error, IPOPT solve-success, joint-limit saturation, and
solve time per point.

Usage: ``python3 validation/check_arm_ik_coverage.py`` (isaac_g1_ik conda env; no
``--headless``/app-launcher flags — no sim in this script at all). ``--grid_res`` and
``--side`` are the only knobs; defaults sweep both arms at the same grid density
``check_arm_reachability.py`` used (14 per axis) for direct comparability.
"""

import argparse
import os
import time

import numpy as np

from _standalone_arm_ik_import import load_arm_ik

arm_ik_module = load_arm_ik()
G1ArmIK = arm_ik_module.G1ArmIK
ARM_JOINT_NAMES = arm_ik_module.ARM_JOINT_NAMES

# Copied from g1_arm_env.py's _GOAL_BOUNDS (2026-07-27) — see module docstring on why
# this isn't imported directly. Ground-relative (env-origin) frame.
_GOAL_BOUNDS = {
    "left":  {"x": (0.20, 0.42), "y": (0.08, 0.40), "z": (0.9, 1.15)},
    "right": {"x": (0.20, 0.42), "y": (-0.40, -0.08), "z": (0.9, 1.15)},
}

# Copied from unitree.py's UNITREE_G1_29DOF_CFG.init_state (2026-07-27): pos=(0,0,0.8),
# no root rotation -> pelvis sits 0.8m directly above the env origin.
_PELVIS_HEIGHT_M = 0.8

# Copied from the same init_state's joint_pos (arm entries only; wrist_pitch/yaw not
# listed there, so they default to 0) — the sim's real resting arm pose, not all-zero.
_DEFAULT_ARM_JOINT_POS = {
    "left_shoulder_pitch_joint": 0.3, "left_shoulder_roll_joint": 0.25, "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.97, "left_wrist_roll_joint": 0.15, "left_wrist_pitch_joint": 0.0, "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.3, "right_shoulder_roll_joint": -0.25, "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.97, "right_wrist_roll_joint": -0.15, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
}


def _ground_to_pelvis(point_ground: np.ndarray) -> np.ndarray:
    out = point_ground.copy()
    out[2] -= _PELVIS_HEIGHT_M
    return out


def _se3(position: np.ndarray) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, 3] = position
    return tf


def _default_q() -> np.ndarray:
    return np.array([_DEFAULT_ARM_JOINT_POS[name] for name in ARM_JOINT_NAMES])


def sweep_side(ik, side: str, grid_res: int) -> dict:
    other_side = "right" if side == "left" else "left"
    q_default = _default_q()
    other_target_pelvis = ik.fk_wrist(q_default, other_side)

    bounds = _GOAL_BOUNDS[side]
    xs = np.linspace(*bounds["x"], grid_res)
    ys = np.linspace(*bounds["y"], grid_res)
    zs = np.linspace(*bounds["z"], grid_res)
    grid_ground = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    n = grid_ground.shape[0]
    errs_cm = np.zeros(n)
    solved_ok = np.zeros(n, dtype=bool)
    within_limits = np.zeros(n, dtype=bool)
    solve_times_ms = np.zeros(n)
    tol_limit = 1e-4

    print(f"[check_arm_ik_coverage] {side} arm: sweeping {n} grid points ({grid_res}^3)...")
    for i in range(n):
        target_pelvis = _ground_to_pelvis(grid_ground[i])
        left_target = target_pelvis if side == "left" else other_target_pelvis
        right_target = target_pelvis if side == "right" else other_target_pelvis

        ik.reset(q_default)
        t0 = time.perf_counter()
        sol_q, _ = ik.solve_ik(_se3(left_target), _se3(right_target), current_q=q_default)
        solve_times_ms[i] = (time.perf_counter() - t0) * 1000.0
        solved_ok[i] = ik.last_solve_succeeded

        achieved = ik.fk_wrist(sol_q, side)
        errs_cm[i] = np.linalg.norm(achieved - target_pelvis) * 100.0

        below = sol_q >= (ik.model.lowerPositionLimit - tol_limit)
        above = sol_q <= (ik.model.upperPositionLimit + tol_limit)
        within_limits[i] = bool(np.all(below) and np.all(above))

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}")

    return {
        "grid_ground": grid_ground,
        "errs_cm": errs_cm,
        "solved_ok": solved_ok,
        "within_limits": within_limits,
        "solve_times_ms": solve_times_ms,
    }


def report(side: str, result: dict) -> None:
    errs_cm = result["errs_cm"]
    solved_ok = result["solved_ok"]
    within_limits = result["within_limits"]
    solve_times_ms = result["solve_times_ms"]
    n = errs_cm.size

    tolerances_cm = [2.0, 5.0, 10.0, 15.0]
    print(f"\n[check_arm_ik_coverage] === {side} arm coverage ({n} points) ===")
    for tol in tolerances_cm:
        coverage = float((errs_cm <= tol).mean()) * 100.0
        print(f"  within {tol:5.1f}cm: {coverage:5.1f}% of goal box")
    print(f"  error: mean={errs_cm.mean():.2f}cm  median={np.median(errs_cm):.2f}cm  "
          f"p90={np.percentile(errs_cm, 90):.2f}cm  max={errs_cm.max():.2f}cm")
    print(f"  IPOPT solve success rate: {solved_ok.mean() * 100:.1f}%")
    print(f"  solutions within joint limits: {within_limits.mean() * 100:.1f}%")
    print(f"  solve time: mean={solve_times_ms.mean():.2f}ms  p90={np.percentile(solve_times_ms, 90):.2f}ms  "
          f"max={solve_times_ms.max():.2f}ms")

    gate_coverage = float((errs_cm <= 2.0).mean()) * 100.0
    gate_pass = gate_coverage >= 95.0
    print(f"  GATE (>=95% within 2cm): {'PASS' if gate_pass else 'FAIL'} ({gate_coverage:.1f}%)")

    grid = result["grid_ground"]
    xm, ym, zm = np.median(grid[:, 0]), np.median(grid[:, 1]), np.median(grid[:, 2])
    print("  per-octant breakdown (mean/max error, split at box median):")
    for x_half, x_mask in (("low", grid[:, 0] < xm), ("high", grid[:, 0] >= xm)):
        for y_half, y_mask in (("low", grid[:, 1] < ym), ("high", grid[:, 1] >= ym)):
            for z_half, z_mask in (("low", grid[:, 2] < zm), ("high", grid[:, 2] >= zm)):
                mask = x_mask & y_mask & z_mask
                if not mask.any():
                    continue
                d = errs_cm[mask]
                cov = float((d <= 2.0).mean()) * 100.0
                print(f"    x={x_half:4s} y={y_half:4s} z={z_half:4s}  n={mask.sum():4d}  "
                      f"mean={d.mean():6.2f}cm  max={d.max():6.2f}cm  within2cm={cov:5.1f}%")

    out_dir = "validation/arm_ik_coverage"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{side}_summary.md")
    lines = [
        f"# G1 29dof Arm IK Coverage Check — {side} arm",
        "",
        f"Goal bounds (ground-relative): `{_GOAL_BOUNDS[side]}`",
        f"Grid points swept: {n}",
        "",
        "| Tolerance | Coverage |",
        "|---|---|",
    ]
    for tol in tolerances_cm:
        coverage = float((errs_cm <= tol).mean()) * 100.0
        lines.append(f"| {tol:.1f}cm | {coverage:.1f}% |")
    lines += [
        "",
        f"Error: mean={errs_cm.mean():.2f}cm, median={np.median(errs_cm):.2f}cm, "
        f"p90={np.percentile(errs_cm, 90):.2f}cm, max={errs_cm.max():.2f}cm",
        f"IPOPT solve success rate: {solved_ok.mean() * 100:.1f}%",
        f"Solutions within joint limits: {within_limits.mean() * 100:.1f}%",
        f"Solve time: mean={solve_times_ms.mean():.2f}ms, p90={np.percentile(solve_times_ms, 90):.2f}ms, "
        f"max={solve_times_ms.max():.2f}ms",
        "",
        f"Gate (>=95% within 2cm): {'PASS' if gate_pass else 'FAIL'} ({gate_coverage:.1f}%)",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  summary written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 dense IK accuracy sweep against the RL arm goal box.")
    parser.add_argument("--side", type=str, default="both", choices=["left", "right", "both"])
    parser.add_argument("--grid_res", type=int, default=14, help="Grid points per axis (matches check_arm_reachability.py's default).")
    args = parser.parse_args()

    ik = G1ArmIK()
    sides = ("left", "right") if args.side == "both" else (args.side,)
    for side in sides:
        result = sweep_side(ik, side, args.grid_res)
        report(side, result)


if __name__ == "__main__":
    main()
