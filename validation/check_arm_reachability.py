# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checks how much of `_GOAL_BOUNDS` (the G1 29dof arm policy's goal workspace) is
actually kinematically reachable, independent of any trained policy.

Rewritten 2026-07-21 for the 29dof pivot — replaces the 23dof-era, 5-DOF version that
used to live at this exact path (preserved on the `23_dof` git branch) — see
`g1_arm_env.py`'s module docstring and `29dof_implementation_plan.md` Phase 3.2.
`_GOAL_BOUNDS` in `g1_arm_env.py` reuses the old 5-DOF task's own reachability-validated
numeric box as a STARTING HYPOTHESIS (reasoning: position reach is governed mostly by
upper-arm/forearm length, which the wrist addition shouldn't change) — that's a
hypothesis, not a verified fact for this asset's actual kinematic chain, hence this
script. Run BEFORE trusting a goal box for a real training run — a hand-picked box in
the 23dof phase turned out only ~47% reachable, cost weeks before anyone checked (see
`lessons_learned.md`).

**Result, left arm, 2026-07-21** (see `validation/arm_reachability/left_summary.md`):
97.0% of the box covered within 2cm (the actual `goal_threshold`), mean nearest-reachable
distance 0.61cm — better than the 5-DOF box's own best score (~65% at 2cm, after two
rounds of reshaping). The reuse hypothesis held; no reshaping needed. Re-run for the
right arm and after any future change to the arm's joint limits or kinematic chain.

Method: samples a large number of random joint configurations within the arm's real
hardware limits (the same limits training/eval already clamp to), reads off where the
end-effector actually ends up, and builds a point cloud of the arm's true reachable
workspace — no RL policy involved at all, pure kinematics via the physics sim. Then
checks, for a dense grid of points inside `_GOAL_BOUNDS`, how close the nearest
reachable point is.

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    python validation/check_arm_reachability.py --arm left --headless

Output: printed coverage percentages at multiple tolerances, plus a coarse per-octant
breakdown, and a summary written to validation/arm_reachability/<arm>_summary.md.
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be launched before all other imports
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Kinematic reachability check for the G1 29dof arm policy's goal workspace.")
parser.add_argument("--arm", type=str, default="left", choices=["left", "right"], help="Which arm's workspace to check.")
parser.add_argument("--num_envs", type=int, default=4096, help="Parallel envs per sampling batch.")
parser.add_argument("--num_batches", type=int, default=100, help="Number of random-sample batches (total samples = num_envs * num_batches).")
parser.add_argument("--grid_res", type=int, default=14, help="Grid points per axis when checking goal-box coverage (grid_res^3 total).")
parser.add_argument("--seed", type=int, default=42, help="Fixed seed for reproducibility.")
parser.add_argument(
    "--locked_wrist", action="store_true",
    help="Check reachability for the G1-Arm-Left/Right-LockedWrist-v0 configuration "
    "(wrist_pitch/wrist_yaw held at default, 5 sampled joints instead of 7) instead of "
    "the standard 7-DOF arm. Added 2026-07-24 after that variant's eval_arm.py success "
    "rate collapsed to ~2-3% (vs. ~28% for the 7-DOF baseline) — this checks how much "
    "of that is the goal box becoming less reachable with 2 fewer DOF, vs. something "
    "else. Writes to a separate <arm>_lockedwrist_summary.md, does not overwrite the "
    "existing 7-DOF result.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything after sim is up
# ---------------------------------------------------------------------------
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

import g1_locomotion.tasks  # noqa: F401 — registers gym envs
from g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env import (
    _GOAL_BOUNDS,
    G1ArmEnv,
    G1ArmLeftEnvCfg_PLAY,
    G1ArmLeftLockedWristEnvCfg,
    G1ArmRightEnvCfg_PLAY,
)


def main():
    torch.manual_seed(args_cli.seed)

    if args_cli.locked_wrist:
        if args_cli.arm != "left":
            raise NotImplementedError("--locked_wrist is only registered for the left arm so far.")
        env_cfg = G1ArmLeftLockedWristEnvCfg()
    else:
        env_cfg = (G1ArmLeftEnvCfg_PLAY if args_cli.arm == "left" else G1ArmRightEnvCfg_PLAY)()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    # Disable everything not relevant to a pure kinematics check — smaller, faster, and
    # removes confounds (wobble/noise would just add jitter to the position readout).
    env_cfg.enable_root_wobble = False
    env_cfg.joint_pos_noise = 0.0
    env_cfg.joint_vel_noise = 0.0

    print(f"[Reachability] Building env (arm={args_cli.arm}, num_envs={args_cli.num_envs})...")
    env = G1ArmEnv(env_cfg)

    arm = env._arm_groups[0]
    jt = arm["joint_tensor"]
    ee_idx = arm["ee_idx"]
    n_joints = len(jt)  # 7

    device = env.device
    reachable_points = []

    with torch.inference_mode():
        obs, _ = env.reset()
        for b in range(args_cli.num_batches):
            # Sample uniformly random joint angles within the real hardware limits —
            # the same limits _apply_action/_get_rewards already clamp to, so this is
            # exactly the set of configurations the policy is allowed to reach.
            rand01 = torch.rand((env.num_envs, n_joints), device=device)
            lo = env._arm_hw_limits[:, 0].unsqueeze(0)
            hi = env._arm_hw_limits[:, 1].unsqueeze(0)
            joint_pos_sample = lo + rand01 * (hi - lo)

            full_joint_pos = env.robot.data.joint_pos.clone()
            full_joint_pos[:, jt] = joint_pos_sample
            joint_vel = torch.zeros_like(full_joint_pos)
            env.robot.write_joint_state_to_sim(full_joint_pos, joint_vel)
            # Hold the sampled pose as the position target too, so the PD controller
            # doesn't fight it back toward wherever it was before — target == actual,
            # nothing to correct, this is a pure kinematics readout not a dynamics test.
            env.robot.set_joint_position_target(joint_pos_sample, joint_ids=jt)
            env.filtered_actions[:] = 0.0
            env.previous_actions[:] = 0.0

            # A couple of zero-action steps to let cached body_pos_w refresh and any
            # self-collision contact response (if the sampled config self-intersects)
            # settle before reading the end-effector position.
            zero_actions = torch.zeros((env.num_envs, env.cfg.action_space), device=device)
            for _ in range(3):
                env.step(zero_actions)

            ee_pos_world = env.robot.data.body_pos_w[:, ee_idx, :]
            ee_pos_local = (ee_pos_world - env.scene.env_origins).cpu().numpy()
            reachable_points.append(ee_pos_local)

            if (b + 1) % 20 == 0:
                print(f"[Reachability] batch {b + 1}/{args_cli.num_batches}")

    env.close()

    cloud = np.concatenate(reachable_points, axis=0)  # (num_envs*num_batches, 3)
    print(f"[Reachability] Collected {cloud.shape[0]} reachable-workspace samples.")

    tree = cKDTree(cloud)

    bounds = _GOAL_BOUNDS[args_cli.arm]
    xs = np.linspace(*bounds["x"], args_cli.grid_res)
    ys = np.linspace(*bounds["y"], args_cli.grid_res)
    zs = np.linspace(*bounds["z"], args_cli.grid_res)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    nearest_dist_m, _ = tree.query(grid, k=1)
    nearest_dist_cm = nearest_dist_m * 100.0

    tolerances_cm = [2.0, 5.0, 10.0, 15.0]
    lines = [
        "# G1 29dof Arm Workspace Reachability Check",
        "",
        f"Arm: `{args_cli.arm}` — Goal bounds: `{bounds}` (starting hypothesis, reused "
        "numerically from the 23dof-era 5-DOF task — see g1_arm_env.py's module docstring)",
        f"Reachable-workspace samples: {cloud.shape[0]} (random joint configs within real hardware limits)",
        f"Goal-box grid points checked: {grid.shape[0]} ({args_cli.grid_res}^3)",
        "",
        "| Tolerance | Coverage (% of goal box within this distance of a reachable point) |",
        "|---|---|",
    ]
    print("\n[Reachability] Coverage of the goal workspace by the arm's actual reachable set:")
    for tol in tolerances_cm:
        coverage = float((nearest_dist_cm <= tol).mean()) * 100.0
        print(f"  within {tol:5.1f}cm: {coverage:5.1f}% of goal box covered")
        lines.append(f"| {tol:.1f}cm | {coverage:.1f}% |")

    print(f"\n[Reachability] Nearest-reachable-point distance over the goal grid: "
          f"mean={nearest_dist_cm.mean():.2f}cm  median={np.median(nearest_dist_cm):.2f}cm  "
          f"p90={np.percentile(nearest_dist_cm, 90):.2f}cm  max={nearest_dist_cm.max():.2f}cm")
    lines += [
        "",
        f"Nearest-reachable-point distance over the grid: mean={nearest_dist_cm.mean():.2f}cm, "
        f"median={np.median(nearest_dist_cm):.2f}cm, p90={np.percentile(nearest_dist_cm, 90):.2f}cm, "
        f"max={nearest_dist_cm.max():.2f}cm",
        "",
        "## Per-octant breakdown (split the box at its own x/y/z median)",
        "",
        "Which corner/region of the box is hardest to reach, if any — high mean distance",
        "here means that region is systematically far from anything the arm can reach.",
        "",
        "| x half | y half | z half | n points | mean dist (cm) | max dist (cm) |",
        "|---|---|---|---|---|---|",
    ]
    print("\n[Reachability] Per-octant breakdown (mean/max nearest-reachable distance):")
    xm, ym, zm = np.median(xs), np.median(ys), np.median(zs)
    for x_half, x_mask in (("low", grid[:, 0] < xm), ("high", grid[:, 0] >= xm)):
        for y_half, y_mask in (("low", grid[:, 1] < ym), ("high", grid[:, 1] >= ym)):
            for z_half, z_mask in (("low", grid[:, 2] < zm), ("high", grid[:, 2] >= zm)):
                mask = x_mask & y_mask & z_mask
                if not mask.any():
                    continue
                d = nearest_dist_cm[mask]
                print(f"  x={x_half:4s} y={y_half:4s} z={z_half:4s}  n={mask.sum():4d}  "
                      f"mean={d.mean():6.2f}cm  max={d.max():6.2f}cm")
                lines.append(f"| {x_half} | {y_half} | {z_half} | {mask.sum()} | {d.mean():.2f} | {d.max():.2f} |")

    out_dir = "validation/arm_reachability"
    os.makedirs(out_dir, exist_ok=True)
    suffix = "_lockedwrist" if args_cli.locked_wrist else ""
    out_path = os.path.join(out_dir, f"{args_cli.arm}{suffix}_summary.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[Reachability] Summary written to: {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
