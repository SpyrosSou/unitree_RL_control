# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isolation test (2026-07-27, user request): does the IK+tau_ff stack work well with
the base genuinely FIXED (bolted to the world, no walking policy, no locomotion env at
all — the closest match in this repo to how `xr_teleoperate`'s own arm-only usage would
look)? Directly answers the open question from `ik_arm_integration_plan.md`'s Phase 2
notes: is the walking policy's active balancing (continuous weight-shifting even at zero
command) responsible for the achieved-vs-kinematic tracking gap, or is that gap (and the
occasional catastrophic kinematic solve, e.g. the ~221cm case) a property of the
solver/model alone, independent of our walking integration?

No `ManagerBasedRLEnv`, no loco policy, no `G1LocomotionEnvCfg_PLAY` — a minimal
`InteractiveScene` with the robot's `fix_root_link=True` (literally bolted to the
world), following the exact low-level pattern `validation/check_ik_accuracy.py` already
uses in this repo for the same kind of isolation (base fixed vs floating), but driving
`g1_locomotion.controllers.arm_ik.G1ArmIK` instead of that script's older analytic
`StandingArmIKReachDisturbance`. Uses `UNITREE_G1_29DOF_CFG` (this project's own,
deploy.yaml-matched actuator config, 40/10 arm gain already built in) rather than
`isaaclab_assets`' generic `G1_MINIMAL_CFG`.

Usage:
    conda activate isaac_g1_ik
    cd ~/Elm/Code/g1_locomotion

    python validation/eval_arm_ik_fixed_base.py --arm left --num_targets 15 --hold_steps 350 --headless

Reports the same style of real-vs-kinematic coverage as `eval_arm_ik_standing.py`, for
direct side-by-side comparison.
"""

# ---------------------------------------------------------------------------
# Isaac Sim must be launched before all other imports
# ---------------------------------------------------------------------------
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fixed-base (no walking) IK accuracy isolation test.")
parser.add_argument("--arm", type=str, default="left", choices=["left", "right"])
parser.add_argument("--num_targets", type=int, default=15)
parser.add_argument("--hold_steps", type=int, default=350)
parser.add_argument("--settle_steps", type=int, default=20, help="No walking policy to settle -- just a few steps for physics to init.")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything after sim is up
# ---------------------------------------------------------------------------
import csv
import datetime
import os

import numpy as np
import torch

from g1_locomotion.assets.robots.unitree import UNITREE_G1_29DOF_CFG
from g1_locomotion.controllers.arm_ik import ARM_JOINT_NAMES, G1ArmIK
from g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env import (
    _GOAL_BOUNDS,
    _LEFT_ARM_JOINTS,
    _LEFT_EE_BODY,
    _RIGHT_ARM_JOINTS,
    _RIGHT_EE_BODY,
)
from g1_locomotion.utils.eval_meta import write_eval_meta

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

ARM_MAX_JOINT_DELTA_PER_STEP = 0.06
_KINEMATIC_RETRY_M = 0.20
_KINEMATIC_HARD_REJECT_M = 0.30
_SIM_DT = 0.005
_DECIMATION = 4  # matches the loco task's 50Hz control rate, for comparability


@configclass
class _FixedBaseSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    robot: ArticulationCfg = None


def _make_robot_cfg() -> ArticulationCfg:
    robot_cfg = UNITREE_G1_29DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.spawn.articulation_props.fix_root_link = True
    robot_cfg.spawn.activate_contact_sensors = False
    return robot_cfg


def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    assert list(_LEFT_ARM_JOINTS) + list(_RIGHT_ARM_JOINTS) == ARM_JOINT_NAMES, (
        "g1_arm_env.py joint lists and arm_ik.py's ARM_JOINT_NAMES have diverged."
    )

    print("[FixedBase] Building fixed-base scene (no walking policy, no locomotion env)...")
    sim = SimulationContext(SimulationCfg(dt=_SIM_DT, render_interval=_DECIMATION, device=args_cli.device))
    scene_cfg = _FixedBaseSceneCfg(num_envs=1, env_spacing=2.5)
    scene_cfg.robot = _make_robot_cfg()
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot: Articulation = scene["robot"]
    device = str(sim.device)

    all_arm_ids, _ = robot.find_joints(list(_LEFT_ARM_JOINTS) + list(_RIGHT_ARM_JOINTS))
    all_arm_joint_ids_robot = torch.tensor(all_arm_ids, dtype=torch.long, device=device)
    l_ee_ids, _ = robot.find_bodies(_LEFT_EE_BODY)
    r_ee_ids, _ = robot.find_bodies(_RIGHT_EE_BODY)
    left_ee_body_id, right_ee_body_id = l_ee_ids[0], r_ee_ids[0]

    side = args_cli.arm
    other_ee_body_id = right_ee_body_id if side == "left" else left_ee_body_id
    active_ee_body_id = left_ee_body_id if side == "left" else right_ee_body_id
    n_per_arm = len(_LEFT_ARM_JOINTS)
    side_slice = slice(0, n_per_arm) if side == "left" else slice(n_per_arm, 2 * n_per_arm)

    arm_ik = G1ArmIK()
    default_joint_pos = robot.data.default_joint_pos.clone()

    def to_pelvis(world_pos: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor) -> np.ndarray:
        return quat_apply_inverse(root_quat, world_pos - root_pos).detach().cpu().numpy()

    def se3(p: np.ndarray) -> np.ndarray:
        tf = np.eye(4)
        tf[:3, 3] = p
        return tf

    retry_pending = [False]
    did_retry = [False]

    def step_sim(target_world: torch.Tensor | None):
        env_target = default_joint_pos[0:1, all_arm_joint_ids_robot].clone()
        tauff_full = torch.zeros(1, all_arm_joint_ids_robot.numel(), device=device)

        achieved_err = kinematic_err = None
        if target_world is not None:
            root_pos = robot.data.root_pos_w[0]
            root_quat = robot.data.root_quat_w[0]
            target_pelvis = to_pelvis(target_world, root_pos, root_quat)
            other_pelvis = to_pelvis(robot.data.body_pos_w[0, other_ee_body_id], root_pos, root_quat)
            left_pelvis = target_pelvis if side == "left" else other_pelvis
            right_pelvis = other_pelvis if side == "left" else target_pelvis

            current_q_full = robot.data.joint_pos[0, all_arm_joint_ids_robot].detach().cpu().numpy()
            sol_q, sol_tauff = arm_ik.solve_ik(se3(left_pelvis), se3(right_pelvis), current_q=current_q_full)
            kinematic_err = float(np.linalg.norm(arm_ik.fk_wrist(sol_q, side) - target_pelvis))

            if retry_pending[0]:
                retry_pending[0] = False
                if kinematic_err > _KINEMATIC_RETRY_M:
                    did_retry[0] = True
                    default_q_full = default_joint_pos[0, all_arm_joint_ids_robot].detach().cpu().numpy()
                    arm_ik.reset(default_q_full)
                    retry_q, retry_tauff = arm_ik.solve_ik(se3(left_pelvis), se3(right_pelvis), current_q=default_q_full)
                    retry_err = float(np.linalg.norm(arm_ik.fk_wrist(retry_q, side) - target_pelvis))
                    sol_q, sol_tauff, kinematic_err = retry_q, retry_tauff, retry_err

            sol_q_t = torch.tensor(sol_q, dtype=torch.float32, device=device)
            sol_slice = sol_q_t[side_slice]
            current = robot.data.joint_pos[0, all_arm_joint_ids_robot[side_slice]]

            if kinematic_err > _KINEMATIC_HARD_REJECT_M:
                env_target[:, side_slice] = current
            else:
                delta = (sol_slice - current).clamp(-ARM_MAX_JOINT_DELTA_PER_STEP, ARM_MAX_JOINT_DELTA_PER_STEP)
                env_target[:, side_slice] = current + delta
                tauff_full[0] = torch.tensor(sol_tauff, dtype=torch.float32, device=device)

            achieved_ee_w = robot.data.body_pos_w[0, active_ee_body_id]
            achieved_err = (target_world - achieved_ee_w).norm().item()

        robot.set_joint_position_target(env_target, joint_ids=all_arm_joint_ids_robot)
        robot.set_joint_effort_target(tauff_full, joint_ids=all_arm_joint_ids_robot)
        for _ in range(_DECIMATION):
            scene.write_data_to_sim()
            sim.step(render=not args_cli.headless)
            scene.update(_SIM_DT)
        return achieved_err, kinematic_err

    print(f"[FixedBase] Settling ({args_cli.settle_steps} steps)...")
    for _ in range(args_cli.settle_steps):
        step_sim(None)

    bounds = _GOAL_BOUNDS[side]
    rng = np.random.default_rng(args_cli.seed)
    targets_ground = np.stack([
        rng.uniform(*bounds["x"], args_cli.num_targets),
        rng.uniform(*bounds["y"], args_cli.num_targets),
        rng.uniform(*bounds["z"], args_cli.num_targets),
    ], axis=-1)

    # Fixed base: root never moves, so x/y need no per-step yaw rotation (identity
    # orientation, bolted in place). BUG FIXED 2026-07-27 (found via a real run that
    # showed EVERY target failing by a similar ~large margin -- the tell that this was
    # a systematic frame bug, not a genuine reach-difficulty finding): z in
    # _GOAL_BOUNDS is GROUND-referenced (absolute height above the floor), not
    # pelvis-referenced -- must add it to ground height, not root_pos0's own z (which
    # is the pelvis height, ~0.8m, and was silently stacking on top of the target's
    # already-absolute height).
    root_pos0 = robot.data.root_pos_w[0].clone()
    ground_z_w = scene.env_origins[0, 2].clone()

    rows = []
    print(f"[FixedBase] Sweeping {args_cli.num_targets} targets, side={side}, hold_steps={args_cli.hold_steps}")
    for i, tgt_ground in enumerate(targets_ground):
        target_world = torch.tensor(
            [root_pos0[0] + tgt_ground[0], root_pos0[1] + tgt_ground[1], ground_z_w + tgt_ground[2]],
            dtype=torch.float32, device=device,
        )

        did_retry[0] = False
        arm_ik.reset(robot.data.joint_pos[0, all_arm_joint_ids_robot].detach().cpu().numpy())
        retry_pending[0] = True

        achieved_err = kinematic_err = None
        for _ in range(args_cli.hold_steps):
            achieved_err, kinematic_err = step_sim(target_world)

        rows.append({
            "target_x": tgt_ground[0], "target_y": tgt_ground[1], "target_z": tgt_ground[2],
            "achieved_err_cm": achieved_err * 100.0, "kinematic_err_cm": kinematic_err * 100.0,
            "retried": int(did_retry[0]),
        })
        print(f"  [{i + 1}/{args_cli.num_targets}] target={tgt_ground.round(3)} "
              f"achieved={achieved_err * 100:.1f}cm kinematic={kinematic_err * 100:.1f}cm retried={did_retry[0]}")

    achieved = np.array([r["achieved_err_cm"] for r in rows])
    kinematic = np.array([r["kinematic_err_cm"] for r in rows])
    n = len(rows)

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("validation", "arm_ik_fixed_base", stamp)
    os.makedirs(out_dir, exist_ok=True)
    write_eval_meta(out_dir, args_cli, __file__)
    csv_path = os.path.join(out_dir, "goal_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_retried = sum(r["retried"] for r in rows)
    print(f"\n[FixedBase] ===== SUMMARY (side={side}, base=FIXED, n={n}) =====")
    print(f"targets that triggered the once-per-target retry: {n_retried}/{n}")
    for tol in (2.0, 5.0, 10.0, 15.0):
        print(f"  within {tol:4.0f}cm  real={100 * (achieved <= tol).mean():5.1f}%  "
              f"kinematic={100 * (kinematic <= tol).mean():5.1f}%")
    print(f"  real error:      mean={achieved.mean():.2f}cm  median={np.median(achieved):.2f}cm  max={achieved.max():.2f}cm")
    print(f"  kinematic error: mean={kinematic.mean():.2f}cm  median={np.median(kinematic):.2f}cm  max={kinematic.max():.2f}cm")
    print(f"[FixedBase] Per-goal CSV: {csv_path}")


if __name__ == "__main__":
    main()
    # See eval_arm_ik_standing.py's identical note: skip simulation_app.close() --
    # confirmed elsewhere in this session that Kit teardown can hang after it, and by
    # this point every side effect that matters is already on disk.
    os._exit(0)
