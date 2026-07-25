# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone diagnostic (2026-07-24): does mdp.ArmMotionDisturbance actually apply
disturbance of increasing magnitude across phases 0/1/2/3, both as a *scripted target*
(env._arm_motion_targets) and as *actual simulated joint motion* (robot.data.joint_pos)?

Built after finding and fixing a real bug where --pin_disturbance_phase silently had no
effect on an env's very first episode (see mdp/events.py's ArmMotionDisturbance.__init__
fix). That fix was about WHICH phase gets selected — this script checks the separate
question of whether the phases, once correctly selected, actually produce different,
real, physically-applied disturbance magnitudes at all.

Forces rel_standing_envs=1.0 (ArmMotionDisturbance only disturbs standing-commanded
envs — this must be near 1.0 or most envs would never get disturbed regardless of
phase). Drives legs with a trained walking checkpoint so the robot behaves sensibly
while this measures the arms specifically.

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion
    python testing/general_testing/check_arm_disturbance_magnitude.py \\
        --checkpoint logs/rsl_rl/walking/ablation_reward_weights/2026-07-24_03-13-16/model_4400.pt \\
        --phase 0
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--phase", type=int, required=True, choices=[0, 1, 2, 3])
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=300, help="300 @ 50Hz = 6s")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import g1_locomotion.tasks  # noqa: F401
from g1_locomotion.tasks.manager_based.g1_locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from g1_locomotion.tasks.manager_based.g1_locomotion.g1_locomotion_env_cfg import (
    G1LocomotionArmDisturbanceEnvCfg_PLAY,
)


def main():
    env_cfg = G1LocomotionArmDisturbanceEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42
    # ArmMotionDisturbance only disturbs standing-commanded envs (literal command norm
    # < threshold, not the is_standing_env flag — see events.py) — force ALL envs into
    # that regime so this measures disturbance on every env, not a lucky 2% slice.
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    # debug_vis=True (the class default) spawns a debug-arrow USD prim fetched from a
    # remote Nucleus/S3 asset path — with no local cache and a slow/unreachable network
    # path this hangs for up to a 300s timeout per env launch, even in --headless mode.
    # Not needed for this script (no viewport to look at the arrow in anyway).
    env_cfg.commands.base_velocity.debug_vis = False
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "base_external_force_torque"):
        env_cfg.events.base_external_force_torque = None

    n = args_cli.phase
    env_cfg.events.arm_motion_disturbance.params["phase_step_boundaries"] = tuple([0] * n)
    env_cfg.events.arm_motion_disturbance.params["phase_step_offset"] = 0

    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(base_env)

    agent_cfg = BasePPORunnerCfg()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(base_env.device))
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=base_env.device)

    robot = base_env.scene["robot"]
    arm_joint_ids = base_env._arm_motion_joint_ids
    default_arm_pos = robot.data.default_joint_pos[:, arm_joint_ids]

    print(f"[Check] Phase {n} pinned. Tracking {len(arm_joint_ids)} arm joints.")
    print(f"[Check] rel_standing_envs=1.0, num_envs={args_cli.num_envs}, steps={args_cli.steps}")

    with torch.inference_mode():
        obs, _ = env.reset()

        max_target_dev = torch.zeros(args_cli.num_envs, device=base_env.device)
        max_actual_dev = torch.zeros(args_cli.num_envs, device=base_env.device)

        for i in range(args_cli.steps):
            action = policy(obs)
            obs, _, _, _ = env.step(action)

            target_dev = (base_env._arm_motion_targets - default_arm_pos).abs().mean(dim=-1)
            actual_pos = robot.data.joint_pos[:, arm_joint_ids]
            actual_dev = (actual_pos - default_arm_pos).abs().mean(dim=-1)

            max_target_dev = torch.maximum(max_target_dev, target_dev)
            max_actual_dev = torch.maximum(max_actual_dev, actual_dev)

            if i % 50 == 0:
                print(
                    f"[Check] step={i:4d}  env0 mean|target_dev_from_default|={target_dev[0].item():.4f}rad  "
                    f"env0 mean|actual_dev_from_default|={actual_dev[0].item():.4f}rad"
                )

        print()
        print("[Check] === RESULT (max over the run, mean across envs) ===")
        print(f"[Check] scripted target max deviation from default: {max_target_dev.mean().item():.4f} rad "
              f"({torch.rad2deg(max_target_dev.mean()).item():.2f} deg)")
        print(f"[Check] actual simulated  max deviation from default: {max_actual_dev.mean().item():.4f} rad "
              f"({torch.rad2deg(max_actual_dev.mean()).item():.2f} deg)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
