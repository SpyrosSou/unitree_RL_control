# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal, standalone diagnostic (2026-07-23): does the walking policy produce real
world-frame displacement, independent of ALL of g1_full_demo.py's custom wiring?

Deliberately bypasses g1_full_demo.py entirely — reuses the exact same env cfg class
and command-pinning mechanism eval_walking.py already used to get its "0% fall, 0.601
m/s achieved at 0.6 commanded" result, but prints raw robot.data.root_pos_w every step,
so there is no ambiguity about whether the printed number reflects real translation.

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion
    python testing/general_testing/check_real_displacement.py \\
        --checkpoint logs/rsl_rl/walking/arm_disturbance/2026-07-22_23-30-26/model_5999.pt
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=500, help="500 @ 50Hz = 10s")
parser.add_argument("--forward_speed", type=float, default=0.6)
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
    # Matching eval_walking.py's _build_base_env() exactly this time (2026-07-23 —
    # first version of this script skipped these, which could easily have explained a
    # bad reading on its own, independent of whether the policy actually works):
    env_cfg.observations.policy.enable_corruption = True
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "base_external_force_torque"):
        env_cfg.events.base_external_force_torque = None
    base_velocity = env_cfg.commands.base_velocity
    if hasattr(base_velocity, "heading_command"):
        base_velocity.heading_command = False
    base_velocity.rel_standing_envs = 0.0
    # debug_vis=True (the class default) spawns a debug-arrow USD prim fetched from a
    # remote Nucleus/S3 asset path — with no local cache and a slow/unreachable network
    # path this hangs for up to a 300s timeout per env launch, even in --headless mode
    # (confirmed 2026-07-24 — see check_arm_disturbance_magnitude.py's identical fix).
    base_velocity.debug_vis = False

    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(base_env)

    agent_cfg = BasePPORunnerCfg()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(base_env.device))
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=base_env.device)

    command_term = base_env.command_manager.get_term("base_velocity")
    command_term.cfg.ranges.lin_vel_x = (args_cli.forward_speed, args_cli.forward_speed)
    command_term.cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_term.cfg.ranges.ang_vel_z = (0.0, 0.0)

    with torch.inference_mode():
        obs, _ = env.reset()

        start_pos = base_env.scene["robot"].data.root_pos_w[:, :2].clone()
        print(f"[Check] start pos (env 0): {start_pos[0].tolist()}")
        print(f"[Check] commanded forward speed: {args_cli.forward_speed} m/s, running {args_cli.steps} steps "
              f"({args_cli.steps / 50:.1f}s @ 50Hz)")

        robot = base_env.scene["robot"]
        leg_joint_ids, leg_joint_names = robot.find_joints([".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"])
        leg_joint_ids_t = torch.tensor(leg_joint_ids, dtype=torch.long, device=base_env.device)
        print(f"[Check] tracking {len(leg_joint_ids)} leg joints: {leg_joint_names}")

        for i in range(args_cli.steps):
            # Forced every step, not just once after reset() — a periodic resample
            # mid-test (base_velocity's own resampling_time_range) could re-roll
            # is_standing_env again otherwise, the exact bug found in g1_full_demo.py.
            command_term.is_standing_env[:] = False
            action = policy(obs)
            obs, _, _, _ = env.step(action)
            if i % 50 == 0:
                pos = base_env.scene["robot"].data.root_pos_w[:, :2]
                lin_vel = base_env.scene["robot"].data.root_lin_vel_b[:, 0]
                disp = (pos - start_pos).norm(dim=-1)
                leg_vel = robot.data.joint_vel[:, leg_joint_ids_t]
                leg_vel_mag = leg_vel.abs().mean(dim=-1)
                leg_pos = robot.data.joint_pos[:, leg_joint_ids_t]
                leg_pos_default = robot.data.default_joint_pos[:, leg_joint_ids_t]
                leg_dev = (leg_pos - leg_pos_default).abs().mean(dim=-1)
                print(f"[Check] step={i:4d}  env0 pos={pos[0].tolist()}  "
                      f"env0 disp_from_start={disp[0].item():.3f}m  env0 lin_vel_b_x={lin_vel[0].item():.3f}  "
                      f"env0 mean|leg_vel|={leg_vel_mag[0].item():.3f}rad/s  env0 mean|leg_dev_from_default|={leg_dev[0].item():.3f}rad")

        final_pos = base_env.scene["robot"].data.root_pos_w[:, :2]
        total_disp = (final_pos - start_pos).norm(dim=-1)
        expected_disp = args_cli.forward_speed * (args_cli.steps / 50.0)
        print()
        print(f"[Check] === RESULT ===")
        print(f"[Check] Expected displacement if walking perfectly: {expected_disp:.2f}m")
        for i in range(args_cli.num_envs):
            print(f"[Check] env {i}: actual displacement = {total_disp[i].item():.3f}m")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
