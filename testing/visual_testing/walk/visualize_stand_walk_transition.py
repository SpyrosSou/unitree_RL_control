# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visual-only stand<->walk transition check (2026-07-27) — no metrics logged, just
live-switches the commanded velocity from stand-still to a walking speed and back on a
small number of envs, so the transition moment itself can be watched directly, rather
than comparing two separately-run clips (one held-still, one held-walking) and having to
mentally stitch them together.

Small env count by default (4) so all of them fit comfortably in view at once — same
reasoning as check_real_displacement.py's default.

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion

    python testing/visual_testing/walk/visualize_stand_walk_transition.py \\
        --checkpoint chosen_checkpoints/walking_latest.pt --arm_disturbance
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visual stand<->walk transition check.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained walking checkpoint (.pt).")
parser.add_argument(
    "--arm_disturbance", action="store_true",
    help="Build the env from G1LocomotionArmDisturbanceEnvCfg_PLAY instead of the plain "
    "G1LocomotionEnvCfg_PLAY — pass this for a checkpoint trained on "
    "G1-Locomotion-Velocity-ArmDisturbance-v0 (every current walking checkpoint).",
)
parser.add_argument("--num_envs", type=int, default=4, help="Small on purpose — easy to watch all at once.")
parser.add_argument("--walk_speed", type=float, default=0.6, help="Forward speed (m/s) during the walk phase.")
parser.add_argument("--stand_steps", type=int, default=250, help="250 @ 50Hz = 5s standing.")
parser.add_argument("--walk_steps", type=int, default=500, help="500 @ 50Hz = 10s walking.")
parser.add_argument("--cycles", type=int, default=2, help="How many stand->walk->stand cycles to run.")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything after sim is up
# ---------------------------------------------------------------------------
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import g1_locomotion.tasks  # noqa: F401 — registers gym envs
from g1_locomotion.tasks.manager_based.g1_locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from g1_locomotion.tasks.manager_based.g1_locomotion.g1_locomotion_env_cfg import (
    G1LocomotionArmDisturbanceEnvCfg_PLAY,
    G1LocomotionEnvCfg_PLAY,
)


def main():
    env_cfg_cls = G1LocomotionArmDisturbanceEnvCfg_PLAY if args_cli.arm_disturbance else G1LocomotionEnvCfg_PLAY
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    # Matches eval_walking.py's _build_base_env() exactly — same reasoning: the _PLAY
    # config disables observation noise for clean visualization, but that's meaningfully
    # easier than real training/deployment, so restore it. Random pushes disabled so
    # what you see is the policy's own transition behavior, not push recovery.
    env_cfg.observations.policy.enable_corruption = True
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "base_external_force_torque"):
        env_cfg.events.base_external_force_torque = None
    base_velocity = env_cfg.commands.base_velocity
    if hasattr(base_velocity, "heading_command"):
        base_velocity.heading_command = False
    base_velocity.rel_standing_envs = 0.0
    base_velocity.debug_vis = False

    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    wrapped_env = RslRlVecEnvWrapper(base_env)

    agent_cfg = BasePPORunnerCfg()
    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=str(base_env.device))
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=base_env.device)

    command_term = base_env.command_manager.get_term("base_velocity")

    def set_cmd(vx: float, vy: float, wz: float):
        command_term.cfg.ranges.lin_vel_x = (vx, vx)
        command_term.cfg.ranges.lin_vel_y = (vy, vy)
        command_term.cfg.ranges.ang_vel_z = (wz, wz)

    with torch.inference_mode():
        set_cmd(0.0, 0.0, 0.0)
        obs, _ = wrapped_env.reset()
        # See eval_walking.py's own comment on this same line: reset() also re-rolls
        # is_standing_env, which force-zeroes the command for flagged envs every step
        # regardless of what set_cmd() just set — force it off so the commanded speed
        # actually takes effect for every env.
        command_term.is_standing_env[:] = False

        for cycle in range(args_cli.cycles):
            print(f"[Viz] Cycle {cycle + 1}/{args_cli.cycles}: STAND for {args_cli.stand_steps} steps "
                  f"({args_cli.stand_steps / 50:.1f}s)")
            set_cmd(0.0, 0.0, 0.0)
            for _ in range(args_cli.stand_steps):
                command_term.is_standing_env[:] = False
                action = policy(obs)
                obs, _, _, _ = wrapped_env.step(action)

            print(f"[Viz] Cycle {cycle + 1}/{args_cli.cycles}: WALK at {args_cli.walk_speed} m/s for "
                  f"{args_cli.walk_steps} steps ({args_cli.walk_steps / 50:.1f}s)")
            set_cmd(args_cli.walk_speed, 0.0, 0.0)
            for _ in range(args_cli.walk_steps):
                command_term.is_standing_env[:] = False
                action = policy(obs)
                obs, _, _, _ = wrapped_env.step(action)

        print("[Viz] Cycles complete — holding STAND indefinitely so you can keep watching. Ctrl+C to exit.")
        set_cmd(0.0, 0.0, 0.0)
        while simulation_app.is_running():
            command_term.is_standing_env[:] = False
            action = policy(obs)
            obs, _, _, _ = wrapped_env.step(action)

    base_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
