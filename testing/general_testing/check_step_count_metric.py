# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal, standalone diagnostic (2026-07-28): why does
WalkingMetricsCsvWrapper._update_stepping_metrics report an implausible ~90% of ticks as
a "real step" (908/1000 for walking_latest.pt's own stand_still eval)? Prints the raw
contact-sensor fields (current_contact_time, current_air_time, last_air_time) and the
exact boolean logic used to count a step, every tick, for one env's two feet — so the
bug gets found from real data instead of guessed at again.

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion
    python testing/general_testing/check_step_count_metric.py \\
        --checkpoint chosen_checkpoints/walking_latest.pt --steps 200
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument(
    "--phase", type=int, default=None,
    help="Arm-disturbance phase to pin (0-3, 3=max). Default: unpinned/natural cycling — "
    "matches the exact condition the original buggy 908/1000 result was measured under "
    "(no --phases was passed to eval_walking.py for that run, and a fresh short session "
    "sticks at phase 0 the whole time regardless — see policy_status.md). Pass --phase 3 "
    "separately to also check under the disturbance condition actually visually inspected.",
)
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

_FOOT_BODY_NAME_PATTERN = ".*_ankle_roll_link"
_STEP_AIR_TIME_THRESHOLD_S = 0.05


def main():
    env_cfg = G1LocomotionArmDisturbanceEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 42
    env_cfg.observations.policy.enable_corruption = True
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "base_external_force_torque"):
        env_cfg.events.base_external_force_torque = None
    base_velocity = env_cfg.commands.base_velocity
    if hasattr(base_velocity, "heading_command"):
        base_velocity.heading_command = False
    # Matching eval_walking.py's _build_base_env() exactly (2026-07-28 fix — first
    # version of this script had this backwards, rel_standing_envs=1.0 +
    # is_standing_env=True, which is NOT the condition the buggy 908/1000 result was
    # actually measured under): rel_standing_envs=0.0, zero velocity achieved purely via
    # explicit ranges below, is_standing_env forced False every step.
    base_velocity.rel_standing_envs = 0.0
    base_velocity.debug_vis = False

    if args_cli.phase is not None and hasattr(env_cfg.events, "arm_motion_disturbance"):
        # Same mechanism eval_walking.py's --phases uses (see its phase sweep): a
        # phase_step_boundaries tuple of N zeros forces _phase_index() to always fall
        # through to len(boundaries) == N, regardless of common_step_counter.
        env_cfg.events.arm_motion_disturbance.params["phase_step_boundaries"] = tuple([0] * args_cli.phase)
        env_cfg.events.arm_motion_disturbance.params["phase_step_offset"] = 0

    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(base_env)

    agent_cfg = BasePPORunnerCfg()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=str(base_env.device))
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=base_env.device)

    command_term = base_env.command_manager.get_term("base_velocity")
    command_term.cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_term.cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_term.cfg.ranges.ang_vel_z = (0.0, 0.0)

    robot = base_env.scene["robot"]
    contact_sensor = base_env.scene.sensors["contact_forces"]
    foot_body_ids_robot, foot_names = robot.find_bodies(_FOOT_BODY_NAME_PATTERN)
    foot_body_ids_sensor, _ = contact_sensor.find_bodies(_FOOT_BODY_NAME_PATTERN)
    print(f"[Check] feet tracked: {foot_names}  robot_ids={foot_body_ids_robot}  sensor_ids={foot_body_ids_sensor}")
    print(f"[Check] control dt (env.step_dt): {base_env.step_dt}")

    step_count = 0
    with torch.inference_mode():
        obs, _ = env.reset()
        command_term.is_standing_env[:] = False

        for i in range(args_cli.steps):
            command_term.is_standing_env[:] = False
            action = policy(obs)
            obs, _, _, _ = env.step(action)

            last_air_time = contact_sensor.data.last_air_time[0, foot_body_ids_sensor]
            current_contact_time = contact_sensor.data.current_contact_time[0, foot_body_ids_sensor]
            current_air_time = contact_sensor.data.current_air_time[0, foot_body_ids_sensor]
            control_dt = base_env.step_dt

            # 2026-07-28 fix, found from this exact script's own first run: current_contact_time
            # reads 0 both right at touchdown AND every tick still fully airborne — must also
            # require > 0.0 to isolate the single genuine touchdown tick.
            just_touched_down = (current_contact_time > 0.0) & (current_contact_time <= control_dt)
            was_a_real_step = last_air_time > _STEP_AIR_TIME_THRESHOLD_S
            step_this_tick = bool((just_touched_down & was_a_real_step).any().item())
            if step_this_tick:
                step_count += 1

            if i < 40 or step_this_tick:
                line = (
                    f"[Check] step={i:4d}  "
                    f"last_air_time={last_air_time.tolist()}  "
                    f"current_contact_time={current_contact_time.tolist()}  "
                    f"current_air_time={current_air_time.tolist()}"
                )
                if step_this_tick:
                    line += (
                        f"  just_touched_down={just_touched_down.tolist()}  "
                        f"was_a_real_step={was_a_real_step.tolist()}  -> COUNTED"
                    )
                print(line)

    print()
    print(f"[Check] === RESULT: {step_count} steps counted over {args_cli.steps} ticks ===")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
