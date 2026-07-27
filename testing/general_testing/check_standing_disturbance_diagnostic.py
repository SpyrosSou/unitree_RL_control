# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-off instrumented diagnostic (2026-07-27) for the "no visible arm movement, small
steps/rotation while standing" observation in eval_walking.py's stand_still bucket.
Writes a per-step CSV instead of relying on terminal prints — avoids the stdout-
buffering-when-piped issue that's bitten this project before (see eval_arm_long_hold.py's
own history), and is easier to read back than scrollback either way.

Checks, per step, per env:
  - which arm-disturbance phase is currently active (computed the same way
    ArmMotionDisturbance._phase_index does, from common_step_counter and the PLAY cfg's
    own (100, 300, 600) boundaries)
  - arm joint deviation from default (mean |actual - default| over the arm joints) — is
    the disturbance mechanism actually moving the arms at all
  - base linear speed and mean |leg joint vel| — is the robot actually stepping/moving
    despite a zero commanded velocity (the "small steps" observation)

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion
    python testing/general_testing/check_standing_disturbance_diagnostic.py \\
        --checkpoint chosen_checkpoints/walking_latest.pt --headless

Output:
    <checkpoint_dir>/standing_disturbance_diagnostic.csv (one row per step, env 0 only)
    plus a short text summary printed AND written alongside it.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Instrumented standing/arm-disturbance diagnostic.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=800, help="800 @ 50Hz = 16s, covers phase 0->1->2->3 (settles by step 600).")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything after sim is up
# ---------------------------------------------------------------------------
import csv
import os

import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import g1_locomotion.tasks  # noqa: F401 — registers gym envs
from g1_locomotion.tasks.manager_based.g1_locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg
from g1_locomotion.tasks.manager_based.g1_locomotion.g1_locomotion_env_cfg import G1LocomotionArmDisturbanceEnvCfg_PLAY


def main():
    env_cfg = G1LocomotionArmDisturbanceEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
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

    # Confirm what phase boundaries this cfg actually carries at construction time —
    # written into the summary so there's no ambiguity about which values were live.
    phase_boundaries = env_cfg.events.arm_motion_disturbance.params.get("phase_step_boundaries")
    phase_offset = env_cfg.events.arm_motion_disturbance.params.get("phase_step_offset")

    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    wrapped_env = RslRlVecEnvWrapper(base_env)

    agent_cfg = BasePPORunnerCfg()
    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=str(base_env.device))
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=base_env.device)

    command_term = base_env.command_manager.get_term("base_velocity")
    command_term.cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_term.cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_term.cfg.ranges.ang_vel_z = (0.0, 0.0)

    robot = base_env.scene["robot"]
    arm_joint_ids = base_env._arm_motion_joint_ids
    arm_default = robot.data.default_joint_pos[:, arm_joint_ids]
    leg_joint_ids, _ = robot.find_joints([".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"])
    leg_joint_ids_t = torch.tensor(leg_joint_ids, dtype=torch.long, device=base_env.device)

    def phase_index(step: int) -> int:
        s = step + phase_offset
        for i, boundary in enumerate(phase_boundaries):
            if s < boundary:
                return i
        return len(phase_boundaries)

    rows = []
    with torch.inference_mode():
        obs, _ = wrapped_env.reset()
        command_term.is_standing_env[:] = False

        for step in range(args_cli.steps):
            command_term.is_standing_env[:] = False
            action = policy(obs)
            obs, _, _, _ = wrapped_env.step(action)

            arm_pos = robot.data.joint_pos[:, arm_joint_ids]
            arm_dev = (arm_pos - arm_default).abs().mean(dim=-1)
            leg_vel = robot.data.joint_vel[:, leg_joint_ids_t].abs().mean(dim=-1)
            base_speed = robot.data.root_lin_vel_b[:, :2].norm(dim=-1)

            rows.append({
                "step": step,
                "phase": phase_index(step),
                "env0_arm_dev_from_default_rad": arm_dev[0].item(),
                "env0_mean_leg_vel_rad_s": leg_vel[0].item(),
                "env0_base_speed_m_s": base_speed[0].item(),
            })

    base_env.close()

    checkpoint_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    csv_path = os.path.join(checkpoint_dir, "standing_disturbance_diagnostic.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "phase", "env0_arm_dev_from_default_rad",
                                                "env0_mean_leg_vel_rad_s", "env0_base_speed_m_s"])
        writer.writeheader()
        writer.writerows(rows)

    # Coarse per-phase summary, written alongside the raw CSV.
    summary_lines = [
        f"Checkpoint: {args_cli.checkpoint}",
        f"phase_step_boundaries={phase_boundaries}  phase_step_offset={phase_offset}",
        f"steps={args_cli.steps}  num_envs={args_cli.num_envs}",
        "",
    ]
    for phase in sorted(set(r["phase"] for r in rows)):
        phase_rows = [r for r in rows if r["phase"] == phase]
        n = len(phase_rows)
        mean_arm_dev = sum(r["env0_arm_dev_from_default_rad"] for r in phase_rows) / n
        mean_leg_vel = sum(r["env0_mean_leg_vel_rad_s"] for r in phase_rows) / n
        mean_base_speed = sum(r["env0_base_speed_m_s"] for r in phase_rows) / n
        summary_lines.append(
            f"phase {phase}: steps {phase_rows[0]['step']}-{phase_rows[-1]['step']} (n={n}) | "
            f"mean_arm_dev_from_default={mean_arm_dev:.4f} rad | "
            f"mean_leg_vel={mean_leg_vel:.4f} rad/s | mean_base_speed={mean_base_speed:.4f} m/s"
        )
    # First-second-vs-rest split, to separate "settling from a randomized reset pose"
    # from steady-state behavior.
    first_50 = rows[:50]
    rest = rows[50:]
    if rest:
        summary_lines += [
            "",
            f"first 1s (steps 0-49): mean_leg_vel="
            f"{sum(r['env0_mean_leg_vel_rad_s'] for r in first_50) / len(first_50):.4f} rad/s | "
            f"mean_base_speed={sum(r['env0_base_speed_m_s'] for r in first_50) / len(first_50):.4f} m/s",
            f"rest (steps 50-{args_cli.steps - 1}): mean_leg_vel="
            f"{sum(r['env0_mean_leg_vel_rad_s'] for r in rest) / len(rest):.4f} rad/s | "
            f"mean_base_speed={sum(r['env0_base_speed_m_s'] for r in rest) / len(rest):.4f} m/s",
        ]

    summary_path = os.path.join(checkpoint_dir, "standing_disturbance_diagnostic_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    for line in summary_lines:
        print(f"[Diag] {line}")
    print(f"[Diag] Raw CSV: {csv_path}")
    print(f"[Diag] Summary: {summary_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
