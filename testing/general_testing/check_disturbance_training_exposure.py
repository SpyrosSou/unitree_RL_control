# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verifies (not infers) that mdp.ArmMotionDisturbance's phase-mixing curriculum
actually reaches and applies high-phase disturbance under conditions representative of
late-stage training — built 2026-07-27 after finding that mdp.ArmMotionDisturbance
samples each env's disturbance phase ONCE per episode, at reset, weighted toward
whatever "frontier" common_step_counter allows at that moment (see events.py's
_sample_phase_mix). A short eval session starting fresh at common_step_counter=0 always
samples phase 0 (100% of the time, since frontier=0 there) and typically never resets
again — which is why check_standing_disturbance_diagnostic.py showed flat, ~zero
disturbance regardless of elapsed eval steps. That's a real gap in eval methodology, not
proof either way about what a full training run actually did.

This script closes that gap for verification purposes: manually sets
base_env.common_step_counter to a large value (matching where a real training run would
be, given its known total iteration count and num_steps_per_env=24) BEFORE calling
reset(), so _sample_phase_mix draws from the fully-unlocked frontier the same way
training's own many resets would have, once past the phase-boundary curriculum. With
enough parallel envs, this lets us directly measure: (a) does the phase distribution
across envs match the theoretical geometric weighting (confirms the sampling mechanism
itself is correct), and (b) does ACTUAL simulated joint motion (not just the internal
scripted target) reach the expected per-phase amplitude (confirms the disturbance is
physically applied, not just computed and dropped). This does not retroactively prove
what a specific already-completed training run's own random draws happened to sample —
no phase distribution was logged during training, so that's not recoverable — but it
does confirm the mechanism is capable of, and actually does, produce full-range
disturbance once a training run has passed the phase boundaries, which combined with
knowing how far past those boundaries a real run gets is the strongest verification
available without having logged it live.

Usage:
    conda activate isaac_g1_control
    cd ~/Elm/Code/g1_locomotion
    python testing/general_testing/check_disturbance_training_exposure.py \\
        --checkpoint chosen_checkpoints/walking_latest.pt \\
        --equivalent_common_step 624000 \\
        --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify disturbance phase-mixing under training-representative conditions.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64, help="Enough envs to see a real phase distribution.")
parser.add_argument(
    "--equivalent_common_step", type=int, default=624000,
    help="common_step_counter to force before reset — should be well past the training "
    "recipe's own phase boundaries (3000/15000/40000). Default matches walking_latest.pt's "
    "documented provenance: ~26000 total iterations (15998 base-recipe + 10000 warm-started) "
    "* num_steps_per_env=24.",
)
parser.add_argument("--steps", type=int, default=100, help="Steps to run after reset to measure actual applied motion.")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything after sim is up
# ---------------------------------------------------------------------------
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
    base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    base_velocity.ranges.ang_vel_z = (0.0, 0.0)

    # Use the REAL training recipe's own default phase boundaries (3000, 15000, 40000),
    # not the PLAY cfg's fast (100, 300, 600) override — this needs to represent what
    # training actually used, not the quick-visual-check variant.
    env_cfg.events.arm_motion_disturbance.params.pop("phase_step_boundaries", None)
    env_cfg.events.arm_motion_disturbance.params.pop("phase_step_offset", None)

    base_env = ManagerBasedRLEnv(cfg=env_cfg)
    wrapped_env = RslRlVecEnvWrapper(base_env)

    agent_cfg = BasePPORunnerCfg()
    runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=str(base_env.device))
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=base_env.device)

    robot = base_env.scene["robot"]
    arm_joint_ids = base_env._arm_motion_joint_ids
    default_arm_pos = robot.data.default_joint_pos[:, arm_joint_ids]

    # The manager framework instantiates class-based terms in place — cfg.func starts as
    # the class (mdp.ArmMotionDisturbance) and gets replaced with the actual instance at
    # prepare_terms() time, so this is how to reach live instance state (_episode_phase
    # etc.), not cfg.params (which only holds the constructor-time arguments).
    disturbance_term = base_env.event_manager.get_term_cfg("arm_motion_disturbance").func
    print(f"[Verify] Actual phase_step_boundaries in use: {disturbance_term._phase_step_boundaries}")

    with torch.inference_mode():
        # Force common_step_counter to a training-representative value BEFORE reset, so
        # _sample_phase_mix (called from within reset's event dispatch) computes its
        # frontier against this value, not the fresh-construction 0.
        base_env.common_step_counter = args_cli.equivalent_common_step
        obs, _ = wrapped_env.reset()

        sampled_phase = disturbance_term._episode_phase.clone()
        n = args_cli.num_envs
        phase_counts = {p: int((sampled_phase == p).sum().item()) for p in range(4)}
        print(f"[Verify] common_step_counter forced to {args_cli.equivalent_common_step} before reset.")
        print(f"[Verify] Sampled phase distribution across {n} envs: {phase_counts} "
              f"({ {p: f'{c/n:.1%}' for p, c in phase_counts.items()} })")
        expected_frontier = 3
        expected_weights = [0.5 ** (expected_frontier - i) for i in range(expected_frontier + 1)]
        expected_total = sum(expected_weights)
        expected_fracs = {i: w / expected_total for i, w in enumerate(expected_weights)}
        print(f"[Verify] Theoretical expected fractions (frontier=3): "
              f"{ {k: f'{v:.1%}' for k, v in expected_fracs.items()} }")

        num_arm_joints = len(arm_joint_ids)
        max_dev_by_phase = {p: torch.zeros(n, device=base_env.device) for p in range(4)}
        # Per-joint running max (not averaged across joints) — each arm joint does its
        # own independent random walk, so different joints hit their own ceiling at
        # different moments; averaging across joints before taking the time-max
        # systematically understates how close any single joint actually gets.
        max_dev_per_joint_by_phase = {p: torch.zeros(n, num_arm_joints, device=base_env.device) for p in range(4)}
        for _ in range(args_cli.steps):
            action = policy(obs)
            obs, _, _, _ = wrapped_env.step(action)
            actual_pos = robot.data.joint_pos[:, arm_joint_ids]
            per_joint_dev = (actual_pos - default_arm_pos).abs()
            actual_dev = per_joint_dev.mean(dim=-1)
            for p in range(4):
                mask = sampled_phase == p
                if bool(mask.any().item()):
                    max_dev_by_phase[p] = torch.where(
                        mask, torch.maximum(max_dev_by_phase[p], actual_dev), max_dev_by_phase[p]
                    )
                    mask2d = mask.unsqueeze(-1).expand(-1, num_arm_joints)
                    max_dev_per_joint_by_phase[p] = torch.where(
                        mask2d, torch.maximum(max_dev_per_joint_by_phase[p], per_joint_dev), max_dev_per_joint_by_phase[p]
                    )

    base_env.close()

    checkpoint_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    out_path = os.path.join(checkpoint_dir, "disturbance_training_exposure_summary.txt")
    lines = [
        f"Checkpoint: {args_cli.checkpoint}",
        f"Forced common_step_counter before reset: {args_cli.equivalent_common_step}",
        f"phase_step_boundaries in use: {disturbance_term._phase_step_boundaries}",
        f"num_envs: {n}  steps measured: {args_cli.steps}",
        "",
        f"Sampled phase distribution: {phase_counts} ({ {p: f'{c/n:.1%}' for p, c in phase_counts.items()} })",
        f"Theoretical expected (frontier=3): { {k: f'{v:.1%}' for k, v in expected_fracs.items()} }",
        "",
        "Actual simulated max |joint_pos - default| reached, by sampled phase "
        "(expected amplitude ceiling per phase: 0.00 / 0.24 / 0.45 / 0.70 rad):",
    ]
    for p in range(4):
        mask = sampled_phase == p
        count = int(mask.sum().item())
        if count == 0:
            lines.append(f"  phase {p}: 0 envs sampled this phase — no data")
            continue
        vals = max_dev_by_phase[p][mask]
        per_joint_vals = max_dev_per_joint_by_phase[p][mask]  # (count, num_arm_joints)
        lines.append(
            f"  phase {p} (n={count}): mean_across_joints_max={vals.mean().item():.4f} rad, "
            f"max_across_joints_max={vals.max().item():.4f} rad | "
            f"BEST SINGLE JOINT (any env, any joint): {per_joint_vals.max().item():.4f} rad | "
            f"mean of each env's own single-best joint: {per_joint_vals.max(dim=-1).values.mean().item():.4f} rad"
        )

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    for line in lines:
        print(f"[Verify] {line}")
    print(f"[Verify] Summary written to: {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
