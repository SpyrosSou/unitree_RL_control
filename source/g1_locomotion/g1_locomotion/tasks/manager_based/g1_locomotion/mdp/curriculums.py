# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ported verbatim from ``unitree_rl_lab``, 2026-07-21 — see 29dof_implementation_plan.md.

Command-range curricula: ramp the commanded velocity range up as the policy's own
tracking reward shows it's keeping up, instead of a fixed range for the whole run.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    # FIXED 2026-07-22 (user request): was `mean(episode_sums) / env.max_episode_length_s`
    # — dividing by the FIXED full episode budget (20s) regardless of how long each env's
    # episode actually ran. With most episodes ending early via bad_orientation (falling),
    # that divided a small "reward accumulated since last reset" by the full 20s max,
    # making the ratio artificially tiny no matter how good tracking was while the robot
    # was actually upright — structurally blocking promotion as long as fall rate stayed
    # high, regardless of true tracking quality. Diagnosed as the reason this curriculum
    # never progressed past its starting tier in every run examined that day, including a
    # checkpoint that scored a perfect 0% eval fall rate — it was never actually testing
    # tracking quality, just accidentally gating on survival duration. Now divides each
    # env's accumulated reward by ITS OWN actual elapsed episode time
    # (env.episode_length_buf, already tracked by IsaacLab), giving a true per-step
    # reward rate regardless of how early that env's episode ended. clamp(min=1) avoids
    # a divide-by-zero for an env that reset on this exact step.
    elapsed_s = env.episode_length_buf[env_ids].clamp(min=1).float() * env.step_dt
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids] / elapsed_s)

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    # FIXED 2026-07-29: same normalization bug lin_vel_cmd_levels had (fixed there
    # 2026-07-22, see its comment above) — dividing by the fixed full episode budget
    # instead of each env's actual elapsed time structurally blocks promotion while
    # fall rate is high, regardless of true tracking quality. Fixed now so that IF this
    # term ever gets wired into a CurriculumCfg (it never has been — as of 2026-07-29
    # no recipe expands ang_vel_z past its (-0.1, 0.1) starting range, which is why
    # sustained turning is effectively untrained), it actually works.
    elapsed_s = env.episode_length_buf[env_ids].clamp(min=1).float() * env.step_dt
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids] / elapsed_s)

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
