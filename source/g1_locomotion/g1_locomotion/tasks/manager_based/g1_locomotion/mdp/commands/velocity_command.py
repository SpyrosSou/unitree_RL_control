# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ported verbatim from ``unitree_rl_lab``, 2026-07-21 — see 29dof_implementation_plan.md.

2026-07-29: grew a real command class (``UniformLevelVelocityCommand``) to support
``zero_command_snap_threshold`` — previously this module was cfg-only (stock
``UniformVelocityCommand`` behavior, the cfg just added the ``limit_ranges`` field the
curricula clamp against)."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class UniformLevelVelocityCommand(UniformVelocityCommand):
    """``UniformVelocityCommand`` plus an optional exact-zero dead-zone snap.

    ADDED 2026-07-29 (standing package): with the stock sampler, exact (0,0,0) commands
    only ever come from the ``rel_standing_envs`` slice (2% in the base recipe), while
    sampled commands with norm just under 0.1 land in a contradictory-incentive band —
    ``track_lin_vel_xy`` still demands e.g. 0.08 m/s of motion there, while
    ``feet_contact_without_cmd`` (gate: norm < 0.1) simultaneously rewards planted feet
    and ``feet_gait`` (gate: norm > 0.1) is off. Snapping sub-threshold samples to exact
    zero removes that contradiction AND multiplies the training exposure of the literal
    zero-command case the ``stand_still`` eval (and the arm-integration use case)
    actually needs — policy_status.md's 2026-07-29 continuation-regression entry traces
    stand-still brittleness to exactly this rarely-sampled, never-anchored corner.
    Snapped envs also satisfy ``ArmMotionDisturbance``'s literal command-norm standing
    gate, so they receive arm-disturbance training too.

    ``zero_command_snap_threshold=0.0`` (the default) is a strict no-op — every
    pre-existing task keeps stock behavior.
    """

    cfg: UniformLevelVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        threshold = self.cfg.zero_command_snap_threshold
        if threshold <= 0.0:
            return
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        snap = torch.norm(self.vel_command_b[env_ids], dim=1) < threshold
        self.vel_command_b[env_ids[snap]] = 0.0


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformLevelVelocityCommand

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING

    # 2026-07-29: sampled commands with norm below this are snapped to exact (0,0,0) at
    # resample time — see UniformLevelVelocityCommand's docstring. 0.0 = off (stock).
    zero_command_snap_threshold: float = 0.0
