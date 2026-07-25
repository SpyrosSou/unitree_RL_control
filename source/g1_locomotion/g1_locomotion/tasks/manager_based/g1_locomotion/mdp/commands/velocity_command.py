# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ported verbatim from ``unitree_rl_lab``, 2026-07-21 — see 29dof_implementation_plan.md."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
