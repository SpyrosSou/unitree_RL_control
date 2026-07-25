# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot asset configurations owned by this project (as opposed to imported from
isaaclab_assets). Currently just the 29dof G1, ported from ``unitree_rl_lab`` since its
actuator gains are what the 29dof training recipe (``tasks/manager_based/
g1_locomotion``) and the real hardware's own ``deploy.yaml`` both agree on — see
``29dof_implementation_plan.md`` (repo root) for why this was chosen over
``isaaclab_assets``'s own generic ``G1_29DOF_CFG``.
"""
