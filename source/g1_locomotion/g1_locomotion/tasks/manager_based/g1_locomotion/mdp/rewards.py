# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for the G1 locomotion task.

Add project-specific reward terms here. All standard Isaac Lab locomotion rewards
are already available via ``isaaclab_tasks.manager_based.locomotion.velocity.mdp``
(imported in the parent env cfg through ``mdp/__init__.py``).

Example:
    def my_custom_reward(env: ManagerBasedRLEnv, ...) -> torch.Tensor:
        ...
"""

