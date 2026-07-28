# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 29dof arm — residual RL on top of a numerical IK baseline. See
``g1_arm_residual_env.py``'s module docstring for the full design (Phase 4 of
``ik_arm_integration_plan.md``).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="G1-Arm-Residual-Left-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm_residual.g1_arm_residual_env:G1ArmResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_residual_env:G1ArmResidualLeftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmResidualLeftPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-Residual-Left-Play-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm_residual.g1_arm_residual_env:G1ArmResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_residual_env:G1ArmResidualLeftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmResidualLeftPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-Residual-Right-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm_residual.g1_arm_residual_env:G1ArmResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_residual_env:G1ArmResidualRightEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmResidualRightPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-Residual-Right-Play-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm_residual.g1_arm_residual_env:G1ArmResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_residual_env:G1ArmResidualRightEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmResidualRightPPORunnerCfg",
    },
)
