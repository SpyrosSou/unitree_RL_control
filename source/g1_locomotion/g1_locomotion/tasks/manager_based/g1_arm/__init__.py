# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# ---------------------------------------------------------------------------
# Left arm  (train + play)
# ---------------------------------------------------------------------------
gym.register(
    id="G1-Arm-IK-Left-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmIKLeftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmIKLeftPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-IK-Left-Play-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmIKLeftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmIKLeftPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Left arm — overnight sweep experiments (2026-07-07), see known_issues.md.
# Each isolates exactly one change vs. G1-Arm-IK-Left-v0's baseline.
# ---------------------------------------------------------------------------
gym.register(
    id="G1-Arm-IK-Left-RewardShape-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmIKLeftRewardShapeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmIKLeftRewardShapePPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-IK-Left-GoalCurriculum-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmIKLeftGoalCurriculumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmIKLeftGoalCurriculumPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-IK-Left-WideNet-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmIKLeftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmIKLeftWideNetPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-IK-Left-Entropy-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmIKLeftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmIKLeftEntropyPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Right arm  (train + play)
# ---------------------------------------------------------------------------
gym.register(
    id="G1-Arm-IK-Right-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmIKRightEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmIKRightPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-IK-Right-Play-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmIKRightEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmIKRightPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Both arms  (train + play)
# ---------------------------------------------------------------------------
gym.register(
    id="G1-Arm-IK-Both-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmIKBothEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmIKBothPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-IK-Both-Play-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmIKEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmIKBothEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmIKBothPPORunnerCfg",
    },
)
