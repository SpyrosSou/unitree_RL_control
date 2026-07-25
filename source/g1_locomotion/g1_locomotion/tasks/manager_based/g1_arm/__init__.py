# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""G1 29dof arm policy — see 29dof_implementation_plan.md Phase 3.

Task ids deliberately don't say "IK" (2026-07-21, user request) — the deployed policy is
pure RL joint-space control, no inverse kinematics involved. See g1_arm_env.py's module
docstring for the full rationale and what changed from the 23dof-era 5-DOF/"Arm-IK"-named
task this was adapted from.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# ---------------------------------------------------------------------------
# Left arm  (train + play)
# ---------------------------------------------------------------------------
gym.register(
    id="G1-Arm-Left-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-Left-Play-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftPPORunnerCfg",
    },
)

# 2026-07-24: wrist_pitch/wrist_yaw locked at default, 5 controlled joints instead of
# 7 — see G1ArmLeftLockedWristEnvCfg's own docstring.
gym.register(
    id="G1-Arm-Left-LockedWrist-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftLockedWristEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftLockedWristPPORunnerCfg",
    },
)

# 2026-07-24 walking-fix-style ablations for the arm task — see G1ArmLeftAblation*
# classes in g1_arm_env.py/agents/rsl_rl_ppo_cfg.py for what each isolates.
gym.register(
    id="G1-Arm-Left-Ablation-EntropyCoef-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftAblationEntropyCoefPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-Left-Ablation-ExpScale-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmLeftAblationExpScaleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmLeftAblationExpScalePPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Right arm  (train + play)
# ---------------------------------------------------------------------------
gym.register(
    id="G1-Arm-Right-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmRightEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmRightPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-Right-Play-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmRightEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmRightPPORunnerCfg",
    },
)

# ---------------------------------------------------------------------------
# Both arms  (train + play)
# ---------------------------------------------------------------------------
gym.register(
    id="G1-Arm-Both-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmBothEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmBothPPORunnerCfg",
    },
)

gym.register(
    id="G1-Arm-Both-Play-v0",
    entry_point="g1_locomotion.tasks.manager_based.g1_arm.g1_arm_env:G1ArmEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_arm_env:G1ArmBothEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1ArmBothPPORunnerCfg",
    },
)
